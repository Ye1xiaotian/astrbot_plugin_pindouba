"""Pixel-to-character-art renderer.

Four render modes:
- bead: full-coverage square-pixel mosaic (the plugin default) - every grid
  cell is one solid square keeping its averaged source color.
- braille: 2x4-dot braille matrix (8 sub-pixels per char) with Floyd-Steinberg
  dithering - by far the most detail at an equal character count.
- shading: luminance mapped onto a character gradient with error diffusion.
- line: Sobel edge magnitude computed at 3x resolution, percentile threshold,
  then max-pooling into the character grid (keeps thin strokes alive).

Every cell keeps the averaged color of its source pixels so the PNG painter
can color characters (or individual braille dots) accordingly.
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

SHADING_CHARS = "@%#*+=-:. "
"""Char gradient from densest (darkest pixels) to lightest (brightest)."""
LINE_EDGE_CHAR = "#"
LINE_BG_CHAR = " "
BRAILLE_BASE = 0x2800
BRAILLE_DOTS = (
    # (dx, dy, bit) - standard braille dot layout inside a 2x4 cell
    (0, 0, 0x01),
    (0, 1, 0x02),
    (0, 2, 0x04),
    (1, 0, 0x08),
    (1, 1, 0x10),
    (1, 2, 0x20),
    (0, 3, 0x40),
    (1, 3, 0x80),
)
BEAD_PALETTE = (
    # Classic perler-bead style palette (56 colors): quantizing bead colors
    # to this set is what gives the mosaic its authentic handmade look.
    # Neutrals
    (255, 255, 255),  # pure white
    (250, 240, 225),  # warm white
    (236, 238, 236),  # white
    (216, 216, 214),  # light gray
    (160, 162, 158),  # gray
    (128, 130, 132),  # mid gray
    (105, 108, 106),  # dark gray
    (48, 49, 47),  # black
    # Red / pink / orange
    (230, 30, 38),  # red
    (150, 20, 30),  # dark red
    (146, 64, 44),  # brick
    (240, 98, 40),  # deep orange
    (255, 127, 39),  # orange
    (255, 180, 100),  # light orange
    (252, 142, 112),  # salmon
    (255, 152, 171),  # pink
    (236, 78, 150),  # magenta
    (236, 0, 140),  # hot pink
    # Yellow
    (255, 201, 14),  # yellow
    (255, 215, 0),  # gold
    (222, 190, 80),  # mustard
    (255, 234, 166),  # pale yellow
    # Brown / skin
    (111, 63, 40),  # dark brown
    (140, 86, 52),  # chestnut
    (159, 106, 63),  # brown
    (196, 98, 43),  # light brown
    (211, 157, 98),  # tan
    (188, 120, 80),  # deep tan
    (238, 190, 148),  # light skin
    (240, 160, 130),  # rosy skin
    (247, 213, 180),  # pale skin
    (252, 232, 214),  # very pale skin
    # Green
    (52, 92, 64),  # dark forest
    (0, 129, 54),  # green
    (84, 168, 70),  # light green
    (146, 220, 160),  # mint
    (178, 211, 125),  # pale green
    (128, 128, 66),  # olive
    (0, 168, 132),  # teal
    # Blue / cyan
    (30, 30, 60),  # midnight
    (20, 36, 86),  # navy
    (0, 110, 184),  # blue
    (0, 160, 233),  # azure
    (64, 170, 227),  # light blue
    (150, 208, 241),  # pale blue
    (210, 228, 242),  # ice blue
    (70, 110, 150),  # steel blue
    (108, 132, 190),  # periwinkle
    (70, 60, 180),  # indigo
    (0, 190, 200),  # cyan
    # Purple
    (90, 20, 90),  # plum
    (120, 70, 160),  # purple
    (138, 43, 226),  # violet
    (178, 118, 200),  # light purple
    (216, 178, 242),  # lavender
    (230, 190, 250),  # lilac
)
MONO_FONTS = (
    "consola.ttf",  # Windows (Consolas)
    "cour.ttf",  # Windows (Courier New)
    "DejaVuSansMono.ttf",  # Linux
    "Menlo.ttc",  # macOS
    "Monaco.ttf",  # macOS
    "Arial.ttf",  # generic fallback
)
BEAD_PITCH = 12  # square cell size in px
BEAD_PAD = 6  # canvas margin in px
MAX_ASPECT_RATIO = 6.0
"""Reject images more elongated than this (checked after cropping): the
max-rows clamp would squeeze them into an unrecognizably narrow strip."""


@dataclass
class AsciiArt:
    """A character-art grid; each cell keeps the averaged color of its pixels."""

    rows: list[list[tuple[str, tuple[int, int, int]]]]
    kind: str = "ascii"
    """ "ascii" (font-painted) or "braille" (dot-painted). """

    @property
    def width(self) -> int:
        """Grid width in characters (0 when empty)."""
        return len(self.rows[0]) if self.rows else 0

    @property
    def height(self) -> int:
        """Grid height in characters."""
        return len(self.rows)

    @property
    def text(self) -> str:
        """The art as a plain multi-line string (characters only)."""
        return "\n".join("".join(ch for ch, _ in row) for row in self.rows)


def render_ascii_art(
    image_path: str | Path,
    mode: str = "braille",
    width: int = 64,
    crop_box: tuple[float, float, float, float] | None = None,
    edge_percent: float = 8.0,
    max_rows: int = 110,
    bead_palette: bool = False,
    bead_dither: bool = False,
) -> AsciiArt:
    """Render an image file into a character grid.

    Args:
        image_path: Path to the source image.
        mode: "bead" (perler-bead mosaic, full coverage), "braille" (2x4 dot
            matrix + dithering), "shading" (luminance gradient + error
            diffusion), or "line" (Sobel edges, best for clean lineart).
        width: Target width in characters (beads for bead mode).
        crop_box: Optional (x1, y1, x2, y2) fractions in [0, 1] to crop the
            source image (e.g. a subject bounding box) before rendering.
        edge_percent: Line-mode adaptive threshold: keep the top `x`% of
            gradient magnitudes as edges.
        max_rows: Upper bound of grid rows; the art shrinks (aspect kept)
            when it would exceed this. Portrait photos need a generous cap
            so the requested width is not silently reduced.
        bead_palette: Bead mode only - quantize bead colors to the classic
            bead palette (handmade look, but colors deviate from the source).
        bead_dither: Bead mode only - diffuse the palette quantization error
            to neighboring cells (Floyd-Steinberg). Smooths gradients that
            would band; only meaningful together with bead_palette.

    Returns:
        The rendered AsciiArt grid.

    Raises:
        ValueError: On unsupported mode or degenerate image.
    """
    if mode not in ("bead", "braille", "shading", "line"):
        raise ValueError(f"unsupported render mode: {mode}")
    with Image.open(image_path) as src:
        img = src.convert("RGB")

    if crop_box:
        w, h = img.size
        left = max(0, int(crop_box[0] * w))
        top = max(0, int(crop_box[1] * h))
        right = min(w, int(crop_box[2] * w))
        bottom = min(h, int(crop_box[3] * h))
        if right - left >= 8 and bottom - top >= 8:
            img = img.crop((left, top, right, bottom))

    w, h = img.size
    if w < 2 or h < 2:
        raise ValueError("image too small to render")
    if h > w * MAX_ASPECT_RATIO or w > h * MAX_ASPECT_RATIO:
        raise ValueError(f"image aspect ratio too extreme ({w}x{h}); crop it and retry")

    if mode == "bead":
        return _render_bead(
            img,
            cols=max(8, width),
            max_rows=max_rows,
            palette=bead_palette,
            dither=bead_dither,
        )
    if mode == "braille":
        return _render_braille(img, cols=max(8, width), max_rows=max_rows)

    cols = max(8, min(width, w))
    # Char cells are ~2x taller than wide; compress rows to keep the aspect.
    rows = max(4, round(h / w * cols * 0.5))
    if rows > max_rows:
        rows = max_rows
        cols = max(8, round(rows / (h / w) / 0.5))

    if mode == "line":
        return _render_line(img, cols, rows, edge_percent)
    return _render_shading(img, cols, rows)


def paint_ascii_art_png(
    art: AsciiArt,
    out_path: str | Path,
    font_size: int = 14,
    color: bool = True,
    bg: tuple[int, int, int] = (18, 18, 18),
    fg: tuple[int, int, int] = (232, 232, 232),
) -> str:
    """Paint an AsciiArt grid onto a PNG.

    Bead art is drawn as a full-coverage square-pixel mosaic (each cell is
    one solid square); braille art as fixed-position dot matrices; other
    kinds are painted with a monospace font, one character per fixed cell,
    so alignment never depends on the chat client's font.

    Args:
        art: The character grid to paint.
        out_path: Destination PNG path.
        font_size: Monospace font size in pixels (ASCII kinds).
        color: When True each character/dot uses its source pixel color.
        bg: Canvas background color.
        fg: Character color used when `color` is False.

    Returns:
        The destination path (as given).
    """
    if art.kind == "bead":
        return _paint_bead_png(art, out_path, color=color, bg=bg, fg=fg)
    font = _load_mono_font(font_size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    cell_w = max(1, round(probe.textlength("M", font=font)))
    ascent, descent = font.getmetrics()
    cell_h = max(1, ascent + descent)

    pad = 10
    canvas = Image.new(
        "RGB",
        (art.width * cell_w + pad * 2, art.height * cell_h + pad * 2),
        bg,
    )
    draw = ImageDraw.Draw(canvas)

    if art.kind == "braille":
        dot_w = max(2.0, cell_w * 0.44)
        dot_h = max(2.0, cell_h * 0.18)
        for r, row in enumerate(art.rows):
            for c, (ch, rgb) in enumerate(row):
                bits = ord(ch) - BRAILLE_BASE
                if bits <= 0:
                    continue
                fill = rgb if color else fg
                for dx, dy, bit in BRAILLE_DOTS:
                    if bits & bit:
                        cx = pad + (c * 2 + dx + 0.5) * cell_w / 2
                        cy = pad + (r * 4 + dy + 0.5) * cell_h / 4
                        draw.rectangle(
                            [
                                cx - dot_w / 2,
                                cy - dot_h / 2,
                                cx + dot_w / 2,
                                cy + dot_h / 2,
                            ],
                            fill=fill,
                        )
    else:
        for r, row in enumerate(art.rows):
            y = pad + r * cell_h
            for c, (ch, rgb) in enumerate(row):
                if ch == " ":
                    continue
                draw.text(
                    (pad + c * cell_w, y), ch, fill=rgb if color else fg, font=font
                )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


# --------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------- #


def _render_bead(
    img: Image.Image,
    cols: int,
    max_rows: int,
    palette: bool = False,
    dither: bool = False,
) -> AsciiArt:
    """Render as a square-pixel mosaic: every grid cell is one colored square.

    Full coverage - background included - so the whole image stays visible.
    Cells sit on a square grid; each cell's color is the average of its
    source pixels, optionally quantized to the bead palette with
    Floyd-Steinberg error diffusion.

    Args:
        img: Source image (RGB).
        cols: Cell columns.
        max_rows: Upper bound of cell rows; the mosaic shrinks (aspect kept)
            when it would exceed this.
        palette: Quantize cell colors to the classic bead palette.
        dither: Palette only - push each cell's quantization error onto its
            yet-unvisited neighbors, so flat gradients blend between palette
            colors instead of banding.

    Returns:
        The bead AsciiArt grid (kind "bead"; chars are placeholders, the
        painter draws solid squares).
    """
    w, h = img.size
    rows = max(4, round(h / w * cols))
    if rows > max_rows:
        rows = max_rows
        cols = max(8, round(rows / (h / w)))
    small = img.resize((cols, rows), Image.LANCZOS)
    raw = small.tobytes()

    # Float RGB working copy: dithering errors accumulate past 0-255 before
    # the next cell's nearest-palette lookup.
    cells = [[raw[3 * i], raw[3 * i + 1], raw[3 * i + 2]] for i in range(cols * rows)]

    rows_out: list[list[tuple[str, tuple[int, int, int]]]] = []
    for r in range(rows):
        row: list[tuple[str, tuple[int, int, int]]] = []
        for c in range(cols):
            i = r * cols + c
            color = _nearest_bead_color(cells[i]) if palette else tuple(cells[i])
            row.append(("■", color))
            if not (palette and dither):
                continue
            err = [cells[i][k] - color[k] for k in range(3)]
            fanout = []
            if c + 1 < cols:
                fanout.append((i + 1, 7))
            if r + 1 < rows:
                base = i + cols
                if c > 0:
                    fanout.append((base - 1, 3))
                fanout.append((base, 5))
                if c + 1 < cols:
                    fanout.append((base + 1, 1))
            for j, weight in fanout:
                for k in range(3):
                    cells[j][k] += err[k] * weight / 16
        rows_out.append(row)
    return AsciiArt(rows=rows_out, kind="bead")


def _render_braille(img: Image.Image, cols: int, max_rows: int) -> AsciiArt:
    """Render via 2x4 braille dot matrix with Floyd-Steinberg dithering.

    Dots are square, so the dot grid samples the image at its native aspect
    (the 2x4 cell shape cancels the tall-character correction).

    Args:
        img: Source image (RGB).
        cols: Target width in braille characters (2 dots each).
        max_rows: Upper bound of character rows.

    Returns:
        The braille AsciiArt grid.
    """
    w, h = img.size
    dot_cols = cols * 2
    dot_rows = max(4, round(h / w * dot_cols))
    char_rows = -(-dot_rows // 4)  # ceil
    if char_rows > max_rows:
        scale = (max_rows * 4) / dot_rows
        dot_rows = max_rows * 4
        dot_cols = max(8, round(dot_cols * scale))
        char_rows = max_rows
    small = img.resize((dot_cols, dot_rows), Image.LANCZOS)
    gray = small.convert("L")
    luma = gray.tobytes()
    raw_rgb = small.tobytes()

    # Polarity: make the subject (the side differing from the border) the
    # "on" dots, then dither.
    invert = _border_mean(luma, dot_cols, dot_rows) >= 128
    onness = [255 - v for v in luma] if invert else list(luma)
    on = _dither_binary(onness, dot_cols, dot_rows)

    rows_out: list[list[tuple[str, tuple[int, int, int]]]] = []
    for cr in range(char_rows):
        row: list[tuple[str, tuple[int, int, int]]] = []
        for cc in range(cols):
            bits = 0
            rs = gs = bs = count = 0
            for dx, dy, bit in BRAILLE_DOTS:
                x, y = cc * 2 + dx, cr * 4 + dy
                if x >= dot_cols or y >= dot_rows:
                    continue
                i = y * dot_cols + x
                rs += raw_rgb[3 * i]
                gs += raw_rgb[3 * i + 1]
                bs += raw_rgb[3 * i + 2]
                count += 1
                if on[i]:
                    bits |= bit
            color = (
                (rs // count, gs // count, bs // count) if count else (232, 232, 232)
            )
            row.append((chr(BRAILLE_BASE + bits), color))
        rows_out.append(row)
    return AsciiArt(rows=rows_out, kind="braille")


def _render_shading(img: Image.Image, cols: int, rows: int) -> AsciiArt:
    """Render via luminance gradient mapping with error diffusion.

    Args:
        img: Source image (RGB).
        cols: Target grid width.
        rows: Target grid height.

    Returns:
        The shaded AsciiArt grid.
    """
    small = img.resize((cols, rows), Image.LANCZOS)
    luma = list(small.convert("L").tobytes())
    raw_rgb = small.tobytes()
    indices = _dither_char_indices(luma, cols, rows, len(SHADING_CHARS))

    rows_out: list[list[tuple[str, tuple[int, int, int]]]] = []
    for r in range(rows):
        row: list[tuple[str, tuple[int, int, int]]] = []
        for c in range(cols):
            i = r * cols + c
            color = (raw_rgb[3 * i], raw_rgb[3 * i + 1], raw_rgb[3 * i + 2])
            row.append((SHADING_CHARS[indices[i]], color))
        rows_out.append(row)
    return AsciiArt(rows=rows_out)


def _render_line(
    img: Image.Image, cols: int, rows: int, edge_percent: float
) -> AsciiArt:
    """Render via Sobel edges detected at 3x resolution and max-pooled.

    Detecting on an upscaled grid keeps 1-2px strokes alive; the percentile
    threshold adapts to each image instead of a magic constant.

    Args:
        img: Source image (RGB).
        cols: Target grid width.
        rows: Target grid height.
        edge_percent: Percentage of strongest-gradient pixels kept as edges.

    Returns:
        The line AsciiArt grid.
    """
    scale = 3
    big = img.resize((cols * scale, rows * scale), Image.LANCZOS).convert("L")
    blurred = big.filter(ImageFilter.GaussianBlur(1.2))
    kx = (-1, 0, 1, -2, 0, 2, -1, 0, 1)
    ky = (-1, -2, -1, 0, 0, 0, 1, 2, 1)
    gx = ImageChops.lighter(
        blurred.filter(ImageFilter.Kernel((3, 3), kx, scale=1)),
        blurred.filter(ImageFilter.Kernel((3, 3), tuple(-v for v in kx), scale=1)),
    )
    gy = ImageChops.lighter(
        blurred.filter(ImageFilter.Kernel((3, 3), ky, scale=1)),
        blurred.filter(ImageFilter.Kernel((3, 3), tuple(-v for v in ky), scale=1)),
    )
    mag = ImageChops.add(gx, gy).tobytes()

    hist = ImageChops.add(gx, gy).histogram()
    threshold = _percentile_threshold(hist, cols * rows * scale * scale, edge_percent)

    small = img.resize((cols, rows), Image.LANCZOS)
    raw_rgb = small.tobytes()

    rows_out: list[list[tuple[str, tuple[int, int, int]]]] = []
    for r in range(rows):
        row: list[tuple[str, tuple[int, int, int]]] = []
        for c in range(cols):
            edge = False
            for dy in range(scale):
                base = (r * scale + dy) * (cols * scale) + c * scale
                for dx in range(scale):
                    if mag[base + dx] >= threshold:
                        edge = True
                        break
                if edge:
                    break
            i = r * cols + c
            color = (raw_rgb[3 * i], raw_rgb[3 * i + 1], raw_rgb[3 * i + 2])
            row.append((LINE_EDGE_CHAR if edge else LINE_BG_CHAR, color))
        rows_out.append(row)
    return AsciiArt(rows=rows_out)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _nearest_bead_color(
    color: tuple[int, int, int] | list[float],
) -> tuple[int, int, int]:
    """Quantize a color to the closest entry of the bead palette.

    Distance is green-weighted, (2dr)^2 + (4dg)^2 + (3db)^2: human vision
    resolves greens better than reds and blues, so holding the green channel
    tight picks more faithful skies and skin tones.

    Args:
        color: An RGB color (ints, or float cells while dithering).

    Returns:
        The palette color with the smallest weighted distance.
    """
    r, g, b = color
    return min(
        BEAD_PALETTE,
        key=lambda p: 4 * (p[0] - r) ** 2 + 16 * (p[1] - g) ** 2 + 9 * (p[2] - b) ** 2,
    )


def _paint_bead_png(
    art: AsciiArt,
    out_path: str | Path,
    color: bool,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
) -> str:
    """Paint a square-pixel mosaic: every cell is one solid, full-pitch square.

    Squares tile the plane edge to edge, so no board color shows between
    cells; `bg` only remains visible in the canvas margin.

    Args:
        art: The bead grid (kind "bead").
        out_path: Destination PNG path.
        color: Use per-cell source colors (False paints all cells with fg).
        bg: Canvas margin color.
        fg: Cell color used when `color` is False.

    Returns:
        The destination path (as given).
    """
    width = art.width * BEAD_PITCH + BEAD_PAD * 2
    height = art.height * BEAD_PITCH + BEAD_PAD * 2
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(art.rows):
        y = BEAD_PAD + r * BEAD_PITCH
        for c, (_, rgb) in enumerate(row):
            x = BEAD_PAD + c * BEAD_PITCH
            draw.rectangle(
                [x, y, x + BEAD_PITCH - 1, y + BEAD_PITCH - 1],
                fill=rgb if color else fg,
            )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


def _dither_binary(values: list[int], w: int, h: int) -> list[int]:
    """Floyd-Steinberg dithering into 0/255 in place.

    Args:
        values: Per-pixel "on-ness" intensities (0-255, may exceed slightly).
        w: Grid width.
        h: Grid height.

    Returns:
        The same list quantized to 0/255.
    """
    for y in range(h):
        base = y * w
        for x in range(w):
            i = base + x
            old = values[i]
            new = 255 if old >= 128 else 0
            err = old - new
            values[i] = new
            if x + 1 < w:
                values[i + 1] += err * 7 // 16
            if y + 1 < h:
                below = i + w
                if x > 0:
                    values[below - 1] += err * 3 // 16
                values[below] += err * 5 // 16
                if x + 1 < w:
                    values[below + 1] += err // 16
    return values


def _dither_char_indices(luma: list[int], w: int, h: int, levels: int) -> list[int]:
    """Floyd-Steinberg error diffusion for multi-level char quantization.

    Args:
        luma: Per-pixel luminance (0-255); modified in place with errors.
        w: Grid width.
        h: Grid height.
        levels: Number of char gradient levels.

    Returns:
        Per-pixel char gradient indices (0..levels-1).
    """
    out = [0] * (w * h)
    for y in range(h):
        base = y * w
        for x in range(w):
            i = base + x
            v = luma[i]
            idx = v * (levels - 1) // 255
            out[i] = idx
            err = v - (idx * 255 // (levels - 1))
            if x + 1 < w:
                luma[i + 1] += err * 7 // 16
            if y + 1 < h:
                below = i + w
                if x > 0:
                    luma[below - 1] += err * 3 // 16
                luma[below] += err * 5 // 16
                if x + 1 < w:
                    luma[below + 1] += err // 16
    return out


def _border_mean(luma: bytes, w: int, h: int) -> float:
    """Mean luminance of the image border (used for polarity detection).

    Args:
        luma: Raw luminance bytes.
        w: Grid width.
        h: Grid height.

    Returns:
        The border mean luminance in 0-255.
    """
    vals = list(luma[:w]) + list(luma[(h - 1) * w :])
    for y in range(1, h - 1):
        vals.append(luma[y * w])
        vals.append(luma[y * w + w - 1])
    return sum(vals) / max(1, len(vals))


def _percentile_threshold(hist: list[int], total: int, percent: float) -> int:
    """Find the value keeping only the top `percent`% of the histogram.

    Args:
        hist: 256-bin histogram.
        total: Total pixel count.
        percent: Percentage of pixels to keep above the threshold.

    Returns:
        The threshold value in 0-255.
    """
    target = total * percent / 100
    acc = 0
    for v in range(255, -1, -1):
        acc += hist[v]
        if acc >= target:
            return v
    return 255


def _load_mono_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the first available monospace font, falling back to PIL default."""
    for name in MONO_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()
