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
# Official MARD 221-color card: the de-facto standard bead numbering in CN
# retail, so a chart/materials list can be shopped code-by-code. Data from
# github.com/HansBug/pindou-color-data (MIT), cross-validated across public
# sources; RGB values are screen references, not physical color guarantees.
BEAD_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("A1", (250, 245, 205)),
    ("A2", (252, 254, 214)),
    ("A3", (252, 255, 146)),
    ("A4", (247, 236, 92)),
    ("A5", (255, 228, 75)),
    ("A6", (253, 169, 81)),
    ("A7", (250, 140, 79)),
    ("A8", (249, 224, 69)),
    ("A9", (249, 156, 95)),
    ("A10", (244, 126, 54)),
    ("A11", (254, 219, 153)),
    ("A12", (253, 162, 118)),
    ("A13", (254, 198, 103)),
    ("A14", (248, 88, 66)),
    ("A15", (251, 246, 94)),
    ("A16", (254, 255, 151)),
    ("A17", (253, 225, 115)),
    ("A18", (252, 191, 128)),
    ("A19", (253, 126, 119)),
    ("A20", (249, 214, 110)),
    ("A21", (250, 227, 147)),
    ("A22", (237, 248, 120)),
    ("A23", (225, 201, 189)),
    ("A24", (243, 246, 169)),
    ("A25", (255, 215, 133)),
    ("A26", (254, 200, 50)),
    ("B1", (223, 241, 57)),
    ("B2", (100, 243, 67)),
    ("B3", (159, 246, 133)),
    ("B4", (95, 223, 52)),
    ("B5", (57, 225, 88)),
    ("B6", (100, 224, 164)),
    ("B7", (63, 174, 124)),
    ("B8", (29, 158, 84)),
    ("B9", (42, 80, 55)),
    ("B10", (154, 209, 186)),
    ("B11", (98, 112, 50)),
    ("B12", (26, 110, 61)),
    ("B13", (200, 232, 125)),
    ("B14", (172, 232, 76)),
    ("B15", (48, 83, 53)),
    ("B16", (192, 237, 156)),
    ("B17", (158, 179, 62)),
    ("B18", (230, 237, 79)),
    ("B19", (38, 183, 142)),
    ("B20", (202, 237, 207)),
    ("B21", (23, 98, 104)),
    ("B22", (10, 66, 65)),
    ("B23", (52, 59, 26)),
    ("B24", (232, 250, 166)),
    ("B25", (78, 132, 109)),
    ("B26", (144, 124, 53)),
    ("B27", (208, 224, 175)),
    ("B28", (158, 229, 187)),
    ("B29", (198, 223, 95)),
    ("B30", (227, 251, 177)),
    ("B31", (178, 230, 148)),
    ("B32", (146, 173, 96)),
    ("C1", (232, 255, 231)),
    ("C2", (171, 248, 254)),
    ("C3", (158, 224, 248)),
    ("C4", (68, 205, 251)),
    ("C5", (6, 171, 227)),
    ("C6", (84, 167, 233)),
    ("C7", (57, 119, 204)),
    ("C8", (15, 82, 189)),
    ("C9", (51, 73, 195)),
    ("C10", (61, 187, 227)),
    ("C11", (42, 222, 211)),
    ("C12", (30, 51, 78)),
    ("C13", (205, 231, 254)),
    ("C14", (214, 253, 252)),
    ("C15", (33, 197, 196)),
    ("C16", (24, 88, 162)),
    ("C17", (2, 209, 243)),
    ("C18", (33, 50, 68)),
    ("C19", (24, 134, 144)),
    ("C20", (26, 112, 169)),
    ("C21", (190, 221, 252)),
    ("C22", (107, 177, 187)),
    ("C23", (200, 226, 249)),
    ("C24", (126, 197, 249)),
    ("C25", (169, 232, 224)),
    ("C26", (66, 173, 209)),
    ("C27", (208, 222, 239)),
    ("C28", (189, 206, 237)),
    ("C29", (54, 74, 137)),
    ("D1", (172, 183, 239)),
    ("D2", (134, 141, 211)),
    ("D3", (54, 83, 175)),
    ("D4", (22, 44, 126)),
    ("D5", (179, 78, 198)),
    ("D6", (179, 123, 220)),
    ("D7", (135, 88, 169)),
    ("D8", (227, 210, 254)),
    ("D9", (214, 186, 245)),
    ("D10", (48, 26, 73)),
    ("D11", (188, 186, 226)),
    ("D12", (220, 153, 206)),
    ("D13", (181, 3, 143)),
    ("D14", (136, 40, 147)),
    ("D15", (47, 30, 142)),
    ("D16", (226, 228, 240)),
    ("D17", (199, 211, 249)),
    ("D18", (154, 100, 184)),
    ("D19", (216, 194, 217)),
    ("D20", (156, 52, 173)),
    ("D21", (148, 5, 149)),
    ("D22", (56, 57, 149)),
    ("D23", (250, 219, 248)),
    ("D24", (118, 138, 225)),
    ("D25", (73, 80, 194)),
    ("D26", (214, 198, 235)),
    ("E1", (246, 212, 203)),
    ("E2", (252, 193, 221)),
    ("E3", (246, 189, 232)),
    ("E4", (233, 99, 158)),
    ("E5", (241, 85, 159)),
    ("E6", (236, 64, 114)),
    ("E7", (198, 54, 116)),
    ("E8", (253, 219, 233)),
    ("E9", (229, 117, 199)),
    ("E10", (211, 57, 151)),
    ("E11", (247, 218, 212)),
    ("E12", (248, 147, 191)),
    ("E13", (181, 2, 106)),
    ("E14", (250, 212, 191)),
    ("E15", (245, 201, 202)),
    ("E16", (251, 244, 236)),
    ("E17", (247, 227, 236)),
    ("E18", (251, 203, 219)),
    ("E19", (246, 187, 209)),
    ("E20", (215, 198, 206)),
    ("E21", (192, 157, 164)),
    ("E22", (181, 139, 159)),
    ("E23", (147, 125, 138)),
    ("E24", (222, 190, 229)),
    ("F1", (255, 146, 128)),
    ("F2", (247, 61, 72)),
    ("F3", (239, 77, 62)),
    ("F4", (249, 43, 64)),
    ("F5", (227, 3, 40)),
    ("F6", (145, 54, 53)),
    ("F7", (145, 25, 50)),
    ("F8", (187, 1, 38)),
    ("F9", (224, 103, 122)),
    ("F10", (135, 70, 40)),
    ("F11", (111, 50, 29)),
    ("F12", (248, 81, 109)),
    ("F13", (244, 92, 69)),
    ("F14", (252, 173, 178)),
    ("F15", (213, 5, 39)),
    ("F16", (248, 192, 169)),
    ("F17", (232, 155, 125)),
    ("F18", (208, 126, 74)),
    ("F19", (190, 69, 74)),
    ("F20", (198, 148, 149)),
    ("F21", (242, 187, 198)),
    ("F22", (247, 195, 208)),
    ("F23", (236, 128, 109)),
    ("F24", (224, 157, 175)),
    ("F25", (232, 72, 84)),
    ("G1", (255, 228, 211)),
    ("G2", (252, 198, 172)),
    ("G3", (241, 196, 165)),
    ("G4", (220, 179, 135)),
    ("G5", (231, 179, 78)),
    ("G6", (243, 160, 20)),
    ("G7", (152, 80, 58)),
    ("G8", (75, 43, 28)),
    ("G9", (228, 182, 133)),
    ("G10", (218, 140, 66)),
    ("G11", (218, 200, 152)),
    ("G12", (254, 201, 147)),
    ("G13", (178, 113, 75)),
    ("G14", (139, 104, 76)),
    ("G15", (252, 249, 224)),
    ("G16", (242, 216, 193)),
    ("G17", (121, 84, 78)),
    ("G18", (255, 228, 214)),
    ("G19", (221, 125, 65)),
    ("G20", (165, 69, 47)),
    ("G21", (179, 133, 97)),
    ("H1", (251, 251, 251)),
    ("H2", (255, 255, 255)),
    ("H3", (180, 180, 180)),
    ("H4", (135, 135, 135)),
    ("H5", (70, 70, 72)),
    ("H6", (44, 44, 44)),
    ("H7", (1, 1, 1)),
    ("H8", (231, 214, 220)),
    ("H9", (239, 237, 238)),
    ("H10", (236, 234, 235)),
    ("H11", (205, 205, 205)),
    ("H12", (253, 246, 238)),
    ("H13", (244, 239, 209)),
    ("H14", (206, 215, 212)),
    ("H15", (152, 166, 166)),
    ("H16", (27, 18, 19)),
    ("H17", (240, 238, 239)),
    ("H18", (252, 255, 248)),
    ("H19", (242, 238, 229)),
    ("H20", (150, 160, 159)),
    ("H21", (255, 251, 225)),
    ("H22", (202, 202, 218)),
    ("H23", (155, 156, 148)),
    ("M1", (187, 198, 182)),
    ("M2", (144, 153, 148)),
    ("M3", (105, 126, 128)),
    ("M4", (224, 212, 188)),
    ("M5", (208, 203, 174)),
    ("M6", (176, 170, 134)),
    ("M7", (176, 167, 150)),
    ("M8", (174, 128, 130)),
    ("M9", (168, 135, 100)),
    ("M10", (198, 178, 187)),
    ("M11", (157, 118, 147)),
    ("M12", (100, 75, 81)),
    ("M13", (199, 146, 102)),
    ("M14", (195, 116, 99)),
    ("M15", (116, 125, 122)),
)
BEAD_PALETTE = tuple(rgb for _, rgb in BEAD_COLORS)
BEAD_CODE_BY_RGB = {rgb: code for code, rgb in BEAD_COLORS}
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

BEAD_CHART_CELL = 30  # chart cell size in px; codes need room to stay legible
BEAD_CHART_MIN_CELL = 20  # below this codes get unreadable; give up instead
BEAD_CHART_MAX_DIM = 6000  # cap for either chart side (grid extent, no pad)


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


def _nearest_bead_index(color: tuple[int, int, int] | list[float]) -> int:
    """Index of the closest bead-palette entry.

    Distance is green-weighted, (2dr)^2 + (4dg)^2 + (3db)^2: human vision
    resolves greens better than reds and blues, so holding the green channel
    tight picks more faithful skies and skin tones.

    Args:
        color: An RGB color (ints, or float cells while dithering).

    Returns:
        The palette index with the smallest weighted distance.
    """
    r, g, b = color
    return min(
        range(len(BEAD_PALETTE)),
        key=lambda i: (
            4 * (BEAD_PALETTE[i][0] - r) ** 2
            + 16 * (BEAD_PALETTE[i][1] - g) ** 2
            + 9 * (BEAD_PALETTE[i][2] - b) ** 2
        ),
    )


def _nearest_bead_color(
    color: tuple[int, int, int] | list[float],
) -> tuple[int, int, int]:
    """Quantize a color to the closest entry of the bead palette.

    Args:
        color: An RGB color (ints, or float cells while dithering).

    Returns:
        The palette color with the smallest weighted distance.
    """
    return BEAD_PALETTE[_nearest_bead_index(color)]


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


def bead_material_counts(art: AsciiArt) -> dict[str, int] | None:
    """Count beads per MARD code over the whole bead grid.

    Args:
        art: The bead grid (kind "bead").

    Returns:
        code -> bead count over all cells (background included), or None
        when any cell color is not on the palette (grid not quantized).
    """
    counts: dict[str, int] = {}
    for row in art.rows:
        for _, rgb in row:
            code = BEAD_CODE_BY_RGB.get(tuple(rgb))
            if code is None:
                return None
            counts[code] = counts.get(code, 0) + 1
    return counts


def bead_material_lines(counts: dict[str, int]) -> list[str]:
    """Format the materials list as user-facing caption lines.

    Args:
        counts: code -> bead count (from bead_material_counts).

    Returns:
        Lines with the totals, the top-5 consumers, and (when present) a
        note listing low-count colors worth buying as separate small bags.
    """
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = [f"材料清单：需要 {len(counts)} 种颜色，共 {total:,} 颗"]
    lines.append("高耗色：" + "、".join(f"{code}×{n:,}" for code, n in ranked[:5]))
    minor = [code for code, n in ranked if n <= 10]
    if minor:
        lines.append(f"另有 {len(minor)} 种少量色（每种不超过 10 颗），建议单独分装")
    return lines


def paint_bead_chart_png(art: AsciiArt, out_path: str | Path) -> str | None:
    """Paint the per-cell code chart: every bead square carries its MARD code.

    Args:
        art: The bead grid (kind "bead", palette-quantized).
        out_path: Destination PNG path.

    Returns:
        The destination path, or None when the grid is too large for a
        legible chart (the cell cannot shrink past BEAD_CHART_MIN_CELL).
    """
    m = max(art.width, art.height)
    cell = BEAD_CHART_CELL
    if m * cell > BEAD_CHART_MAX_DIM:
        cell = BEAD_CHART_MAX_DIM // m
    if cell < BEAD_CHART_MIN_CELL:
        return None
    font = _load_mono_font(max(9, int(cell * 0.42)))
    width = art.width * cell + BEAD_PAD * 2
    height = art.height * cell + BEAD_PAD * 2
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(art.rows):
        y = BEAD_PAD + r * cell
        for c, (_, rgb) in enumerate(row):
            x = BEAD_PAD + c * cell
            draw.rectangle(
                [x, y, x + cell - 1, y + cell - 1],
                fill=rgb,
                outline=(150, 150, 150),
            )
            code = BEAD_CODE_BY_RGB.get(tuple(rgb))
            if code is None:
                continue
            # Text color flips against the cell so codes stay readable on
            # both dark and light beads.
            luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            fg = (20, 20, 20) if luma > 140 else (245, 245, 245)
            tb = draw.textbbox((0, 0), code, font=font)
            draw.text(
                (
                    x + (cell - tb[2] + tb[0]) / 2 - tb[0],
                    y + (cell - tb[3] + tb[1]) / 2 - tb[1],
                ),
                code,
                fill=fg,
                font=font,
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
