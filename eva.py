"""EVA-style renderer: the "one last image" treatment of a photo.

Everything EVA lives in this one file on purpose: renderer.py stays
untouched. The pipeline replicates the look popularized by the open
"one last image" generator (github.com/itorr/one-last-image, MIT) -
the "One Last Kiss" cover style:

1. luma + 3x3 box denoise;
2. line response = 128 + luma - 7x7 box blur (unsharp sketch);
3. level cut at dark=118/light=128 flattens it into dark strokes on
   a white field;
4. a chunky tone pass marks dark regions (quarter-scale blur, weight 80);
5. a diagonal gradient (yellow-orange -> red -> magenta-purple -> cyan)
   colors by inverted luminance: dark strokes/tones drink the gradient,
   light areas stay white;
6. a soften down/upscale pass finishes the print-like look.

No HUD frame, no pixel grid - the output is just the stylized image at
the source aspect ratio. Fully deterministic.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    from .renderer import MAX_ASPECT_RATIO
except (  # Fallback when the plugin dir is imported as a plain module (tests).
    ImportError
):
    from renderer import MAX_ASPECT_RATIO

# Gradient color stops (fraction -> sRGB), diagonal top-left to bottom-right.
_GRADIENT_STOPS = (
    (0.0, (251, 186, 48)),
    (0.4, (252, 114, 53)),
    (0.6, (252, 53, 78)),
    (0.7, (207, 54, 223)),
    (0.8, (55, 181, 217)),
    (1.0, (62, 182, 218)),
)
_DARK_CUT = 118  # level-cut floor; with _LIGHT_CUT this flattens the sketch
_LIGHT_CUT = 128
_SHADE_LIMIT = 108  # luma below this counts as a tone region
_SHADE_LIGHT = 80  # how dark tone regions paint


def render_eva_png(
    image_path: str | Path,
    out_path: str | Path,
    crop_box: tuple[float, float, float, float] | None = None,
    max_dim: int = 1280,
) -> str:
    """Render an image in the EVA "one last image" style: white paper,
    diagonal gradient-colored strokes and tones, softened.

    Args:
        image_path: Path to the source image.
        out_path: Destination PNG path.
        crop_box: Optional (x1, y1, x2, y2) fractions in [0, 1] to crop
            the source first.
        max_dim: Longest output side in px; larger images are scaled down,
            smaller ones keep their native size.

    Returns:
        The destination path (as given).

    Raises:
        ValueError: On degenerate image size or extreme aspect ratio,
            mirroring the bead renderer's guards.
    """
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
        raise ValueError(f"image aspect ratio too extreme ({w}x{h})")

    scale = min(1.0, max_dim / max(w, h))
    w, h = max(2, round(w * scale)), max(2, round(h * scale))
    img = img.resize((w, h), Image.LANCZOS)

    # Luma with a 3x3 box denoise; the 7x7 blur feeds the sketch pass.
    lum = img.convert("L").filter(ImageFilter.BoxBlur(1))
    blur7 = lum.filter(ImageFilter.BoxBlur(3))  # radius 3 == 7x7 box
    y = 128.0 + np.asarray(lum, dtype=np.float64) - np.asarray(blur7, dtype=np.float64)

    # Level cut: flatten the sketch into dark strokes on a white field.
    y = np.clip((y - _DARK_CUT) * (255.0 / (255 - _LIGHT_CUT - _DARK_CUT)), 0, 255)

    # Chunky tone pass: dark regions, blurred at quarter scale for the
    # cel-shading banding, painted up to _SHADE_LIGHT.
    tone = np.asarray(lum, dtype=np.float64) < _SHADE_LIMIT
    tone_img = Image.fromarray((tone * 255).astype(np.uint8))
    small = tone_img.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR)
    tone_smooth = (
        np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.float64) / 255.0
    )

    # Diagonal gradient color field.
    stops_f = np.array([f for f, _ in _GRADIENT_STOPS])
    stops_rgb = np.array([c for _, c in _GRADIENT_STOPS], dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    diag = ((xx / max(1, w - 1) + yy / max(1, h - 1)) / 2.0).ravel()
    grad = np.stack(
        [np.interp(diag, stops_f, stops_rgb[:, ch]) for ch in range(3)], axis=1
    ).reshape(h, w, 3)

    # Dark strokes/tones drink the gradient; light areas stay white.
    alpha = np.maximum(255.0 - y, tone_smooth * _SHADE_LIGHT)[..., None] / 255.0
    out = (1.0 - alpha) * 255.0 + alpha * grad

    # Soften pass: down then up scaling gives the print-like softness.
    softened = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    softened = softened.resize(
        (max(2, round(w / 1.4)), max(2, round(h / 1.3))), Image.BILINEAR
    ).resize((w, h), Image.LANCZOS)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    softened.save(out_path)
    return str(out_path)
