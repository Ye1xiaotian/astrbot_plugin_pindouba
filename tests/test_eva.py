"""Unit tests for the EVA-style renderer (need Pillow and numpy)."""

import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

try:
    from PIL import Image
except ImportError:  # Pillow missing; renderer cannot be tested.
    raise SystemExit("Pillow is required: pip install pillow")

from eva import render_eva_png  # noqa: E402  (after the Pillow check)


def make_gradient_image(path: Path, size=(256, 64)):
    """Left-to-right black-to-white smooth gradient."""
    row = Image.new("L", (size[0], 1))
    for x in range(size[0]):
        row.putpixel((x, 0), round(x / (size[0] - 1) * 255))
    row.resize(size, Image.NEAREST).convert("RGB").save(path)
    return path


class EvaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.image = make_gradient_image(self.tmp / "grad.png")

    def test_output_is_scaled_source_without_frame(self):
        # 1200x1600 scales to 960x1280 (max_dim 1280); the output IS the
        # image - no HUD frame, no extra margins.
        big = self.tmp / "big.png"
        Image.new("RGB", (1200, 1600), (128, 128, 128)).save(big)
        out = self.tmp / "native.png"
        render_eva_png(big, out)
        with Image.open(out) as png:
            self.assertEqual(png.size, (960, 1280))

    def test_light_areas_stay_white(self):
        bright = self.tmp / "bright.png"
        Image.new("RGB", (200, 100), (240, 240, 240)).save(bright)
        out = self.tmp / "white.png"
        render_eva_png(bright, out)
        with Image.open(out) as png:
            px = list(png.getdata())
            mean = sum(sum(p) for p in px) / (len(px) * 3)
            self.assertGreater(mean, 245)  # paper stays white

    def test_dark_areas_carry_the_diagonal_gradient(self):
        # Flat dark regions take a light gradient wash; the hue still
        # tracks the diagonal: warm (R > B) top-left, cool (B > R)
        # bottom-right.
        dark = self.tmp / "dark.png"
        Image.new("RGB", (200, 100), (10, 10, 10)).save(dark)
        out = self.tmp / "grad.png"
        render_eva_png(dark, out)
        with Image.open(out) as png:
            w, h = png.size
            tl = png.getpixel((4, 4))
            br = png.getpixel((w - 5, h - 5))
            self.assertGreater(tl[0] - tl[2], 30)  # warm stop
            self.assertGreater(br[2] - br[0], 30)  # cool stop

    def test_render_is_deterministic(self):
        a, b = self.tmp / "a.png", self.tmp / "b.png"
        render_eva_png(self.image, a)
        render_eva_png(self.image, b)
        self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_aspect_guard_raises(self):
        tall = self.tmp / "tall.png"
        Image.new("RGB", (40, 400), (128, 128, 128)).save(tall)
        with self.assertRaises(ValueError):
            render_eva_png(tall, self.tmp / "out.png")

    def test_crop_box_changes_output(self):
        half = self.tmp / "half.png"
        Image.new("RGB", (200, 100), (20, 200, 20)).save(half)
        full = self.tmp / "full.png"
        cropped = self.tmp / "cropped.png"
        render_eva_png(half, full)
        render_eva_png(half, cropped, crop_box=(0.0, 0.0, 0.5, 1.0))
        with Image.open(full) as f, Image.open(cropped) as c:
            self.assertNotEqual(f.size, c.size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
