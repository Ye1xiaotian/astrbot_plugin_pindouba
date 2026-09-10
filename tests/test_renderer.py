"""Unit tests for the pixel-mapping renderer (need Pillow)."""

import sys
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

try:
    from PIL import Image
except ImportError:  # Pillow missing; renderer cannot be tested.
    raise SystemExit("Pillow is required: pip install pillow")

from renderer import BEAD_PALETTE, AsciiArt, paint_ascii_art_png, render_ascii_art


def make_test_image(path: Path, left: int = 0, right: int = 255, size=(100, 50)):
    """Half-dark / half-bright horizontal split image."""
    img = Image.new("L", size, right)
    img.paste(Image.new("L", (size[0] // 2, size[1]), left), (0, 0))
    img.convert("RGB").save(path)
    return path


class ShadingTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        self.image = make_test_image(self.tmp / "split.png")

    def test_aspect_correction(self):
        art = render_ascii_art(self.image, mode="shading", width=40)
        # 100x50 image at width 40 -> rows = 50/100 * 40 * 0.5 = 10
        self.assertEqual((art.width, art.height), (40, 10))

    def test_dark_side_maps_to_dense_char(self):
        art = render_ascii_art(self.image, mode="shading", width=40)
        self.assertEqual(art.rows[5][5][0], "@")  # dark half center
        self.assertEqual(art.rows[5][34][0], " ")  # bright half center

    def test_text_property_joins_rows(self):
        art = render_ascii_art(self.image, mode="shading", width=40)
        self.assertEqual(len(art.text.splitlines()), 10)

    def test_max_rows_clamps_and_keeps_aspect(self):
        art = render_ascii_art(self.image, mode="shading", width=80, max_rows=8)
        self.assertEqual(art.height, 8)
        self.assertLessEqual(art.width, 80)

    def test_crop_box_limits_region(self):
        # Crop to the dark half only; every sampled cell must stay dark.
        art = render_ascii_art(
            self.image, mode="shading", width=40, crop_box=(0.0, 0.0, 0.5, 1.0)
        )
        self.assertEqual(art.width, 40)
        for row in art.rows:
            for ch, _ in row:
                self.assertEqual(ch, "@")


class LineTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        img = Image.new("L", (100, 100), 255)
        img.paste(Image.new("L", (40, 40), 0), (30, 30))
        img.convert("RGB").save(self.tmp / "square.png")
        self.image = self.tmp / "square.png"

    def test_line_mode_marks_edges(self):
        art = render_ascii_art(self.image, mode="line", width=40)
        text = art.text
        self.assertIn("#", text)
        self.assertIn(" ", text)

    def test_unsupported_mode_raises(self):
        with self.assertRaises(ValueError):
            render_ascii_art(self.image, mode="nope")


class BrailleTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        img = Image.new("L", (100, 100), 255)
        img.paste(Image.new("L", (40, 40), 0), (30, 30))
        img.convert("RGB").save(self.tmp / "square.png")
        self.image = self.tmp / "square.png"

    def test_braille_dims_double_resolution(self):
        art = render_ascii_art(self.image, mode="braille", width=40)
        # dot grid 80x80 -> char grid 40x20 (2x4 dots per char)
        self.assertEqual((art.width, art.height), (40, 20))
        self.assertEqual(art.kind, "braille")

    def test_braille_dark_subject_on_bright_border(self):
        art = render_ascii_art(self.image, mode="braille", width=40)
        # center of the dark square: every dot on -> 0xFF
        self.assertEqual(ord(art.rows[10][20][0]), 0x28FF)
        # far outside corner: blank braille
        self.assertEqual(ord(art.rows[1][1][0]), 0x2800)

    def test_braille_text_output(self):
        art = render_ascii_art(self.image, mode="braille", width=40)
        self.assertEqual(len(art.text.splitlines()), art.height)

    def test_braille_paint_produces_png(self):
        art = render_ascii_art(self.image, mode="braille", width=40)
        out = self.tmp / "braille.png"
        paint_ascii_art_png(art, out, color=True)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)

    def test_braille_max_rows_clamps(self):
        art = render_ascii_art(self.image, mode="braille", width=40, max_rows=8)
        self.assertEqual(art.height, 8)


class BeadTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        self.image = make_test_image(self.tmp / "split.png")

    def test_bead_dims_square_grid(self):
        art = render_ascii_art(self.image, mode="bead", width=40)
        # square beads: rows = h/w * cols = 50/100 * 40 = 20
        self.assertEqual((art.width, art.height), (40, 20))
        self.assertEqual(art.kind, "bead")

    def test_bead_full_coverage(self):
        art = render_ascii_art(self.image, mode="bead", width=40)
        for row in art.rows:
            for ch, color in row:
                self.assertEqual(ch, "●")
                self.assertGreaterEqual(color[0], 0)

    def test_bead_bright_side_is_bright(self):
        # Full coverage means the bright half keeps bright bead colors
        # (no polarity): RGB of white side stays near 255.
        art = render_ascii_art(self.image, mode="bead", width=40)
        _, (r, g, b) = art.rows[10][30]
        self.assertGreater(min(r, g, b), 200)

    def test_bead_default_keeps_true_colors(self):
        # No palette quantization by default: pure white stays pure white.
        art = render_ascii_art(self.image, mode="bead", width=40)
        _, (r, g, b) = art.rows[10][30]
        self.assertEqual((r, g, b), (255, 255, 255))

    def test_bead_colors_quantized_to_palette(self):
        art = render_ascii_art(
            self.image, mode="bead", width=40, bead_palette=True
        )
        palette = set(BEAD_PALETTE)
        for row in art.rows:
            for _, color in row:
                self.assertIn(color, palette)

    def test_bead_paint_produces_png(self):
        art = render_ascii_art(self.image, mode="bead", width=40)
        out = self.tmp / "bead.png"
        paint_ascii_art_png(art, out, color=True)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)

    def test_bead_max_rows_clamps(self):
        art = render_ascii_art(self.image, mode="bead", width=80, max_rows=10)
        self.assertEqual(art.height, 10)
        self.assertLessEqual(art.width, 80)


class PaintTests(unittest.TestCase):
    def test_paint_produces_png(self):
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        art = render_ascii_art(
            make_test_image(tmp / "src.png"), mode="shading", width=20
        )
        out = tmp / "out.png"
        paint_ascii_art_png(art, out, color=True)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)
        with Image.open(out) as rendered:
            self.assertEqual(rendered.format, "PNG")


class AsciiArtTests(unittest.TestCase):
    def test_empty_grid(self):
        art = AsciiArt(rows=[])
        self.assertEqual((art.width, art.height, art.text), (0, 0, ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
