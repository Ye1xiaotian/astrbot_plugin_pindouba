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

from renderer import (  # noqa: E402  (imports follow the Pillow availability check)
    BEAD_PAD,
    BEAD_PALETTE,
    BEAD_PITCH,
    AsciiArt,
    _nearest_bead_color,
    paint_ascii_art_png,
    render_ascii_art,
)


def make_test_image(path: Path, left: int = 0, right: int = 255, size=(100, 50)):
    """Half-dark / half-bright horizontal split image."""
    img = Image.new("L", size, right)
    img.paste(Image.new("L", (size[0] // 2, size[1]), left), (0, 0))
    img.convert("RGB").save(path)
    return path


def make_gradient_image(path: Path, size=(256, 64)):
    """Left-to-right black-to-white smooth gradient (dithering tests)."""
    row = Image.new("L", (size[0], 1))
    for x in range(size[0]):
        row.putpixel((x, 0), round(x / (size[0] - 1) * 255))
    row.resize(size, Image.NEAREST).convert("RGB").save(path)
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
                self.assertEqual(ch, "■")
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
        art = render_ascii_art(self.image, mode="bead", width=40, bead_palette=True)
        palette = set(BEAD_PALETTE)
        for row in art.rows:
            for _, color in row:
                self.assertIn(color, palette)

    def test_bead_dither_stays_in_palette(self):
        art = render_ascii_art(
            make_gradient_image(self.tmp / "grad.png"),
            mode="bead",
            width=40,
            bead_palette=True,
            bead_dither=True,
        )
        palette = set(BEAD_PALETTE)
        for row in art.rows:
            for _, color in row:
                self.assertIn(color, palette)

    def test_bead_dither_smooths_gradient(self):
        # A smooth gradient bands into few flat colors without dithering;
        # error diffusion blends neighboring palette colors instead, so the
        # dithered grid uses noticeably more distinct palette entries.
        image = make_gradient_image(self.tmp / "grad.png")

        def distinct_colors(**kw):
            art = render_ascii_art(image, mode="bead", width=40, **kw)
            return len({color for row in art.rows for _, color in row})

        plain = distinct_colors(bead_palette=True)
        dithered = distinct_colors(bead_palette=True, bead_dither=True)
        self.assertGreater(dithered, plain)

    def test_bead_dither_without_palette_is_noop(self):
        # Dithering only diffuses quantization error; without the palette
        # quantizer there is no error, so the flag must change nothing.
        image = make_gradient_image(self.tmp / "grad.png")
        plain = render_ascii_art(image, mode="bead", width=40)
        dithered = render_ascii_art(image, mode="bead", width=40, bead_dither=True)
        self.assertEqual(plain.rows, dithered.rows)

    def test_nearest_bead_color_weights_green(self):
        # (0, 0, 72) is euclidean-closest to navy (20, 36, 86), but the
        # green-weighted metric keeps green tight and picks midnight
        # (30, 30, 60) instead; exact palette hits must map to themselves.
        self.assertEqual(_nearest_bead_color((0, 0, 72)), (30, 30, 60))
        self.assertEqual(_nearest_bead_color((0, 129, 54)), (0, 129, 54))

    def test_bead_paint_produces_png(self):
        art = render_ascii_art(self.image, mode="bead", width=40)
        out = self.tmp / "bead.png"
        paint_ascii_art_png(art, out, color=True)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)

    def test_bead_paint_squares_tile_cells(self):
        # Full-pitch squares: every pixel inside a cell (not just its center)
        # carries the cell color; bg only remains in the canvas margin.
        art = render_ascii_art(self.image, mode="bead", width=40)
        out = self.tmp / "bead_squares.png"
        paint_ascii_art_png(art, out, color=True)
        with Image.open(out) as png:
            self.assertEqual(
                png.size,
                (
                    art.width * BEAD_PITCH + BEAD_PAD * 2,
                    art.height * BEAD_PITCH + BEAD_PAD * 2,
                ),
            )
            for r, c in [(10, 5), (10, 30)]:  # dark half / bright half cells
                expected = art.rows[r][c][1]
                for dx, dy in [(0, 0), (BEAD_PITCH - 1, BEAD_PITCH - 1)]:
                    px = png.getpixel(
                        (
                            BEAD_PAD + c * BEAD_PITCH + dx,
                            BEAD_PAD + r * BEAD_PITCH + dy,
                        )
                    )
                    self.assertEqual(px, expected)

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


class AspectGuardTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())

    def _save(self, size):
        img = Image.new("L", size, 128)
        path = self.tmp / f"img_{size[0]}x{size[1]}.png"
        img.convert("RGB").save(path)
        return path

    def test_extreme_portrait_rejected(self):
        with self.assertRaises(ValueError):
            render_ascii_art(self._save((40, 400)), mode="bead", width=40)

    def test_extreme_landscape_rejected(self):
        with self.assertRaises(ValueError):
            render_ascii_art(self._save((400, 40)), mode="bead", width=40)

    def test_crop_box_rescues_tall_image(self):
        # The guard applies after cropping: a tall screenshot whose subject
        # box cuts it back to a sane aspect still renders.
        art = render_ascii_art(
            self._save((40, 400)), mode="bead", width=40, crop_box=(0.0, 0.0, 1.0, 0.2)
        )
        self.assertGreater(art.width, 0)

    def test_borderline_aspect_passes(self):
        art = render_ascii_art(self._save((60, 360)), mode="bead", width=40)
        self.assertEqual(art.kind, "bead")


class AsciiArtTests(unittest.TestCase):
    def test_empty_grid(self):
        art = AsciiArt(rows=[])
        self.assertEqual((art.width, art.height, art.text), (0, 0, ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
