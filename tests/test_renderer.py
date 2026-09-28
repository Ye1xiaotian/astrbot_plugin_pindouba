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
    BEAD_CODE_BY_RGB,
    BEAD_COLORS,
    BEAD_PAD,
    BEAD_PALETTE,
    BEAD_PITCH,
    AsciiArt,
    _ciede2000,
    _nearest_bead_color,
    _nearest_bead_index,
    _srgb_to_lab,
    bead_material_counts,
    bead_material_lines,
    merge_minor_bead_colors,
    paint_ascii_art_png,
    paint_bead_chart_png,
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

    def test_nearest_bead_color_contract(self):
        # Exact palette hits map to themselves, the color/index helpers stay
        # consistent, and quantized results are always palette members.
        probes = [(0, 0, 72), (250, 30, 90), (12.0, 200.5, 33.0), (255, 255, 254)]
        for probe in probes:
            idx = _nearest_bead_index(probe)
            self.assertEqual(_nearest_bead_color(probe), BEAD_PALETTE[idx])
        for rgb in BEAD_PALETTE[:10]:
            self.assertEqual(_nearest_bead_color(rgb), rgb)

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


class Ciede2000Tests(unittest.TestCase):
    def test_sharma_reference_vectors(self):
        # Official CIEDE2000 test pairs (Sharma, Wu, Dalal 2005, table 1).
        pairs = [
            ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
            ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
            ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
            ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
            ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
            ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
            ((50.0, 2.5, 0.0), (58.0, 24.0, 15.0), 19.4535),
        ]
        for lab1, lab2, expected in pairs:
            self.assertAlmostEqual(_ciede2000(lab1, lab2), expected, places=3)

    def test_srgb_to_lab_anchor_colors(self):
        L, a, b = _srgb_to_lab((255, 255, 255))
        self.assertAlmostEqual(L, 100.0, places=1)
        self.assertAlmostEqual(a, 0.0, places=1)
        self.assertAlmostEqual(b, 0.0, places=1)
        L, a, b = _srgb_to_lab((255, 0, 0))
        self.assertAlmostEqual(L, 53.24, places=1)
        self.assertAlmostEqual(a, 80.09, places=1)
        self.assertAlmostEqual(b, 67.20, places=1)
        L, a, b = _srgb_to_lab((0, 0, 0))
        self.assertAlmostEqual(L, 0.0, places=1)

    def test_nearest_index_clamps_out_of_range_dither_cells(self):
        # Dithering can push accumulated error past 0-255; the lookup must
        # clamp instead of raising on a fractional power.
        idx = _nearest_bead_index((-4.2, 260.5, 128.0))
        self.assertTrue(0 <= idx < len(BEAD_PALETTE))

    def test_numpy_path_matches_exhaustive_scalar_ciede2000(self):
        # The vectorized lookup must pick the same palette entry as a
        # brute-force scalar CIEDE2000 scan over all 221 entries; the
        # scalar oracle is built independently of the array path.
        import random

        palette_labs = tuple(_srgb_to_lab(rgb) for rgb in BEAD_PALETTE)
        rng = random.Random(42)
        for _ in range(200):
            rgb = tuple(rng.randrange(256) for _ in range(3))
            lab = _srgb_to_lab(rgb)
            exhaustive = min(
                range(len(BEAD_PALETTE)),
                key=lambda i: _ciede2000(lab, palette_labs[i]),
            )
            self.assertEqual(_nearest_bead_index(rgb), exhaustive, msg=f"rgb={rgb}")


class MardPaletteTests(unittest.TestCase):
    def test_palette_is_the_official_mard_221(self):
        # 221 unique purchasable codes; every entry round-trips through the
        # rgb -> code map that the materials list relies on.
        self.assertEqual(len(BEAD_COLORS), 221)
        codes = [code for code, _ in BEAD_COLORS]
        self.assertEqual(len(set(codes)), 221)
        for _, rgb in BEAD_COLORS:
            self.assertEqual(len(rgb), 3)
            self.assertTrue(all(0 <= v <= 255 for v in rgb))
            self.assertEqual(BEAD_CODE_BY_RGB[rgb], _)
            self.assertIn(rgb, BEAD_PALETTE)

    def test_quantized_render_is_countable(self):
        import tempfile

        image = make_gradient_image(Path(tempfile.mkdtemp()) / "grad.png")
        art = render_ascii_art(image, mode="bead", width=40, bead_palette=True)
        counts = bead_material_counts(art)
        self.assertIsNotNone(counts)
        self.assertEqual(sum(counts.values()), art.width * art.height)
        self.assertTrue(set(counts) <= {code for code, _ in BEAD_COLORS})


class BeadChartTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())

    @staticmethod
    def fake_bead_art(colors2d):
        return AsciiArt(
            rows=[[("■", rgb) for rgb in row] for row in colors2d],
            kind="bead",
        )

    def test_material_counts_sum_equals_cells(self):
        a, b = BEAD_COLORS[0][1], BEAD_COLORS[50][1]
        art = self.fake_bead_art([[a, b, a], [b, a, b]])
        counts = bead_material_counts(art)
        self.assertEqual(sum(counts.values()), 6)
        self.assertEqual(counts, {BEAD_CODE_BY_RGB[a]: 3, BEAD_CODE_BY_RGB[b]: 3})

    def test_material_counts_none_when_not_quantized(self):
        art = self.fake_bead_art([[(1, 2, 3), (255, 255, 255)]])
        self.assertIsNone(bead_material_counts(art))

    def test_material_lines_ranking_and_format(self):
        counts = {"A1": 5000, "B2": 1200, "C3": 800, "D4": 300, "E5": 100, "F6": 7}
        lines = bead_material_lines(counts)
        self.assertEqual(lines[0], "材料清单：需要 6 种颜色，共 7,407 颗")
        self.assertIn("A1×5,000", lines[1])
        self.assertIn("E5×100", lines[1])
        self.assertNotIn("F6", lines[1])  # only the top 5 make the cut
        self.assertIn("1 种少量色", lines[2])
        self.assertIn("单独分装", lines[2])

    def test_material_lines_without_minor_colors(self):
        lines = bead_material_lines({"A1": 500, "B2": 60})
        self.assertEqual(len(lines), 2)

    def test_merge_folds_minors_into_nearest_major(self):
        # 12+12 cells for the palette-backed majors, a light and a dark
        # minor nearby; each minor must fold into the closer major, with
        # every cell count preserved.
        light = BEAD_COLORS[0][1]
        dark = next(rgb for _, rgb in BEAD_COLORS if sum(rgb) < 200)
        near_light = tuple(min(255, v + 10) for v in light)
        near_dark = tuple(max(0, v - 10) for v in dark)
        art = self.fake_bead_art(
            [[light] * 12 + [near_light] * 3 + [dark] * 12 + [near_dark] * 4]
        )
        merged = merge_minor_bead_colors(art)
        self.assertEqual(merged, 2)
        rgbs = {rgb for row in art.rows for _, rgb in row}
        self.assertEqual(rgbs, {light, dark})
        counts = bead_material_counts(art)
        self.assertEqual(sum(counts.values()), 31)

    def test_merge_skips_grids_with_no_major(self):
        art = self.fake_bead_art([[(255, 0, 0)] * 5, [(0, 255, 0)] * 5])
        before = [row[:] for row in art.rows]
        self.assertEqual(merge_minor_bead_colors(art), 0)
        self.assertEqual(art.rows, before)

    def test_merge_noop_when_no_minors(self):
        light, dark = (250, 245, 205), (20, 20, 20)
        art = self.fake_bead_art([[light] * 12 + [dark] * 12])
        before = [row[:] for row in art.rows]
        self.assertEqual(merge_minor_bead_colors(art), 0)
        self.assertEqual(art.rows, before)

    def test_dithered_render_merges_clean(self):
        # Dithering is the minor-color factory; a merged pass must leave no
        # color at or below the threshold while keeping the cell total.
        image = make_gradient_image(self.tmp / "grad.png")
        art = render_ascii_art(
            image, mode="bead", width=40, bead_palette=True, bead_dither=True
        )
        merged = merge_minor_bead_colors(art)
        self.assertGreater(merged, 0)
        counts = bead_material_counts(art)
        self.assertEqual(sum(counts.values()), art.width * art.height)
        self.assertTrue(all(n > 10 for n in counts.values()))

    def test_chart_paints_one_code_per_cell(self):
        a, b = BEAD_COLORS[0][1], BEAD_COLORS[100][1]
        art = self.fake_bead_art([[a, b, a, b], [b, a, b, a], [a, a, b, b]])
        out = paint_bead_chart_png(art, self.tmp / "chart.png")
        self.assertIsNotNone(out)
        with Image.open(out) as png:
            self.assertEqual(png.size, (4 * 30 + BEAD_PAD * 2, 3 * 30 + BEAD_PAD * 2))

    def test_chart_shrinks_cell_for_large_grids(self):
        art = self.fake_bead_art([[(255, 255, 255)] * 20 for _ in range(250)])
        out = paint_bead_chart_png(art, self.tmp / "big.png")
        self.assertIsNotNone(out)
        with Image.open(out) as png:
            self.assertEqual(
                png.size, (20 * 24 + BEAD_PAD * 2, 250 * 24 + BEAD_PAD * 2)
            )

    def test_chart_gives_up_on_oversized_grids(self):
        # 400 rows cannot fit within the canvas cap at a legible cell size.
        art = self.fake_bead_art([[(255, 255, 255)] * 20 for _ in range(400)])
        self.assertIsNone(paint_bead_chart_png(art, self.tmp / "huge.png"))


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
