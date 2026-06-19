"""Unit tests for preview_heightmap.py — palette, palette application, and legend."""

import numpy as np
import pytest
from PIL import Image

from tools.preview_heightmap import LEGEND_H, _apply_palette, _make_legend, _make_palette


class TestPalette:
    def test_palette_has_256_entries(self):
        palette = _make_palette()
        assert len(palette) == 256

    def test_all_entries_are_rgb_tuples(self):
        palette = _make_palette()
        for i, entry in enumerate(palette):
            assert len(entry) == 3, f"Entry {i} is not a 3-tuple: {entry}"

    def test_all_values_in_byte_range(self):
        palette = _make_palette()
        for i, (r, g, b) in enumerate(palette):
            assert 0 <= r <= 255, f"Entry {i} red={r} out of range"
            assert 0 <= g <= 255, f"Entry {i} green={g} out of range"
            assert 0 <= b <= 255, f"Entry {i} blue={b} out of range"

    def test_ocean_is_blue(self):
        """Pixel value 0 should map to a predominantly blue colour."""
        palette = _make_palette()
        r, g, b = palette[0]
        assert b > r and b > g

    def test_peak_is_near_white(self):
        """Pixel value 255 should map to near-white (snow)."""
        palette = _make_palette()
        r, g, b = palette[255]
        assert r > 200 and g > 200 and b > 200


class TestApplyPalette:
    def test_output_shape(self):
        arr = np.zeros((10, 20), dtype=np.uint8)
        rgb = _apply_palette(arr)
        assert rgb.shape == (10, 20, 3)

    def test_output_dtype(self):
        arr = np.zeros((5, 5), dtype=np.uint8)
        rgb = _apply_palette(arr)
        assert rgb.dtype == np.uint8

    def test_maps_correct_colour(self):
        """Value 0 → ocean blue; value 255 → snow white."""
        palette = _make_palette()
        arr = np.array([[0, 255]], dtype=np.uint8)
        rgb = _apply_palette(arr)
        assert tuple(rgb[0, 0]) == palette[0]
        assert tuple(rgb[0, 1]) == palette[255]


class TestLegend:
    def test_legend_dimensions(self):
        legend = _make_legend(200, 1338.0)
        assert legend.size == (200, LEGEND_H)

    def test_legend_mode_is_rgb(self):
        legend = _make_legend(150, 1338.0)
        assert legend.mode == "RGB"

    def test_legend_non_uniform(self):
        """Legend should not be a flat single colour (contains gradient bar)."""
        legend = _make_legend(256, 1338.0)
        arr = np.array(legend)
        assert arr.std() > 0
