"""Unit tests for generate_synthetic_dem.py core functions."""

import numpy as np
import pytest
import rasterio
import rasterio.crs
from rasterio.transform import from_bounds
from PIL import Image, ImageDraw

# Import functions under test
from tools.generate_synthetic_dem import (
    _build_coastline_mask,
    _build_elevation_grid,
    TERRAIN_FEATURES,
    WIDTH,
    HEIGHT,
    LON_MIN, LAT_MIN, LON_MAX, LAT_MAX,
)


class TestCoastlineMask:
    def test_has_land(self):
        mask = _build_coastline_mask()
        assert np.any(mask), "Mask should have at least some land pixels"

    def test_has_ocean(self):
        mask = _build_coastline_mask()
        assert np.any(~mask), "Mask should have at least some ocean pixels"

    def test_shape(self):
        mask = _build_coastline_mask()
        assert mask.shape == (HEIGHT, WIDTH)

    def test_land_fraction_reasonable(self):
        mask = _build_coastline_mask()
        frac = mask.sum() / mask.size
        # Puerto Rico island should cover 20–80% of the bounding box
        assert 0.20 < frac < 0.80, f"Unexpected land fraction: {frac:.2f}"


class TestElevationGrid:
    def test_shape(self):
        elev = _build_elevation_grid()
        assert elev.shape == (HEIGHT, WIDTH)

    def test_all_non_negative(self):
        elev = _build_elevation_grid()
        assert elev.min() >= 0.0, "Raw elevation grid should have no negatives"

    def test_range_is_positive(self):
        elev = _build_elevation_grid()
        assert elev.max() > 0.0

    def test_cerro_punta_is_highest(self):
        """Cerro Punta (lon=-66.593, lat=18.173) should be near the global max."""
        from tools.generate_synthetic_dem import _lon_to_px, _lat_to_py
        elev = _build_elevation_grid()
        px = int(_lon_to_px(-66.593))
        py = int(_lat_to_py(18.173))
        window = elev[max(0, py-10):py+10, max(0, px-10):px+10]
        assert window.max() >= 1200, (
            f"Peak near Cerro Punta should be ≥1200 m, got {window.max():.0f}"
        )


class TestSyntheticGeoTiff:
    """Tests write a fresh GeoTIFF to a temp dir — independent of any cached DEM."""

    def _make_tif(self, tmp_path):
        mask = _build_coastline_mask()
        elev = _build_elevation_grid()
        elev[~mask] = -1.0
        tif = tmp_path / "synthetic_test.tif"
        transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, WIDTH, HEIGHT)
        with rasterio.open(
            tif, "w", driver="GTiff", height=HEIGHT, width=WIDTH,
            count=1, dtype="float32",
            crs=rasterio.crs.CRS.from_epsg(4326),
            transform=transform,
            nodata=-1.0,
        ) as dst:
            dst.write(elev, 1)
        return tif

    def test_output_is_valid_geotiff(self, tmp_path):
        tif = self._make_tif(tmp_path)
        with rasterio.open(tif) as src:
            assert src.count == 1
            assert src.crs is not None
            assert src.nodata == -1.0

    def test_elevation_range_positive(self, tmp_path):
        tif = self._make_tif(tmp_path)
        with rasterio.open(tif) as src:
            data = src.read(1)
            land = data[data != src.nodata]
            assert land.max() > 1000, f"Expected PR terrain above 1000 m, got {land.max():.0f}"
