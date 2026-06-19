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
    BATHYMETRY_FEATURES,
    ISLAND_COASTLINES,
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
        # PR main island + 6 smaller islands inside a 3.4° × 4.0° box → small fraction
        assert 0.05 < frac < 0.50, f"Unexpected land fraction: {frac:.2f}"


class TestElevationGrid:
    # _build_elevation_grid() now returns (land_elev, ocean_depth)
    def test_shape(self):
        land_elev, _ = _build_elevation_grid()
        assert land_elev.shape == (HEIGHT, WIDTH)

    def test_land_all_non_negative(self):
        land_elev, _ = _build_elevation_grid()
        assert land_elev.min() >= 0.0, "Land elevation grid should have no negatives"

    def test_land_range_is_positive(self):
        land_elev, _ = _build_elevation_grid()
        assert land_elev.max() > 0.0

    def test_cerro_punta_is_highest(self):
        """Cerro Punta (lon=-66.593, lat=18.173) should be near the global max."""
        from tools.generate_synthetic_dem import _lon_to_px, _lat_to_py
        land_elev, _ = _build_elevation_grid()
        px = int(_lon_to_px(-66.593))
        py = int(_lat_to_py(18.173))
        window = land_elev[max(0, py-10):py+10, max(0, px-10):px+10]
        assert window.max() >= 1200, (
            f"Peak near Cerro Punta should be ≥1200 m, got {window.max():.0f}"
        )


class TestBathymetry:
    def test_ocean_depth_grid_shape(self):
        _, ocean_depth = _build_elevation_grid()
        assert ocean_depth.shape == (HEIGHT, WIDTH)

    def test_ocean_depth_non_positive(self):
        """All bathymetric values should be ≤ 0 (sea level or below)."""
        _, ocean_depth = _build_elevation_grid()
        assert ocean_depth.max() <= 0.0

    def test_bathy_min_is_meaningful(self):
        """At least some depths should be below -50 m."""
        _, ocean_depth = _build_elevation_grid()
        assert ocean_depth.min() < -50.0, (
            f"Expected depths below -50 m; min={ocean_depth.min():.1f}"
        )

    def test_ocean_pixels_get_negative_depth(self):
        """Ocean pixels in combined DEM should be negative."""
        mask = _build_coastline_mask()
        land_elev, ocean_depth = _build_elevation_grid()
        combined = land_elev.copy()
        combined[~mask] = ocean_depth[~mask]
        assert combined[~mask].max() <= 0.0, "Ocean pixels should be ≤ 0"

    def test_bathymetry_features_list_not_empty(self):
        assert len(BATHYMETRY_FEATURES) > 0


class TestSyntheticGeoTiff:
    """Tests write a fresh GeoTIFF to a temp dir — independent of any cached DEM."""

    def _make_tif(self, tmp_path):
        mask = _build_coastline_mask()
        land_elev, ocean_depth = _build_elevation_grid()
        combined = land_elev.copy()
        combined[~mask] = ocean_depth[~mask]
        tif = tmp_path / "synthetic_test.tif"
        transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, WIDTH, HEIGHT)
        with rasterio.open(
            tif, "w", driver="GTiff", height=HEIGHT, width=WIDTH,
            count=1, dtype="float32",
            crs=rasterio.crs.CRS.from_epsg(4326),
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(combined, 1)
        return tif

    def test_output_is_valid_geotiff(self, tmp_path):
        tif = self._make_tif(tmp_path)
        with rasterio.open(tif) as src:
            assert src.count == 1
            assert src.crs is not None
            assert src.nodata == -9999.0

    def test_elevation_range_positive(self, tmp_path):
        tif = self._make_tif(tmp_path)
        with rasterio.open(tif) as src:
            data = src.read(1)
            land = data[data > 0]
            assert land.max() > 1000, f"Expected PR terrain above 1000 m, got {land.max():.0f}"


class TestOffshoreIslands:
    """Verify offshore island polygons produce land pixels in the mask."""

    def _px(self, lon, lat):
        from tools.generate_synthetic_dem import _lon_to_px, _lat_to_py
        return int(_lon_to_px(lon)), int(_lat_to_py(lat))

    def _check_island(self, lon, lat, name):
        mask = _build_coastline_mask()
        px, py = self._px(lon, lat)
        assert mask[py, px], f"{name} centre pixel should be land"

    def test_vieques(self):
        self._check_island(-65.45, 18.13, "Vieques")

    def test_culebra(self):
        self._check_island(-65.28, 18.33, "Culebra")

    def test_mona(self):
        self._check_island(-67.89, 18.11, "Mona")

    def test_island_coastlines_count(self):
        assert len(ISLAND_COASTLINES) == 7  # PR main + 6 offshore

    def test_bathymetry_features_have_trenches(self):
        depths = [feat[2] for feat in BATHYMETRY_FEATURES]
        assert min(depths) <= -8000, "PR Trench (~8376 m) should be represented"
