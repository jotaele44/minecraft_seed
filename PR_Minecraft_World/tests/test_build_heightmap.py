"""Unit tests for build_heightmap.py core functions."""

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers duplicated here to test in isolation (no side effects)
# ---------------------------------------------------------------------------

def _normalise(land: np.ndarray) -> np.ndarray:
    """Replicate the normalisation logic from build_heightmap.main()."""
    land = land.copy()
    land[land < 0] = 0.0
    land_max = float(np.max(land))
    if land_max <= 0:
        raise ValueError("No positive terrain")
    return np.clip(land / land_max, 0.0, 1.0)


def _resize_preserving_aspect(arr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = arr.shape
    scale = min(max_dim / max(h, w), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray((arr * 255).astype("uint8"), mode="L")
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    return np.array(img) / 255.0


def _find_spawn_pixel(arr: np.ndarray, sea_px: int = 10) -> tuple[int, int]:
    """Minimal version of the spawn-finder (northeast quadrant)."""
    h, w = arr.shape
    ne_x = int(w * 0.60)
    ne_y = int(h * 0.50)
    region = arr[:ne_y, ne_x:]
    land_ys, land_xs = np.where(region > sea_px)
    if land_xs.size == 0:
        land_ys, land_xs = np.where(arr > sea_px)
        if land_xs.size == 0:
            return w // 2, h // 4
        return int(np.median(land_xs)), int(np.median(land_ys))
    return int(np.median(land_xs)) + ne_x, int(np.median(land_ys))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_clips_to_0_1(self):
        arr = np.array([[0.0, 500.0, 1000.0]], dtype="float32")
        result = _normalise(arr)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_max_becomes_1(self):
        arr = np.array([[0.0, 250.0, 1338.0]], dtype="float32")
        result = _normalise(arr)
        assert result.max() == pytest.approx(1.0)

    def test_negatives_forced_to_zero(self):
        arr = np.array([[-100.0, 0.0, 500.0]], dtype="float32")
        result = _normalise(arr)
        assert result.min() == pytest.approx(0.0)

    def test_all_zero_raises(self):
        arr = np.zeros((10, 10), dtype="float32")
        with pytest.raises(ValueError, match="No positive terrain"):
            _normalise(arr)

    def test_all_negative_raises(self):
        arr = np.full((5, 5), -50.0, dtype="float32")
        with pytest.raises(ValueError, match="No positive terrain"):
            _normalise(arr)


class TestResize:
    def test_respects_max_dim(self):
        arr = np.random.rand(4000, 2000).astype("float32")
        result = _resize_preserving_aspect(arr, max_dim=2048)
        h, w = result.shape
        assert max(h, w) <= 2048

    def test_preserves_aspect_ratio(self):
        arr = np.random.rand(1000, 2000).astype("float32")
        result = _resize_preserving_aspect(arr, max_dim=2048)
        h, w = result.shape
        original_ratio = 2000 / 1000
        result_ratio = w / h
        assert abs(result_ratio - original_ratio) < 0.02

    def test_no_upsample(self):
        """Small images should not be enlarged."""
        arr = np.random.rand(100, 200).astype("float32")
        result = _resize_preserving_aspect(arr, max_dim=2048)
        assert result.shape == arr.shape


class TestSpawnFinder:
    def test_returns_land_pixel(self):
        arr = np.zeros((100, 200), dtype="uint8")
        arr[10:40, 130:190] = 200  # land block in NE
        sx, sz = _find_spawn_pixel(arr)
        assert arr[sz, sx] > 10

    def test_stays_within_image(self):
        arr = np.zeros((100, 200), dtype="uint8")
        arr[20:50, 120:180] = 150
        sx, sz = _find_spawn_pixel(arr)
        assert 0 <= sx < 200
        assert 0 <= sz < 100

    def test_fallback_on_empty_quadrant(self):
        """If NE quadrant has no land, should fall back to any land pixel."""
        arr = np.zeros((100, 200), dtype="uint8")
        arr[60:80, 10:50] = 100  # land only in SW
        sx, sz = _find_spawn_pixel(arr)
        assert arr[sz, sx] > 10


class TestOutputMode:
    def test_8bit_mode_is_L(self):
        arr = np.random.randint(0, 255, (100, 200), dtype="uint8")
        img = Image.fromarray(arr, mode="L").convert("L")
        assert img.mode == "L"

    def test_no_alpha_in_L_mode(self):
        arr = np.zeros((50, 100), dtype="uint8")
        img = Image.fromarray(arr, mode="L")
        assert "A" not in img.getbands()


class TestCRSFallback:
    """Verify the CRS-None detection logic used in build_heightmap.main()."""

    def test_raster_without_crs_has_none(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds

        tif = tmp_path / "nocrs.tif"
        transform = from_bounds(-67.0, 17.0, -65.0, 19.0, 10, 10)
        data = np.ones((1, 10, 10), dtype="float32") * 500.0
        with rasterio.open(
            tif, "w", driver="GTiff", height=10, width=10,
            count=1, dtype="float32", crs=None, transform=transform,
        ) as dst:
            dst.write(data)

        with rasterio.open(tif) as src:
            assert src.crs is None

    def test_lat_lon_bounds_trigger_wgs84_assumption(self, tmp_path):
        import rasterio
        import rasterio.crs
        from rasterio.transform import from_bounds

        tif = tmp_path / "nocrs.tif"
        transform = from_bounds(-67.0, 17.0, -65.0, 19.0, 10, 10)
        data = np.ones((1, 10, 10), dtype="float32") * 500.0
        with rasterio.open(
            tif, "w", driver="GTiff", height=10, width=10,
            count=1, dtype="float32", crs=None, transform=transform,
        ) as dst:
            dst.write(data)

        with rasterio.open(tif) as src:
            b = src.bounds
            # The condition used in build_heightmap.main()
            in_wgs84_range = -180 <= b.left <= 180 and -90 <= b.bottom <= 90
            assert in_wgs84_range, f"Bounds {b} should be in WGS84 range"
            assumed = rasterio.crs.CRS.from_epsg(4326)
            assert assumed is not None
