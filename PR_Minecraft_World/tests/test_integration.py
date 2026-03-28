"""
Integration smoke test — runs the full synthetic pipeline in a temp directory.

Verifies:
  - fetch_dem.py generates a valid GeoTIFF via synthetic path
  - build_heightmap.py produces HEIGHTMAP_OK and a valid PNG
  - validate_outputs.py exits 0 with JSON_OK + HEIGHTMAP_VALID in output
  - preview_heightmap.py produces a preview PNG
"""

import subprocess
import sys
from pathlib import Path

import pytest
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(script: str, extra: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [PYTHON, str(ROOT / "tools" / script)] + (extra or [])
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


class TestFullPipeline:
    """Runs the synthetic DEM path end-to-end."""

    @pytest.fixture(autouse=True)
    def clean_outputs(self):
        """Remove generated outputs before each test run."""
        targets = [
            ROOT / "data" / "raw" / "puerto_rico_official_dem.tif",
            ROOT / "output" / "heightmap" / "puerto_rico_heightmap.png",
            ROOT / "output" / "heightmap" / "puerto_rico_heightmap_16bit.png",
            ROOT / "output" / "heightmap" / "puerto_rico_preview.png",
            ROOT / "output" / "heightmap" / "heightmap_metadata.json",
            ROOT / "output" / "logs" / "source_audit.txt",
            ROOT / "output" / "worldpainter" / "import_settings.txt",
        ]
        for t in targets:
            t.unlink(missing_ok=True)
        yield
        # Cleanup after test so committed outputs stay clean
        for t in targets:
            t.unlink(missing_ok=True)

    def test_fetch_produces_valid_geotiff(self):
        result = _run("fetch_dem.py")
        assert result.returncode == 0, f"fetch_dem.py failed:\n{result.stderr}"
        tif = ROOT / "data" / "raw" / "puerto_rico_official_dem.tif"
        assert tif.exists() and tif.stat().st_size > 1024
        with rasterio.open(tif) as src:
            assert src.count >= 1
            assert src.crs is not None

    def test_build_produces_valid_png(self):
        _run("fetch_dem.py")
        result = _run("build_heightmap.py")
        assert result.returncode == 0, f"build_heightmap.py failed:\n{result.stderr}"
        assert "HEIGHTMAP_OK" in result.stdout or "HEIGHTMAP_OK" in result.stderr

        png = ROOT / "output" / "heightmap" / "puerto_rico_heightmap.png"
        assert png.exists()
        img = Image.open(png)
        assert img.mode == "L"
        assert max(img.size) <= 2048

    def test_validate_passes(self):
        _run("fetch_dem.py")
        _run("build_heightmap.py")
        result = _run("validate_outputs.py")
        assert result.returncode == 0, f"validate_outputs.py failed:\n{result.stderr}"
        combined = result.stdout + result.stderr
        assert "JSON_OK" in combined
        assert "HEIGHTMAP_VALID" in combined

    def test_preview_produced(self):
        _run("fetch_dem.py")
        _run("build_heightmap.py")
        result = _run("preview_heightmap.py")
        assert result.returncode == 0, f"preview_heightmap.py failed:\n{result.stderr}"
        preview = ROOT / "output" / "heightmap" / "puerto_rico_preview.png"
        assert preview.exists()
        img = Image.open(preview)
        assert img.mode == "RGB"
        # Preview should be taller than heightmap (legend strip added)
        hm = Image.open(ROOT / "output" / "heightmap" / "puerto_rico_heightmap.png")
        assert img.height > hm.height
