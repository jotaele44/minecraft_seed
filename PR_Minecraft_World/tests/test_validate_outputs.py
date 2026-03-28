"""Unit tests for validate_outputs.py (imported, not CLI-invoked)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# We test the functions directly by monkey-patching the module paths
import tools.validate_outputs as vo


def _make_valid_png(path: Path, w: int = 200, h: int = 100) -> None:
    """Write a valid L-mode PNG with land content."""
    arr = np.zeros((h, w), dtype="uint8")
    arr[20:80, 20:180] = 150  # land area
    Image.fromarray(arr, mode="L").save(path)


def _make_valid_meta(path: Path, png_path: Path, w: int = 200, h: int = 100) -> None:
    meta = {
        "input_raster": "data/raw/dem.tif",  # relative
        "source_min_m": 1.5,
        "source_max_m": 1338.0,
        "output_width": w,
        "output_height": h,
        "sea_level_block": 62,
        "output_mode": "L",
        "spawn_x": 160,
        "spawn_y": 63,
        "spawn_z": 30,
    }
    path.write_text(json.dumps(meta), encoding="utf-8")


class TestValidateJson:
    def test_valid_meta_passes(self, tmp_path, monkeypatch):
        png = tmp_path / "hm.png"
        meta_path = tmp_path / "meta.json"
        _make_valid_png(png)
        _make_valid_meta(meta_path, png)

        monkeypatch.setattr(vo, "HEIGHTMAP", png)
        monkeypatch.setattr(vo, "META", meta_path)
        result = vo.validate_json()
        assert result["output_mode"] == "L"

    def test_missing_key_fails(self, tmp_path, monkeypatch):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"output_mode": "L"}), encoding="utf-8")
        monkeypatch.setattr(vo, "META", meta_path)
        with pytest.raises(SystemExit):
            vo.validate_json()

    def test_absolute_path_fails(self, tmp_path, monkeypatch):
        meta_path = tmp_path / "meta.json"
        meta = {
            "input_raster": "/absolute/path/dem.tif",
            "source_min_m": 0.0, "source_max_m": 1338.0,
            "output_width": 200, "output_height": 100,
            "sea_level_block": 62, "output_mode": "L",
            "spawn_x": 10, "spawn_y": 63, "spawn_z": 10,
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        monkeypatch.setattr(vo, "META", meta_path)
        with pytest.raises(SystemExit):
            vo.validate_json()


class TestValidateHeightmap:
    def _base_meta(self, w, h):
        return {
            "input_raster": "data/raw/x.tif",
            "source_min_m": 0.0, "source_max_m": 1338.0,
            "output_width": w, "output_height": h,
            "sea_level_block": 62, "output_mode": "L",
            "spawn_x": 0, "spawn_y": 63, "spawn_z": 0,
        }

    def test_valid_png_passes(self, tmp_path, monkeypatch):
        png = tmp_path / "hm.png"
        _make_valid_png(png, 200, 100)
        meta = self._base_meta(200, 100)
        monkeypatch.setattr(vo, "HEIGHTMAP", png)
        vo.validate_heightmap(meta)  # should not raise

    def test_wrong_mode_fails(self, tmp_path, monkeypatch):
        png = tmp_path / "hm.png"
        Image.new("RGB", (100, 50)).save(png)
        meta = self._base_meta(100, 50)
        monkeypatch.setattr(vo, "HEIGHTMAP", png)
        with pytest.raises(SystemExit):
            vo.validate_heightmap(meta)

    def test_oversized_fails(self, tmp_path, monkeypatch):
        png = tmp_path / "hm.png"
        arr = np.ones((100, 3000), dtype="uint8") * 100
        Image.fromarray(arr, mode="L").save(png)
        meta = self._base_meta(3000, 100)
        monkeypatch.setattr(vo, "HEIGHTMAP", png)
        with pytest.raises(SystemExit):
            vo.validate_heightmap(meta)

    def test_blank_png_fails(self, tmp_path, monkeypatch):
        png = tmp_path / "hm.png"
        Image.fromarray(np.zeros((100, 200), dtype="uint8"), mode="L").save(png)
        meta = self._base_meta(200, 100)
        monkeypatch.setattr(vo, "HEIGHTMAP", png)
        with pytest.raises(SystemExit):
            vo.validate_heightmap(meta)
