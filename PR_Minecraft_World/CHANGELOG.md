# Changelog

All notable changes to the Puerto Rico Minecraft pipeline are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — 2026-03-28

### Added
- `config.toml` — all runtime constants now configurable without editing code
- `tools/_config.py` — shared config loader with hardcoded defaults as fallback
- `tools/__init__.py` — makes `tools/` a proper importable package
- `tools/run_pipeline.py` — single orchestration script (`--skip-fetch`, `--skip-build`, `--force`, `--bits`, `--verbose`)
- `tools/preview_heightmap.py` — terrain colormap preview PNG with elevation legend
- 16-bit heightmap output (`--bits 16`) for better vertical resolution in WorldPainter
- Concrete spawn coordinates (block X/Y/Z) computed from heightmap and written to metadata
- `output/worldpainter/worldpainter_import.properties` — CLI-ready WorldPainter config
- `pyproject.toml` — project metadata and `pip install -e .` entry points
- `Makefile` — `make fetch / build / validate / preview / run / clean`
- `.python-version` — pyenv version hint (`3.11`)
- `.pre-commit-config.yaml` — ruff lint + format hooks
- `Dockerfile` — reproducible container with GDAL pre-installed
- `LICENSE` — MIT
- `CHANGELOG.md` — this file
- `tests/` — pytest unit and integration tests (12 unit + 1 integration)
- `.github/workflows/ci.yml` — GitHub Actions: install → pipeline → pytest

### Changed
- `tools/fetch_dem.py`:
  - Removed unused imports (`os`, `time`, `hashlib`)
  - HEAD probe before full download (avoids streaming HTML error pages)
  - Rasterio probe validates file integrity immediately after download
  - Existing file probe before skip (catches corrupt files from prior runs)
  - Retry with exponential backoff on transient errors (up to 4 attempts)
  - Download progress reporting every 50 MiB
  - All `print()` replaced with `logging`
  - Config values loaded from `config.toml`
- `tools/build_heightmap.py`:
  - Output filename: `puerto_rico_heightmap_2048.png` → `puerto_rico_heightmap.png`
  - `input_raster` in metadata now stores relative path (not absolute)
  - Audit log uses timestamped run header (`--- RUN ISO8601 ---`) to avoid duplication
  - `data/processed/` directory removed (was declared but never written to)
  - NetCDF subdataset detection now inspects band dtypes, not just name keywords
  - Spawn coordinates computed and written to metadata (`spawn_x`, `spawn_y`, `spawn_z`)
  - Optional 16-bit PNG output alongside 8-bit
  - Memory warning for rasters > 500 MB; reprojection uses all CPU threads
  - All `print()` replaced with `logging`
  - Config values loaded from `config.toml`
- `tools/validate_outputs.py`:
  - All validation logic moved into `main()` (was module-level; broke imports)
  - Updated hardcoded filename reference to `puerto_rico_heightmap.png`
  - All `print()` replaced with `logging`
- `tools/generate_synthetic_dem.py`:
  - Fixed incorrect HEIGHT comment (`≈ 802`, not `≈ 496`)
  - Audit log now uses atomic write (temp file → rename)
  - All `print()` replaced with `logging`
- `README.md`:
  - Fixed output filename reference
  - Added sections: Synthetic DEM fallback, Minecraft version compatibility, Docker, Configuration
  - Removed stale `HEIGHTMAP_OK` from validation expected output
- `.gitignore` — removed `data/processed/` (directory deleted)
- `requirements.txt` — added `pytest==8.3.2`, `pre-commit==3.7.1`

### Fixed
- Absolute path stored in `heightmap_metadata.json` → now relative
- Audit log appended duplicate entries on each pipeline run → now per-run sections
- `_2048` in output filename was misleading (actual size 2033×839) → removed
- Wrong HEIGHT comment in `generate_synthetic_dem.py` (said 496, was 802)
- Unused `data/processed/` directory removed from structure

### Removed
- `data/processed/` directory (unused; reprojected intermediate stays in memory)

---

## [1.0.0] — 2026-03-27

### Added
- Initial pipeline: `fetch_dem.py` → `build_heightmap.py` → `validate_outputs.py`
- `generate_synthetic_dem.py` — offline fallback DEM from known geographic features
- 8-bit grayscale PNG heightmap (2033×839) suitable for WorldPainter import
- `output/worldpainter/import_settings.txt` — step-by-step GUI import guide
- `output/logs/source_audit.txt` — provenance log
- `output/heightmap/heightmap_metadata.json` — raster and output metadata
