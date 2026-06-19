# Puerto Rico Minecraft World

Builds a Minecraft Java Edition world shaped like Puerto Rico from an official DEM, using a 2048-pixel grayscale heightmap and WorldPainter.

## Pipeline

```
fetch_dem.py → build_heightmap.py → validate_outputs.py → preview_heightmap.py
```

1. Download official Puerto Rico DEM (NOAA CUDEM → USGS 3DEP → synthetic fallback)
2. Reproject, normalise, and export an 8-bit (or 16-bit) grayscale PNG heightmap
3. Validate all outputs against quality gates
4. Generate a terrain colourmap preview image
5. Import into WorldPainter and export as a Minecraft Java world

## Quick Start

```bash
# macOS / Linux
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make run          # full pipeline in one command
```

```powershell
# Windows PowerShell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tools/run_pipeline.py
```

## Step-by-step

```bash
python tools/fetch_dem.py        # download or generate DEM
python tools/build_heightmap.py  # produce PNG heightmap
python tools/validate_outputs.py # confirm quality gates
python tools/preview_heightmap.py # render colourmap preview
```

Expected validation output:
```
INFO JSON_OK
INFO   Size:          2033 x 839 px
INFO   Mode:          L
INFO   Pixel range:   0 – 255
INFO   Land fraction: 62.6%
INFO HEIGHTMAP_VALID
```

## Outputs

| File | Description |
|---|---|
| `output/heightmap/puerto_rico_heightmap.png` | 8-bit grayscale, mode L, max 2048 px |
| `output/heightmap/puerto_rico_heightmap_16bit.png` | 16-bit variant (better vertical resolution) |
| `output/heightmap/puerto_rico_preview.png` | Terrain colourmap with spawn marker and legend |
| `output/heightmap/heightmap_metadata.json` | Raster stats, spawn coords, Minecraft settings |
| `output/worldpainter/import_settings.txt` | Step-by-step WorldPainter GUI + CLI guide |
| `output/worldpainter/worldpainter_import.properties` | WorldPainter CLI config file |
| `output/logs/source_audit.txt` | Provenance log (per-run, timestamped) |

## WorldPainter Import

See `output/worldpainter/import_settings.txt` for full instructions.

**GUI summary:**
1. File → Import → Import Height Map...
2. Select `output/heightmap/puerto_rico_heightmap.png`
3. Mapping: **Linear** | Smoothing: **off** | Water level: **62** | Max height: **255**
4. Place spawn at the coordinates in `heightmap_metadata.json` (`spawn_x`, `spawn_z`)
5. File → Export → Export as Minecraft Java world

**CLI (if your WorldPainter version supports it):**
```bash
worldpainter -import output/heightmap/puerto_rico_heightmap.png \
             --config output/worldpainter/worldpainter_import.properties
```

## Configuration

All tunable constants live in `config.toml`:

```toml
[minecraft]
version = "pre-1.18"   # or "1.18+"
sea_level_block = 62   # use 63 for 1.18+
max_height = 255       # use 384 for 1.18+

[heightmap]
target_max_dim = 2048  # max output dimension in pixels
bits = 8               # 8 or 16
```

## Minecraft Version Compatibility

| Setting | pre-1.18 | 1.18+ |
|---|---|---|
| Sea level block | 62 | 63 |
| World height | 0 – 255 | −64 – 320 |
| Max height (WP) | 255 | 384 |

Update `config.toml` `[minecraft]` section before building for 1.18+.

## Synthetic DEM Fallback

When NOAA CUDEM and USGS 3DEP are both unreachable, `fetch_dem.py` automatically calls `generate_synthetic_dem.py`, which builds a GeoTIFF from:

- Puerto Rico coastline polygon (34 cartographic reference points)
- Gaussian elevation peaks at 26 known locations:
  - Cerro Punta 1338 m, Tres Picachos 1246 m, El Yunque 1065 m, etc.
  - Broad Cordillera Central spine

The synthetic DEM produces a correctly shaped, topographically approximate island. For production quality, replace with an official DEM (see `output/logs/source_audit.txt`).

## Docker

```bash
docker build -t pr-minecraft .
docker run --rm -v $(pwd)/output:/app/output pr-minecraft
```

The container runs the full pipeline and writes outputs to your local `output/` directory.

## Development

```bash
pip install -r requirements.txt   # includes pytest and pre-commit
pre-commit install                 # enable ruff lint/format on commit
make test                          # run pytest suite
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| NOAA/USGS fetch fails | Script auto-falls back to synthetic DEM |
| Heightmap is blank | Check `MIN`/`MAX` in `source_audit.txt` |
| WorldPainter import curved/distorted | Confirm mode L, Linear mapping, smoothing off |
| Wrong sea level in world | Update `sea_level_block` in `config.toml` |
| Large raster OOM | Pipeline warns at >500 MB; use a machine with ≥8 GB RAM |
