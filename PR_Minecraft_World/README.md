# Puerto Rico Minecraft World

This project builds a Minecraft-compatible Puerto Rico island world from an official DEM.

## Pipeline
1. Download official Puerto Rico DEM data
2. Convert DEM to Minecraft-safe grayscale heightmap
3. Validate outputs
4. Import heightmap into WorldPainter
5. Set spawn and export the world

## Outputs
- output/heightmap/puerto_rico_heightmap_2048.png
- output/heightmap/heightmap_metadata.json
- output/worldpainter/import_settings.txt
- output/logs/source_audit.txt

## Setup

macOS / Linux:
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Fetch DEM
```bash
python tools/fetch_dem.py
```

Attempts NOAA CUDEM first, falls back to USGS 3DEP automatically.

## Build Heightmap
```bash
python tools/build_heightmap.py
```

## Validation
Run:
```bash
python tools/validate_outputs.py
```

Expected:
```
HEIGHTMAP_OK
JSON_OK
HEIGHTMAP_VALID
```

## WorldPainter Import
See `output/worldpainter/import_settings.txt` for step-by-step import instructions.

## Troubleshooting
- **NOAA fetch fails**: script automatically switches to USGS 3DEP
- **Heightmap is blank**: check `output/logs/source_audit.txt` for min/max elevation values
- **WorldPainter import looks curved or distorted**: verify PNG is grayscale L mode with no alpha and linear mapping is selected; re-import without smoothing
- **Output exceeds 2048**: the script will fail loudly — check the resize logic
