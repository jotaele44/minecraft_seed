"""
validate_outputs.py — Verify that all pipeline outputs meet quality gates.

Usage:
    python tools/validate_outputs.py

Expected output:
    JSON_OK
    HEIGHTMAP_VALID
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEIGHTMAP = ROOT / "output" / "heightmap" / "puerto_rico_heightmap_2048.png"
META = ROOT / "output" / "heightmap" / "heightmap_metadata.json"
IMPORT = ROOT / "output" / "worldpainter" / "import_settings.txt"
AUDIT = ROOT / "output" / "logs" / "source_audit.txt"


def fail(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def check_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"Missing {label}: {path}")


# ---------------------------------------------------------------------------
# File existence checks
# ---------------------------------------------------------------------------
check_file_exists(HEIGHTMAP, "heightmap PNG")
check_file_exists(META, "metadata JSON")
check_file_exists(IMPORT, "WorldPainter import settings")
check_file_exists(AUDIT, "source audit log")

# ---------------------------------------------------------------------------
# Metadata JSON validation
# ---------------------------------------------------------------------------
try:
    meta = json.loads(META.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    fail(f"heightmap_metadata.json is not valid JSON: {exc}")

required_keys = ["input_raster", "source_min_m", "source_max_m", "output_width",
                 "output_height", "sea_level_block", "output_mode"]
for key in required_keys:
    if key not in meta:
        fail(f"Missing key '{key}' in heightmap_metadata.json")

if meta.get("output_mode") != "L":
    fail(f"Metadata reports output_mode='{meta.get('output_mode')}', expected 'L'")

print("JSON_OK")

# ---------------------------------------------------------------------------
# Heightmap PNG validation
# ---------------------------------------------------------------------------
try:
    from PIL import Image
except ImportError:
    fail("Pillow not installed — run: pip install Pillow==10.4.0")

img = Image.open(HEIGHTMAP)

if img.mode != "L":
    fail(f"Heightmap image mode is '{img.mode}', must be 'L' (8-bit grayscale, no alpha)")

w, h = img.size
if max(w, h) > 2048:
    fail(f"Heightmap dimensions {w}x{h} exceed maximum of 2048 px")

if min(w, h) < 1:
    fail(f"Heightmap has zero-size dimension: {w}x{h}")

# Sanity check: image must have both dark (ocean) and bright (land) pixels
import numpy as np
arr = np.array(img)
if arr.max() == 0:
    fail("Heightmap is entirely black — no land terrain found")
if arr.min() == arr.max():
    fail(f"Heightmap is flat (all pixels = {arr.min()}) — normalization may have failed")

land_fraction = float((arr > 10).sum()) / arr.size
if land_fraction < 0.01:
    fail(f"Heightmap has very little land ({land_fraction*100:.2f}% > 10) — check DEM coverage")

print(f"  Size:          {w} x {h} px")
print(f"  Mode:          {img.mode}")
print(f"  Pixel range:   {int(arr.min())} – {int(arr.max())}")
print(f"  Land fraction: {land_fraction*100:.1f}%")
print("HEIGHTMAP_VALID")
