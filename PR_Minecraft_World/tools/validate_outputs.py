"""
validate_outputs.py — Verify all pipeline outputs meet quality gates.

Usage:
    python tools/validate_outputs.py [--verbose]

Expected output on success:
    JSON_OK
    HEIGHTMAP_VALID
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tools._config import ROOT

log = logging.getLogger(__name__)

HEIGHTMAP = ROOT / "output" / "heightmap" / "puerto_rico_heightmap.png"
META      = ROOT / "output" / "heightmap" / "heightmap_metadata.json"
IMPORT    = ROOT / "output" / "worldpainter" / "import_settings.txt"
AUDIT     = ROOT / "output" / "logs"        / "source_audit.txt"


def _fail(msg: str, code: int = 1) -> None:
    log.error(msg)
    sys.exit(code)


def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        _fail(f"Missing {label}: {path}")


def validate_json() -> dict:
    """Validate heightmap_metadata.json; return parsed dict."""
    _check_exists(META, "metadata JSON")
    try:
        meta = json.loads(META.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"heightmap_metadata.json is not valid JSON: {exc}")

    required = [
        "input_raster", "source_min_m", "source_max_m",
        "output_width", "output_height", "sea_level_block", "output_mode",
        "spawn_x", "spawn_y", "spawn_z",
    ]
    for key in required:
        if key not in meta:
            _fail(f"Missing key '{key}' in heightmap_metadata.json")

    if meta["output_mode"] != "L":
        _fail(f"output_mode is '{meta['output_mode']}', expected 'L'")

    if meta.get("input_raster", "").startswith("/"):
        _fail("input_raster in metadata is an absolute path — should be relative")

    log.info("JSON_OK")
    return meta


def validate_heightmap(meta: dict) -> None:
    """Validate the PNG heightmap against quality gates."""
    _check_exists(HEIGHTMAP, "heightmap PNG")

    img = Image.open(HEIGHTMAP)

    if img.mode != "L":
        _fail(f"PNG mode is '{img.mode}', must be 'L' (8-bit grayscale, no alpha)")

    w, h = img.size
    if max(w, h) > 2048:
        _fail(f"PNG dimensions {w}×{h} exceed 2048 px")
    if min(w, h) < 1:
        _fail(f"PNG has zero-size dimension: {w}×{h}")

    arr = np.array(img)
    if arr.max() == 0:
        _fail("PNG is entirely black — no land terrain found")
    if arr.min() == arr.max():
        _fail(f"PNG is flat (all pixels = {arr.min()}) — normalisation failed")

    land_frac = float((arr > 10).sum()) / arr.size
    if land_frac < 0.01:
        _fail(
            f"Only {land_frac*100:.2f}% of pixels > 10 — check DEM covers Puerto Rico land area"
        )

    # Cross-check dimensions against metadata
    if w != meta["output_width"] or h != meta["output_height"]:
        _fail(
            f"PNG dimensions {w}×{h} do not match metadata "
            f"({meta['output_width']}×{meta['output_height']})"
        )

    log.info("  Size:          %d x %d px", w, h)
    log.info("  Mode:          %s", img.mode)
    log.info("  Pixel range:   %d – %d", int(arr.min()), int(arr.max()))
    log.info("  Land fraction: %.1f%%", land_frac * 100)
    log.info("HEIGHTMAP_VALID")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pipeline outputs.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    _check_exists(IMPORT, "WorldPainter import settings")
    _check_exists(AUDIT,  "source audit log")

    meta = validate_json()
    validate_heightmap(meta)


if __name__ == "__main__":
    main()
