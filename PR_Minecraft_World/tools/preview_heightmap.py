"""
preview_heightmap.py — Render a terrain colourmap preview of the heightmap PNG.

Produces output/heightmap/puerto_rico_preview.png:
  - Terrain colours: ocean → lowland → highland → peak/snow
  - Red dot marking the computed spawn position
  - Elevation legend strip with labels (uses PIL default font only)

Usage:
    python tools/preview_heightmap.py [--verbose]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools._config import ROOT

log = logging.getLogger(__name__)

HEIGHTMAP = ROOT / "output" / "heightmap" / "puerto_rico_heightmap.png"
META      = ROOT / "output" / "heightmap" / "heightmap_metadata.json"
PREVIEW   = ROOT / "output" / "heightmap" / "puerto_rico_preview.png"

LEGEND_H = 42  # pixels for legend strip below island


# ---------------------------------------------------------------------------
# Colourmap
# ---------------------------------------------------------------------------

def _make_palette() -> list[tuple[int, int, int]]:
    """Return a 256-entry RGB list mapping pixel value → terrain colour."""
    palette: list[tuple[int, int, int]] = []

    def lerp(a: int, b: int, t: float) -> int:
        return int(a + (b - a) * t)

    def blend(c1: tuple, c2: tuple, t: float) -> tuple[int, int, int]:
        return (lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t))

    # Colour stops (pixel value → RGB)
    stops = [
        (0,   (10,  40, 180)),   # deep ocean
        (8,   (30,  80, 200)),   # coastal water
        (12,  (195, 225, 145)),  # coastal lowland
        (50,  (120, 195,  80)),  # lowland green
        (110, (180, 160,  70)),  # mid elevation
        (170, (140, 100,  50)),  # highland
        (220, (110,  75,  40)),  # mountain brown
        (240, (200, 200, 210)),  # near-peak
        (255, (255, 255, 255)),  # snow peak
    ]

    for i in range(256):
        # Find the two bracketing stops
        lo_v, lo_c = stops[0]
        hi_v, hi_c = stops[1]
        for j in range(len(stops) - 1):
            if stops[j][0] <= i <= stops[j + 1][0]:
                lo_v, lo_c = stops[j]
                hi_v, hi_c = stops[j + 1]
                break
        span = hi_v - lo_v
        t = (i - lo_v) / span if span > 0 else 0.0
        palette.append(blend(lo_c, hi_c, t))

    return palette


_PALETTE = _make_palette()


def _apply_palette(arr: np.ndarray) -> np.ndarray:
    """Map an H×W uint8 array through the terrain palette → H×W×3 RGB."""
    lut = np.array(_PALETTE, dtype=np.uint8)   # 256 × 3
    return lut[arr]


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def _make_legend(width: int, land_max_m: float) -> Image.Image:
    """Return a legend strip of size (width, LEGEND_H)."""
    legend = Image.new("RGB", (width, LEGEND_H), (235, 235, 235))
    draw = ImageDraw.Draw(legend)
    font = ImageFont.load_default()

    # Gradient bar (top 20 px)
    bar_h = 20
    for x in range(width):
        val = int(x / width * 255)
        r, g, b = _PALETTE[val]
        for y in range(bar_h):
            draw.point((x, y), fill=(r, g, b))

    # Border on bar
    draw.rectangle([0, 0, width - 1, bar_h - 1], outline=(100, 100, 100))

    # Labels below bar
    mid_m = land_max_m / 2
    labels = [
        (4,               f"0 m"),
        (width // 2 - 20, f"{mid_m:.0f} m"),
        (width - 80,      f"{land_max_m:.0f} m (Cerro Punta)"),
    ]
    text_y = bar_h + 4
    for lx, text in labels:
        draw.text((lx, text_y), text, fill=(40, 40, 40), font=font)

    return legend


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate terrain preview image.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not HEIGHTMAP.exists():
        log.error("Heightmap not found: %s  —  run build_heightmap.py first.", HEIGHTMAP)
        sys.exit(1)

    # Load metadata for spawn coords and land_max_m
    land_max_m = 1338.0
    spawn_x = spawn_z = None
    if META.exists():
        try:
            meta = json.loads(META.read_text(encoding="utf-8"))
            land_max_m = meta.get("land_max_m", 1338.0)
            spawn_x = meta.get("spawn_x")
            spawn_z = meta.get("spawn_z")
        except Exception as exc:
            log.warning("Could not read metadata: %s", exc)

    log.info("Loading heightmap: %s", HEIGHTMAP)
    img = Image.open(HEIGHTMAP).convert("L")
    arr = np.array(img)
    w, h = img.size

    # Apply terrain colourmap
    log.info("Applying terrain colourmap …")
    rgb_arr = _apply_palette(arr)
    coloured = Image.fromarray(rgb_arr, mode="RGB")

    # Mark spawn point with a red cross
    if spawn_x is not None and spawn_z is not None:
        draw = ImageDraw.Draw(coloured)
        sx, sz = int(spawn_x), int(spawn_z)
        r = 6  # cross arm length
        draw.line([(sx - r, sz), (sx + r, sz)], fill=(255, 30, 30), width=2)
        draw.line([(sx, sz - r), (sx, sz + r)], fill=(255, 30, 30), width=2)
        draw.ellipse([(sx - 3, sz - 3), (sx + 3, sz + 3)], outline=(255, 30, 30), width=1)
        log.info("Spawn marked at pixel (%d, %d)", sx, sz)

    # Build legend and combine
    legend = _make_legend(w, land_max_m)
    combined = Image.new("RGB", (w, h + LEGEND_H))
    combined.paste(coloured, (0, 0))
    combined.paste(legend,   (0, h))

    combined.save(PREVIEW)
    log.info("Preview saved: %s  (%dx%d)", PREVIEW, combined.width, combined.height)


if __name__ == "__main__":
    main()
