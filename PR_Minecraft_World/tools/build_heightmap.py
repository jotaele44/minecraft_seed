"""
build_heightmap.py — Convert an official Puerto Rico DEM GeoTIFF into a
WorldPainter-compatible 8-bit grayscale PNG heightmap.

Usage:
    python tools/build_heightmap.py
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "output"
LOG_DIR = OUT_DIR / "logs"
HEIGHTMAP_DIR = OUT_DIR / "heightmap"
WORLDPAINTER_DIR = OUT_DIR / "worldpainter"

TARGET_MAX_DIM = 2048
SEA_LEVEL_BLOCK = 62

for d in [PROC_DIR, LOG_DIR, HEIGHTMAP_DIR, WORLDPAINTER_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def fail(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def find_input_raster() -> Path:
    candidates = list(RAW_DIR.glob("*.tif")) + list(RAW_DIR.glob("*.tiff"))
    # Also accept NetCDF files (NOAA CUDEM is sometimes distributed as .nc)
    candidates += list(RAW_DIR.glob("*.nc"))
    if not candidates:
        fail("No GeoTIFF or NetCDF found in data/raw/. Run fetch_dem.py first.")
    # Choose the largest file as the primary candidate
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def open_raster(path: Path):
    """Open a raster with rasterio, handling NetCDF subdatasets if needed."""
    try:
        src = rasterio.open(path)
        # Quick check — if it's a NetCDF with multiple subdatasets, pick the elevation band
        if hasattr(src, "subdatasets") and src.subdatasets:
            # prefer a subdataset whose name mentions 'Band1', 'z', or 'elevation'
            sd = src.subdatasets
            chosen = sd[0]
            for s in sd:
                sl = s.lower()
                if any(kw in sl for kw in ["z:", "elevation", "band1"]):
                    chosen = s
                    break
            src.close()
            src = rasterio.open(chosen)
        return src
    except Exception as exc:
        fail(f"Cannot open raster {path}: {exc}")


def main() -> None:
    in_raster = find_input_raster()
    print(f"Input raster: {in_raster}")

    audit_lines = []
    audit_lines.append(f"INPUT={in_raster.name}")

    src = open_raster(in_raster)
    audit_lines.append(f"CRS={src.crs}")
    audit_lines.append(f"WIDTH={src.width}")
    audit_lines.append(f"HEIGHT={src.height}")
    audit_lines.append(f"NODATA={src.nodata}")
    audit_lines.append(f"BOUNDS={src.bounds}")
    print(f"  CRS:    {src.crs}")
    print(f"  Size:   {src.width} x {src.height}")
    print(f"  Nodata: {src.nodata}")
    print(f"  Bounds: {src.bounds}")

    data = src.read(1).astype("float32")

    if src.nodata is not None:
        data[data == src.nodata] = np.nan

    data[~np.isfinite(data)] = np.nan

    # Reproject to Web Mercator for stable, square-pixel resizing
    dst_crs = "EPSG:3857"
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )

    reproj = np.full((height, width), np.nan, dtype="float32")

    reproject(
        source=data,
        destination=reproj,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    src.close()

    arr = reproj

    valid = np.isfinite(arr)
    if not np.any(valid):
        fail("Raster contains no valid cells after reprojection.")

    src_min = float(np.nanmin(arr))
    src_max = float(np.nanmax(arr))
    audit_lines.append(f"MIN={src_min:.4f}")
    audit_lines.append(f"MAX={src_max:.4f}")
    print(f"  Elevation range after reprojection: {src_min:.2f} m – {src_max:.2f} m")

    # Force underwater / nodata cells to 0 (ocean baseline).
    # This preserves the island silhouette cleanly.
    land = arr.copy()
    land[~np.isfinite(land)] = 0.0
    land[land < 0.0] = 0.0

    # Normalize only positive (land) terrain to 0–1.
    land_max = float(np.max(land))
    if land_max <= 0.0:
        fail("No positive terrain found after ocean baseline conversion. "
             "Check that the DEM covers Puerto Rico land area.")

    norm = land / land_max
    norm = np.clip(norm, 0.0, 1.0)
    print(f"  Land max elevation: {land_max:.2f} m (normalised to 1.0)")

    # Resize to fit within TARGET_MAX_DIM while preserving aspect ratio.
    h, w = norm.shape
    scale = min(TARGET_MAX_DIM / max(h, w), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    print(f"  Resize: {w}x{h} → {new_w}x{new_h}")

    # Convert to 8-bit grayscale image (no alpha).
    img = Image.fromarray((norm * 255).astype("uint8"), mode="L")
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

    # Guarantee mode L (no alpha channel).
    img = img.convert("L")

    if max(img.size) > TARGET_MAX_DIM:
        fail(f"Resized image {img.size} exceeds {TARGET_MAX_DIM}px — aborting.")

    heightmap_path = HEIGHTMAP_DIR / "puerto_rico_heightmap_2048.png"
    img.save(heightmap_path)
    print(f"  Saved heightmap: {heightmap_path}")

    # Machine-readable metadata
    metadata = {
        "input_raster": str(in_raster),
        "source_min_m": round(src_min, 4),
        "source_max_m": round(src_max, 4),
        "land_max_m": round(land_max, 4),
        "output_width": img.width,
        "output_height": img.height,
        "sea_level_block": SEA_LEVEL_BLOCK,
        "output_mode": img.mode,
        "notes": [
            "Underwater and nodata cells were forced to 0 baseline (ocean)",
            "Land elevation normalized: 0 m → pixel 0, land_max_m → pixel 255",
            "Output is 8-bit grayscale (mode L) without alpha channel",
            "Designed for WorldPainter heightmap import with linear mapping",
        ],
    }
    meta_path = HEIGHTMAP_DIR / "heightmap_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Append audit info
    audit_lines.append(f"LAND_MAX={land_max:.4f}")
    audit_lines.append(f"OUTPUT_WIDTH={img.width}")
    audit_lines.append(f"OUTPUT_HEIGHT={img.height}")
    audit_lines.append(f"OUTPUT_FILE={heightmap_path}")

    # Preserve existing source audit lines; merge if file already exists
    audit_path = LOG_DIR / "source_audit.txt"
    existing = ""
    if audit_path.exists():
        existing = audit_path.read_text(encoding="utf-8").strip()
    combined = existing + "\n" + "\n".join(audit_lines) if existing else "\n".join(audit_lines)
    write_text(audit_path, combined.strip() + "\n")

    # WorldPainter import instructions
    write_text(
        WORLDPAINTER_DIR / "import_settings.txt",
        "\n".join([
            "WorldPainter Heightmap Import Settings",
            "======================================",
            "",
            "Heightmap file: output/heightmap/puerto_rico_heightmap_2048.png",
            "Format:         8-bit grayscale PNG (no alpha)",
            "Mapping:        Linear",
            "Smoothing:      DISABLED (uncheck 'smooth terrain' on import)",
            "Water level:    62 (Minecraft block height)",
            "World type:     Surface",
            "",
            "Import steps (WorldPainter GUI):",
            "  1. File > Import > Import Height Map...",
            "  2. Select: output/heightmap/puerto_rico_heightmap_2048.png",
            "  3. Verify 'Grayscale' is selected (not RGBA)",
            "  4. Set 'Maximum height' to 255 (or your preferred ceiling)",
            "  5. Set 'Water level' to 62",
            "  6. Uncheck 'Smooth terrain'",
            "  7. Leave scale as default (one pixel = one block)",
            "  8. Click Import",
            "",
            "Spawn recommendation:",
            "  Place spawn in the northeast quadrant (San Juan / Loíza area)",
            "  Ensure spawn is on solid land above water level 62",
            "  Use WorldPainter Spawn Point tool to confirm placement",
            "",
            "Export:",
            "  File > Export > Export as Minecraft world...",
            "  Choose Minecraft Java Edition format",
            "  Set seed to any value (terrain is heightmap-driven)",
            "",
            "If the import looks curved or distorted:",
            "  - Confirm PNG mode is L (grayscale, no alpha)",
            "  - Confirm Linear mapping is selected (not logarithmic)",
            "  - Re-import with Smooth terrain DISABLED",
        ]) + "\n"
    )

    print("\nHEIGHTMAP_OK")
    print(str(heightmap_path))


if __name__ == "__main__":
    main()
