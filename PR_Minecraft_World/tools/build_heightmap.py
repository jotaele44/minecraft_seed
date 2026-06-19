"""
build_heightmap.py — Convert a Puerto Rico DEM GeoTIFF into a
WorldPainter-compatible grayscale PNG heightmap.

Usage:
    python tools/build_heightmap.py [--bits {8,16}] [--verbose]
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.crs
from rasterio.enums import Resampling as RasterioResampling
from rasterio.warp import calculate_default_transform, reproject
from PIL import Image

from tools._config import ROOT, load_config

log = logging.getLogger(__name__)

RAW_DIR    = ROOT / "data" / "raw"
OUT_DIR    = ROOT / "output"
LOG_DIR    = OUT_DIR / "logs"
HM_DIR     = OUT_DIR / "heightmap"
WP_DIR     = OUT_DIR / "worldpainter"

_RESAMPLING_MAP = {
    "bilinear": RasterioResampling.bilinear,
    "nearest":  RasterioResampling.nearest,
    "cubic":    RasterioResampling.cubic,
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fail(msg: str, code: int = 1) -> None:
    log.error(msg)
    sys.exit(code)


def _find_input_raster() -> Path:
    candidates = (
        list(RAW_DIR.glob("*.tif"))
        + list(RAW_DIR.glob("*.tiff"))
        + list(RAW_DIR.glob("*.nc"))
    )
    if not candidates:
        _fail("No GeoTIFF or NetCDF found in data/raw/. Run fetch_dem.py first.")
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def _open_raster(path: Path) -> rasterio.DatasetReader:
    """Open a raster, resolving NetCDF subdatasets by inspecting band dtypes."""
    try:
        src = rasterio.open(path)
    except Exception as exc:
        _fail(f"Cannot open raster {path}: {exc}")

    if not (hasattr(src, "subdatasets") and src.subdatasets):
        return src

    # NetCDF with subdatasets: pick the one whose band has a float/int dtype
    # and the widest value range (most likely elevation).
    log.debug("NetCDF subdatasets: %s", src.subdatasets)
    src.close()
    best_sd = src.subdatasets[0]
    best_range = -1.0
    for sd in src.subdatasets:
        try:
            with rasterio.open(sd) as probe:
                if probe.dtypes[0] not in ("float32", "float64", "int16", "int32"):
                    continue
                data = probe.read(1).astype("float64")
                nd = probe.nodata
                if nd is not None:
                    data = data[data != nd]
                if data.size == 0:
                    continue
                rng = float(data.max() - data.min())
                log.debug("  subdataset %s range=%.1f", sd, rng)
                if rng > best_range:
                    best_range = rng
                    best_sd = sd
        except Exception:
            continue

    log.info("Selected NetCDF subdataset: %s (range=%.1f)", best_sd, best_range)
    try:
        return rasterio.open(best_sd)
    except Exception as exc:
        _fail(f"Cannot open NetCDF subdataset {best_sd}: {exc}")


def _find_spawn_pixel(arr: np.ndarray, sea_px: int = 10) -> tuple[int, int]:
    """
    Find a safe land pixel in the configured spawn quadrant.
    Returns (pixel_x, pixel_y) → Minecraft block (X, Z).
    """
    h, w = arr.shape

    # Build quadrant slice
    def _ne():  return arr[:h // 2,  w * 6 // 10:]
    def _nw():  return arr[:h // 2,  :w * 4 // 10]
    def _se():  return arr[h // 2:,  w * 6 // 10:]
    def _sw():  return arr[h // 2:,  :w * 4 // 10]
    def _ctr(): return arr[h // 4:h * 3 // 4, w // 4:w * 3 // 4]

    offsets = {
        "northeast": (_ne,  w * 6 // 10, 0),
        "northwest": (_nw,  0,            0),
        "southeast": (_se,  w * 6 // 10, h // 2),
        "southwest": (_sw,  0,            h // 2),
        "center":    (_ctr, w // 4,       h // 4),
    }

    cfg = load_config()
    quadrant = cfg["spawn"].get("quadrant", "northeast")
    region_fn, ox, oy = offsets.get(quadrant, offsets["northeast"])
    region = region_fn()

    land_ys, land_xs = np.where(region > sea_px)
    if land_xs.size == 0:
        log.warning("No land in %s quadrant; falling back to any land pixel.", quadrant)
        land_ys, land_xs = np.where(arr > sea_px)
        if land_xs.size == 0:
            return w // 2, h // 4
        ox, oy = 0, 0

    cx = int(np.median(land_xs)) + ox
    cy = int(np.median(land_ys)) + oy
    return cx, cy


def _write_worldpainter_settings(
    hm_name: str,
    spawn_x: int,
    spawn_y: int,
    spawn_z: int,
    sea_level: int,
    max_height: int,
    mc_version: str,
    m_per_block_x: float = 0.0,
    m_per_block_z: float = 0.0,
    m_per_block_y: float = 0.0,
    include_bathy: bool = False,
    m_per_block_y_land: float = 0.0,
    m_per_block_y_ocean: float = 0.0,
    max_ocean_depth_m: float = 0.0,
) -> None:
    lines = [
        "WorldPainter Heightmap Import Settings",
        "======================================",
        "",
        f"Heightmap file:  output/heightmap/{hm_name}",
        "Format:          8-bit grayscale PNG (no alpha)  [or 16-bit if _16bit variant used]",
        "Mapping:         Linear",
        "Smoothing:       DISABLED (uncheck 'smooth terrain' on import)",
        f"Water level:     {sea_level} (Minecraft block height)",
        f"Maximum height:  {max_height}",
        "World type:      Surface",
        "",
        f"Minecraft version target: {mc_version}",
        "  pre-1.18 → water level 62, max height 255",
        "  1.18+    → water level 63, max height 384  (update values above)",
        "",
        "Pixel value → terrain mapping:",
        *(
            [
                f"  Pixels  0 – {sea_level-1:3d}  →  ocean depth "
                f"(0 = deepest ~{max_ocean_depth_m:.0f} m, {sea_level-1} = near surface)",
                f"  Pixel  {sea_level:3d}        →  sea level / coastline",
                f"  Pixels {sea_level:3d} – {max_height:3d}  →  land elevation "
                f"(0 m at coast → peak at {max_height})",
            ] if include_bathy else
            [
                "  Pixel   0          →  ocean floor (flat)",
                f"  Pixel  {max_height:3d}        →  land peak",
            ]
        ),
        "",
        "Scale conversion (block ↔ real world):",
        f"  1 block (X/Z)    ≈ {m_per_block_x:.1f} m horizontal",
        *(
            [
                f"  1 block (Y) land  ≈ {m_per_block_y_land:.2f} m elevation",
                f"  1 block (Y) ocean ≈ {m_per_block_y_ocean:.2f} m depth",
                f"  Real-world land elevation  = (block_Y - {sea_level}) × {m_per_block_y_land:.2f} m",
                f"  Real-world ocean depth     = ({sea_level} - block_Y) × {m_per_block_y_ocean:.2f} m",
            ] if include_bathy else
            [
                f"  1 block (Y)       ≈ {m_per_block_y:.2f} m elevation",
                f"  Real-world elevation = block_Y × {m_per_block_y:.2f} m",
            ]
        ),
        "",
        "Import steps (WorldPainter GUI):",
        "  1. File > Import > Import Height Map...",
        f"  2. Select: output/heightmap/{hm_name}",
        "  3. Verify 'Grayscale' is selected (not RGBA)",
        f"  4. Set 'Maximum height' to {max_height}",
        f"  5. Set 'Water level' to {sea_level}",
        "  6. Uncheck 'Smooth terrain'",
        "  7. Leave scale as default (1 pixel = 1 block)",
        "  8. Click Import",
        "",
        "CLI alternative (WorldPainter command-line):",
        f"  worldpainter -import output/heightmap/{hm_name} \\",
        "               --config output/worldpainter/worldpainter_import.properties",
        "",
        "Computed spawn (set this in WorldPainter Spawn Point tool):",
        f"  Block X = {spawn_x}",
        f"  Block Y = {spawn_y}  (approximate surface; WorldPainter will adjust)",
        f"  Block Z = {spawn_z}",
        "  Region:  northeast quadrant (San Juan / Loíza area)",
        "",
        "Export:",
        "  File > Export > Export as Minecraft world...",
        "  Choose Minecraft Java Edition format",
        "",
        "Troubleshooting:",
        "  Curved/distorted import → confirm PNG is mode L, mapping is Linear, smoothing off",
        "  Blank world            → check source_audit.txt for elevation range",
        "  Wrong sea level        → update water level to match your Minecraft version",
    ]
    (WP_DIR / "import_settings.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(bits_override: int | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Puerto Rico Minecraft heightmap.")
    parser.add_argument(
        "--bits", type=int, choices=[8, 16], default=None,
        help="Output bit depth (overrides config.toml). Default: 8.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cfg = load_config()
    hm_cfg = cfg["heightmap"]
    mc_cfg = cfg["minecraft"]

    target_max_dim = hm_cfg["target_max_dim"]
    bits = bits_override or args.bits or hm_cfg["bits"]
    sea_level = mc_cfg["sea_level_block"]
    max_height = mc_cfg["max_height"]
    mc_version = mc_cfg["version"]
    resample_alg = _RESAMPLING_MAP.get(hm_cfg["resampling"], RasterioResampling.bilinear)

    for d in [LOG_DIR, HM_DIR, WP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    in_raster = _find_input_raster()
    log.info("Input raster: %s", in_raster)

    # Per-run audit section
    run_ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    audit: list[str] = [f"--- RUN {run_ts} ---"]
    audit.append(f"INPUT={in_raster.relative_to(ROOT)}")

    # -----------------------------------------------------------------
    # Open and inspect
    # -----------------------------------------------------------------
    src = _open_raster(in_raster)
    log.info("  CRS:    %s", src.crs)
    log.info("  Size:   %d x %d", src.width, src.height)
    log.info("  Nodata: %s", src.nodata)
    log.info("  Bounds: %s", src.bounds)
    audit += [
        f"CRS={src.crs}",
        f"WIDTH={src.width}",
        f"HEIGHT={src.height}",
        f"NODATA={src.nodata}",
        f"BOUNDS={src.bounds}",
    ]

    # NOAA NetCDF files (e.g. crm_vol9.nc) often have no embedded CRS.
    # If the bounds fall within WGS84 lat/lon range, assume EPSG:4326.
    if src.crs is None:
        b = src.bounds
        if -180 <= b.left <= 180 and -90 <= b.bottom <= 90:
            log.warning(
                "Source raster has no CRS; bounds %s look like WGS84 — assuming EPSG:4326.", b
            )
            _src_crs = rasterio.crs.CRS.from_epsg(4326)
        else:
            _fail(
                "Source raster has no embedded CRS and bounds don't look like lat/lon. "
                "Cannot reproject."
            )
    else:
        _src_crs = src.crs
    audit.append(f"CRS_ASSUMED={src.crs is None}")

    # Memory warning for large files
    file_mb = in_raster.stat().st_size / (1 << 20)
    if file_mb > 500:
        log.warning(
            "Large raster (%.0f MiB) — reprojection may use significant RAM.", file_mb
        )

    # -----------------------------------------------------------------
    # Read band 1, mask nodata
    # -----------------------------------------------------------------
    import os
    data = src.read(1).astype("float32")
    if src.nodata is not None:
        data[data == src.nodata] = np.nan
    data[~np.isfinite(data)] = np.nan

    # -----------------------------------------------------------------
    # Reproject to Web Mercator (stable square pixels)
    # -----------------------------------------------------------------
    dst_crs = "EPSG:3857"
    transform, rp_w, rp_h = calculate_default_transform(
        _src_crs, dst_crs, src.width, src.height, *src.bounds
    )
    src_transform = src.transform
    src_crs = _src_crs
    src.close()

    reproj = np.full((rp_h, rp_w), np.nan, dtype="float32")
    reproject(
        source=data,
        destination=reproj,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=resample_alg,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        num_threads=os.cpu_count() or 1,
    )

    valid = np.isfinite(reproj)
    if not np.any(valid):
        _fail("Raster contains no valid cells after reprojection.")

    src_min = float(np.nanmin(reproj))
    src_max = float(np.nanmax(reproj))
    log.info("  Elevation range: %.2f m – %.2f m", src_min, src_max)
    audit += [f"MIN={src_min:.4f}", f"MAX={src_max:.4f}"]

    # -----------------------------------------------------------------
    # Normalise: dual mode (land + ocean) or legacy (land only)
    # -----------------------------------------------------------------
    include_bathy = cfg["heightmap"].get("include_bathymetry", False)
    max_ocean_depth = float(cfg["heightmap"].get("max_ocean_depth_m", 500))

    filled = reproj.copy()
    filled[~np.isfinite(filled)] = 0.0

    land_max = float(np.max(np.maximum(filled, 0.0)))
    if land_max <= 0.0:
        _fail(
            "No positive terrain found. Check that the DEM covers Puerto Rico. "
            "Inspect MIN/MAX in source_audit.txt."
        )

    norm = np.zeros_like(filled, dtype="float32")

    if include_bathy:
        # norm is a [0, 1] fraction multiplied by 255 to get pixel value.
        # Ocean:  [-max_ocean_depth, 0] → pixel [0,          sea_level - 1]
        # Land:   [0, land_max]         → pixel [sea_level,  max_height   ]
        ocean_mask = filled < 0.0
        land_mask  = ~ocean_mask

        norm[ocean_mask] = np.clip(
            (filled[ocean_mask] + max_ocean_depth) / max_ocean_depth * (sea_level / 255.0),
            0.0,
            (sea_level - 1) / 255.0,
        )
        norm[land_mask] = sea_level / 255.0 + np.clip(
            filled[land_mask] / land_max * ((max_height - sea_level) / 255.0),
            0.0,
            (max_height - sea_level) / 255.0,
        )
        log.info(
            "  Land max: %.2f m (→ pixel %d)  |  ocean capped at %.0f m (→ pixel 0)",
            land_max, max_height, max_ocean_depth,
        )
        log.info(
            "  Pixel mapping: 0–%d = ocean, %d = coast, %d–%d = land",
            sea_level - 1, sea_level, sea_level, max_height,
        )
    else:
        # Legacy: flat ocean floor at pixel 0
        filled[filled < 0.0] = 0.0
        norm = np.clip(filled / land_max, 0.0, 1.0)
        log.info("  Land max: %.2f m (→ pixel %d)", land_max, max_height)

    # -----------------------------------------------------------------
    # Resize to target_max_dim (preserve aspect ratio)
    # -----------------------------------------------------------------
    rh, rw = norm.shape
    scale = min(target_max_dim / max(rh, rw), 1.0)
    new_w = max(1, int(round(rw * scale)))
    new_h = max(1, int(round(rh * scale)))
    log.info("  Resize: %dx%d → %dx%d", rw, rh, new_w, new_h)

    # -----------------------------------------------------------------
    # 8-bit output (always produced)
    # -----------------------------------------------------------------
    img8 = Image.fromarray((norm * 255).astype("uint8"), mode="L")
    if (new_w, new_h) != (rw, rh):
        img8 = img8.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    img8 = img8.convert("L")

    if max(img8.size) > target_max_dim:
        _fail(f"Resized image {img8.size} exceeds {target_max_dim}px — aborting.")

    hm_path = HM_DIR / "puerto_rico_heightmap.png"
    img8.save(hm_path)
    log.info("  Saved 8-bit heightmap: %s", hm_path)

    # -----------------------------------------------------------------
    # 16-bit output (when requested or always alongside 8-bit)
    # -----------------------------------------------------------------
    hm_16_path = HM_DIR / "puerto_rico_heightmap_16bit.png"
    arr_16 = (norm * 65535).astype(np.uint16)
    img16_src = Image.fromarray(arr_16.astype(np.int32), mode="I")
    if (new_w, new_h) != (rw, rh):
        img16_src = img16_src.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    img16_src.save(hm_16_path)
    log.info("  Saved 16-bit heightmap: %s", hm_16_path)

    # -----------------------------------------------------------------
    # Spawn coordinate
    # -----------------------------------------------------------------
    arr8 = np.array(img8)
    # With bathymetry, land pixels start at sea_level value (62); use that as threshold
    spawn_sea_px = sea_level if include_bathy else 10
    spawn_px, spawn_pz = _find_spawn_pixel(arr8, sea_px=spawn_sea_px)
    spawn_y = sea_level + 1
    log.info("  Spawn: block X=%d, Y=%d, Z=%d", spawn_px, spawn_y, spawn_pz)

    # -----------------------------------------------------------------
    # Scale conversion factors (block ↔ real-world metres)
    # The reprojected raster spans the bounding box in EPSG:3857 metres.
    # Dividing that span by the output pixel count gives m/block.
    # With dual normalization, land and ocean have different vertical scales.
    # -----------------------------------------------------------------
    reproj_width_m  = abs(transform.c + transform.a * rp_w - transform.c)
    reproj_height_m = abs(transform.f + transform.e * rp_h - transform.f)
    m_per_block_x = reproj_width_m  / img8.width  if img8.width  > 0 else 0.0
    m_per_block_z = reproj_height_m / img8.height if img8.height > 0 else 0.0
    land_px_count  = max_height - sea_level   # pixel levels allocated to land
    ocean_px_count = sea_level                # pixel levels allocated to ocean
    m_per_block_y_land  = land_max / land_px_count  if land_px_count  > 0 else 0.0
    m_per_block_y_ocean = (max_ocean_depth / ocean_px_count if ocean_px_count > 0 else 0.0
                           ) if include_bathy else None
    log.info(
        "  Scale: %.1f m/block (X/Z)  |  land %.2f m/block (Y)  |  ocean %s m/block (Y)",
        m_per_block_x,
        m_per_block_y_land,
        f"{m_per_block_y_ocean:.2f}" if m_per_block_y_ocean is not None else "n/a",
    )

    # -----------------------------------------------------------------
    # Metadata (relative paths)
    # -----------------------------------------------------------------
    scale_section: dict = {
        "m_per_block_x": round(m_per_block_x, 2),
        "m_per_block_z": round(m_per_block_z, 2),
        "m_per_block_y_land": round(m_per_block_y_land, 4),
        "description": (
            "Multiply block coords by m_per_block values to get real-world metres. "
            f"Sea level = block Y {sea_level}. "
            f"Pixels 0–{sea_level-1} = ocean depth, "
            f"pixels {sea_level}–{max_height} = land elevation."
            if include_bathy else
            "Multiply block coords by m_per_block_x/z to get real-world metres."
        ),
    }
    if m_per_block_y_ocean is not None:
        scale_section["m_per_block_y_ocean"] = round(m_per_block_y_ocean, 4)
        scale_section["max_ocean_depth_m"] = max_ocean_depth

    metadata = {
        "input_raster": str(in_raster.relative_to(ROOT)),
        "source_min_m": round(src_min, 4),
        "source_max_m": round(src_max, 4),
        "land_max_m": round(land_max, 4),
        "output_width": img8.width,
        "output_height": img8.height,
        "sea_level_block": sea_level,
        "max_height": max_height,
        "minecraft_version": mc_version,
        "output_mode": img8.mode,
        "spawn_x": spawn_px,
        "spawn_y": spawn_y,
        "spawn_z": spawn_pz,
        "bathymetry_enabled": include_bathy,
        "scale": scale_section,
        "notes": [
            (
                f"Dual normalisation: pixels 0–{sea_level-1} = ocean depth "
                f"(0=deepest, {sea_level-1}=near-surface), "
                f"pixel {sea_level} = coast/sea level, "
                f"pixels {sea_level}–{max_height} = land elevation"
            ) if include_bathy else
            "Underwater and nodata cells forced to pixel 0 (flat ocean floor)",
            "Output is 8-bit grayscale (mode L) without alpha channel",
            "16-bit variant also available: puerto_rico_heightmap_16bit.png",
            "Spawn coordinates are pixel-to-block (X=col, Z=row); Y is approximate surface",
            "Designed for WorldPainter heightmap import with linear mapping",
        ],
    }
    meta_path = HM_DIR / "heightmap_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------
    # Audit log — append new run section atomically
    # -----------------------------------------------------------------
    audit += [
        f"LAND_MAX={land_max:.4f}",
        f"OUTPUT_WIDTH={img8.width}",
        f"OUTPUT_HEIGHT={img8.height}",
        f"SPAWN_X={spawn_px}",
        f"SPAWN_Z={spawn_pz}",
        f"OUTPUT_FILE={hm_path.relative_to(ROOT)}",
    ]
    audit_path = LOG_DIR / "source_audit.txt"
    existing = audit_path.read_text(encoding="utf-8").rstrip() if audit_path.exists() else ""
    new_section = "\n".join(audit)
    combined = (existing + "\n\n" + new_section).lstrip()
    tmp = audit_path.with_suffix(".tmp")
    tmp.write_text(combined + "\n", encoding="utf-8")
    tmp.rename(audit_path)

    # -----------------------------------------------------------------
    # WorldPainter import settings
    # -----------------------------------------------------------------
    _write_worldpainter_settings(
        hm_name=hm_path.name,
        spawn_x=spawn_px,
        spawn_y=spawn_y,
        spawn_z=spawn_pz,
        sea_level=sea_level,
        max_height=max_height,
        mc_version=mc_version,
        m_per_block_x=m_per_block_x,
        m_per_block_z=m_per_block_z,
        m_per_block_y=m_per_block_y_land,
        include_bathy=include_bathy,
        m_per_block_y_land=m_per_block_y_land,
        m_per_block_y_ocean=m_per_block_y_ocean or 0.0,
        max_ocean_depth_m=max_ocean_depth if include_bathy else 0.0,
    )

    log.info("\nHEIGHTMAP_OK")
    log.info("%s", hm_path)


if __name__ == "__main__":
    main()
