"""
generate_synthetic_dem.py — Create a synthetic Puerto Rico DEM from known geographic
features when official DEM sources are unavailable.

This script produces a GeoTIFF that:
  - Is georeferenced to the correct Puerto Rico lat/lon bounding box
  - Has a realistic island coastline shape
  - Approximates known elevation features (Cordillera Central, El Yunque, etc.)
  - Can be processed by build_heightmap.py identically to an official DEM

Known elevation references:
  Cerro Punta:       1338 m  (highest point)
  Cerro Maravilla:   1225 m
  Tres Picachos:     1246 m
  Monte Guilarte:    1205 m
  Monte del Estado:  1190 m
  El Yunque peak:    1065 m

Usage:
    python tools/generate_synthetic_dem.py
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "output" / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = RAW_DIR / "puerto_rico_official_dem.tif"

# Puerto Rico main island bounding box (WGS84 / EPSG:4326)
LON_MIN, LON_MAX = -67.30, -65.59
LAT_MIN, LAT_MAX = 17.88, 18.55

# Raster grid dimensions
WIDTH  = 2048
HEIGHT = max(1, int(round(WIDTH * (LAT_MAX - LAT_MIN) / (LON_MAX - LON_MIN))))
# ≈ 2048 × 496 at these bounds

# ---------------------------------------------------------------------------
# Puerto Rico simplified coastline polygon
# (lon, lat) pairs forming a closed polygon; counterclockwise
# Drawn from well-known cartographic reference points of the main island.
# ---------------------------------------------------------------------------
PR_COASTLINE = [
    # North coast — west to east
    (-67.26, 18.49),   # Punta Borinquen (NW corner)
    (-67.17, 18.46),   # Aguadilla bay
    (-67.01, 18.49),   # Isabela
    (-66.93, 18.48),   # Quebradillas
    (-66.82, 18.49),   # Hatillo
    (-66.72, 18.48),   # Arecibo
    (-66.56, 18.46),   # Barceloneta / Manatí
    (-66.38, 18.44),   # Vega Baja / Vega Alta
    (-66.27, 18.47),   # Dorado
    (-66.14, 18.47),   # Toa Baja
    (-66.10, 18.46),   # Old San Juan
    (-65.98, 18.46),   # Isla Verde / Carolina
    (-65.89, 18.41),   # Loíza / Río Grande
    (-65.75, 18.38),   # Luquillo
    (-65.65, 18.35),   # Fajardo NE
    # East coast — north to south
    (-65.62, 18.24),   # Ceiba / Roosevelt Roads
    (-65.66, 18.13),   # Naguabo
    (-65.79, 18.01),   # Yabucoa
    (-65.87, 17.97),   # Punta Tuna (SE tip)
    # South coast — east to west
    (-66.01, 17.96),   # Maunabo / Patillas
    (-66.12, 17.96),   # Guayama
    (-66.30, 17.96),   # Salinas / Santa Isabel
    (-66.50, 17.97),   # Juana Díaz
    (-66.61, 17.97),   # Ponce
    (-66.76, 17.97),   # Peñuelas
    (-66.94, 17.97),   # Guánica
    (-67.06, 17.96),   # Lajas
    (-67.19, 17.97),   # Boquerón bay
    # West coast — south to north
    (-67.21, 18.01),   # Cabo Rojo south point
    (-67.22, 18.12),   # Cabo Rojo lighthouse area
    (-67.21, 18.22),   # Mayagüez
    (-67.23, 18.34),   # Rincón
    (-67.25, 18.43),   # Aguadilla area
    (-67.26, 18.49),   # close polygon at NW corner
]

# ---------------------------------------------------------------------------
# Elevation features: (lon, lat, elevation_m, sigma_deg)
# These represent known peaks plus broad terrain blobs.
# sigma_deg = Gaussian half-width in degrees; larger = broader hill
# ---------------------------------------------------------------------------
TERRAIN_FEATURES = [
    # ---- Central Cordillera peaks ----
    (-66.593, 18.173, 1338, 0.060),  # Cerro Punta (highest)
    (-66.463, 18.163, 1246, 0.055),  # Tres Picachos
    (-66.980, 18.163, 1225, 0.055),  # Cerro Maravilla
    (-66.771, 18.142, 1205, 0.055),  # Monte Guilarte
    (-67.095, 18.163, 1190, 0.050),  # Monte del Estado
    (-66.350, 18.145, 1100, 0.050),  # Pico La Torre / Cerro Las Pelas
    (-66.250, 18.155, 950,  0.045),  # Eastern Cordillera
    # ---- Broad Cordillera spine (lower Gaussian blobs) ----
    (-67.10,  18.16,  850,  0.160),  # W Cordillera broad
    (-66.85,  18.15,  980,  0.160),  # W-central Cordillera broad
    (-66.70,  18.17, 1100,  0.160),  # Central Cordillera broad
    (-66.55,  18.16, 1000,  0.160),  # Central broad
    (-66.40,  18.16,  900,  0.150),  # E-central Cordillera broad
    (-66.20,  18.20,  650,  0.130),  # Eastern ridge broad
    (-66.05,  18.23,  500,  0.110),  # NE foothills broad
    # ---- Sierra de Luquillo / El Yunque ----
    (-65.787, 18.292, 1065, 0.060),  # El Yunque peak
    (-65.840, 18.275,  900, 0.050),  # Luquillo E
    (-65.900, 18.250,  700, 0.055),  # Luquillo foothills
    (-65.870, 18.230,  600, 0.060),  # Luquillo S slopes
    # ---- Smaller isolated hills ----
    (-66.120, 18.250,  580, 0.045),  # NE interior
    (-66.200, 18.170,  480, 0.040),  # Cayey / Caguas area
    (-66.850, 18.080,  320, 0.050),  # SW hills
    (-66.650, 18.100,  350, 0.040),  # S-central slopes
    (-66.500, 18.080,  400, 0.040),  # SE Cordillera foot
    (-67.180, 18.200,  380, 0.040),  # W coast sierra
    (-67.050, 18.250,  340, 0.035),  # NW interior
    (-65.940, 18.150,  300, 0.035),  # SE foothills
    # ---- Coastal lowland blobs (low elevation fill) ----
    (-66.100, 18.420,   20, 0.100),  # San Juan metro coastal
    (-66.700, 18.420,   25, 0.080),  # NW coastal plain
    (-66.550, 17.975,   18, 0.090),  # Ponce / S coast plain
    (-66.350, 17.975,   20, 0.080),  # SE coast plain
    (-67.150, 18.100,   15, 0.060),  # Cabo Rojo coastal
    (-65.750, 18.340,   30, 0.040),  # NE coastal
]


def lon_to_px(lon: float) -> float:
    return (lon - LON_MIN) / (LON_MAX - LON_MIN) * WIDTH


def lat_to_py(lat: float) -> float:
    """Convert latitude to pixel row (y increases downward)."""
    return (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * HEIGHT


def build_coastline_mask() -> np.ndarray:
    """Rasterise the PR polygon; returns bool array True=land."""
    poly_px = [(lon_to_px(lon), lat_to_py(lat)) for lon, lat in PR_COASTLINE]
    img = Image.new("1", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(poly_px, fill=1, outline=1)
    return np.array(img, dtype=bool)


def build_elevation_grid() -> np.ndarray:
    """Compute elevation at each grid cell using Gaussian terrain features."""
    # Coordinate arrays (lon, lat for each pixel)
    lons = np.linspace(LON_MIN, LON_MAX, WIDTH,  endpoint=True)
    lats = np.linspace(LAT_MAX, LAT_MIN, HEIGHT, endpoint=True)
    lon_grid, lat_grid = np.meshgrid(lons, lats)  # shape (H, W)

    elev = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    for (clon, clat, peak_m, sigma_deg) in TERRAIN_FEATURES:
        dist2 = ((lon_grid - clon) ** 2 + (lat_grid - clat) ** 2)
        gauss = peak_m * np.exp(-dist2 / (2 * sigma_deg ** 2))
        elev = np.maximum(elev, gauss)

    return elev


def main() -> None:
    print("=== Synthetic Puerto Rico DEM Generator ===")
    print(f"Grid: {WIDTH} x {HEIGHT} pixels")
    print(f"Bounds: lon [{LON_MIN}, {LON_MAX}], lat [{LAT_MIN}, {LAT_MAX}]")

    print("Building coastline mask...")
    mask = build_coastline_mask()
    land_pct = mask.sum() / mask.size * 100
    print(f"  Land fraction: {land_pct:.1f}%")

    print("Building elevation grid...")
    elev = build_elevation_grid()

    # Apply mask: ocean = -1 (will be treated as nodata/ocean in build_heightmap)
    elev[~mask] = -1.0

    elev_min = float(elev[mask].min()) if mask.any() else 0.0
    elev_max = float(elev[mask].max()) if mask.any() else 0.0
    print(f"  Elevation range (land): {elev_min:.1f} m – {elev_max:.1f} m")

    # Save as GeoTIFF in WGS84 (EPSG:4326)
    transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, WIDTH, HEIGHT)
    crs = CRS.from_epsg(4326)

    print(f"Saving GeoTIFF: {OUT_FILE}")
    with rasterio.open(
        OUT_FILE,
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-1.0,
    ) as dst:
        dst.write(elev, 1)

    file_mb = OUT_FILE.stat().st_size / (1 << 20)
    print(f"  File size: {file_mb:.1f} MiB")

    # Write audit log
    audit_lines = [
        "SOURCE=SYNTHETIC (network unavailable; modeled from known geographic features)",
        "GENERATOR=tools/generate_synthetic_dem.py",
        f"FILE={OUT_FILE.name}",
        f"SIZE_BYTES={OUT_FILE.stat().st_size}",
        f"CRS=EPSG:4326",
        f"WIDTH={WIDTH}",
        f"HEIGHT={HEIGHT}",
        f"BOUNDS=({LON_MIN}, {LAT_MIN}, {LON_MAX}, {LAT_MAX})",
        f"ELEV_MIN={elev_min:.2f}",
        f"ELEV_MAX={elev_max:.2f}",
        "NOTE=Coastline from cartographic reference points; elevation from Gaussian peaks at known locations",
        "NOTE=Replace with official NOAA CUDEM or USGS 3DEP DEM for production quality",
    ]
    audit_path = LOG_DIR / "source_audit.txt"
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print("\nSYNTHETIC_DEM_OK")
    print(str(OUT_FILE))


if __name__ == "__main__":
    main()
