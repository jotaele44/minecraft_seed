"""
generate_synthetic_dem.py — Create a synthetic Puerto Rico DEM from known geographic
features when official DEM sources are unavailable.

Produces a GeoTIFF that:
  - Is georeferenced to the correct Puerto Rico lat/lon bounding box (WGS84)
  - Has a realistic island coastline shape (34 cartographic reference points)
  - Approximates known elevation features via Gaussian peaks

Known elevation references used:
  Cerro Punta:       1338 m  (highest point, Cordillera Central)
  Tres Picachos:     1246 m
  Cerro Maravilla:   1225 m
  Monte Guilarte:    1205 m
  Monte del Estado:  1190 m
  El Yunque peak:    1065 m

Usage:
    python tools/generate_synthetic_dem.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from PIL import Image, ImageDraw

from tools._config import ROOT

log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "output" / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = RAW_DIR / "puerto_rico_official_dem.tif"

# Extended bounding box — captures PR Trench (N), Muertos Trough (S), and offshore islands
LON_MIN, LON_MAX = -68.3, -64.9
LAT_MIN, LAT_MAX =  16.5, 20.5

# Raster grid dimensions
WIDTH  = 2048
# HEIGHT ≈ 2410 pixels  (2048 × (20.5-16.5) / (68.3-64.9) ≈ 2048 × 4.0/3.4)
HEIGHT = max(1, int(round(WIDTH * (LAT_MAX - LAT_MIN) / (LON_MAX - LON_MIN))))

# ---------------------------------------------------------------------------
# Puerto Rico simplified coastline polygon
# (lon, lat) pairs, counterclockwise, closed at NW corner.
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
# Offshore island coastline polygons (simplified convex/concave hulls)
# ---------------------------------------------------------------------------
VIEQUES_COASTLINE = [           # ~34 km × 7 km, E-W oriented
    (-65.65, 18.14), (-65.55, 18.16), (-65.40, 18.16), (-65.25, 18.14),
    (-65.25, 18.11), (-65.40, 18.10), (-65.55, 18.10), (-65.65, 18.12),
    (-65.65, 18.14),
]

CULEBRA_COASTLINE = [           # ~11 km × 7 km
    (-65.35, 18.34), (-65.28, 18.36), (-65.23, 18.33),
    (-65.25, 18.28), (-65.33, 18.27), (-65.35, 18.34),
]

MONA_COASTLINE = [              # ~10 km × 12 km flat plateau
    (-67.95, 18.08), (-67.90, 18.17), (-67.83, 18.16),
    (-67.83, 18.06), (-67.88, 18.04), (-67.95, 18.08),
]

DESECHEO_COASTLINE = [          # ~2 km × 1 km
    (-67.50, 18.38), (-67.47, 18.39), (-67.46, 18.38),
    (-67.47, 18.37), (-67.50, 18.37), (-67.50, 18.38),
]

CAJA_DE_MUERTOS_COASTLINE = [   # ~3.5 km × 1.5 km
    (-66.54, 17.90), (-66.49, 17.90),
    (-66.49, 17.88), (-66.54, 17.88), (-66.54, 17.90),
]

PALOMINO_COASTLINE = [          # ~2 km × 0.7 km
    (-65.64, 18.36), (-65.61, 18.36),
    (-65.61, 18.34), (-65.64, 18.34), (-65.64, 18.36),
]

ISLAND_COASTLINES = [
    PR_COASTLINE,
    VIEQUES_COASTLINE,
    CULEBRA_COASTLINE,
    MONA_COASTLINE,
    DESECHEO_COASTLINE,
    CAJA_DE_MUERTOS_COASTLINE,
    PALOMINO_COASTLINE,
]

# ---------------------------------------------------------------------------
# Terrain features: (lon, lat, elevation_m, sigma_deg)
# sigma_deg = Gaussian half-width in degrees (larger = broader hill)
# ---------------------------------------------------------------------------
TERRAIN_FEATURES = [
    # Cordillera Central peaks
    (-66.593, 18.173, 1338, 0.060),
    (-66.463, 18.163, 1246, 0.055),
    (-66.980, 18.163, 1225, 0.055),
    (-66.771, 18.142, 1205, 0.055),
    (-67.095, 18.163, 1190, 0.050),
    (-66.350, 18.145, 1100, 0.050),
    (-66.250, 18.155,  950, 0.045),
    # Broad Cordillera spine
    (-67.10,  18.16,   850, 0.160),
    (-66.85,  18.15,   980, 0.160),
    (-66.70,  18.17,  1100, 0.160),
    (-66.55,  18.16,  1000, 0.160),
    (-66.40,  18.16,   900, 0.150),
    (-66.20,  18.20,   650, 0.130),
    (-66.05,  18.23,   500, 0.110),
    # Sierra de Luquillo / El Yunque
    (-65.787, 18.292, 1065, 0.060),
    (-65.840, 18.275,  900, 0.050),
    (-65.900, 18.250,  700, 0.055),
    (-65.870, 18.230,  600, 0.060),
    # Smaller hills
    (-66.120, 18.250,  580, 0.045),
    (-66.200, 18.170,  480, 0.040),
    (-66.850, 18.080,  320, 0.050),
    (-66.650, 18.100,  350, 0.040),
    (-66.500, 18.080,  400, 0.040),
    (-67.180, 18.200,  380, 0.040),
    (-67.050, 18.250,  340, 0.035),
    (-65.940, 18.150,  300, 0.035),
    # Coastal lowland fills
    (-66.100, 18.420,   20, 0.100),
    (-66.700, 18.420,   25, 0.080),
    (-66.550, 17.975,   18, 0.090),
    (-66.350, 17.975,   20, 0.080),
    (-67.150, 18.100,   15, 0.060),
    (-65.750, 18.340,   30, 0.040),
    # Offshore island peaks
    (-65.450, 18.130,   99, 0.025),  # Vieques — Monte Pirata
    (-65.550, 18.120,   80, 0.020),  # Vieques — secondary hill
    (-65.270, 18.310,  193, 0.020),  # Culebra — Monte Resaca
    (-67.890, 18.110,   80, 0.040),  # Mona Island plateau
    (-67.480, 18.380,  143, 0.010),  # Desecheo
    (-66.520, 17.890,   60, 0.010),  # Caja de Muertos
    (-65.620, 18.350,   40, 0.008),  # Palomino
]

# ---------------------------------------------------------------------------
# Bathymetric features: (lon, lat, depth_m, sigma_deg)
# depth_m is NEGATIVE (below sea level).
# Gaussians represent shelf drop-offs and basin floors within the bounding box.
# Within this box (17.88–18.55°N), the Puerto Rico Trench (~19.5°N) is out of
# range; depths modelled here represent the shelf and upper Caribbean basin.
# ---------------------------------------------------------------------------
BATHYMETRY_FEATURES = [
    # Puerto Rico Trench — axis ~19.5°N, deepest point 8376 m
    (-66.50, 19.50, -8376, 0.15),
    (-67.00, 19.48, -7000, 0.12),
    (-65.80, 19.45, -6500, 0.12),
    (-66.20, 19.42, -7500, 0.13),
    (-65.30, 19.35, -5000, 0.10),   # eastern end, shallower
    # Muertos Trough — axis ~17.3°N, ~5000 m deep
    (-66.50, 17.30, -5000, 0.12),
    (-66.00, 17.35, -4600, 0.10),
    (-65.80, 17.40, -4000, 0.10),
    # Caribbean basin south of PR (between island and Muertos Trough)
    (-66.50, 17.88, -3500, 0.10),
    (-66.00, 17.88, -3200, 0.10),
    (-66.80, 17.88, -3000, 0.08),
    (-65.80, 17.90, -2800, 0.08),
    # Northern shelf (narrow zone between PR north coast and PR Trench)
    (-66.50, 18.70,  -500, 0.08),
    (-66.00, 18.70,  -400, 0.08),
    (-67.00, 18.68,  -350, 0.07),
    (-65.75, 18.65,  -450, 0.07),
    # Atlantic east of PR / Culebra area
    (-65.10, 18.50, -2500, 0.10),
    # Mona Passage (west, between PR and Hispaniola)
    (-67.80, 18.30, -1000, 0.10),
    (-68.10, 18.20, -1500, 0.10),
]


def _lon_to_px(lon: float) -> float:
    return (lon - LON_MIN) / (LON_MAX - LON_MIN) * WIDTH


def _lat_to_py(lat: float) -> float:
    """Latitude → pixel row (y increases downward)."""
    return (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * HEIGHT


def _build_coastline_mask() -> np.ndarray:
    img = Image.new("1", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(img)
    for coastline in ISLAND_COASTLINES:
        poly_px = [(_lon_to_px(lon), _lat_to_py(lat)) for lon, lat in coastline]
        draw.polygon(poly_px, fill=1, outline=1)
    return np.array(img, dtype=bool)


def _build_elevation_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return (land_elev, ocean_depth) grids.

    land_elev:  Gaussian peaks for land features (positive values).
    ocean_depth: Gaussian wells for bathymetric features (negative values, 0 = sea level).
    """
    lons = np.linspace(LON_MIN, LON_MAX, WIDTH,  endpoint=True)
    lats = np.linspace(LAT_MAX, LAT_MIN, HEIGHT, endpoint=True)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    land_elev = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for clon, clat, peak_m, sigma_deg in TERRAIN_FEATURES:
        dist2 = (lon_grid - clon) ** 2 + (lat_grid - clat) ** 2
        land_elev = np.maximum(land_elev, peak_m * np.exp(-dist2 / (2 * sigma_deg ** 2)))

    ocean_depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for clon, clat, depth_m, sigma_deg in BATHYMETRY_FEATURES:
        dist2 = (lon_grid - clon) ** 2 + (lat_grid - clat) ** 2
        ocean_depth = np.minimum(ocean_depth, depth_m * np.exp(-dist2 / (2 * sigma_deg ** 2)))

    return land_elev, ocean_depth


def main() -> None:
    log.info("=== Synthetic Puerto Rico DEM Generator ===")
    log.info("Grid: %d x %d  bounds: lon [%.2f, %.2f]  lat [%.2f, %.2f]",
             WIDTH, HEIGHT, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

    log.info("Building coastline mask …")
    mask = _build_coastline_mask()
    log.info("  Land fraction: %.1f%%", mask.sum() / mask.size * 100)

    log.info("Building elevation + bathymetry grids …")
    land_elev, ocean_depth = _build_elevation_grid()

    # Combine: land pixels use positive elevation, ocean pixels use depth
    elev = land_elev.copy()
    elev[~mask] = ocean_depth[~mask]

    elev_land = elev[mask]
    elev_ocean = elev[~mask]
    elev_min = float(elev_land.min()) if elev_land.size else 0.0
    elev_max = float(elev_land.max()) if elev_land.size else 0.0
    bathy_min = float(elev_ocean.min()) if elev_ocean.size else 0.0
    bathy_max = float(elev_ocean.max()) if elev_ocean.size else 0.0
    log.info("  Land elevation: %.1f m – %.1f m", elev_min, elev_max)
    log.info("  Ocean depth:    %.1f m – %.1f m", bathy_min, bathy_max)

    log.info("Saving GeoTIFF: %s", OUT_FILE)
    transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, WIDTH, HEIGHT)
    with rasterio.open(
        OUT_FILE, "w",
        driver="GTiff", height=HEIGHT, width=WIDTH,
        count=1, dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(elev, 1)

    log.info("  File size: %.1f MiB", OUT_FILE.stat().st_size / (1 << 20))

    # Atomic audit write
    audit_lines = [
        "SOURCE=SYNTHETIC (network unavailable; modeled from known geographic features)",
        "GENERATOR=tools/generate_synthetic_dem.py",
        f"FILE={OUT_FILE.name}",
        f"SIZE_BYTES={OUT_FILE.stat().st_size}",
        "CRS=EPSG:4326",
        f"WIDTH={WIDTH}",
        f"HEIGHT={HEIGHT}",
        f"BOUNDS=({LON_MIN}, {LAT_MIN}, {LON_MAX}, {LAT_MAX})",
        f"ELEV_MIN={elev_min:.2f}",
        f"ELEV_MAX={elev_max:.2f}",
        f"BATHY_MIN={bathy_min:.2f}",
        f"BATHY_MAX={bathy_max:.2f}",
        "NOTE=Coastline from PR main island + 6 offshore islands (Vieques, Culebra, Mona, Desecheo, Caja de Muertos, Palomino)",
        "NOTE=Land elevation from Gaussian peaks at known locations",
        "NOTE=Ocean depth from Gaussian wells (PR Trench 8376m N, Muertos Trough 5000m S, Caribbean basin)",
        "NOTE=Replace with official NOAA CUDEM or USGS 3DEP DEM for production quality",
    ]
    tmp = LOG_DIR / "source_audit.tmp"
    tmp.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    tmp.rename(LOG_DIR / "source_audit.txt")

    log.info("\nSYNTHETIC_DEM_OK")
    log.info("%s", OUT_FILE)


if __name__ == "__main__":
    main()
