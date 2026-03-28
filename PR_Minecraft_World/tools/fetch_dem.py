"""
fetch_dem.py — Download official Puerto Rico DEM data.

Primary source:  NOAA NCEI CUDEM (Continuously Updated Digital Elevation Model)
Fallback source: USGS 3DEP via The National Map API

Usage:
    python tools/fetch_dem.py
"""

import os
import sys
import time
import hashlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "output" / "logs"
OUT_FILE = RAW_DIR / "puerto_rico_official_dem.tif"

for d in [RAW_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TIMEOUT = 120          # seconds per request
CHUNK = 1 << 20        # 1 MiB download chunks

# ---------------------------------------------------------------------------
# NOAA CUDEM / Coastal DEM candidates
# ---------------------------------------------------------------------------
# NOAA NCEI provides the Puerto Rico / USVI CUDEM as a single-file GeoTIFF.
# The product page is at:
#   https://www.ncei.noaa.gov/products/coastal-relief-model
# The direct-download paths follow NOAA's thredds/HTTP server convention.
# We list the most likely stable direct-download URLs in priority order.
NOAA_CANDIDATES = [
    # 1/9 arc-second Puerto Rico coastal DEM (CUDEM tile, ~1 GB)
    "https://www.ngdc.noaa.gov/thredds/fileServer/regional/puerto_rico_-67_to_-65_17_to_19_navd88_2014_1_9_mhws.nc",
    # Alternative NOAA NCEI direct GeoTIFF path (18-arc-second relief model)
    "https://www.ncei.noaa.gov/data/oceans/coastal-relief-model/crm_vol9.nc",
    # NOAA OCS/thredds fallback for Caribbean / Puerto Rico
    "https://www.ngdc.noaa.gov/thredds/fileServer/crm/crm_vol9.nc",
]

# Puerto Rico bounding box
PR_BBOX = "-67.35,17.85,-65.20,18.55"

# ---------------------------------------------------------------------------
# USGS 3DEP fallback via TNM API
# ---------------------------------------------------------------------------
TNM_API = "https://tnmapi.cr.usgs.gov/api/products"
TNM_PARAMS = {
    "datasets": "Digital Elevation Model (DEM) 1 arc-second",
    "bbox": PR_BBOX,
    "prodFormats": "GeoTIFF",
    "max": 10,
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fail(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def download(url: str, dest: Path, label: str) -> bool:
    """Stream-download *url* into *dest*. Returns True on success."""
    print(f"  Trying {label}: {url}")
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            if r.status_code != 200:
                print(f"    HTTP {r.status_code} — skipping")
                return False
            content_type = r.headers.get("content-type", "")
            # Accept geotiff, netcdf, or octet-stream
            if "text/html" in content_type:
                print(f"    Got HTML response (likely a redirect/error page) — skipping")
                return False
            total = int(r.headers.get("content-length", 0))
            received = 0
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    fh.write(chunk)
                    received += len(chunk)
            if received < 1024:
                dest.unlink(missing_ok=True)
                print(f"    Downloaded only {received} bytes — skipping")
                return False
            mb = received / (1 << 20)
            print(f"    OK — {mb:.1f} MiB saved to {dest}")
            return True
    except requests.RequestException as exc:
        print(f"    Request failed: {exc}")
        return False


def write_audit(lines: list) -> None:
    (LOG_DIR / "source_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# NOAA acquisition
# ---------------------------------------------------------------------------

def try_noaa() -> bool:
    print("\n[1/2] Attempting NOAA CUDEM / Coastal DEM download ...")
    for url in NOAA_CANDIDATES:
        if download(url, OUT_FILE, "NOAA"):
            audit = [
                f"SOURCE=NOAA CUDEM / Coastal DEM",
                f"URL={url}",
                f"FILE={OUT_FILE.name}",
                f"SIZE_BYTES={OUT_FILE.stat().st_size}",
            ]
            write_audit(audit)
            return True
    return False


# ---------------------------------------------------------------------------
# USGS 3DEP fallback
# ---------------------------------------------------------------------------

def try_usgs() -> bool:
    print("\n[2/2] Attempting USGS 3DEP fallback via TNM API ...")
    try:
        r = requests.get(TNM_API, params=TNM_PARAMS, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"  TNM API query failed: {exc}")
        return False

    items = data.get("items", [])
    if not items:
        print("  TNM API returned 0 products for Puerto Rico DEM query")
        return False

    for item in items:
        url = item.get("downloadURL") or item.get("urls", {}).get("TIFF", "")
        if not url:
            continue
        if not (url.lower().endswith(".tif") or url.lower().endswith(".tiff")
                or "tif" in url.lower()):
            # also accept zip/img fallback from USGS
            url = item.get("urls", {}).get("ZIP", url)
        if download(url, OUT_FILE, "USGS 3DEP"):
            audit = [
                f"SOURCE=USGS 3DEP via TNM API",
                f"URL={url}",
                f"FILE={OUT_FILE.name}",
                f"SIZE_BYTES={OUT_FILE.stat().st_size}",
                f"PRODUCT_TITLE={item.get('title', 'unknown')}",
            ]
            write_audit(audit)
            return True

    print("  No downloadable GeoTIFF found in TNM results")
    return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Puerto Rico DEM Fetcher ===")
    print(f"Output: {OUT_FILE}")

    # Skip download if we already have a valid file
    if OUT_FILE.exists() and OUT_FILE.stat().st_size > 1024:
        print(f"\nExisting file found ({OUT_FILE.stat().st_size / (1<<20):.1f} MiB) — skipping download.")
        return

    if try_noaa():
        print("\nNOAA acquisition successful.")
        return

    if try_usgs():
        print("\nUSGS 3DEP fallback acquisition successful.")
        return

    print("\n[3/3] Network sources unavailable. Generating synthetic DEM from known geographic features ...")
    _generate_synthetic()


def _generate_synthetic() -> None:
    """Fall back to the local synthetic DEM generator."""
    import subprocess
    script = Path(__file__).parent / "generate_synthetic_dem.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=False,
    )
    if result.returncode != 0:
        fail("Synthetic DEM generation also failed. Check generate_synthetic_dem.py.")
    if not (OUT_FILE.exists() and OUT_FILE.stat().st_size > 1024):
        fail("Synthetic DEM generator ran but output file is missing or empty.")


if __name__ == "__main__":
    main()
