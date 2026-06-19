"""
fetch_dem.py — Download official Puerto Rico DEM data.

Acquisition order:
  1. NOAA NCEI CUDEM (Continuously Updated Digital Elevation Model)
  2. USGS 3DEP via The National Map API
  3. Local synthetic DEM generator (offline fallback)

Usage:
    python tools/fetch_dem.py [--force]
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import requests
import rasterio

from tools._config import ROOT, load_config

log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "output" / "logs"
OUT_FILE = RAW_DIR / "puerto_rico_official_dem.tif"

for _d in [RAW_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NOAA CUDEM candidates
# Each URL should be verified against the live NOAA NCEI catalog before use.
# Add a "# verified: YYYY-MM-DD" comment when confirming a URL still works.
# Product page: https://www.ncei.noaa.gov/products/coastal-relief-model
# ---------------------------------------------------------------------------
NOAA_CANDIDATES = [
    # 1/9 arc-second Puerto Rico CUDEM tile (~1 GB NetCDF)
    # verified: needs re-check against current NOAA NCEI THREDDS catalog
    "https://www.ngdc.noaa.gov/thredds/fileServer/regional/puerto_rico_-67_to_-65_17_to_19_navd88_2014_1_9_mhws.nc",
    # 18 arc-second Caribbean relief model vol 9 (NetCDF)
    # verified: needs re-check
    "https://www.ncei.noaa.gov/data/oceans/coastal-relief-model/crm_vol9.nc",
    # NOAA THREDDS alternate path
    # verified: needs re-check
    "https://www.ngdc.noaa.gov/thredds/fileServer/crm/crm_vol9.nc",
]

# USGS 3DEP TNM API
TNM_API = "https://tnmapi.cr.usgs.gov/api/products"

# Transient HTTP status codes that warrant a retry
_TRANSIENT_STATUSES = {500, 502, 503, 504}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fail(msg: str, code: int = 1) -> None:
    log.error(msg)
    sys.exit(code)


def _probe_raster(path: Path) -> bool:
    """Return True if *path* can be opened by rasterio as a valid raster."""
    try:
        with rasterio.open(path):
            pass
        return True
    except Exception as exc:
        log.debug("Raster probe failed for %s: %s", path, exc)
        return False


def _write_audit(lines: list[str]) -> None:
    """Atomically write audit lines to source_audit.txt."""
    tmp = LOG_DIR / "source_audit.tmp"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.rename(LOG_DIR / "source_audit.txt")


def _download(url: str, dest: Path, cfg: dict) -> bool:
    """
    Stream-download *url* to *dest*.

    - Sends a HEAD probe first (cheap validity check).
    - Retries up to 4 times with exponential backoff on transient errors.
    - Validates the written file is a real raster before returning True.
    """
    timeout = cfg["dem"]["fetch_timeout_s"]
    chunk_size = cfg["dem"]["chunk_bytes"]
    log.info("  Trying: %s", url)

    # HEAD probe — skip if the URL clearly won't serve a raster
    try:
        head = requests.head(url, timeout=15, allow_redirects=True)
        ct = head.headers.get("content-type", "")
        if head.status_code == 404:
            log.info("    HEAD 404 — skipping")
            return False
        if "text/html" in ct:
            log.info("    HEAD returned text/html — skipping (likely error page)")
            return False
        log.debug("    HEAD %s content-type=%s", head.status_code, ct)
    except requests.RequestException as exc:
        log.debug("    HEAD probe failed: %s", exc)
        # Continue — some servers reject HEAD; try GET anyway

    # GET with retry + exponential backoff
    for attempt in range(4):
        if attempt:
            wait = 2 ** attempt
            log.info("    Retry %d/3 in %ds …", attempt, wait)
            import time as _time
            _time.sleep(wait)
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code in _TRANSIENT_STATUSES:
                    log.warning("    HTTP %s (transient) — will retry", r.status_code)
                    continue
                if r.status_code != 200:
                    log.info("    HTTP %s — skipping", r.status_code)
                    return False
                ct = r.headers.get("content-type", "")
                if "text/html" in ct:
                    log.info("    GET returned text/html — skipping")
                    return False

                total = int(r.headers.get("content-length", 0))
                received = 0
                next_report = 50 * (1 << 20)  # report every 50 MiB
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        fh.write(chunk)
                        received += len(chunk)
                        if total and received >= next_report:
                            pct = received / total * 100
                            log.info(
                                "    %.0f MiB / %.0f MiB (%.0f%%)",
                                received / (1 << 20),
                                total / (1 << 20),
                                pct,
                            )
                            next_report += 50 * (1 << 20)

            if received < 1024:
                dest.unlink(missing_ok=True)
                log.info("    Only %d bytes received — skipping", received)
                return False

            log.info("    Downloaded %.1f MiB", received / (1 << 20))

            # Validate the file is actually a raster
            if not _probe_raster(dest):
                log.warning("    File failed raster validation — removing")
                dest.unlink(missing_ok=True)
                return False

            return True

        except requests.RequestException as exc:
            log.warning("    Request failed: %s", exc)
            if attempt < 3:
                continue
            return False

    return False


# ---------------------------------------------------------------------------
# NOAA acquisition
# ---------------------------------------------------------------------------

def _try_noaa(cfg: dict) -> bool:
    log.info("\n[1/2] Attempting NOAA CUDEM / Coastal DEM …")
    for url in NOAA_CANDIDATES:
        if _download(url, OUT_FILE, cfg):
            _write_audit([
                "SOURCE=NOAA CUDEM / Coastal DEM",
                f"URL={url}",
                f"FILE={OUT_FILE.name}",
                f"SIZE_BYTES={OUT_FILE.stat().st_size}",
            ])
            return True
    return False


# ---------------------------------------------------------------------------
# USGS 3DEP fallback
# ---------------------------------------------------------------------------

def _try_usgs(cfg: dict) -> bool:
    log.info("\n[2/2] Attempting USGS 3DEP via TNM API …")
    bbox = cfg["dem"]["bbox"]
    params = {
        "datasets": "Digital Elevation Model (DEM) 1 arc-second",
        "bbox": bbox,
        "prodFormats": "GeoTIFF",
        "max": 10,
    }
    try:
        r = requests.get(TNM_API, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("  TNM API query failed: %s", exc)
        return False

    items = data.get("items", [])
    if not items:
        log.info("  TNM API returned 0 products")
        return False

    for item in items:
        url = item.get("downloadURL") or item.get("urls", {}).get("TIFF", "")
        if not url:
            continue
        if "tif" not in url.lower():
            url = item.get("urls", {}).get("ZIP", url)
        if _download(url, OUT_FILE, cfg):
            _write_audit([
                "SOURCE=USGS 3DEP via TNM API",
                f"URL={url}",
                f"FILE={OUT_FILE.name}",
                f"SIZE_BYTES={OUT_FILE.stat().st_size}",
                f"PRODUCT_TITLE={item.get('title', 'unknown')}",
            ])
            return True

    log.info("  No downloadable GeoTIFF found in TNM results")
    return False


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def _generate_synthetic() -> None:
    import os
    log.info("\n[3/3] Falling back to synthetic DEM generator …")
    script = Path(__file__).parent / "generate_synthetic_dem.py"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run([sys.executable, str(script)], env=env)
    if result.returncode != 0:
        _fail("Synthetic DEM generation failed. Check generate_synthetic_dem.py.")
    if not (OUT_FILE.exists() and OUT_FILE.stat().st_size > 1024):
        _fail("Synthetic generator ran but output is missing or empty.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Puerto Rico DEM data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a valid file already exists.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cfg = load_config()
    log.info("=== Puerto Rico DEM Fetcher ===")
    log.info("Output: %s", OUT_FILE)

    # Skip if a valid raster already exists (unless --force)
    if not args.force and OUT_FILE.exists():
        if OUT_FILE.stat().st_size > 1024 and _probe_raster(OUT_FILE):
            log.info(
                "Existing valid file found (%.1f MiB) — skipping. Use --force to re-fetch.",
                OUT_FILE.stat().st_size / (1 << 20),
            )
            return
        else:
            log.warning("Existing file is corrupt or empty — removing and re-fetching.")
            OUT_FILE.unlink(missing_ok=True)

    if _try_noaa(cfg):
        log.info("\nNOAA acquisition successful.")
        return
    if _try_usgs(cfg):
        log.info("\nUSGS 3DEP acquisition successful.")
        return
    _generate_synthetic()


if __name__ == "__main__":
    main()
