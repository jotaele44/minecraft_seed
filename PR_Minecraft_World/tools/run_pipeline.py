"""
run_pipeline.py — Orchestrate the full Puerto Rico DEM → Minecraft heightmap pipeline.

Steps:
  1. fetch_dem      — download or generate DEM
  2. build_heightmap — reproject, normalise, export PNG
  3. validate_outputs — quality gates
  4. preview_heightmap — terrain colourmap image

Usage:
    python tools/run_pipeline.py [options]

Options:
    --skip-fetch     Skip DEM acquisition (use existing raw file)
    --skip-build     Skip heightmap build (use existing PNG)
    --force          Force re-download even if a valid DEM exists
    --bits {8,16}    Output bit depth (default: from config.toml)
    --verbose        Enable debug logging
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from tools._config import ROOT

# Ensure child processes can import the tools package regardless of how
# the orchestrator was invoked (direct python call, make, CI, etc.)
_CHILD_ENV = {**os.environ, "PYTHONPATH": str(ROOT)}

log = logging.getLogger(__name__)

_SCRIPTS = {
    "fetch":    ROOT / "tools" / "fetch_dem.py",
    "build":    ROOT / "tools" / "build_heightmap.py",
    "validate": ROOT / "tools" / "validate_outputs.py",
    "preview":  ROOT / "tools" / "preview_heightmap.py",
}

_OUTPUTS = {
    "fetch":    ROOT / "data" / "raw" / "puerto_rico_official_dem.tif",
    "build":    ROOT / "output" / "heightmap" / "puerto_rico_heightmap.png",
    "validate": None,
    "preview":  ROOT / "output" / "heightmap" / "puerto_rico_preview.png",
}


def _run_step(name: str, extra_args: list[str], verbose: bool) -> tuple[bool, float]:
    """Run one pipeline step. Returns (success, elapsed_seconds)."""
    script = _SCRIPTS[name]
    cmd = [sys.executable, str(script)] + extra_args
    if verbose:
        cmd.append("--verbose")

    log.info("\n>>> %s", " ".join(str(c) for c in cmd))
    t0 = time.monotonic()
    result = subprocess.run(cmd, env=_CHILD_ENV)
    elapsed = time.monotonic() - t0
    success = result.returncode == 0
    return success, elapsed


def _fmt_output(name: str) -> str:
    path = _OUTPUTS[name]
    if path is None:
        return "—"
    if path.exists():
        mb = path.stat().st_size / (1 << 20)
        sz = f"{mb:.1f} MiB" if mb >= 0.1 else f"{path.stat().st_size} B"
        w = h = None
        if path.suffix == ".png":
            try:
                from PIL import Image
                with Image.open(path) as img:
                    w, h = img.size
            except Exception:
                pass
        dims = f" ({w}×{h})" if w else ""
        return f"{path.relative_to(ROOT)} {sz}{dims}"
    return f"{path.relative_to(ROOT)} (not found)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full PR Minecraft heightmap pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-fetch",  action="store_true", help="Skip DEM fetch step.")
    parser.add_argument("--skip-build",  action="store_true", help="Skip heightmap build step.")
    parser.add_argument("--force",       action="store_true", help="Force re-download.")
    parser.add_argument("--bits", type=int, choices=[8, 16], help="Heightmap bit depth.")
    parser.add_argument("--verbose",     action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    steps = []
    if not args.skip_fetch:
        fetch_args = ["--force"] if args.force else []
        steps.append(("fetch", fetch_args))
    if not args.skip_build:
        build_args = []
        if args.bits:
            build_args += ["--bits", str(args.bits)]
        steps.append(("build", build_args))
    steps.append(("validate", []))
    steps.append(("preview",  []))

    results: dict[str, tuple[bool, float]] = {}
    failed = False

    for name, extra in steps:
        ok, elapsed = _run_step(name, extra, args.verbose)
        results[name] = (ok, elapsed)
        if not ok:
            log.error("Step '%s' FAILED (exit code non-zero).", name)
            failed = True
            break

    # Summary table
    log.info("\n%s", "=" * 60)
    log.info("%-12s  %-8s  %-8s  %s", "Step", "Status", "Time", "Output")
    log.info("%s", "-" * 60)
    all_steps = ["fetch", "build", "validate", "preview"]
    for name in all_steps:
        if name in results:
            ok, elapsed = results[name]
            status = "OK" if ok else "FAILED"
            log.info(
                "%-12s  %-8s  %5.1fs  %s",
                name, status, elapsed, _fmt_output(name),
            )
        elif args.skip_fetch and name == "fetch":
            log.info("%-12s  %-8s  %5s  %s", name, "SKIPPED", "—", "")
        elif args.skip_build and name == "build":
            log.info("%-12s  %-8s  %5s  %s", name, "SKIPPED", "—", "")
    log.info("%s", "=" * 60)

    if failed:
        sys.exit(1)
    log.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
