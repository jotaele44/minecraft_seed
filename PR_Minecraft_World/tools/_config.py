"""
_config.py — Shared configuration loader.

Reads config.toml from the project root (Python 3.11+ tomllib).
Falls back to hardcoded defaults if the file is absent or a key is missing.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_DEFAULTS: dict = {
    "dem": {
        "bbox": "-67.35,17.85,-65.20,18.55",
        "fetch_timeout_s": 120,
        "chunk_bytes": 1_048_576,
    },
    "heightmap": {
        "target_max_dim": 2048,
        "bits": 8,
        "resampling": "bilinear",
        "include_bathymetry": True,
        "max_ocean_depth_m": 9000,
    },
    "minecraft": {
        "version": "pre-1.18",
        "sea_level_block": 62,
        "max_height": 255,
    },
    "spawn": {
        "quadrant": "northeast",
    },
}


def load_config() -> dict:
    """Return merged config: user config.toml values layered over defaults."""
    config_path = ROOT / "config.toml"
    user: dict = {}
    if config_path.exists():
        with open(config_path, "rb") as fh:
            user = tomllib.load(fh)

    result: dict = {}
    for section, defaults in _DEFAULTS.items():
        result[section] = {**defaults, **user.get(section, {})}
    return result
