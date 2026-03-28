"""
conftest.py — Add project root to sys.path so `tools` is importable in tests
without requiring a package install.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
