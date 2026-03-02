"""Smoke tests for top-level package imports."""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def test_key_packages_can_be_imported() -> None:
    """Ensure core top-level packages are importable."""
    for package in ("mcp_server", "core", "ingestion", "libs", "observability"):
        importlib.import_module(package)
