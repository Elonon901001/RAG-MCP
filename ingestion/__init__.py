"""Compatibility package proxy to src/ingestion."""

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent.parent / "src" / "ingestion"
__path__ = [str(_pkg_dir)] if _pkg_dir.exists() else []
