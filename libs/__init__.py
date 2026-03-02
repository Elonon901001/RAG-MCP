"""Compatibility package proxy to src/libs."""

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent.parent / "src" / "libs"
__path__ = [str(_pkg_dir)] if _pkg_dir.exists() else []
