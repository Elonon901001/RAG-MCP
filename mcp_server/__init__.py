"""Compatibility package proxy to src/mcp_server."""

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent.parent / "src" / "mcp_server"
__path__ = [str(_pkg_dir)] if _pkg_dir.exists() else []
