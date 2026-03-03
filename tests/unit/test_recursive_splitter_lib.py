"""Unit tests for recursive splitter default implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


def test_factory_creates_recursive_splitter() -> None:
    splitter = SplitterFactory.create({"splitter": {"strategy": "recursive"}})
    assert isinstance(splitter, RecursiveSplitter)


def test_recursive_splitter_splits_markdown_without_breaking_fenced_code() -> None:
    text = (
        "# Title\n\n"
        "intro paragraph that is somewhat long and should be split.\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "ending paragraph"
    )
    splitter = RecursiveSplitter(config={"chunk_size": 60, "chunk_overlap": 0})

    chunks = splitter.split_text(text)

    assert chunks
    joined = "".join(chunks)
    assert "```python" in joined and "return a + b" in joined and "```" in joined
    assert joined.count("```python") == 1


def test_recursive_splitter_rejects_invalid_config() -> None:
    splitter = RecursiveSplitter(config={"chunk_size": 10, "chunk_overlap": 10})
    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        splitter.split_text("hello world")
