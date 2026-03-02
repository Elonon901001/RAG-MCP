"""Unit tests for splitter factory routing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class FakeRecursiveSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [f"recursive:{text}"]


class FakeSemanticSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [f"semantic:{text}"]


class FakeFixedSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [f"fixed:{text}"]


def test_splitter_factory_routes_by_strategy() -> None:
    SplitterFactory.register("recursive", FakeRecursiveSplitter)
    SplitterFactory.register("semantic", FakeSemanticSplitter)
    SplitterFactory.register("fixed", FakeFixedSplitter)

    recursive = SplitterFactory.create({"splitter": {"strategy": "recursive"}})
    semantic = SplitterFactory.create({"splitter": {"strategy": "semantic"}})
    fixed = SplitterFactory.create({"splitter": {"strategy": "fixed"}})

    assert isinstance(recursive, FakeRecursiveSplitter)
    assert isinstance(semantic, FakeSemanticSplitter)
    assert isinstance(fixed, FakeFixedSplitter)
    assert recursive.split_text("a") == ["recursive:a"]
    assert semantic.split_text("a") == ["semantic:a"]
    assert fixed.split_text("a") == ["fixed:a"]


def test_splitter_factory_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported splitter strategy: unknown"):
        SplitterFactory.create({"splitter": {"strategy": "unknown"}})
