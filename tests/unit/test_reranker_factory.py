"""Unit tests for reranker factory and fallback behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.reranker.base_reranker import BaseReranker, NoneReranker
from libs.reranker.reranker_factory import RerankerFactory


class ReverseReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, object]],
        trace: object | None = None,
    ) -> list[dict[str, object]]:
        return list(reversed(candidates))


def test_none_reranker_preserves_input_order() -> None:
    candidates = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    reranker = NoneReranker(config={"backend": "none"})

    ranked = reranker.rerank("test query", candidates)

    assert ranked == candidates


def test_reranker_factory_routes_to_registered_backend() -> None:
    RerankerFactory.register("reverse", ReverseReranker)
    settings = {"rerank": {"backend": "reverse", "top_n": 5}}

    reranker = RerankerFactory.create(settings)

    assert isinstance(reranker, ReverseReranker)
    assert reranker.config["top_n"] == 5


def test_reranker_factory_uses_strategy_field_for_backward_compatibility() -> None:
    settings = {"rerank": {"strategy": "none"}}

    reranker = RerankerFactory.create(settings)

    assert isinstance(reranker, NoneReranker)


def test_reranker_factory_rejects_unknown_backend() -> None:
    settings = {"rerank": {"backend": "unknown"}}

    with pytest.raises(ValueError, match="Unsupported reranker backend: unknown"):
        RerankerFactory.create(settings)
