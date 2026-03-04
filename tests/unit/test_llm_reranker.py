"""Unit tests for LLM reranker behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.reranker_factory import RerankerFactory


class FakeLLM:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_reranker_factory_creates_llm_backend() -> None:
    settings = {
        "rerank": {
            "backend": "llm",
            "prompt_template": "Rank candidates.",
            "llm_client": FakeLLM('{"ranked_ids":["c1"]}'),
        }
    }
    reranker = RerankerFactory.create(settings)
    assert isinstance(reranker, LLMReranker)


def test_llm_reranker_reorders_by_ranked_ids() -> None:
    fake_llm = FakeLLM('{"ranked_ids":["c3","c1","c2"]}')
    reranker = LLMReranker(
        config={"prompt_template": "RERANK", "llm_client": fake_llm, "top_n": 3}
    )
    candidates = [
        {"id": "c1", "text": "a"},
        {"id": "c2", "text": "b"},
        {"id": "c3", "text": "c"},
    ]

    ranked = reranker.rerank("query", candidates)

    assert [item["id"] for item in ranked] == ["c3", "c1", "c2"]
    assert reranker.last_fallback_reason is None
    assert fake_llm.calls
    assert "RERANK" in fake_llm.calls[0][1]["content"]


def test_llm_reranker_raises_on_invalid_schema() -> None:
    reranker = LLMReranker(
        config={"prompt_template": "RERANK", "llm_client": FakeLLM('{"ranked_ids":[1,2]}')}
    )
    candidates = [{"id": "c1", "text": "a"}]

    with pytest.raises(ValueError, match="Invalid rerank response schema"):
        reranker.rerank("query", candidates)


def test_llm_reranker_raises_on_unknown_ranked_id() -> None:
    reranker = LLMReranker(
        config={"prompt_template": "RERANK", "llm_client": FakeLLM('{"ranked_ids":["cX"]}')}
    )
    candidates = [{"id": "c1", "text": "a"}]

    with pytest.raises(ValueError, match="unknown ids"):
        reranker.rerank("query", candidates)


def test_llm_reranker_returns_fallback_signal_on_llm_error() -> None:
    reranker = LLMReranker(
        config={"prompt_template": "RERANK", "llm_client": FakeLLM(RuntimeError("network"))}
    )
    candidates = [{"id": "c1"}, {"id": "c2"}]

    ranked = reranker.rerank("query", candidates)

    assert ranked == candidates
    assert reranker.last_fallback_reason is not None
    assert reranker.last_fallback_reason.startswith("llm_call_failed")
