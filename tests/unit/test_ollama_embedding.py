"""Unit tests for Ollama embedding provider with mocked HTTP."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.ollama_embedding import OllamaEmbedding


def test_factory_creates_ollama_embedding() -> None:
    embedding = EmbeddingFactory.create(
        {"embedding": {"provider": "ollama", "model": "nomic-embed-text"}}
    )
    assert isinstance(embedding, OllamaEmbedding)


def test_ollama_embedding_batch_calls_and_returns_vectors() -> None:
    embedding = OllamaEmbedding(
        config={"model": "nomic-embed-text", "base_url": "http://localhost:11434"}
    )
    calls: list[dict[str, object]] = []

    def fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append({"url": url, "payload": payload})
        prompt = str(payload["prompt"])
        return {"embedding": [float(len(prompt)), 1.0]}

    embedding._post_json = fake_post_json  # type: ignore[method-assign]
    vectors = embedding.embed(["a", "abcd"])

    assert len(calls) == 2
    assert all(str(c["url"]).endswith("/api/embeddings") for c in calls)
    assert calls[0]["payload"]["model"] == "nomic-embed-text"  # type: ignore[index]
    assert vectors == [[1.0, 1.0], [4.0, 1.0]]


def test_ollama_embedding_rejects_empty_input() -> None:
    embedding = OllamaEmbedding(config={"model": "nomic-embed-text"})
    with pytest.raises(ValueError, match=r"\[ollama:ValidationError\]"):
        embedding.embed([])


def test_ollama_embedding_handles_long_input_policy() -> None:
    embedding = OllamaEmbedding(
        config={
            "model": "nomic-embed-text",
            "max_input_length": 5,
            "truncate_long_input": False,
        }
    )
    with pytest.raises(ValueError, match=r"max_input_length=5"):
        embedding.embed(["123456"])

    trunc_embedding = OllamaEmbedding(
        config={
            "model": "nomic-embed-text",
            "max_input_length": 5,
            "truncate_long_input": True,
        }
    )

    def fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        assert payload["prompt"] == "12345"
        return {"embedding": [1.0]}

    trunc_embedding._post_json = fake_post_json  # type: ignore[method-assign]
    assert trunc_embedding.embed(["123456"]) == [[1.0]]


def test_ollama_embedding_wraps_connection_and_timeout_errors() -> None:
    embedding = OllamaEmbedding(config={"model": "nomic-embed-text"})

    import libs.embedding.ollama_embedding as module

    def raise_url_error(*args: object, **kwargs: object) -> object:
        raise URLError("refused")

    def raise_timeout(*args: object, **kwargs: object) -> object:
        raise TimeoutError("timeout")

    old_urlopen = module.urlopen
    try:
        module.urlopen = raise_url_error  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match=r"\[ollama:ConnectionError\]"):
            embedding._post_json("http://localhost:11434/api/embeddings", {"model": "x", "prompt": "p"})

        module.urlopen = raise_timeout  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match=r"\[ollama:TimeoutError\]"):
            embedding._post_json("http://localhost:11434/api/embeddings", {"model": "x", "prompt": "p"})
    finally:
        module.urlopen = old_urlopen  # type: ignore[assignment]
