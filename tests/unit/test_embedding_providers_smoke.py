"""Smoke tests for OpenAI/Azure embedding providers with mocked HTTP."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAIEmbedding


def test_embedding_factory_routes_to_openai_and_azure() -> None:
    openai = EmbeddingFactory.create(
        {
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "k-openai",
            }
        }
    )
    azure = EmbeddingFactory.create(
        {
            "embedding": {
                "provider": "azure",
                "model": "text-embedding-ada-002",
                "endpoint": "https://azure.example",
                "api_key": "k-azure",
            }
        }
    )
    assert isinstance(openai, OpenAIEmbedding)
    assert isinstance(azure, AzureEmbedding)


def test_openai_embedding_embed_uses_mocked_http() -> None:
    embedding = OpenAIEmbedding(
        config={"model": "text-embedding-3-small", "api_key": "k-openai"}
    )

    def fake_post_json(url: str, payload: dict, headers: dict) -> dict:
        assert url.endswith("/embeddings")
        assert payload["model"] == "text-embedding-3-small"
        assert payload["input"] == ["a", "bc"]
        assert headers["Authorization"] == "Bearer k-openai"
        return {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
            ]
        }

    embedding._post_json = fake_post_json  # type: ignore[method-assign]
    vectors = embedding.embed(["a", "bc"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_azure_embedding_embed_uses_mocked_http() -> None:
    embedding = AzureEmbedding(
        config={
            "model": "text-embedding-ada-002",
            "endpoint": "https://azure.example",
            "api_version": "2024-10-21",
            "api_key": "k-azure",
        }
    )

    def fake_post_json(url: str, payload: dict, headers: dict) -> dict:
        assert "openai/deployments/text-embedding-ada-002/embeddings" in url
        assert "api-version=2024-10-21" in url
        assert payload == {"input": ["hello"]}
        assert headers["api-key"] == "k-azure"
        return {"data": [{"embedding": [0.9, 0.8]}]}

    embedding._post_json = fake_post_json  # type: ignore[method-assign]
    vectors = embedding.embed(["hello"])
    assert vectors == [[0.9, 0.8]]


def test_embed_rejects_empty_input() -> None:
    embedding = OpenAIEmbedding(
        config={"model": "text-embedding-3-small", "api_key": "k-openai"}
    )
    with pytest.raises(ValueError, match=r"\[openai:ValidationError\]"):
        embedding.embed([])


def test_embed_handles_long_input_by_configurable_policy() -> None:
    embedding = OpenAIEmbedding(
        config={
            "model": "text-embedding-3-small",
            "api_key": "k-openai",
            "max_input_length": 5,
            "truncate_long_input": False,
        }
    )
    with pytest.raises(ValueError, match=r"max_input_length=5"):
        embedding.embed(["123456"])

    trunc_embedding = OpenAIEmbedding(
        config={
            "model": "text-embedding-3-small",
            "api_key": "k-openai",
            "max_input_length": 5,
            "truncate_long_input": True,
        }
    )

    def fake_post_json(url: str, payload: dict, headers: dict) -> dict:
        assert payload["input"] == ["12345"]
        return {"data": [{"embedding": [1.0]}]}

    trunc_embedding._post_json = fake_post_json  # type: ignore[method-assign]
    vectors = trunc_embedding.embed(["123456"])
    assert vectors == [[1.0]]
