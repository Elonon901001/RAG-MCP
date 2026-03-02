"""Unit tests for embedding factory routing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    def embed(
        self,
        texts: list[str],
        trace: object | None = None,
    ) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def test_embedding_factory_routes_to_registered_provider() -> None:
    EmbeddingFactory.register("fake", FakeEmbedding)
    settings = {"embedding": {"provider": "fake", "model": "fake-embedding-model"}}

    embedding = EmbeddingFactory.create(settings)
    vectors = embedding.embed(["a", "abcd"])

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.config["model"] == "fake-embedding-model"
    assert vectors == [[1.0, 1.0], [4.0, 1.0]]


def test_embedding_factory_rejects_unknown_provider() -> None:
    settings = {"embedding": {"provider": "unknown"}}

    with pytest.raises(ValueError, match="Unsupported embedding provider: unknown"):
        EmbeddingFactory.create(settings)
