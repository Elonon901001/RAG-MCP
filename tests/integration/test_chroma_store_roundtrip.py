"""Integration tests for ChromaStore upsert/query roundtrip."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.vector_store.vector_store_factory import VectorStoreFactory


@pytest.mark.integration
def test_chroma_store_roundtrip_with_persistence_and_filters() -> None:
    persist_directory = PROJECT_ROOT / "tests" / "tmp_write_probe" / f"chroma-{uuid.uuid4().hex}"
    persist_directory.mkdir(parents=True, exist_ok=True)
    settings = {
        "vector_store": {
            "provider": "chroma",
            "persist_directory": str(persist_directory),
        }
    }

    store = VectorStoreFactory.create(settings)
    store.upsert(
        [
            {
                "id": "chunk-1",
                "vector": [1.0, 0.0, 0.0],
                "metadata": {"collection": "alpha", "source": "doc-a"},
            },
            {
                "id": "chunk-2",
                "vector": [0.0, 1.0, 0.0],
                "metadata": {"collection": "beta", "source": "doc-b"},
            },
            {
                "id": "chunk-3",
                "vector": [0.8, 0.2, 0.0],
                "metadata": {"collection": "alpha", "source": "doc-c"},
            },
        ]
    )

    results = store.query(vector=[1.0, 0.0, 0.0], top_k=2, filters=None)
    assert len(results) == 2
    assert results[0]["id"] == "chunk-1"
    assert results[0]["score"] >= results[1]["score"]

    filtered = store.query(
        vector=[1.0, 0.0, 0.0],
        top_k=5,
        filters={"collection": "alpha"},
    )
    assert [item["id"] for item in filtered] == ["chunk-1", "chunk-3"]

    reloaded = VectorStoreFactory.create(settings)
    reloaded_results = reloaded.query(vector=[1.0, 0.0, 0.0], top_k=3, filters=None)
    assert [item["id"] for item in reloaded_results] == ["chunk-1", "chunk-3", "chunk-2"]
