"""Contract tests for vector store interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._records: list[dict[str, Any]] = []

    def upsert(self, records: list[dict[str, Any]], trace: object | None = None) -> None:
        for record in records:
            assert isinstance(record["id"], str)
            assert isinstance(record["vector"], list)
            assert all(isinstance(v, float) for v in record["vector"])
            assert isinstance(record.get("metadata", {}), dict)
        self._records.extend(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
        trace: object | None = None,
    ) -> list[dict[str, Any]]:
        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)
        assert isinstance(top_k, int)
        assert filters is None or isinstance(filters, dict)

        results: list[dict[str, Any]] = []
        for record in self._records[:top_k]:
            results.append(
                {
                    "id": record["id"],
                    "score": 1.0,
                    "metadata": record.get("metadata", {}),
                }
            )
        return results


def test_vector_store_contract_upsert_and_query_shape() -> None:
    VectorStoreFactory.register("fake", FakeVectorStore)
    store = VectorStoreFactory.create({"vector_store": {"provider": "fake"}})

    store.upsert(
        [
            {"id": "chunk-1", "vector": [0.1, 0.2], "metadata": {"source": "doc-1"}},
            {"id": "chunk-2", "vector": [0.2, 0.3], "metadata": {"source": "doc-2"}},
        ]
    )
    results = store.query(vector=[0.1, 0.2], top_k=1, filters={"source": "doc-1"})

    assert isinstance(results, list)
    assert len(results) == 1
    assert set(results[0].keys()) == {"id", "score", "metadata"}
    assert isinstance(results[0]["id"], str)
    assert isinstance(results[0]["score"], float)
    assert isinstance(results[0]["metadata"], dict)
