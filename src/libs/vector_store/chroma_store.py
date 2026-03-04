"""Chroma-like vector store backend with local persistence.

This implementation provides the expected VectorStore behavior for the
project's default `chroma` provider without requiring an external database
service during local development and tests.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreRecord, VectorStoreResult


class ChromaStore(BaseVectorStore):
    """Local vector store adapter used by the `chroma` provider."""

    _store_filename = "store.json"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._persist_directory = Path(str(self.config.get("persist_directory", "data/db/chroma")))
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._store_path = self._persist_directory / self._store_filename
        self._records: dict[str, dict[str, Any]] = {}
        self._vector_dim: int | None = None
        self._load_from_disk()

    def upsert(
        self,
        records: list[VectorStoreRecord],
        trace: object | None = None,
    ) -> None:
        """Insert or update vector records and persist to disk."""
        if not isinstance(records, list):
            raise ValueError("[chroma:ValidationError] records must be a list.")
        for record in records:
            normalized = self._normalize_record(record)
            self._records[normalized["id"]] = normalized
        self._save_to_disk()

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
        trace: object | None = None,
    ) -> list[VectorStoreResult]:
        """Query nearest records for one vector with optional metadata filtering."""
        query_vector = self._normalize_query_vector(vector)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("[chroma:ValidationError] top_k must be a positive integer.")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("[chroma:ValidationError] filters must be a dict or None.")

        scored_results: list[VectorStoreResult] = []
        for record in self._records.values():
            if not self._matches_filters(record.get("metadata", {}), filters):
                continue
            score = self._cosine_similarity(query_vector, record["vector"])
            scored_results.append(
                {
                    "id": record["id"],
                    "score": float(score),
                    "metadata": dict(record.get("metadata", {})),
                }
            )

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return scored_results[:top_k]

    def _normalize_query_vector(self, vector: list[float]) -> list[float]:
        if not isinstance(vector, list) or not vector:
            raise ValueError("[chroma:ValidationError] query vector must be a non-empty list.")
        if not all(isinstance(value, (int, float)) for value in vector):
            raise ValueError("[chroma:ValidationError] query vector must be numeric.")
        normalized = [float(value) for value in vector]
        if self._vector_dim is not None and len(normalized) != self._vector_dim:
            raise ValueError(
                "[chroma:ValidationError] query vector dimension mismatch with stored vectors."
            )
        return normalized

    def _normalize_record(self, record: VectorStoreRecord) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError("[chroma:ValidationError] record must be a dict.")

        record_id = record.get("id")
        vector = record.get("vector")
        metadata = record.get("metadata", {})

        if not isinstance(record_id, str) or not record_id:
            raise ValueError("[chroma:ValidationError] record.id must be a non-empty string.")
        if not isinstance(vector, list) or not vector:
            raise ValueError("[chroma:ValidationError] record.vector must be a non-empty list.")
        if not all(isinstance(value, (int, float)) for value in vector):
            raise ValueError("[chroma:ValidationError] record.vector must be numeric.")
        if not isinstance(metadata, dict):
            raise ValueError("[chroma:ValidationError] record.metadata must be a dict.")

        normalized_vector = [float(value) for value in vector]
        self._enforce_vector_dimensionality(len(normalized_vector))

        return {
            "id": record_id,
            "vector": normalized_vector,
            "metadata": dict(metadata),
        }

    def _enforce_vector_dimensionality(self, dim: int) -> None:
        if self._vector_dim is None:
            self._vector_dim = dim
            return
        if dim != self._vector_dim:
            raise ValueError(
                "[chroma:ValidationError] all vectors must share the same dimension."
            )

    @staticmethod
    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    @staticmethod
    def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vector_a, vector_b))
        mag_a = math.sqrt(sum(a * a for a in vector_a))
        mag_b = math.sqrt(sum(b * b for b in vector_b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    def _save_to_disk(self) -> None:
        payload = {
            "vector_dim": self._vector_dim,
            "records": list(self._records.values()),
        }
        self._store_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_from_disk(self) -> None:
        if not self._store_path.exists():
            return
        raw = self._store_path.read_text(encoding="utf-8").strip()
        if not raw:
            return
        payload = json.loads(raw)
        vector_dim = payload.get("vector_dim")
        if isinstance(vector_dim, int) and vector_dim > 0:
            self._vector_dim = vector_dim

        records = payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError("[chroma:ValidationError] persisted records payload must be a list.")

        loaded_records: dict[str, dict[str, Any]] = {}
        for raw_record in records:
            normalized = self._normalize_record(raw_record)
            loaded_records[normalized["id"]] = normalized
        self._records = loaded_records
