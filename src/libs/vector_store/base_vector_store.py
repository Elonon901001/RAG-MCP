"""Abstract contract for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

VectorStoreRecord = dict[str, Any]
VectorStoreResult = dict[str, Any]


class BaseVectorStore(ABC):
    """Base interface all vector stores must implement."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def upsert(
        self,
        records: list[VectorStoreRecord],
        trace: 'TraceContext | None' = None,
    ) -> None:
        """Insert or update vector records."""

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
        trace: 'TraceContext | None' = None,
    ) -> list[VectorStoreResult]:
        """Query nearest records for one vector."""
