"""Abstract contract and default fallback for reranker backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

RerankCandidate = dict[str, Any]


class BaseReranker(ABC):
    """Base interface all reranker implementations must satisfy."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: "TraceContext | None" = None,
    ) -> list[RerankCandidate]:
        """Return candidates reordered by relevance to the query."""


class NoneReranker(BaseReranker):
    """Fallback reranker that preserves the input order unchanged."""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: "TraceContext | None" = None,
    ) -> list[RerankCandidate]:
        return list(candidates)
