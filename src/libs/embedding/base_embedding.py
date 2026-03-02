"""Abstract interface for embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseEmbedding(ABC):
    """Base interface for all embedding providers."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        trace: 'TraceContext | None' = None,
    ) -> list[list[float]]:
        """Encode a batch of texts into embedding vectors."""
