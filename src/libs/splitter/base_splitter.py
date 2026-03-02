"""Abstract interface for text splitter backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseSplitter(ABC):
    """Base interface for all splitter strategies."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def split_text(self, text: str, trace: 'TraceContext | None' = None) -> list[str]:
        """Split one text into chunk strings."""
