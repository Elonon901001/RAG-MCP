"""Abstract contract for evaluation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseEvaluator(ABC):
    """Base interface all evaluator implementations must satisfy."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: "TraceContext | None" = None,
    ) -> dict[str, float]:
        """Return metrics for one query against retrieved and golden ids."""
