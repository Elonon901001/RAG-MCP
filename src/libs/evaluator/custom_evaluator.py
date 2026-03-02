"""Lightweight evaluator with deterministic retrieval metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from libs.evaluator.base_evaluator import BaseEvaluator

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class CustomEvaluator(BaseEvaluator):
    """Compute simple retrieval metrics such as hit_rate and mrr."""

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: "TraceContext | None" = None,
    ) -> dict[str, float]:
        del query, trace

        if not retrieved_ids:
            return {"hit_rate": 0.0, "mrr": 0.0}

        golden_set = set(golden_ids)
        if not golden_set:
            return {"hit_rate": 0.0, "mrr": 0.0}

        first_rank: int | None = None
        for idx, item_id in enumerate(retrieved_ids, start=1):
            if item_id in golden_set:
                first_rank = idx
                break

        if first_rank is None:
            return {"hit_rate": 0.0, "mrr": 0.0}

        return {
            "hit_rate": 1.0,
            "mrr": 1.0 / float(first_rank),
        }
