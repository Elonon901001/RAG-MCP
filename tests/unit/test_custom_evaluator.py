"""Unit tests for custom evaluator metrics and factory routing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


def test_custom_evaluator_returns_hit_rate_and_mrr() -> None:
    evaluator = CustomEvaluator(config={"backend": "custom"})

    metrics = evaluator.evaluate(
        query="what is rag",
        retrieved_ids=["doc-a", "doc-b", "doc-c"],
        golden_ids=["doc-b", "doc-z"],
    )

    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5


def test_custom_evaluator_returns_zero_when_no_match() -> None:
    evaluator = CustomEvaluator(config={"backend": "custom"})

    metrics = evaluator.evaluate(
        query="what is rag",
        retrieved_ids=["doc-a", "doc-b"],
        golden_ids=["doc-x"],
    )

    assert metrics == {"hit_rate": 0.0, "mrr": 0.0}


def test_evaluator_factory_routes_to_custom_backend() -> None:
    settings = {"evaluation": {"backend": "custom", "enabled": False}}

    evaluator = EvaluatorFactory.create(settings)

    assert isinstance(evaluator, CustomEvaluator)
    assert evaluator.config["enabled"] is False


def test_evaluator_factory_rejects_unknown_backend() -> None:
    settings = {"evaluation": {"backend": "unknown"}}

    with pytest.raises(ValueError, match="Unsupported evaluator backend: unknown"):
        EvaluatorFactory.create(settings)
