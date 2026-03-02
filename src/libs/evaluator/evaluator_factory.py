"""Factory for creating evaluator backend instances."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator


class EvaluatorFactory:
    """Create evaluator instances based on evaluation backend settings."""

    _registry: dict[str, type[BaseEvaluator]] = {"custom": CustomEvaluator}

    @classmethod
    def register(cls, backend: str, evaluator_cls: type[BaseEvaluator]) -> None:
        cls._registry[backend.lower()] = evaluator_cls

    @classmethod
    def create(cls, settings: Any) -> BaseEvaluator:
        evaluation_settings = cls._extract_evaluation_settings(settings)
        backend = cls._get_backend(evaluation_settings)

        evaluator_cls = cls._registry.get(backend)
        if evaluator_cls is None:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise ValueError(
                f"Unsupported evaluator backend: {backend}. Available backends: {available}"
            )

        return evaluator_cls(config=evaluation_settings)

    @staticmethod
    def _get_backend(evaluation_settings: dict[str, Any]) -> str:
        raw_backend = (
            evaluation_settings.get("backend")
            or evaluation_settings.get("provider")
            or "custom"
        )
        return str(raw_backend).lower()

    @staticmethod
    def _extract_evaluation_settings(settings: Any) -> dict[str, Any]:
        if isinstance(settings, dict):
            evaluation = settings.get("evaluation")
            if isinstance(evaluation, dict):
                return evaluation
            raise ValueError("Missing required settings field: evaluation")

        if isinstance(settings, SimpleNamespace):
            evaluation = getattr(settings, "evaluation", None)
            if isinstance(evaluation, dict):
                return evaluation
            raise ValueError("Missing required settings field: evaluation")

        evaluation = getattr(settings, "evaluation", None)
        if isinstance(evaluation, dict):
            return evaluation

        raise ValueError("Missing required settings field: evaluation")
