"""Factory for creating reranker backend instances."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from libs.reranker.base_reranker import BaseReranker, NoneReranker


class RerankerFactory:
    """Create reranker instances based on settings backend/strategy."""

    _registry: dict[str, type[BaseReranker]] = {"none": NoneReranker}

    @classmethod
    def register(cls, provider: str, reranker_cls: type[BaseReranker]) -> None:
        cls._registry[provider.lower()] = reranker_cls

    @classmethod
    def create(cls, settings: Any) -> BaseReranker:
        rerank_settings = cls._extract_rerank_settings(settings)
        provider = cls._get_provider(rerank_settings)

        reranker_cls = cls._registry.get(provider)
        if reranker_cls is None:
            available = ", ".join(sorted(cls._registry.keys())) or "<none>"
            raise ValueError(
                "Unsupported reranker backend: "
                f"{provider}. Available backends: {available}"
            )

        return reranker_cls(config=rerank_settings)

    @staticmethod
    def _get_provider(rerank_settings: dict[str, Any]) -> str:
        raw_provider = (
            rerank_settings.get("backend")
            or rerank_settings.get("strategy")
            or rerank_settings.get("provider")
            or "none"
        )
        return str(raw_provider).lower()

    @staticmethod
    def _extract_rerank_settings(settings: Any) -> dict[str, Any]:
        if isinstance(settings, dict):
            rerank = settings.get("rerank")
            if isinstance(rerank, dict):
                return rerank
            raise ValueError("Missing required settings field: rerank")

        if isinstance(settings, SimpleNamespace):
            rerank = getattr(settings, "rerank", None)
            if isinstance(rerank, dict):
                return rerank
            raise ValueError("Missing required settings field: rerank")

        rerank = getattr(settings, "rerank", None)
        if isinstance(rerank, dict):
            return rerank

        raise ValueError("Missing required settings field: rerank")
