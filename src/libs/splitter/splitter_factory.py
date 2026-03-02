"""Factory for creating splitter strategy instances."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from libs.splitter.base_splitter import BaseSplitter


class SplitterFactory:
    """Create splitter instances based on strategy settings."""

    _registry: dict[str, type[BaseSplitter]] = {}

    @classmethod
    def register(cls, strategy: str, splitter_cls: type[BaseSplitter]) -> None:
        cls._registry[strategy.lower()] = splitter_cls

    @classmethod
    def create(cls, settings: Any) -> BaseSplitter:
        splitter_settings = cls._extract_splitter_settings(settings)
        strategy = splitter_settings.get('strategy')
        if not strategy:
            raise ValueError('Missing required settings field: splitter.strategy')

        splitter_cls = cls._registry.get(str(strategy).lower())
        if splitter_cls is None:
            available = ', '.join(sorted(cls._registry.keys())) or '<none>'
            raise ValueError(
                f"Unsupported splitter strategy: {strategy}. Available strategies: {available}"
            )

        return splitter_cls(config=splitter_settings)

    @staticmethod
    def _extract_splitter_settings(settings: Any) -> dict[str, Any]:
        if isinstance(settings, dict):
            splitter = settings.get('splitter')
            if isinstance(splitter, dict):
                return splitter
            raise ValueError('Missing required settings field: splitter')

        if isinstance(settings, SimpleNamespace):
            splitter = getattr(settings, 'splitter', None)
            if isinstance(splitter, dict):
                return splitter
            raise ValueError('Missing required settings field: splitter')

        splitter = getattr(settings, 'splitter', None)
        if isinstance(splitter, dict):
            return splitter

        raise ValueError('Missing required settings field: splitter')
