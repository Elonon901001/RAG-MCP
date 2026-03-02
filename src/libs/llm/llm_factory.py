"""Factory for creating LLM provider instances."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from libs.llm.base_llm import BaseLLM


class LLMFactory:
    """Create LLM instances based on provider settings."""

    _registry: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register(cls, provider: str, llm_cls: type[BaseLLM]) -> None:
        cls._registry[provider.lower()] = llm_cls

    @classmethod
    def create(cls, settings: Any) -> BaseLLM:
        llm_settings = cls._extract_llm_settings(settings)
        provider = llm_settings.get('provider')
        if not provider:
            raise ValueError('Missing required settings field: llm.provider')

        llm_cls = cls._registry.get(str(provider).lower())
        if llm_cls is None:
            available = ', '.join(sorted(cls._registry.keys())) or '<none>'
            raise ValueError(
                f"Unsupported LLM provider: {provider}. Available providers: {available}"
            )

        return llm_cls(config=llm_settings)

    @staticmethod
    def _extract_llm_settings(settings: Any) -> dict[str, Any]:
        if isinstance(settings, dict):
            llm = settings.get('llm')
            if isinstance(llm, dict):
                return llm
            raise ValueError('Missing required settings field: llm')

        if isinstance(settings, SimpleNamespace):
            llm = getattr(settings, 'llm', None)
            if isinstance(llm, dict):
                return llm
            raise ValueError('Missing required settings field: llm')

        llm = getattr(settings, 'llm', None)
        if isinstance(llm, dict):
            return llm

        raise ValueError('Missing required settings field: llm')
