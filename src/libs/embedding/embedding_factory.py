"""Factory for creating embedding provider instances."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from libs.embedding.base_embedding import BaseEmbedding


class EmbeddingFactory:
    """Create embedding instances based on provider settings."""

    _registry: dict[str, type[BaseEmbedding]] = {}
    _defaults_registered = False

    @classmethod
    def register(cls, provider: str, embedding_cls: type[BaseEmbedding]) -> None:
        cls._registry[provider.lower()] = embedding_cls

    @classmethod
    def create(cls, settings: Any) -> BaseEmbedding:
        cls._register_default_providers()
        embedding_settings = cls._extract_embedding_settings(settings)
        provider = embedding_settings.get('provider')
        if not provider:
            raise ValueError('Missing required settings field: embedding.provider')

        embedding_cls = cls._registry.get(str(provider).lower())
        if embedding_cls is None:
            available = ', '.join(sorted(cls._registry.keys())) or '<none>'
            raise ValueError(
                'Unsupported embedding provider: '
                f"{provider}. Available providers: {available}"
            )

        return embedding_cls(config=embedding_settings)

    @classmethod
    def _register_default_providers(cls) -> None:
        if cls._defaults_registered:
            return

        from libs.embedding.azure_embedding import AzureEmbedding
        from libs.embedding.openai_embedding import OpenAIEmbedding

        cls.register("openai", OpenAIEmbedding)
        cls.register("azure", AzureEmbedding)
        cls._defaults_registered = True

    @staticmethod
    def _extract_embedding_settings(settings: Any) -> dict[str, Any]:
        if isinstance(settings, dict):
            embedding = settings.get('embedding')
            if isinstance(embedding, dict):
                return embedding
            raise ValueError('Missing required settings field: embedding')

        if isinstance(settings, SimpleNamespace):
            embedding = getattr(settings, 'embedding', None)
            if isinstance(embedding, dict):
                return embedding
            raise ValueError('Missing required settings field: embedding')

        embedding = getattr(settings, 'embedding', None)
        if isinstance(embedding, dict):
            return embedding

        raise ValueError('Missing required settings field: embedding')
