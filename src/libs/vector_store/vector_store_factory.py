"""Factory for creating vector store backend instances."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from libs.vector_store.base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """Create vector store instances based on provider settings."""

    _registry: dict[str, type[BaseVectorStore]] = {}
    _defaults_registered = False

    @classmethod
    def register(cls, provider: str, vector_store_cls: type[BaseVectorStore]) -> None:
        cls._registry[provider.lower()] = vector_store_cls

    @classmethod
    def create(cls, settings: Any) -> BaseVectorStore:
        cls._register_default_providers()
        vector_store_settings = cls._extract_vector_store_settings(settings)
        provider = vector_store_settings.get('provider')
        if not provider:
            raise ValueError('Missing required settings field: vector_store.provider')

        vector_store_cls = cls._registry.get(str(provider).lower())
        if vector_store_cls is None:
            available = ', '.join(sorted(cls._registry.keys())) or '<none>'
            raise ValueError(
                'Unsupported vector store provider: '
                f"{provider}. Available providers: {available}"
            )

        return vector_store_cls(config=vector_store_settings)

    @classmethod
    def _register_default_providers(cls) -> None:
        if cls._defaults_registered:
            return

        from libs.vector_store.chroma_store import ChromaStore

        cls.register("chroma", ChromaStore)
        cls._defaults_registered = True

    @staticmethod
    def _extract_vector_store_settings(settings: Any) -> dict[str, Any]:
        if isinstance(settings, dict):
            vector_store = settings.get('vector_store')
            if isinstance(vector_store, dict):
                return vector_store
            raise ValueError('Missing required settings field: vector_store')

        if isinstance(settings, SimpleNamespace):
            vector_store = getattr(settings, 'vector_store', None)
            if isinstance(vector_store, dict):
                return vector_store
            raise ValueError('Missing required settings field: vector_store')

        vector_store = getattr(settings, 'vector_store', None)
        if isinstance(vector_store, dict):
            return vector_store

        raise ValueError('Missing required settings field: vector_store')
