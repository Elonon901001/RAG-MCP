"""Configuration loading and validation for application settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    """Application settings container with minimal structured sections."""

    llm: dict[str, Any]
    embedding: dict[str, Any]
    vector_store: dict[str, Any]
    retrieval: dict[str, Any]
    rerank: dict[str, Any]
    evaluation: dict[str, Any]
    observability: dict[str, Any]


def _get_nested_value(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for part in path.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def validate_settings(settings: Settings) -> None:
    """Validate required settings fields and fail with readable field paths."""
    config = {
        'llm': settings.llm,
        'embedding': settings.embedding,
        'vector_store': settings.vector_store,
        'retrieval': settings.retrieval,
        'rerank': settings.rerank,
        'evaluation': settings.evaluation,
        'observability': settings.observability,
    }

    required_paths = (
        'llm.provider',
        'embedding.provider',
        'vector_store.provider',
        'retrieval.top_k',
        'rerank.strategy',
        'evaluation.enabled',
        'observability.log_level',
    )

    for path in required_paths:
        value = _get_nested_value(config, path)
        if value is None:
            raise ValueError(f'Missing required settings field: {path}')


def load_settings(path: str) -> Settings:
    """Load YAML settings from disk and validate required fields."""
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f'Settings file not found: {settings_path}')

    with settings_path.open('r', encoding='utf-8') as f:
        parsed = yaml.safe_load(f) or {}

    if not isinstance(parsed, dict):
        raise ValueError('Settings root must be a mapping/object.')

    settings = Settings(
        llm=parsed.get('llm') or {},
        embedding=parsed.get('embedding') or {},
        vector_store=parsed.get('vector_store') or {},
        retrieval=parsed.get('retrieval') or {},
        rerank=parsed.get('rerank') or {},
        evaluation=parsed.get('evaluation') or {},
        observability=parsed.get('observability') or {},
    )
    validate_settings(settings)
    return settings
