"""Tests for settings loading and validation."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.settings import Settings, load_settings


def _valid_config() -> dict:
    return {
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "vector_store": {"provider": "chroma", "persist_directory": "data/db/chroma"},
        "retrieval": {"top_k": 10},
        "rerank": {"strategy": "none"},
        "evaluation": {"enabled": False},
        "observability": {"log_level": "INFO"},
    }


def _make_workspace_tmp_dir() -> Path:
    tmp_dir = PROJECT_ROOT / "tests" / "fixtures" / "_tmp" / str(uuid4())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def test_load_settings_success() -> None:
    settings_file = _make_workspace_tmp_dir() / "settings.yaml"
    settings_file.write_text(yaml.safe_dump(_valid_config()), encoding="utf-8")

    settings = load_settings(str(settings_file))

    assert isinstance(settings, Settings)
    assert settings.embedding["provider"] == "openai"
    assert settings.retrieval["top_k"] == 10


def test_load_settings_missing_required_field() -> None:
    config = _valid_config()
    del config["embedding"]["provider"]
    settings_file = _make_workspace_tmp_dir() / "settings.yaml"
    settings_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="embedding.provider"):
        load_settings(str(settings_file))
