"""Unit tests for LLM factory routing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


class FakeLLM(BaseLLM):
    def chat(self, messages: list[dict[str, str]]) -> str:
        return "ok"


def test_llm_factory_routes_to_registered_provider() -> None:
    LLMFactory.register("fake", FakeLLM)
    settings = {"llm": {"provider": "fake", "model": "fake-model"}}

    llm = LLMFactory.create(settings)

    assert isinstance(llm, FakeLLM)
    assert llm.config["model"] == "fake-model"


def test_llm_factory_rejects_unknown_provider() -> None:
    settings = {"llm": {"provider": "unknown"}}

    with pytest.raises(ValueError, match="Unsupported LLM provider: unknown"):
        LLMFactory.create(settings)
