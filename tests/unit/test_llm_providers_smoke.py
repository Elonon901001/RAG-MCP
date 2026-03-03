"""Smoke tests for OpenAI-compatible LLM providers using mocked HTTP."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.llm.azure_llm import AzureOpenAILLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAICompatibleLLM


def _valid_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "hello"}]


def test_llm_factory_routes_to_openai_compatible_providers() -> None:
    openai = LLMFactory.create(
        {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "k-openai",
            }
        }
    )
    azure = LLMFactory.create(
        {
            "llm": {
                "provider": "azure",
                "model": "gpt-4o-mini",
                "endpoint": "https://azure.example",
                "api_key": "k-azure",
            }
        }
    )
    deepseek = LLMFactory.create(
        {
            "llm": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_key": "k-deepseek",
            }
        }
    )

    assert isinstance(openai, OpenAICompatibleLLM)
    assert isinstance(azure, AzureOpenAILLM)
    assert isinstance(deepseek, DeepSeekLLM)


def test_openai_llm_chat_validates_message_shape_with_readable_error() -> None:
    llm = OpenAICompatibleLLM(config={"model": "gpt-4o-mini", "api_key": "k"})

    with pytest.raises(ValueError, match=r"\[openai:ValidationError\]"):
        llm.chat([{"role": "user", "content": 1}])  # type: ignore[list-item]


def test_openai_provider_chat_uses_mocked_http_and_parses_text() -> None:
    llm = OpenAICompatibleLLM(
        config={"model": "gpt-4o-mini", "api_key": "k-openai", "temperature": 0.2}
    )

    def fake_post_json(url: str, payload: dict, headers: dict) -> dict:
        assert url.endswith("/chat/completions")
        assert payload["model"] == "gpt-4o-mini"
        assert payload["messages"] == _valid_messages()
        assert headers["Authorization"] == "Bearer k-openai"
        return {"choices": [{"message": {"content": "openai-ok"}}]}

    llm._post_json = fake_post_json  # type: ignore[method-assign]
    result = llm.chat(_valid_messages())

    assert result == "openai-ok"


def test_azure_provider_chat_uses_mocked_http_and_parses_text() -> None:
    llm = AzureOpenAILLM(
        config={
            "model": "gpt-4o-mini",
            "endpoint": "https://azure.example",
            "api_version": "2024-10-21",
            "api_key": "k-azure",
        }
    )

    def fake_post_json(url: str, payload: dict, headers: dict) -> dict:
        assert "openai/deployments/gpt-4o-mini/chat/completions" in url
        assert "api-version=2024-10-21" in url
        assert "model" not in payload
        assert payload["messages"] == _valid_messages()
        assert headers["api-key"] == "k-azure"
        return {"choices": [{"message": {"content": "azure-ok"}}]}

    llm._post_json = fake_post_json  # type: ignore[method-assign]
    result = llm.chat(_valid_messages())

    assert result == "azure-ok"


def test_deepseek_provider_chat_uses_mocked_http_and_parses_text() -> None:
    llm = DeepSeekLLM(config={"model": "deepseek-chat", "api_key": "k-deepseek"})

    def fake_post_json(url: str, payload: dict, headers: dict) -> dict:
        assert url.startswith("https://api.deepseek.com/v1")
        assert payload["model"] == "deepseek-chat"
        assert headers["Authorization"] == "Bearer k-deepseek"
        return {"choices": [{"message": {"content": "deepseek-ok"}}]}

    llm._post_json = fake_post_json  # type: ignore[method-assign]
    result = llm.chat(_valid_messages())

    assert result == "deepseek-ok"
