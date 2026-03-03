"""Unit tests for Ollama LLM with mocked HTTP behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import OllamaLLM


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "hello ollama"}]


def test_factory_creates_ollama_provider() -> None:
    llm = LLMFactory.create({"llm": {"provider": "ollama", "model": "llama3.1"}})
    assert isinstance(llm, OllamaLLM)


def test_ollama_chat_success_with_mocked_http() -> None:
    llm = OllamaLLM(config={"model": "llama3.1", "base_url": "http://localhost:11434"})

    def fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        assert url.endswith("/api/chat")
        assert payload["model"] == "llama3.1"
        assert payload["messages"] == _messages()
        return {"message": {"content": "ollama-ok"}}

    llm._post_json = fake_post_json  # type: ignore[method-assign]
    assert llm.chat(_messages()) == "ollama-ok"


def test_ollama_chat_connection_error_is_readable_and_hides_config() -> None:
    llm = OllamaLLM(config={"model": "llama3.1", "api_key": "should-not-appear"})

    def fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("[ollama:ConnectionError] Failed to reach Ollama service.")

    llm._post_json = fake_post_json  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match=r"\[ollama:ConnectionError\]"):
        llm.chat(_messages())

    with pytest.raises(RuntimeError) as exc_info:
        llm.chat(_messages())
    assert "should-not-appear" not in str(exc_info.value)


def test_ollama_post_json_wraps_timeout_error() -> None:
    llm = OllamaLLM(config={"model": "llama3.1"})

    def raise_timeout(*args: object, **kwargs: object) -> object:
        raise TimeoutError("socket timeout")

    # Patch module symbol used by OllamaLLM._post_json
    import libs.llm.ollama_llm as module

    old_urlopen = module.urlopen
    module.urlopen = raise_timeout  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match=r"\[ollama:TimeoutError\]"):
            llm._post_json("http://localhost:11434/api/chat", {"model": "llama3.1"})
    finally:
        module.urlopen = old_urlopen  # type: ignore[assignment]


def test_ollama_post_json_wraps_url_error() -> None:
    llm = OllamaLLM(config={"model": "llama3.1"})

    def raise_url_error(*args: object, **kwargs: object) -> object:
        raise URLError("connection refused")

    import libs.llm.ollama_llm as module

    old_urlopen = module.urlopen
    module.urlopen = raise_url_error  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match=r"\[ollama:ConnectionError\]"):
            llm._post_json("http://localhost:11434/api/chat", {"model": "llama3.1"})
    finally:
        module.urlopen = old_urlopen  # type: ignore[assignment]
