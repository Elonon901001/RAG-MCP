"""Ollama chat completion implementation."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from libs.llm.base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    """Chat completion client for local Ollama service."""

    provider_name = "ollama"
    default_base_url = "http://localhost:11434"

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Generate response via Ollama chat endpoint."""
        self._validate_messages(messages)

        model = self.config.get("model")
        if not model:
            raise ValueError(
                f"[{self.provider_name}:ValidationError] Missing required field: llm.model"
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if "options" in self.config and isinstance(self.config["options"], dict):
            payload["options"] = self.config["options"]

        response_json = self._post_json(self._build_endpoint(), payload)
        return self._extract_content(response_json)

    def _build_endpoint(self) -> str:
        base_url = str(self.config.get("base_url") or self.default_base_url).rstrip("/")
        return f"{base_url}/api/chat"

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = self._resolve_timeout_seconds()
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"[{self.provider_name}:HTTPError] Request failed with status {exc.code}."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"[{self.provider_name}:ConnectionError] Failed to reach Ollama service."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"[{self.provider_name}:TimeoutError] Request timed out."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"[{self.provider_name}:ParseError] Response is not valid JSON."
            ) from exc

    def _resolve_timeout_seconds(self) -> float:
        timeout = self.config.get("timeout", 30)
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"[{self.provider_name}:ValidationError] Invalid timeout value: {timeout!r}."
            ) from exc
        if timeout_value <= 0:
            raise ValueError(
                f"[{self.provider_name}:ValidationError] Timeout must be > 0."
            )
        return timeout_value

    def _extract_content(self, response_json: dict[str, Any]) -> str:
        try:
            content = response_json["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"[{self.provider_name}:ResponseShapeError] Missing message.content."
            ) from exc
        if not isinstance(content, str):
            raise RuntimeError(
                f"[{self.provider_name}:ResponseShapeError] message.content must be string."
            )
        return content

    def _validate_messages(self, messages: list[dict[str, str]]) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"[{self.provider_name}:ValidationError] messages must be a non-empty list."
            )
        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    f"[{self.provider_name}:ValidationError] messages[{idx}] must be a dict."
                )
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role:
                raise ValueError(
                    f"[{self.provider_name}:ValidationError] messages[{idx}].role must be non-empty string."
                )
            if not isinstance(content, str):
                raise ValueError(
                    f"[{self.provider_name}:ValidationError] messages[{idx}].content must be string."
                )
