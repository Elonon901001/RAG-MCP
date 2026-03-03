"""OpenAI-compatible chat completion implementation."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from libs.llm.base_llm import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    """Chat completion client for OpenAI-compatible providers."""

    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"
    api_key_env_var = "OPENAI_API_KEY"

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from a list of chat messages."""
        self._validate_messages(messages)
        payload = self._build_payload(messages)
        response_json = self._post_json(
            url=self._build_endpoint(),
            payload=payload,
            headers=self._build_headers(),
        )
        return self._extract_message_content(response_json)

    def _build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        model = self.config.get("model")
        if not model:
            raise ValueError(
                f"[{self.provider_name}:ValidationError] Missing required field: llm.model"
            )

        payload: dict[str, Any] = {"model": model, "messages": messages}
        if "temperature" in self.config:
            payload["temperature"] = self.config["temperature"]
        return payload

    def _build_endpoint(self) -> str:
        base_url = str(self.config.get("base_url") or self.default_base_url).rstrip("/")
        return f"{base_url}/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._resolve_api_key()}",
            "Content-Type": "application/json",
        }

    def _resolve_api_key(self) -> str:
        api_key = self.config.get("api_key") or os.getenv(self.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"[{self.provider_name}:ValidationError] Missing API key "
                f"(llm.api_key or env {self.api_key_env_var})."
            )
        return str(api_key)

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        timeout = self._resolve_timeout_seconds()
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"[{self.provider_name}:HTTPError] Request failed with status {exc.code}."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"[{self.provider_name}:ConnectionError] Request failed: {exc.reason}."
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

    def _extract_message_content(self, response_json: dict[str, Any]) -> str:
        try:
            choice = response_json["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"[{self.provider_name}:ResponseShapeError] Missing choices[0].message.content."
            ) from exc

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "".join(parts)

        raise RuntimeError(
            f"[{self.provider_name}:ResponseShapeError] message.content must be string."
        )

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
