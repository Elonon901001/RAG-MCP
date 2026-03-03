"""OpenAI embedding implementation."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from libs.embedding.base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """Embedding client for OpenAI-compatible embeddings API."""

    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"
    default_max_input_length = 8192
    api_key_env_var = "OPENAI_API_KEY"

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        """Encode a batch of texts into embedding vectors."""
        prepared_texts = self._prepare_texts(texts)
        payload = self._build_payload(prepared_texts)
        response_json = self._post_json(
            url=self._build_endpoint(),
            payload=payload,
            headers=self._build_headers(),
        )
        return self._extract_vectors(response_json)

    def _prepare_texts(self, texts: list[str]) -> list[str]:
        if not isinstance(texts, list) or not texts:
            raise ValueError(
                f"[{self.provider_name}:ValidationError] texts must be a non-empty list."
            )

        max_input_length = int(self.config.get("max_input_length", self.default_max_input_length))
        truncate = bool(self.config.get("truncate_long_input", False))
        prepared: list[str] = []
        for idx, text in enumerate(texts):
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"[{self.provider_name}:ValidationError] texts[{idx}] must be non-empty string."
                )
            if len(text) > max_input_length:
                if truncate:
                    text = text[:max_input_length]
                else:
                    raise ValueError(
                        f"[{self.provider_name}:ValidationError] texts[{idx}] exceeds max_input_length={max_input_length}."
                    )
            prepared.append(text)
        return prepared

    def _build_payload(self, texts: list[str]) -> dict[str, Any]:
        model = self.config.get("model")
        if not model:
            raise ValueError(
                f"[{self.provider_name}:ValidationError] Missing required field: embedding.model"
            )
        return {"model": model, "input": texts}

    def _build_endpoint(self) -> str:
        base_url = str(self.config.get("base_url") or self.default_base_url).rstrip("/")
        return f"{base_url}/embeddings"

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
                f"(embedding.api_key or env {self.api_key_env_var})."
            )
        return str(api_key)

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

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
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

    def _extract_vectors(self, response_json: dict[str, Any]) -> list[list[float]]:
        data = response_json.get("data")
        if not isinstance(data, list):
            raise RuntimeError(
                f"[{self.provider_name}:ResponseShapeError] Missing response.data list."
            )
        vectors: list[list[float]] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise RuntimeError(
                    f"[{self.provider_name}:ResponseShapeError] data[{idx}].embedding missing."
                )
            embedding = item["embedding"]
            if not all(isinstance(v, (int, float)) for v in embedding):
                raise RuntimeError(
                    f"[{self.provider_name}:ResponseShapeError] data[{idx}].embedding must be numeric list."
                )
            vectors.append([float(v) for v in embedding])
        return vectors
