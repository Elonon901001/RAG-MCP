"""Ollama embedding implementation."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from libs.embedding.base_embedding import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):
    """Embedding client for local Ollama service."""

    provider_name = "ollama"
    default_base_url = "http://localhost:11434"
    default_max_input_length = 8192

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        """Encode a batch of texts with Ollama embeddings API."""
        prepared_texts = self._prepare_texts(texts)
        vectors: list[list[float]] = []
        for text in prepared_texts:
            payload: dict[str, Any] = {
                "model": self._resolve_model(),
                "prompt": text,
            }
            response_json = self._post_json(self._build_endpoint(), payload)
            vectors.append(self._extract_vector(response_json))
        return vectors

    def _resolve_model(self) -> str:
        model = self.config.get("model")
        if not model:
            raise ValueError(
                f"[{self.provider_name}:ValidationError] Missing required field: embedding.model"
            )
        return str(model)

    def _build_endpoint(self) -> str:
        base_url = str(self.config.get("base_url") or self.default_base_url).rstrip("/")
        return f"{base_url}/api/embeddings"

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

    def _extract_vector(self, response_json: dict[str, Any]) -> list[float]:
        embedding = response_json.get("embedding")
        if isinstance(embedding, list):
            if not all(isinstance(v, (int, float)) for v in embedding):
                raise RuntimeError(
                    f"[{self.provider_name}:ResponseShapeError] embedding must be numeric list."
                )
            return [float(v) for v in embedding]

        data = response_json.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            nested = data[0].get("embedding")
            if isinstance(nested, list) and all(isinstance(v, (int, float)) for v in nested):
                return [float(v) for v in nested]

        raise RuntimeError(
            f"[{self.provider_name}:ResponseShapeError] Missing embedding vector."
        )
