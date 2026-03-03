"""Azure OpenAI embedding implementation."""

from __future__ import annotations

import os
from typing import Any

from libs.embedding.openai_embedding import OpenAIEmbedding


class AzureEmbedding(OpenAIEmbedding):
    """Embedding client for Azure OpenAI deployments."""

    provider_name = "azure"
    default_api_version = "2024-06-01"
    api_key_env_var = "AZURE_OPENAI_API_KEY"

    def _build_endpoint(self) -> str:
        endpoint = self.config.get("endpoint") or self.config.get("azure_endpoint")
        deployment = self.config.get("deployment") or self.config.get("model")
        api_version = self.config.get("api_version") or self.default_api_version
        if not endpoint:
            raise ValueError(
                f"[{self.provider_name}:ValidationError] Missing required field: embedding.endpoint"
            )
        if not deployment:
            raise ValueError(
                f"[{self.provider_name}:ValidationError] Missing required field: embedding.deployment"
            )
        endpoint_text = str(endpoint).rstrip("/")
        return (
            f"{endpoint_text}/openai/deployments/{deployment}/embeddings"
            f"?api-version={api_version}"
        )

    def _build_payload(self, texts: list[str]) -> dict[str, Any]:
        return {"input": texts}

    def _build_headers(self) -> dict[str, str]:
        return {
            "api-key": self._resolve_api_key(),
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
