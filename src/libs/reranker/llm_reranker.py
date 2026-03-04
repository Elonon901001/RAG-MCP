"""LLM-powered reranker implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory
from libs.reranker.base_reranker import BaseReranker, RerankCandidate


class LLMReranker(BaseReranker):
    """Rerank candidates by asking an LLM to output ranked candidate ids."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._last_fallback_reason: str | None = None

    @property
    def last_fallback_reason(self) -> str | None:
        """Last fallback reason if the previous rerank used fallback."""
        return self._last_fallback_reason

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        self._last_fallback_reason = None
        self._validate_inputs(query, candidates)
        llm = self._resolve_llm()
        prompt = self._build_prompt(query, candidates)
        try:
            response_text = llm.chat(
                [
                    {"role": "system", "content": "You are a reranking engine."},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:  # pragma: no cover - exercised in tests with fake errors
            self._last_fallback_reason = f"llm_call_failed: {exc}"
            return list(candidates)

        ranked_ids = self._parse_ranked_ids(response_text)
        return self._reorder_candidates(candidates, ranked_ids)

    def _resolve_llm(self) -> BaseLLM:
        llm_client = self.config.get("llm_client")
        if llm_client is not None:
            if hasattr(llm_client, "chat") and callable(llm_client.chat):
                return llm_client
            raise ValueError("rerank.llm_client must provide callable chat(messages).")

        llm_settings = self.config.get("llm")
        if isinstance(llm_settings, dict):
            return LLMFactory.create({"llm": llm_settings})

        raise ValueError(
            "Missing LLM configuration for llm reranker. "
            "Provide rerank.llm_client or rerank.llm settings."
        )

    def _build_prompt(self, query: str, candidates: list[RerankCandidate]) -> str:
        prompt_template = self.config.get("prompt_template")
        if prompt_template is None:
            prompt_template = self._load_prompt_template()
        if not isinstance(prompt_template, str):
            raise ValueError("rerank.prompt_template must be a string.")

        payload = {
            "query": query,
            "candidate_ids": [str(item["id"]) for item in candidates],
            "candidates": candidates,
            "instruction": (
                'Return strict JSON with shape: {"ranked_ids": ["id1","id2",...]} '
                "without markdown."
            ),
        }
        return f"{prompt_template.strip()}\\n\\n{json.dumps(payload, ensure_ascii=False)}"

    def _load_prompt_template(self) -> str:
        prompt_path = self.config.get("prompt_path", "config/prompts/rerank.txt")
        if not isinstance(prompt_path, str) or not prompt_path:
            raise ValueError("rerank.prompt_path must be a non-empty string.")
        path = Path(prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Rerank prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

    def _parse_ranked_ids(self, response_text: str) -> list[str]:
        if not isinstance(response_text, str) or not response_text.strip():
            raise ValueError("Invalid rerank response: empty content.")
        parsed = self._parse_json_object(response_text)
        ranked_ids = parsed.get("ranked_ids")
        if not isinstance(ranked_ids, list) or not ranked_ids:
            raise ValueError(
                "Invalid rerank response schema: ranked_ids must be a non-empty list."
            )
        if not all(isinstance(item, str) and item for item in ranked_ids):
            raise ValueError("Invalid rerank response schema: ranked_ids must contain strings.")
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError("Invalid rerank response schema: ranked_ids contains duplicates.")
        return ranked_ids

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
            stripped = "\\n".join(lines).strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid rerank response: expected JSON object. {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Invalid rerank response schema: root must be a JSON object.")
        return parsed

    def _reorder_candidates(
        self,
        candidates: list[RerankCandidate],
        ranked_ids: list[str],
    ) -> list[RerankCandidate]:
        by_id: dict[str, RerankCandidate] = {}
        for item in candidates:
            candidate_id = str(item["id"])
            if candidate_id in by_id:
                raise ValueError(f"Invalid candidates: duplicate id {candidate_id}.")
            by_id[candidate_id] = item

        unknown = [candidate_id for candidate_id in ranked_ids if candidate_id not in by_id]
        if unknown:
            raise ValueError(f"Invalid rerank response schema: unknown ids {unknown}.")

        ranked: list[RerankCandidate] = [by_id[candidate_id] for candidate_id in ranked_ids]
        ranked_set = set(ranked_ids)
        ranked.extend(item for item in candidates if str(item["id"]) not in ranked_set)

        top_n_raw = self.config.get("top_n")
        if top_n_raw is None:
            return ranked
        top_n = int(top_n_raw)
        if top_n <= 0:
            raise ValueError("rerank.top_n must be a positive integer when provided.")
        return ranked[:top_n]

    @staticmethod
    def _validate_inputs(query: str, candidates: list[RerankCandidate]) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates must be a non-empty list.")
        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError(f"candidates[{idx}] must be a dict.")
            if "id" not in candidate:
                raise ValueError(f"candidates[{idx}] must include id.")
