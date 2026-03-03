"""DeepSeek chat completion implementation."""

from __future__ import annotations

from libs.llm.openai_llm import OpenAICompatibleLLM


class DeepSeekLLM(OpenAICompatibleLLM):
    """Chat completion client for DeepSeek's OpenAI-compatible API."""

    provider_name = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"
    api_key_env_var = "DEEPSEEK_API_KEY"
