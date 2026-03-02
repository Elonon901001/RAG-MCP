"""Abstract interface for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Base interface for all LLM providers."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Generate a chat response from message history."""
