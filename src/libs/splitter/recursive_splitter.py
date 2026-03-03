"""Recursive text splitter with basic Markdown awareness."""

from __future__ import annotations

import re
from typing import Any

from libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """Split text recursively by separator priority and chunk length."""

    default_separators = ("\n\n", "\n", " ")

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        """Split text into chunks while preserving fenced code blocks."""
        if not isinstance(text, str):
            raise ValueError("[recursive:ValidationError] text must be a string.")
        if not text:
            return []

        chunk_size = self._chunk_size()
        chunk_overlap = self._chunk_overlap(chunk_size)
        separators = self._separators()

        parts = self._split_markdown_aware(text)
        chunks: list[str] = []
        for part in parts:
            if not part.strip():
                continue
            if len(part) <= chunk_size:
                chunks.append(part)
            else:
                chunks.extend(self._recursive_split(part, separators, chunk_size))

        return self._apply_overlap(chunks, chunk_overlap)

    def _chunk_size(self) -> int:
        value = int(self.config.get("chunk_size", 1000))
        if value <= 0:
            raise ValueError("[recursive:ValidationError] chunk_size must be > 0.")
        return value

    def _chunk_overlap(self, chunk_size: int) -> int:
        value = int(self.config.get("chunk_overlap", 100))
        if value < 0:
            raise ValueError("[recursive:ValidationError] chunk_overlap must be >= 0.")
        if value >= chunk_size:
            raise ValueError(
                "[recursive:ValidationError] chunk_overlap must be smaller than chunk_size."
            )
        return value

    def _separators(self) -> list[str]:
        raw = self.config.get("separators")
        if raw is None:
            return list(self.default_separators)
        if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
            raise ValueError(
                "[recursive:ValidationError] separators must be a list of strings."
            )
        return [s for s in raw if s]

    def _split_markdown_aware(self, text: str) -> list[str]:
        # Keep fenced code blocks as independent units to avoid split corruption.
        pattern = r"(```[\s\S]*?```)"
        segments = re.split(pattern, text)
        return [seg for seg in segments if seg]

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
        chunk_size: int,
    ) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        if not separators:
            return self._hard_wrap(text, chunk_size)

        sep = separators[0]
        if sep in text:
            pieces = text.split(sep)
            merged = self._merge_with_separator(pieces, sep, chunk_size)
            output: list[str] = []
            for item in merged:
                if len(item) <= chunk_size:
                    output.append(item)
                else:
                    output.extend(self._recursive_split(item, separators[1:], chunk_size))
            return output

        return self._recursive_split(text, separators[1:], chunk_size)

    def _merge_with_separator(
        self,
        pieces: list[str],
        sep: str,
        chunk_size: int,
    ) -> list[str]:
        result: list[str] = []
        current = ""
        for idx, piece in enumerate(pieces):
            candidate = piece if not current else current + sep + piece
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                current = piece
            if idx == len(pieces) - 1 and current:
                result.append(current)
        return result

    def _hard_wrap(self, text: str, chunk_size: int) -> list[str]:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _apply_overlap(self, chunks: list[str], chunk_overlap: int) -> list[str]:
        if chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        result: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                result.append(chunk)
                continue
            prev_tail = chunks[idx - 1][-chunk_overlap:]
            if prev_tail and not chunk.startswith(prev_tail):
                result.append(prev_tail + chunk)
            else:
                result.append(chunk)
        return result
