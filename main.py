"""Entry point for Smart Knowledge Hub MCP server scaffold."""

from __future__ import annotations

from core.settings import load_settings
from observability.logger import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Load settings and fail fast when config is invalid."""
    try:
        settings = load_settings('config/settings.yaml')
    except Exception as exc:
        LOGGER.error('Failed to load settings: %s', exc)
        raise SystemExit(1) from exc

    LOGGER.info(
        'Configuration loaded successfully (llm=%s, embedding=%s)',
        settings.llm.get('provider'),
        settings.embedding.get('provider'),
    )
    print('Smart Knowledge Hub scaffold is ready.')


if __name__ == '__main__':
    main()
