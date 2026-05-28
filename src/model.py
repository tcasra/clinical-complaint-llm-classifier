"""Model-facing helpers exposed as a small reusable module."""

try:
    from .evaluate import (
        API_KEY_CANDIDATES,
        DEFAULT_API_BASE,
        DEFAULT_EXTRACTION_MODEL,
        DEFAULT_STT_MODEL,
        OpenRouterClient,
        OpenRouterError,
        RecoverableOpenRouterError,
        build_classification_messages,
        extract_message_content,
        normalize_classification_payload,
        resolve_api_key,
    )
except ImportError:  # pragma: no cover - fallback for direct script usage
    from evaluate import (  # type: ignore
        API_KEY_CANDIDATES,
        DEFAULT_API_BASE,
        DEFAULT_EXTRACTION_MODEL,
        DEFAULT_STT_MODEL,
        OpenRouterClient,
        OpenRouterError,
        RecoverableOpenRouterError,
        build_classification_messages,
        extract_message_content,
        normalize_classification_payload,
        resolve_api_key,
    )

__all__ = [
    "API_KEY_CANDIDATES",
    "DEFAULT_API_BASE",
    "DEFAULT_EXTRACTION_MODEL",
    "DEFAULT_STT_MODEL",
    "OpenRouterClient",
    "OpenRouterError",
    "RecoverableOpenRouterError",
    "build_classification_messages",
    "extract_message_content",
    "normalize_classification_payload",
    "resolve_api_key",
]
