"""Dataset access helpers exposed as a small reusable module."""

try:
    from .evaluate import (
        DEFAULT_CSV_PATH,
        DEFAULT_RECORDINGS_DIR,
        allowed_prompt_labels,
        load_metadata,
        load_split_reference_rows,
        select_audio_files,
    )
except ImportError:  # pragma: no cover - fallback for direct script usage
    from evaluate import (  # type: ignore
        DEFAULT_CSV_PATH,
        DEFAULT_RECORDINGS_DIR,
        allowed_prompt_labels,
        load_metadata,
        load_split_reference_rows,
        select_audio_files,
    )

__all__ = [
    "DEFAULT_CSV_PATH",
    "DEFAULT_RECORDINGS_DIR",
    "allowed_prompt_labels",
    "load_metadata",
    "load_split_reference_rows",
    "select_audio_files",
]
