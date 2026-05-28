from __future__ import annotations

import argparse
import base64
import csv
import difflib
import html
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+",
)

import requests


DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_CSV_PATH = Path("data/overview-of-recordings.csv")
DEFAULT_RECORDINGS_DIR = Path("data/recordings")
DEFAULT_OUTPUT_DIR = Path("outputs")
RUN_STATE_FILE = "run_state.json"
RESULTS_JSON_FILE = "results.json"
RESULTS_CSV_FILE = "results.csv"
SUMMARY_JSON_FILE = "summary.json"
STT_RECORDS_JSON_FILE = "stt_records.json"
ERRORS_JSON_FILE = "errors.json"
DEFAULT_STT_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_EXTRACTION_MODEL = "anthropic/claude-haiku-4.5"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_METHODS = ("zero_shot", "few_shot")
DEFAULT_MAX_RECOVERABLE_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_SUBGROUP_FIELDS = (
    "background_noise_audible",
    "audio_clipping",
    "quiet_speaker",
)
DEFAULT_HIGH_RISK_LABELS = (
    "Blurry vision",
    "Hard to breath",
    "Heart hurts",
    "Infected wound",
    "Internal pain",
)
API_KEY_CANDIDATES = (
    "OPENROUTER_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "MPHY_OPENROUTER_API_KEY",
    "MPHY_API_KEY",
)
PUNCTUATION_RE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_RE = re.compile(r"\s+")


class OpenRouterError(RuntimeError):
    pass


class RecoverableOpenRouterError(OpenRouterError):
    pass


@dataclass
class PipelineConfig:
    csv_path: Path
    recordings_dir: Path
    split: str
    output_dir: Path
    file_names: List[str]
    limit: Optional[int]
    language: Optional[str]
    stt_model: str
    extraction_model: str
    api_base: str
    api_key_env: str
    sleep_seconds: float
    timeout_seconds: int
    transcript_source: str
    methods: List[str]
    few_shot_k: int
    few_shot_pool_split: str
    review_threshold: float
    high_risk_review_threshold: float
    high_risk_labels: List[str]
    bootstrap_samples: int
    subgroup_fields: List[str]
    resume_run_dir: Optional[Path]
    concurrency: int = 1
    checkpoint_every: int = 5


def parse_args(argv: Optional[Iterable[str]] = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate zero-shot and few-shot clinical complaint classification with "
            "OpenRouter on transcript text or end-to-end audio."
        )
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_PATH),
        help=f"Path to the metadata CSV. Default: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--recordings-dir",
        default=str(DEFAULT_RECORDINGS_DIR),
        help=f"Path to the recordings root directory. Default: {DEFAULT_RECORDINGS_DIR}",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Recording split under the recordings directory. Default: train",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where reports should be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        help=(
            "Resume an interrupted run from an existing output directory such as "
            "`outputs/20260506T034302Z`."
        ),
    )
    parser.add_argument(
        "--file-name",
        action="append",
        default=[],
        help="Specific file name to process. Repeat this flag to process multiple files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to process after filtering.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="ISO-639-1 language code for transcription. Default: en",
    )
    parser.add_argument(
        "--stt-model",
        default=DEFAULT_STT_MODEL,
        help=f"OpenRouter speech-to-text model. Default: {DEFAULT_STT_MODEL}",
    )
    parser.add_argument(
        "--extraction-model",
        default=DEFAULT_EXTRACTION_MODEL,
        help=f"OpenRouter chat model for classification. Default: {DEFAULT_EXTRACTION_MODEL}",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"OpenRouter API base URL. Default: {DEFAULT_API_BASE}",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=(
            "Primary environment variable to check for the API key. "
            f"Fallbacks: {', '.join(API_KEY_CANDIDATES)}"
        ),
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between files to avoid rate spikes. Default: 0",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="HTTP timeout for each OpenRouter request. Default: 180",
    )
    parser.add_argument(
        "--transcript-source",
        choices=("asr", "gold"),
        default="asr",
        help=(
            "`asr` uses audio -> speech-to-text -> classifier. "
            "`gold` uses the CSV `phrase` text directly."
        ),
    )
    parser.add_argument(
        "--methods",
        default="zero_shot,few_shot",
        help="Comma-separated methods to evaluate. Choices: zero_shot,few_shot",
    )
    parser.add_argument(
        "--few-shot-k",
        type=int,
        default=4,
        help="Number of retrieved few-shot examples to include. Default: 4",
    )
    parser.add_argument(
        "--few-shot-pool-split",
        default="train",
        help="Split to draw few-shot examples from. Default: train",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.60,
        help="Confidence threshold below which a case is sent to human review. Default: 0.60",
    )
    parser.add_argument(
        "--high-risk-review-threshold",
        type=float,
        default=0.80,
        help=(
            "Stricter confidence threshold for high-risk predicted labels. "
            "Default: 0.80"
        ),
    )
    parser.add_argument(
        "--high-risk-label",
        action="append",
        default=[],
        help=(
            "High-risk label requiring stricter review when confidence is low. "
            "Repeat this flag or pass a comma-separated list."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=300,
        help="Bootstrap samples for 95%% confidence intervals. Default: 300",
    )
    parser.add_argument(
        "--subgroup-field",
        action="append",
        default=[],
        help=(
            "Metadata field for subgroup metrics. Repeat this flag or pass a "
            "comma-separated list."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of files to process in parallel via a thread pool. "
            "Threads share an OpenRouter session. Default: 1 (sequential)."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help=(
            "Persist run state and partial results to disk every N completed "
            "files. Default: 5"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    methods = parse_list_argument(args.methods)
    valid_methods = set(DEFAULT_METHODS)
    invalid_methods = [method for method in methods if method not in valid_methods]
    if invalid_methods:
        raise SystemExit(
            f"Unsupported method(s): {', '.join(invalid_methods)}. "
            f"Supported methods: {', '.join(sorted(valid_methods))}"
        )

    high_risk_labels = (
        flatten_repeated_list_argument(args.high_risk_label)
        if args.high_risk_label
        else list(DEFAULT_HIGH_RISK_LABELS)
    )
    subgroup_fields = (
        flatten_repeated_list_argument(args.subgroup_field)
        if args.subgroup_field
        else list(DEFAULT_SUBGROUP_FIELDS)
    )

    return PipelineConfig(
        csv_path=Path(args.csv_path),
        recordings_dir=Path(args.recordings_dir),
        split=args.split,
        output_dir=Path(args.output_dir),
        file_names=args.file_name,
        limit=args.limit,
        language=args.language or None,
        stt_model=args.stt_model,
        extraction_model=args.extraction_model,
        api_base=args.api_base.rstrip("/"),
        api_key_env=args.api_key_env,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        transcript_source=args.transcript_source,
        methods=methods,
        few_shot_k=max(0, args.few_shot_k),
        few_shot_pool_split=args.few_shot_pool_split,
        review_threshold=clamp_float(args.review_threshold, 0.0, 1.0),
        high_risk_review_threshold=clamp_float(
            args.high_risk_review_threshold, 0.0, 1.0
        ),
        high_risk_labels=high_risk_labels,
        bootstrap_samples=max(0, args.bootstrap_samples),
        subgroup_fields=subgroup_fields,
        resume_run_dir=Path(args.resume_run_dir) if args.resume_run_dir else None,
        concurrency=max(1, args.concurrency),
        checkpoint_every=max(1, args.checkpoint_every),
    )


def parse_list_argument(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def flatten_repeated_list_argument(values: Sequence[str]) -> List[str]:
    flattened: List[str] = []
    for value in values:
        flattened.extend(parse_list_argument(value))
    return flattened


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OpenRouterError(
            f"Failed to parse JSON file {path}. The file may be corrupted or truncated."
        ) from exc


def serialize_config(config: PipelineConfig) -> Dict[str, Any]:
    return {
        "csv_path": str(config.csv_path),
        "recordings_dir": str(config.recordings_dir),
        "split": config.split,
        "output_dir": str(config.output_dir),
        "file_names": config.file_names,
        "limit": config.limit,
        "language": config.language,
        "stt_model": config.stt_model,
        "extraction_model": config.extraction_model,
        "api_base": config.api_base,
        "api_key_env": config.api_key_env,
        "sleep_seconds": config.sleep_seconds,
        "timeout_seconds": config.timeout_seconds,
        "transcript_source": config.transcript_source,
        "methods": config.methods,
        "few_shot_k": config.few_shot_k,
        "few_shot_pool_split": config.few_shot_pool_split,
        "review_threshold": config.review_threshold,
        "high_risk_review_threshold": config.high_risk_review_threshold,
        "high_risk_labels": config.high_risk_labels,
        "bootstrap_samples": config.bootstrap_samples,
        "subgroup_fields": config.subgroup_fields,
        "resume_run_dir": str(config.resume_run_dir) if config.resume_run_dir else None,
        "concurrency": config.concurrency,
        "checkpoint_every": config.checkpoint_every,
    }


def create_run_state(
    config: PipelineConfig,
    run_dir: Path,
    status: str,
    missing_metadata_files: Sequence[str],
    error_count: int,
    completed_prediction_rows: int,
    completed_stt_records: int,
    stop_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "run_dir": str(run_dir),
        "updated_at_utc": iso_utc_now(),
        "config": serialize_config(config),
        "missing_metadata_files": list(missing_metadata_files),
        "error_count": error_count,
        "completed_prediction_rows": completed_prediction_rows,
        "completed_stt_records": completed_stt_records,
        "stop_reason": stop_reason,
    }


def resolve_run_dir(config: PipelineConfig) -> Tuple[Path, bool]:
    if config.resume_run_dir:
        run_dir = config.resume_run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory not found: {run_dir}")
        artifact_files = (
            RUN_STATE_FILE,
            RESULTS_JSON_FILE,
            STT_RECORDS_JSON_FILE,
            ERRORS_JSON_FILE,
        )
        is_resuming = any((run_dir / file_name).exists() for file_name in artifact_files)
        return run_dir, is_resuming

    run_dir = config.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, False


def ensure_resume_compatible(existing_state: Dict[str, Any], config: PipelineConfig) -> None:
    if not existing_state:
        return

    previous = existing_state.get("config", {})
    comparable_fields = (
        "csv_path",
        "recordings_dir",
        "split",
        "file_names",
        "language",
        "transcript_source",
        "methods",
        "few_shot_k",
        "few_shot_pool_split",
        "stt_model",
        "extraction_model",
        "review_threshold",
        "high_risk_review_threshold",
        "high_risk_labels",
    )
    mismatches = []
    current = serialize_config(config)
    for field in comparable_fields:
        if previous.get(field) != current.get(field):
            mismatches.append(
                f"{field}: previous={previous.get(field)!r}, current={current.get(field)!r}"
            )

    if mismatches:
        raise OpenRouterError(
            "Resume run directory is not compatible with the current arguments: "
            + "; ".join(mismatches)
        )


def is_budget_related_error(exc: Exception) -> bool:
    if is_content_moderation_error(exc):
        return False
    message = str(exc).lower()
    budget_markers = (
        "insufficient",
        "quota",
        "credit",
        "credits",
        "balance",
        "billing",
        "payment",
        "out of funds",
        "rate limit exceeded due to insufficient",
    )
    return any(marker in message for marker in budget_markers)


def is_content_moderation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    moderation_markers = (
        "flagged",
        "moderation",
        "self-harm",
        "sexual",
        "violence",
        "hate",
        "harassment",
        "content policy",
    )
    return (
        "403" in message
        and any(marker in message for marker in moderation_markers)
    )


def should_return_partial_outputs(exc: Exception) -> bool:
    return is_budget_related_error(exc) or isinstance(exc, RecoverableOpenRouterError)


def error_type_for_exception(exc: Exception) -> str:
    if is_content_moderation_error(exc):
        return "content_moderation"
    if is_budget_related_error(exc):
        return "budget"
    if isinstance(exc, RecoverableOpenRouterError):
        return "recoverable"
    return "fatal"


def build_error_entry(
    file_name: str,
    stage: str,
    method: Optional[str],
    exc: Exception,
    retryable: bool,
) -> Dict[str, Any]:
    return {
        "file_name": file_name,
        "stage": stage,
        "method": method,
        "message": str(exc),
        "timestamp_utc": iso_utc_now(),
        "retryable": retryable,
        "error_type": error_type_for_exception(exc),
    }


def blocked_prediction_keys_from_errors(
    errors: Sequence[Dict[str, Any]],
) -> set[Tuple[str, str]]:
    return {
        (str(error.get("file_name")), str(error.get("method")))
        for error in errors
        if error.get("stage") == "classify"
        and error.get("error_type") == "content_moderation"
        and error.get("file_name")
        and error.get("method")
    }


def blocked_stt_files_from_errors(
    errors: Sequence[Dict[str, Any]],
) -> set[str]:
    return {
        str(error.get("file_name"))
        for error in errors
        if error.get("stage") == "transcribe"
        and error.get("error_type") == "content_moderation"
        and error.get("file_name")
    }


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def resolve_api_key(primary_env_name: str) -> str:
    load_dotenv(Path(".env"))

    candidates = [primary_env_name] + [
        name for name in API_KEY_CANDIDATES if name != primary_env_name
    ]
    for env_name in candidates:
        api_key = os.getenv(env_name)
        if api_key:
            return api_key
    raise OpenRouterError(
        "OpenRouter API key not found. Set one of these environment variables: "
        + ", ".join(candidates)
    )


def normalize_text(value: str) -> str:
    normalized = value.lower()
    normalized = PUNCTUATION_RE.sub(" ", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def exact_normalized_match(left: str, right: str) -> bool:
    return normalize_text(left) == normalize_text(right)


def text_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(
        a=normalize_text(left),
        b=normalize_text(right),
    ).ratio()


def word_overlap(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def guess_audio_format(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower().lstrip(".")
    return "wav" if suffix == "wave" else suffix


def encode_audio(audio_path: Path) -> Dict[str, str]:
    data = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
    return {"data": data, "format": guess_audio_format(audio_path)}


def load_metadata(csv_path: Path) -> Dict[str, Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["file_name"]: row for row in rows}


def allowed_prompt_labels(metadata_by_file: Dict[str, Dict[str, str]]) -> List[str]:
    labels = {
        row["prompt"].strip()
        for row in metadata_by_file.values()
        if row.get("prompt", "").strip()
    }
    return sorted(labels)


def select_audio_files(config: PipelineConfig) -> List[Path]:
    split_dir = config.recordings_dir / config.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Recording split directory not found: {split_dir}")

    if config.file_names:
        candidates = [split_dir / file_name for file_name in config.file_names]
    else:
        candidates = sorted(split_dir.glob("*.wav"))

    missing = [path for path in candidates if not path.exists()]
    if missing:
        missing_text = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Recording files not found: {missing_text}")

    if config.limit is not None:
        return candidates[: config.limit]
    return candidates


def load_split_reference_rows(
    metadata_by_file: Dict[str, Dict[str, str]],
    recordings_dir: Path,
    split: str,
) -> List[Dict[str, str]]:
    split_dir = recordings_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Recording split directory not found: {split_dir}")

    rows: List[Dict[str, str]] = []
    for audio_path in sorted(split_dir.glob("*.wav")):
        metadata_row = metadata_by_file.get(audio_path.name)
        if metadata_row is None:
            continue
        phrase = metadata_row.get("phrase", "").strip()
        prompt = metadata_row.get("prompt", "").strip()
        if not phrase or not prompt:
            continue
        rows.append(
            {
                "file_name": audio_path.name,
                "phrase": phrase,
                "prompt": prompt,
            }
        )
    return rows


def select_few_shot_examples(
    transcript: str,
    few_shot_pool: Sequence[Dict[str, str]],
    k: int,
    exclude_file_name: Optional[str] = None,
) -> List[Dict[str, str]]:
    if k <= 0:
        return []

    scored_candidates: List[Tuple[float, str, str, Dict[str, str]]] = []
    for row in few_shot_pool:
        if exclude_file_name and row.get("file_name") == exclude_file_name:
            continue
        candidate_phrase = row.get("phrase", "")
        score = (2.0 * word_overlap(transcript, candidate_phrase)) + text_similarity(
            transcript, candidate_phrase
        )
        scored_candidates.append(
            (
                score,
                row.get("prompt", ""),
                row.get("file_name", ""),
                row,
            )
        )

    scored_candidates.sort(
        key=lambda item: (-item[0], item[1], item[2])
    )

    selected: List[Dict[str, str]] = []
    seen_labels = set()
    seen_files = set()

    for _, _, _, row in scored_candidates:
        label = row.get("prompt", "")
        file_name = row.get("file_name", "")
        if label in seen_labels or file_name in seen_files:
            continue
        selected.append(row)
        seen_labels.add(label)
        seen_files.add(file_name)
        if len(selected) >= k:
            return selected

    for _, _, _, row in scored_candidates:
        file_name = row.get("file_name", "")
        if file_name in seen_files:
            continue
        selected.append(row)
        seen_files.add(file_name)
        if len(selected) >= k:
            break

    return selected


def canonicalize_label(value: str, label_options: Sequence[str]) -> Optional[str]:
    normalized_lookup = {normalize_text(label): label for label in label_options}
    return normalized_lookup.get(normalize_text(value))


class OpenRouterClient:
    def __init__(self, api_key: str, api_base: str, timeout_seconds: int) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
        self.api_base = api_base

    def _run_with_retries(
        self,
        operation_name: str,
        callback: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        for attempt in range(1, DEFAULT_MAX_RECOVERABLE_RETRIES + 1):
            try:
                return callback()
            except RecoverableOpenRouterError:
                if attempt >= DEFAULT_MAX_RECOVERABLE_RETRIES:
                    raise
                time.sleep(DEFAULT_RETRY_BACKOFF_SECONDS * attempt)
        raise AssertionError(f"Unreachable retry state for {operation_name}")

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.api_base}{endpoint}",
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RecoverableOpenRouterError(
                f"OpenRouter transport error at {endpoint}: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc

        if response.ok:
            try:
                return response.json()
            except ValueError as exc:
                raise RecoverableOpenRouterError(
                    f"OpenRouter returned a non-JSON success response at {endpoint}: "
                    f"{response.text[:500]}"
                ) from exc

        try:
            body = response.json()
        except ValueError:
            body = response.text
        error_cls = RecoverableOpenRouterError if response.status_code in {429, 500, 502, 503, 504} else OpenRouterError
        raise error_cls(
            f"OpenRouter request failed ({response.status_code}) at {endpoint}: {body}"
        )

    def transcribe(
        self,
        audio_path: Path,
        model: str,
        language: Optional[str],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "input_audio": encode_audio(audio_path),
        }
        if language:
            payload["language"] = language
        return self._run_with_retries(
            "transcription",
            lambda: self._post("/audio/transcriptions", payload),
        )

    def classify_transcript(
        self,
        transcript: str,
        model: str,
        label_options: List[str],
        method: str,
        few_shot_examples: Sequence[Dict[str, str]],
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": build_classification_messages(
                transcript=transcript,
                label_options=label_options,
                method=method,
                few_shot_examples=few_shot_examples,
            ),
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "clinical_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "clinical_label": {
                                "type": "string",
                                "enum": label_options,
                                "description": "One label chosen from the allowed dataset labels.",
                            },
                            "confidence": {
                                "type": "number",
                                "description": (
                                    "Self-estimated confidence for the chosen label, between 0 and 1. "
                                    "Values outside that range are clamped client-side."
                                ),
                            },
                            "top_3_labels": {
                                "type": "array",
                                "items": {"type": "string", "enum": label_options},
                                "description": (
                                    "Three labels ordered from most to least likely. "
                                    "Length is enforced client-side."
                                ),
                            },
                            "chief_complaint": {
                                "type": "string",
                                "description": "One sentence summary of the medical issue.",
                            },
                            "symptoms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Symptoms mentioned or implied by the transcript.",
                            },
                            "body_parts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Body parts involved, if any.",
                            },
                            "acuity": {
                                "type": "string",
                                "enum": ["unknown", "low", "medium", "high"],
                                "description": "Rough urgency estimate from transcript alone.",
                            },
                        },
                        "required": [
                            "clinical_label",
                            "confidence",
                            "top_3_labels",
                            "chief_complaint",
                            "symptoms",
                            "body_parts",
                            "acuity",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        }
        def classify_once() -> Dict[str, Any]:
            response = self._post("/chat/completions", payload)
            content = extract_message_content(response)
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise RecoverableOpenRouterError(
                    f"Classification did not return valid JSON: {content}"
                ) from exc
            try:
                normalized = normalize_classification_payload(parsed, label_options)
            except OpenRouterError as exc:
                raise RecoverableOpenRouterError(str(exc)) from exc
            normalized["_raw_response"] = response
            return normalized

        return self._run_with_retries("classification", classify_once)


def build_classification_messages(
    transcript: str,
    label_options: Sequence[str],
    method: str,
    few_shot_examples: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    label_list = "\n".join(f"- {label}" for label in label_options)
    system_prompt = (
        "You classify short patient complaint transcripts into one allowed label. "
        "Return only the requested JSON schema. "
        "Use the label names exactly as written. "
        "The first item in top_3_labels must be the same as clinical_label. "
        "Confidence should be a calibrated estimate between 0 and 1."
    )
    user_lines = [
        "Choose exactly one `clinical_label` from this allowed list:",
        label_list,
        "",
    ]
    if method == "few_shot" and few_shot_examples:
        user_lines.append("Here are labeled examples from the training split:")
        for index, example in enumerate(few_shot_examples, start=1):
            user_lines.extend(
                [
                    f"Example {index}",
                    f"Transcript: {example['phrase']}",
                    f"Label: {example['prompt']}",
                    "",
                ]
            )

    user_lines.extend(
        [
            "Now classify this transcript and extract brief clinical details.",
            f"Transcript: {transcript}",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def normalize_classification_payload(
    payload: Dict[str, Any],
    label_options: Sequence[str],
) -> Dict[str, Any]:
    predicted_label = canonicalize_label(
        str(payload.get("clinical_label", "")),
        label_options,
    )
    if not predicted_label:
        raise OpenRouterError(f"Invalid predicted label returned: {payload}")

    top_3_labels: List[str] = []
    for raw_label in payload.get("top_3_labels", []):
        canonical = canonicalize_label(str(raw_label), label_options)
        if canonical and canonical not in top_3_labels:
            top_3_labels.append(canonical)

    if predicted_label in top_3_labels:
        top_3_labels.remove(predicted_label)
    top_3_labels.insert(0, predicted_label)

    for label in label_options:
        if len(top_3_labels) >= 3:
            break
        if label not in top_3_labels:
            top_3_labels.append(label)

    confidence = safe_float(payload.get("confidence"))
    if confidence is None:
        confidence = 0.0
    confidence = clamp_float(confidence, 0.0, 1.0)

    symptoms = payload.get("symptoms", [])
    if not isinstance(symptoms, list):
        symptoms = []

    body_parts = payload.get("body_parts", [])
    if not isinstance(body_parts, list):
        body_parts = []

    acuity = str(payload.get("acuity", "unknown"))
    if acuity not in {"unknown", "low", "medium", "high"}:
        acuity = "unknown"

    return {
        "clinical_label": predicted_label,
        "confidence": confidence,
        "top_3_labels": top_3_labels[:3],
        "chief_complaint": str(payload.get("chief_complaint", "")).strip(),
        "symptoms": [str(item).strip() for item in symptoms if str(item).strip()],
        "body_parts": [str(item).strip() for item in body_parts if str(item).strip()],
        "acuity": acuity,
    }


def extract_message_content(response_payload: Dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        raise RecoverableOpenRouterError(
            f"No choices returned from chat model: {response_payload}"
        )

    first_choice = choices[0]
    choice_error = first_choice.get("error")
    if isinstance(choice_error, dict) and choice_error:
        code = choice_error.get("code")
        message = choice_error.get("message", "Unknown provider error")
        metadata = choice_error.get("metadata")
        error_text = f"OpenRouter choice-level error ({code}): {message}"
        if metadata:
            error_text += f" | metadata={metadata}"
        numeric_code = safe_float(code)
        if numeric_code in {429.0, 500.0, 502.0, 503.0, 504.0}:
            raise RecoverableOpenRouterError(error_text)
        raise OpenRouterError(error_text)

    message = first_choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        joined = "".join(text_parts).strip()
        if joined:
            return joined

    raise RecoverableOpenRouterError(
        f"Unexpected chat response content shape: {response_payload}"
    )


def should_route_to_human_review(
    predicted_label: str,
    confidence: Optional[float],
    review_threshold: float,
    high_risk_labels: Sequence[str],
    high_risk_review_threshold: float,
) -> Tuple[bool, str]:
    reasons: List[str] = []
    if confidence is None:
        reasons.append("missing_confidence")
    else:
        if confidence < review_threshold:
            reasons.append("low_confidence")
        if (
            predicted_label in high_risk_labels
            and confidence < high_risk_review_threshold
        ):
            reasons.append("high_risk_label_low_confidence")

    return (bool(reasons), "|".join(reasons) if reasons else "auto_accept")


def build_result_row(
    audio_path: Path,
    metadata_row: Dict[str, str],
    transcript_source: str,
    input_transcript: str,
    transcript_payload: Optional[Dict[str, Any]],
    classification_payload: Dict[str, Any],
    config: PipelineConfig,
    method: str,
    few_shot_examples: Sequence[Dict[str, str]],
    latency_seconds: float,
) -> Dict[str, Any]:
    actual_phrase = metadata_row.get("phrase", "")
    predicted_label = classification_payload.get("clinical_label", "").strip()
    actual_label = metadata_row.get("prompt", "")
    stt_usage = transcript_payload.get("usage", {}) if transcript_payload else {}
    chat_usage = classification_payload.get("_raw_response", {}).get("usage", {})
    confidence = safe_float(classification_payload.get("confidence"))
    review_recommended, review_reason = should_route_to_human_review(
        predicted_label=predicted_label,
        confidence=confidence,
        review_threshold=config.review_threshold,
        high_risk_labels=config.high_risk_labels,
        high_risk_review_threshold=config.high_risk_review_threshold,
    )
    top_3_labels = classification_payload.get("top_3_labels", [])

    if transcript_source == "asr":
        predicted_transcript = input_transcript
        transcript_exact_match = exact_normalized_match(predicted_transcript, actual_phrase)
        transcript_similarity = round(
            text_similarity(predicted_transcript, actual_phrase), 4
        )
        transcript_word_overlap = round(
            word_overlap(predicted_transcript, actual_phrase), 4
        )
    else:
        predicted_transcript = ""
        transcript_exact_match = None
        transcript_similarity = None
        transcript_word_overlap = None

    return {
        "file_name": audio_path.name,
        "split": config.split,
        "audio_path": str(audio_path),
        "transcript_source": transcript_source,
        "method": method,
        "speaker_id": metadata_row.get("speaker_id", ""),
        "writer_id": metadata_row.get("writer_id", ""),
        "audio_clipping": metadata_row.get("audio_clipping", ""),
        "background_noise_audible": metadata_row.get("background_noise_audible", ""),
        "quiet_speaker": metadata_row.get("quiet_speaker", ""),
        "overall_quality_of_the_audio": metadata_row.get(
            "overall_quality_of_the_audio", ""
        ),
        "ground_truth_phrase": actual_phrase,
        "input_transcript": input_transcript,
        "predicted_transcript": predicted_transcript,
        "transcript_exact_match": transcript_exact_match,
        "transcript_similarity": transcript_similarity,
        "transcript_word_overlap": transcript_word_overlap,
        "ground_truth_prompt": actual_label,
        "predicted_clinical_label": predicted_label,
        "clinical_label_exact_match": exact_normalized_match(
            predicted_label, actual_label
        ),
        "clinical_label_similarity": round(
            text_similarity(predicted_label, actual_label), 4
        ),
        "clinical_label_word_overlap": round(
            word_overlap(predicted_label, actual_label), 4
        ),
        "confidence": round(confidence, 4) if confidence is not None else None,
        "top_3_labels": " | ".join(top_3_labels),
        "actual_label_in_top_3": actual_label in top_3_labels,
        "human_review_recommended": review_recommended,
        "human_review_reason": review_reason,
        "chief_complaint": classification_payload.get("chief_complaint", ""),
        "symptoms": " | ".join(classification_payload.get("symptoms", [])),
        "body_parts": " | ".join(classification_payload.get("body_parts", [])),
        "acuity": classification_payload.get("acuity", "unknown"),
        "few_shot_example_count": len(few_shot_examples),
        "few_shot_example_file_names": " | ".join(
            example["file_name"] for example in few_shot_examples
        ),
        "few_shot_example_labels": " | ".join(
            example["prompt"] for example in few_shot_examples
        ),
        "stt_model": config.stt_model if transcript_source == "asr" else "",
        "extraction_model": config.extraction_model,
        "chat_latency_seconds": round(latency_seconds, 4),
        "stt_cost": safe_float(stt_usage.get("cost")),
        "chat_cost": safe_float(chat_usage.get("cost")),
        "stt_seconds_billed": safe_float(stt_usage.get("seconds")),
        "stt_input_tokens": stt_usage.get("input_tokens"),
        "stt_output_tokens": stt_usage.get("output_tokens"),
        "chat_input_tokens": chat_usage.get("input_tokens"),
        "chat_output_tokens": chat_usage.get("output_tokens"),
    }


def build_stt_record(
    audio_path: Path,
    metadata_row: Dict[str, str],
    transcript_payload: Dict[str, Any],
) -> Dict[str, Any]:
    predicted_transcript = transcript_payload.get("text", "").strip()
    actual_phrase = metadata_row.get("phrase", "")
    stt_usage = transcript_payload.get("usage", {})
    return {
        "file_name": audio_path.name,
        "ground_truth_phrase": actual_phrase,
        "predicted_transcript": predicted_transcript,
        "transcript_exact_match": exact_normalized_match(
            predicted_transcript, actual_phrase
        ),
        "transcript_similarity": round(
            text_similarity(predicted_transcript, actual_phrase), 4
        ),
        "transcript_word_overlap": round(
            word_overlap(predicted_transcript, actual_phrase), 4
        ),
        "stt_cost": safe_float(stt_usage.get("cost")),
        "stt_seconds_billed": safe_float(stt_usage.get("seconds")),
        "stt_input_tokens": stt_usage.get("input_tokens"),
        "stt_output_tokens": stt_usage.get("output_tokens"),
    }


def transcript_payload_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": record.get("predicted_transcript", ""),
        "usage": {
            "cost": record.get("stt_cost"),
            "seconds": record.get("stt_seconds_billed"),
            "input_tokens": record.get("stt_input_tokens"),
            "output_tokens": record.get("stt_output_tokens"),
        },
    }


def accuracy(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get("clinical_label_exact_match")) for row in rows) / len(rows)


def top_3_accuracy(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get("actual_label_in_top_3")) for row in rows) / len(rows)


def average_confidence(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    values = [
        safe_float(row.get("confidence"))
        for row in rows
        if safe_float(row.get("confidence")) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def build_confusion_matrix(
    rows: Sequence[Dict[str, Any]],
    label_options: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    matrix = {
        label: {predicted_label: 0 for predicted_label in label_options}
        for label in label_options
    }
    for row in rows:
        actual = row.get("ground_truth_prompt", "")
        predicted = row.get("predicted_clinical_label", "")
        if actual not in matrix:
            continue
        if predicted not in matrix[actual]:
            continue
        matrix[actual][predicted] += 1
    return matrix


def compute_per_class_metrics(
    rows: Sequence[Dict[str, Any]],
    label_options: Sequence[str],
) -> List[Dict[str, Any]]:
    matrix = build_confusion_matrix(rows, label_options)
    per_class: List[Dict[str, Any]] = []

    for label in label_options:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        fp = sum(matrix[other_label][label] for other_label in label_options if other_label != label)
        fn = sum(matrix[label][other_label] for other_label in label_options if other_label != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class.append(
            {
                "label": label,
                "support": support,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    return per_class


def macro_metrics(
    rows: Sequence[Dict[str, Any]],
    label_options: Sequence[str],
) -> Dict[str, float]:
    per_class = compute_per_class_metrics(rows, label_options)
    if not per_class:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    macro_precision = sum(item["precision"] for item in per_class) / len(per_class)
    macro_recall = sum(item["recall"] for item in per_class) / len(per_class)
    macro_f1 = sum(item["f1"] for item in per_class) / len(per_class)
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def bootstrap_confidence_interval(
    rows: Sequence[Dict[str, Any]],
    metric_fn: Callable[[Sequence[Dict[str, Any]]], float],
    samples: int,
    seed: int = 13,
) -> Optional[Dict[str, float]]:
    if not rows or samples <= 0:
        return None

    rng = random.Random(seed)
    sample_values: List[float] = []
    for _ in range(samples):
        bootstrap_rows = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        sample_values.append(metric_fn(bootstrap_rows))

    sample_values.sort()
    lower_index = int(0.025 * (len(sample_values) - 1))
    upper_index = int(0.975 * (len(sample_values) - 1))
    return {
        "lower": round(sample_values[lower_index], 4),
        "upper": round(sample_values[upper_index], 4),
    }


def compute_top_confusions(
    rows: Sequence[Dict[str, Any]],
    limit: int = 15,
) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in rows:
        actual = row.get("ground_truth_prompt", "")
        predicted = row.get("predicted_clinical_label", "")
        if actual and predicted and actual != predicted:
            counts[(actual, predicted)] += 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [
        {
            "ground_truth_label": actual,
            "predicted_label": predicted,
            "count": count,
        }
        for (actual, predicted), count in ranked[:limit]
    ]


def compute_subgroup_metrics(
    rows: Sequence[Dict[str, Any]],
    subgroup_fields: Sequence[str],
    label_options: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for field in subgroup_fields:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = str(row.get(field, "") or "unknown")
            groups[value].append(row)

        field_metrics: Dict[str, Dict[str, Any]] = {}
        for value, group_rows in sorted(groups.items()):
            metrics = macro_metrics(group_rows, label_options)
            auto_accept_rows = [
                row for row in group_rows if not row.get("human_review_recommended")
            ]
            field_metrics[value] = {
                "count": len(group_rows),
                "accuracy": round(accuracy(group_rows), 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "top_3_accuracy": round(top_3_accuracy(group_rows), 4),
                "human_review_rate": round(
                    sum(bool(row.get("human_review_recommended")) for row in group_rows)
                    / len(group_rows),
                    4,
                ),
                "auto_accept_accuracy": round(accuracy(auto_accept_rows), 4)
                if auto_accept_rows
                else None,
            }
        results[field] = field_metrics
    return results


def summarize_transcriptions(
    stt_records: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not stt_records:
        return None

    transcript_scores = [record["transcript_similarity"] for record in stt_records]
    transcript_exact = [record["transcript_exact_match"] for record in stt_records]
    word_overlap_scores = [record["transcript_word_overlap"] for record in stt_records]
    costs = [
        value for record in stt_records for value in [record.get("stt_cost")] if isinstance(value, float)
    ]
    seconds_billed = [
        value
        for record in stt_records
        for value in [record.get("stt_seconds_billed")]
        if isinstance(value, float)
    ]
    return {
        "evaluated_examples": len(stt_records),
        "average_transcript_similarity": round(
            sum(transcript_scores) / len(transcript_scores), 4
        ),
        "average_transcript_word_overlap": round(
            sum(word_overlap_scores) / len(word_overlap_scores), 4
        ),
        "transcript_exact_match_rate": round(
            sum(transcript_exact) / len(transcript_exact), 4
        ),
        "estimated_stt_cost": round(sum(costs), 6) if costs else None,
        "stt_seconds_billed": round(sum(seconds_billed), 4) if seconds_billed else None,
    }


def summarize_method_rows(
    rows: Sequence[Dict[str, Any]],
    label_options: Sequence[str],
    bootstrap_samples: int,
    shared_stt_cost: Optional[float] = None,
) -> Dict[str, Any]:
    metrics = macro_metrics(rows, label_options)
    mean_confidence = average_confidence(rows)
    human_review_count = sum(bool(row.get("human_review_recommended")) for row in rows)
    auto_accept_rows = [row for row in rows if not row.get("human_review_recommended")]
    high_risk_rows = [
        row
        for row in rows
        if row.get("human_review_reason") == "high_risk_label_low_confidence"
        or "high_risk_label_low_confidence" in str(row.get("human_review_reason", ""))
    ]
    chat_costs = [
        value for row in rows for value in [row.get("chat_cost")] if isinstance(value, float)
    ]

    accuracy_ci = bootstrap_confidence_interval(rows, accuracy, bootstrap_samples)
    macro_f1_ci = bootstrap_confidence_interval(
        rows,
        lambda sample_rows: macro_metrics(sample_rows, label_options)["macro_f1"],
        bootstrap_samples,
    )

    worst_labels = sorted(
        compute_per_class_metrics(rows, label_options),
        key=lambda item: (item["f1"], item["support"], item["label"]),
    )[:5]

    return {
        "prediction_rows": len(rows),
        "accuracy": round(accuracy(rows), 4),
        "accuracy_ci_95": accuracy_ci,
        "macro_precision": round(metrics["macro_precision"], 4),
        "macro_recall": round(metrics["macro_recall"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "macro_f1_ci_95": macro_f1_ci,
        "top_3_accuracy": round(top_3_accuracy(rows), 4),
        "average_confidence": round(mean_confidence, 4)
        if mean_confidence is not None
        else None,
        "human_review_rate": round(human_review_count / len(rows), 4) if rows else 0.0,
        "auto_accept_rate": round(len(auto_accept_rows) / len(rows), 4) if rows else 0.0,
        "auto_accept_accuracy": round(accuracy(auto_accept_rows), 4)
        if auto_accept_rows
        else None,
        "high_risk_review_rate": round(len(high_risk_rows) / len(rows), 4)
        if rows
        else 0.0,
        "estimated_chat_cost": round(sum(chat_costs), 6) if chat_costs else None,
        "estimated_total_cost_if_run_alone": round(
            (sum(chat_costs) if chat_costs else 0.0) + (shared_stt_cost or 0.0),
            6,
        ),
        "worst_labels_by_f1": worst_labels,
    }


def summarize_results(
    rows: List[Dict[str, Any]],
    missing_metadata: List[str],
    config: PipelineConfig,
    label_options: List[str],
    stt_records: Optional[Sequence[Dict[str, Any]]] = None,
    errors: Optional[Sequence[Dict[str, Any]]] = None,
    run_status: str = "completed",
    stop_reason: Optional[str] = None,
) -> Dict[str, Any]:
    method_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method_groups[row.get("method", "unknown")].append(row)

    stt_summary = summarize_transcriptions(stt_records or [])
    shared_stt_cost = (
        stt_summary.get("estimated_stt_cost")
        if stt_summary and stt_summary.get("estimated_stt_cost") is not None
        else None
    )

    methods_summary = {
        method: summarize_method_rows(
            method_rows,
            label_options=label_options,
            bootstrap_samples=config.bootstrap_samples,
            shared_stt_cost=shared_stt_cost,
        )
        for method, method_rows in sorted(method_groups.items())
    }

    total_chat_cost = sum(
        value
        for row in rows
        for value in [row.get("chat_cost")]
        if isinstance(value, float)
    )
    total_cost = total_chat_cost + (shared_stt_cost or 0.0)

    unique_files = {row.get("file_name") for row in rows if row.get("file_name")}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_status": run_status,
        "stop_reason": stop_reason,
        "split": config.split,
        "transcript_source": config.transcript_source,
        "methods_evaluated": config.methods,
        "evaluated_examples": len(unique_files) if unique_files else len(rows),
        "prediction_rows": len(rows),
        "missing_metadata_files": missing_metadata,
        "error_count": len(errors or []),
        "clinical_label_mode": "closed_set_csv_prompt",
        "allowed_clinical_label_count": len(label_options),
        "few_shot_pool_split": config.few_shot_pool_split,
        "few_shot_k": config.few_shot_k,
        "bootstrap_samples": config.bootstrap_samples,
        "review_policy": {
            "confidence_threshold": config.review_threshold,
            "high_risk_confidence_threshold": config.high_risk_review_threshold,
            "high_risk_labels": config.high_risk_labels,
        },
        "stt_summary": stt_summary,
        "method_metrics": methods_summary,
        "estimated_total_chat_cost": round(total_chat_cost, 6) if rows else None,
        "estimated_total_cost_for_this_run": round(total_cost, 6) if rows else None,
        "stt_model": config.stt_model if config.transcript_source == "asr" else None,
        "extraction_model": config.extraction_model,
    }


def write_json(path: Path, payload: Any) -> None:
    serialized = json.dumps(payload, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_run_artifacts(
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows = read_json_file(run_dir / RESULTS_JSON_FILE, [])
    stt_records = read_json_file(run_dir / STT_RECORDS_JSON_FILE, [])
    errors = read_json_file(run_dir / ERRORS_JSON_FILE, [])
    run_state = read_json_file(run_dir / RUN_STATE_FILE, {})
    return rows, stt_records, errors, run_state


def persist_checkpoint(
    run_dir: Path,
    config: PipelineConfig,
    rows: Sequence[Dict[str, Any]],
    stt_records: Sequence[Dict[str, Any]],
    errors: Sequence[Dict[str, Any]],
    missing_metadata_files: Sequence[str],
    status: str,
    stop_reason: Optional[str] = None,
) -> None:
    write_json(run_dir / RESULTS_JSON_FILE, rows)
    write_json(run_dir / STT_RECORDS_JSON_FILE, stt_records)
    write_json(run_dir / ERRORS_JSON_FILE, errors)
    write_json(
        run_dir / RUN_STATE_FILE,
        create_run_state(
            config=config,
            run_dir=run_dir,
            status=status,
            missing_metadata_files=missing_metadata_files,
            error_count=len(errors),
            completed_prediction_rows=len(rows),
            completed_stt_records=len(stt_records),
            stop_reason=stop_reason,
        ),
    )


def write_confusion_matrix_csv(
    path: Path,
    confusion_matrix: Dict[str, Dict[str, int]],
    label_options: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ground_truth_label"] + list(label_options))
        for actual_label in label_options:
            writer.writerow(
                [actual_label]
                + [confusion_matrix[actual_label][predicted] for predicted in label_options]
            )


def render_confusion_matrix_html(
    path: Path,
    method: str,
    confusion_matrix: Dict[str, Dict[str, int]],
    label_options: Sequence[str],
) -> None:
    max_count = max(
        confusion_matrix[actual][predicted]
        for actual in label_options
        for predicted in label_options
    ) or 1

    header_cells = "".join(
        f"<th>{html.escape(label)}</th>" for label in label_options
    )
    body_rows = []
    for actual in label_options:
        cells = []
        for predicted in label_options:
            count = confusion_matrix[actual][predicted]
            intensity = count / max_count
            background = f"rgba(31, 119, 180, {0.08 + 0.72 * intensity:.3f})"
            cells.append(
                "<td style='background:%s'>%s</td>"
                % (background, html.escape(str(count)))
            )
        body_rows.append(
            "<tr><th>%s</th>%s</tr>"
            % (html.escape(actual), "".join(cells))
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Confusion Matrix - {html.escape(method)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #111827;
      background: #f8fafc;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    table {{
      border-collapse: collapse;
      font-size: 12px;
      background: white;
    }}
    th, td {{
      border: 1px solid #cbd5e1;
      padding: 6px 8px;
      text-align: center;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #e2e8f0;
    }}
    .wrapper {{
      overflow: auto;
      max-width: 100%;
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }}
  </style>
</head>
<body>
  <h1>Confusion Matrix: {html.escape(method)}</h1>
  <div class="wrapper">
    <table>
      <thead>
        <tr><th>Ground truth</th>{header_cells}</tr>
      </thead>
      <tbody>
        {''.join(body_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def flatten_subgroup_metrics(
    subgroup_metrics: Dict[str, Dict[str, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for field, groups in subgroup_metrics.items():
        for value, metrics in groups.items():
            row = {"field": field, "value": value}
            row.update(metrics)
            rows.append(row)
    return rows


def method_slug(method: str) -> str:
    return method.lower()


def write_outputs(
    run_dir: Path,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    label_options: Sequence[str],
    subgroup_fields: Sequence[str],
    stt_records: Sequence[Dict[str, Any]],
    errors: Sequence[Dict[str, Any]],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)

    results_json_path = run_dir / RESULTS_JSON_FILE
    results_csv_path = run_dir / RESULTS_CSV_FILE
    summary_path = run_dir / SUMMARY_JSON_FILE
    write_json(results_json_path, rows)
    write_rows_csv(results_csv_path, rows)
    write_json(run_dir / STT_RECORDS_JSON_FILE, stt_records)
    write_json(run_dir / ERRORS_JSON_FILE, errors)

    artifact_manifest: Dict[str, Dict[str, str]] = {}

    for method in summary.get("methods_evaluated", []):
        method_rows = [row for row in rows if row.get("method") == method]
        slug = method_slug(method)
        confusion = build_confusion_matrix(method_rows, label_options)
        confusion_csv = run_dir / f"confusion_matrix_{slug}.csv"
        confusion_html = run_dir / f"confusion_matrix_{slug}.html"
        per_class_json = run_dir / f"per_class_metrics_{slug}.json"
        per_class_csv = run_dir / f"per_class_metrics_{slug}.csv"
        subgroup_json = run_dir / f"subgroup_metrics_{slug}.json"
        subgroup_csv = run_dir / f"subgroup_metrics_{slug}.csv"
        top_confusions_json = run_dir / f"top_confusions_{slug}.json"

        per_class_metrics = compute_per_class_metrics(method_rows, label_options)
        subgroup_metrics = compute_subgroup_metrics(
            method_rows,
            subgroup_fields=subgroup_fields,
            label_options=label_options,
        )
        top_confusions = compute_top_confusions(method_rows)

        write_confusion_matrix_csv(confusion_csv, confusion, label_options)
        render_confusion_matrix_html(confusion_html, method, confusion, label_options)
        write_json(per_class_json, per_class_metrics)
        write_rows_csv(per_class_csv, per_class_metrics)
        write_json(subgroup_json, subgroup_metrics)
        write_rows_csv(subgroup_csv, flatten_subgroup_metrics(subgroup_metrics))
        write_json(top_confusions_json, top_confusions)

        artifact_manifest[method] = {
            "confusion_matrix_csv": confusion_csv.name,
            "confusion_matrix_html": confusion_html.name,
            "per_class_metrics_json": per_class_json.name,
            "per_class_metrics_csv": per_class_csv.name,
            "subgroup_metrics_json": subgroup_json.name,
            "subgroup_metrics_csv": subgroup_csv.name,
            "top_confusions_json": top_confusions_json.name,
        }

    summary["artifact_files"] = artifact_manifest
    write_json(summary_path, summary)
    return run_dir


@dataclass
class FileTaskResult:
    file_name: str
    new_stt_record: Optional[Dict[str, Any]] = None
    blocked_stt: bool = False
    rows: List[Dict[str, Any]] = field(default_factory=list)
    blocked_predictions: List[Tuple[str, str]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    fatal_exception: Optional[OpenRouterError] = None
    fatal_stage: Optional[str] = None


def _process_one_file_safely(
    *,
    client: OpenRouterClient,
    audio_path: Path,
    metadata_row: Dict[str, str],
    config: PipelineConfig,
    label_options: List[str],
    few_shot_pool: Sequence[Dict[str, str]],
    pending_methods: Sequence[str],
    existing_stt_record: Optional[Dict[str, Any]],
) -> FileTaskResult:
    file_name = audio_path.name
    result = FileTaskResult(file_name=file_name)

    transcript_payload: Optional[Dict[str, Any]] = None
    if config.transcript_source == "asr":
        if existing_stt_record is not None:
            transcript_payload = transcript_payload_from_record(existing_stt_record)
            input_transcript = transcript_payload.get("text", "").strip()
        else:
            try:
                transcript_payload = client.transcribe(
                    audio_path=audio_path,
                    model=config.stt_model,
                    language=config.language,
                )
            except OpenRouterError as exc:
                if is_content_moderation_error(exc):
                    result.errors.append(
                        build_error_entry(
                            file_name=file_name,
                            stage="transcribe",
                            method=None,
                            exc=exc,
                            retryable=False,
                        )
                    )
                    result.blocked_stt = True
                    return result
                result.errors.append(
                    build_error_entry(
                        file_name=file_name,
                        stage="transcribe",
                        method=None,
                        exc=exc,
                        retryable=isinstance(exc, RecoverableOpenRouterError),
                    )
                )
                result.fatal_exception = exc
                result.fatal_stage = "transcribe"
                return result
            input_transcript = transcript_payload.get("text", "").strip()
            result.new_stt_record = build_stt_record(
                audio_path=audio_path,
                metadata_row=metadata_row,
                transcript_payload=transcript_payload,
            )
    else:
        input_transcript = metadata_row.get("phrase", "").strip()

    for method_index, method in enumerate(pending_methods):
        few_shot_examples = (
            select_few_shot_examples(
                transcript=input_transcript,
                few_shot_pool=few_shot_pool,
                k=config.few_shot_k,
                exclude_file_name=audio_path.name,
            )
            if method == "few_shot"
            else []
        )
        started_at = time.perf_counter()
        try:
            classification_payload = client.classify_transcript(
                transcript=input_transcript,
                model=config.extraction_model,
                label_options=label_options,
                method=method,
                few_shot_examples=few_shot_examples,
            )
        except OpenRouterError as exc:
            if is_content_moderation_error(exc):
                for blocked_method in pending_methods[method_index:]:
                    result.errors.append(
                        build_error_entry(
                            file_name=file_name,
                            stage="classify",
                            method=blocked_method,
                            exc=exc,
                            retryable=False,
                        )
                    )
                    result.blocked_predictions.append((file_name, blocked_method))
                return result
            result.errors.append(
                build_error_entry(
                    file_name=file_name,
                    stage="classify",
                    method=method,
                    exc=exc,
                    retryable=isinstance(exc, RecoverableOpenRouterError),
                )
            )
            result.fatal_exception = exc
            result.fatal_stage = "classify"
            return result

        latency_seconds = time.perf_counter() - started_at
        row = build_result_row(
            audio_path=audio_path,
            metadata_row=metadata_row,
            transcript_source=config.transcript_source,
            input_transcript=input_transcript,
            transcript_payload=transcript_payload,
            classification_payload=classification_payload,
            config=config,
            method=method,
            few_shot_examples=few_shot_examples,
            latency_seconds=latency_seconds,
        )
        result.rows.append(row)

    if (
        config.sleep_seconds > 0
        and result.fatal_exception is None
        and (result.rows or result.new_stt_record)
    ):
        time.sleep(config.sleep_seconds)

    return result


def run_pipeline(config: PipelineConfig) -> Path:
    metadata_by_file = load_metadata(config.csv_path)
    label_options = allowed_prompt_labels(metadata_by_file)
    audio_files = select_audio_files(config)
    few_shot_pool = load_split_reference_rows(
        metadata_by_file,
        recordings_dir=config.recordings_dir,
        split=config.few_shot_pool_split,
    )
    run_dir, is_resuming = resolve_run_dir(config)
    rows, stt_records, errors, existing_state = load_existing_run_artifacts(run_dir)
    ensure_resume_compatible(existing_state, config)

    missing_metadata_files = list(existing_state.get("missing_metadata_files", []))
    missing_metadata_set = set(missing_metadata_files)
    processed_prediction_keys = {
        (row.get("file_name"), row.get("method"))
        for row in rows
        if row.get("file_name") and row.get("method")
    }
    blocked_prediction_keys = blocked_prediction_keys_from_errors(errors)
    blocked_stt_files = blocked_stt_files_from_errors(errors)
    stt_record_by_file = {
        record.get("file_name"): record
        for record in stt_records
        if record.get("file_name")
    }

    persist_checkpoint(
        run_dir=run_dir,
        config=config,
        rows=rows,
        stt_records=stt_records,
        errors=errors,
        missing_metadata_files=missing_metadata_files,
        status="running",
    )

    api_key = resolve_api_key(config.api_key_env)
    client = OpenRouterClient(
        api_key=api_key,
        api_base=config.api_base,
        timeout_seconds=config.timeout_seconds,
    )

    work_items: List[
        Tuple[Path, Dict[str, str], List[str], Optional[Dict[str, Any]]]
    ] = []
    metadata_changed = False
    for audio_path in audio_files:
        file_name = audio_path.name
        metadata_row = metadata_by_file.get(file_name)
        if metadata_row is None:
            if file_name not in missing_metadata_set:
                missing_metadata_set.add(file_name)
                missing_metadata_files.append(file_name)
                metadata_changed = True
            continue
        pending_methods = [
            method
            for method in config.methods
            if (file_name, method) not in processed_prediction_keys
            and (file_name, method) not in blocked_prediction_keys
        ]
        needs_stt = (
            config.transcript_source == "asr"
            and file_name not in stt_record_by_file
            and file_name not in blocked_stt_files
        )
        if not pending_methods and not needs_stt:
            continue
        if config.transcript_source == "asr" and file_name in blocked_stt_files:
            continue
        existing_stt = stt_record_by_file.get(file_name)
        work_items.append((audio_path, metadata_row, pending_methods, existing_stt))

    if metadata_changed:
        persist_checkpoint(
            run_dir=run_dir,
            config=config,
            rows=rows,
            stt_records=stt_records,
            errors=errors,
            missing_metadata_files=missing_metadata_files,
            status="running",
        )

    state_lock = threading.Lock()
    fatal_payload: Optional[Tuple[OpenRouterError, str]] = None
    completed = 0
    total_work = len(work_items)
    checkpoint_every = max(1, config.checkpoint_every)
    max_workers = max(1, config.concurrency)
    resume_marker = " (resume)" if is_resuming else ""

    def merge_result(result: FileTaskResult) -> Optional[Tuple[OpenRouterError, str]]:
        nonlocal completed
        with state_lock:
            if result.new_stt_record:
                stt_records.append(result.new_stt_record)
                stt_record_by_file[result.file_name] = result.new_stt_record
            if result.blocked_stt:
                blocked_stt_files.add(result.file_name)
            for row in result.rows:
                rows.append(row)
                processed_prediction_keys.add((row["file_name"], row["method"]))
            for key in result.blocked_predictions:
                blocked_prediction_keys.add(key)
            if result.errors:
                errors.extend(result.errors)
            completed += 1
            new_fatal: Optional[Tuple[OpenRouterError, str]] = None
            if result.fatal_exception is not None:
                new_fatal = (result.fatal_exception, result.fatal_stage or "classify")
            print(
                f"[{completed}/{total_work}] Processed {result.file_name}{resume_marker}",
                flush=True,
            )
            if completed % checkpoint_every == 0 or new_fatal is not None:
                persist_checkpoint(
                    run_dir=run_dir,
                    config=config,
                    rows=rows,
                    stt_records=stt_records,
                    errors=errors,
                    missing_metadata_files=missing_metadata_files,
                    status="running",
                )
            return new_fatal

    if work_items:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: Dict[Future, Path] = {
                executor.submit(
                    _process_one_file_safely,
                    client=client,
                    audio_path=audio_path,
                    metadata_row=metadata_row,
                    config=config,
                    label_options=label_options,
                    few_shot_pool=few_shot_pool,
                    pending_methods=pending_methods,
                    existing_stt_record=existing_stt,
                ): audio_path
                for audio_path, metadata_row, pending_methods, existing_stt in work_items
            }
            for future in as_completed(futures):
                audio_path = futures[future]
                try:
                    result = future.result()
                except CancelledError:
                    continue
                except Exception as exc:
                    result = FileTaskResult(file_name=audio_path.name)
                    result.errors.append(
                        {
                            "file_name": audio_path.name,
                            "stage": "task",
                            "method": None,
                            "message": f"{exc.__class__.__name__}: {exc}",
                            "timestamp_utc": iso_utc_now(),
                            "retryable": False,
                            "error_type": "fatal",
                        }
                    )
                    result.fatal_exception = (
                        exc if isinstance(exc, OpenRouterError) else OpenRouterError(str(exc))
                    )
                    result.fatal_stage = "task"
                new_fatal = merge_result(result)
                if new_fatal is not None and fatal_payload is None:
                    fatal_payload = new_fatal
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()

    if fatal_payload is not None:
        exc, _stage = fatal_payload
        stop_status = (
            "partial_budget_stop" if is_budget_related_error(exc) else "failed"
        )
        with state_lock:
            persist_checkpoint(
                run_dir=run_dir,
                config=config,
                rows=rows,
                stt_records=stt_records,
                errors=errors,
                missing_metadata_files=missing_metadata_files,
                status=stop_status,
                stop_reason=str(exc),
            )
            partial_summary = summarize_results(
                rows=rows,
                missing_metadata=missing_metadata_files,
                config=config,
                label_options=label_options,
                stt_records=stt_records,
                errors=errors,
                run_status=stop_status,
                stop_reason=str(exc),
            )
            write_outputs(
                run_dir=run_dir,
                rows=rows,
                summary=partial_summary,
                label_options=label_options,
                subgroup_fields=config.subgroup_fields,
                stt_records=stt_records,
                errors=errors,
            )
        if should_return_partial_outputs(exc):
            print(json.dumps(partial_summary, indent=2), flush=True)
            if is_budget_related_error(exc):
                message = (
                    "Stopped early due to a budget-related API error. "
                    f"Partial outputs saved to {run_dir}"
                )
            else:
                message = (
                    "Stopped early due to a recoverable API or network error. "
                    f"Partial outputs saved to {run_dir}"
                )
            print(message, flush=True)
            return run_dir
        raise exc

    summary = summarize_results(
        rows=rows,
        missing_metadata=missing_metadata_files,
        config=config,
        label_options=label_options,
        stt_records=stt_records,
        errors=errors,
        run_status="completed",
    )
    persist_checkpoint(
        run_dir=run_dir,
        config=config,
        rows=rows,
        stt_records=stt_records,
        errors=errors,
        missing_metadata_files=missing_metadata_files,
        status="completed",
    )
    run_dir = write_outputs(
        run_dir=run_dir,
        rows=rows,
        summary=summary,
        label_options=label_options,
        subgroup_fields=config.subgroup_fields,
        stt_records=stt_records,
        errors=errors,
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved reports to {run_dir}", flush=True)
    return run_dir


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        config = parse_args(argv)
        run_pipeline(config)
    except Exception as exc:  # pragma: no cover - handled in CLI only
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
