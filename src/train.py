from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    from . import evaluate as pipeline
except ImportError:  # pragma: no cover - fallback for direct script usage
    import evaluate as pipeline  # type: ignore


STAGE_PRESETS = {
    "development": {
        "split": "validate",
        "transcript_source": "gold",
        "description": "Practice run on the validation split using gold transcripts.",
    },
    "final_gold": {
        "split": "test",
        "transcript_source": "gold",
        "description": "Main held-out evaluation on the test split using gold transcripts.",
    },
    "final_asr": {
        "split": "test",
        "transcript_source": "asr",
        "description": "End-to-end audio evaluation on the test split using speech-to-text first.",
    },
}

PLAN_STAGES = {
    "development": ["development"],
    "final_gold": ["final_gold"],
    "final_asr": ["final_asr"],
    "recommended": ["development", "final_gold"],
    "full": ["development", "final_gold", "final_asr"],
}

DEFAULT_STAGE_SAMPLE_SIZE = 10
DEFAULT_STAGE_SAMPLE_SEED = 13


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run beginner-friendly evaluation workflows for the clinical NLP pipeline."
        )
    )
    parser.add_argument(
        "--plan",
        choices=sorted(PLAN_STAGES),
        default="recommended",
        help=(
            "`recommended` runs development on validate, then final_gold on test. "
            "`full` also adds the end-to-end ASR run."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/workflows",
        help="Parent directory where workflow runs are saved.",
    )
    parser.add_argument(
        "--resume-workflow-dir",
        default=None,
        help=(
            "Resume an existing workflow directory such as "
            "`outputs/workflows/20260506T120000Z`."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Always start a new workflow instead of auto-resuming the latest incomplete one.",
    )
    parser.add_argument(
        "--methods",
        default="zero_shot,few_shot",
        help="Comma-separated methods to evaluate for every stage.",
    )
    parser.add_argument(
        "--few-shot-k",
        type=int,
        default=4,
        help="Number of few-shot examples to retrieve. Default: 4",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=300,
        help="Bootstrap samples for 95%% confidence intervals. Default: 300",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between examples for every stage. Default: 0",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code forwarded to STT runs. Default: en",
    )
    parser.add_argument(
        "--development-limit",
        type=int,
        default=DEFAULT_STAGE_SAMPLE_SIZE,
        help="Optional limit for the development stage.",
    )
    parser.add_argument(
        "--final-gold-limit",
        type=int,
        default=DEFAULT_STAGE_SAMPLE_SIZE,
        help="Optional limit for the final_gold stage.",
    )
    parser.add_argument(
        "--final-asr-limit",
        type=int,
        default=DEFAULT_STAGE_SAMPLE_SIZE,
        help="Optional limit for the final_asr stage.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.60,
        help="Human-review confidence threshold. Default: 0.60",
    )
    parser.add_argument(
        "--high-risk-review-threshold",
        type=float,
        default=0.80,
        help="Stricter human-review threshold for high-risk labels. Default: 0.80",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help=(
            "Number of files to process in parallel for every stage. "
            "Default: 8. Lower this if the API rate-limits you."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Persist run state every N completed files. Default: 5",
    )
    parser.add_argument(
        "--stratified",
        dest="stratified",
        action="store_true",
        default=True,
        help=(
            "Sample evenly across complaint labels for stage subsamples. "
            "Default: enabled. Improves per-class metrics on small samples."
        ),
    )
    parser.add_argument(
        "--no-stratified",
        dest="stratified",
        action="store_false",
        help="Disable stratified sampling and use uniform random sampling.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_stage_names(plan: str) -> List[str]:
    return list(PLAN_STAGES[plan])


def workflow_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def limit_for_stage(args: argparse.Namespace, stage_name: str) -> Optional[int]:
    mapping = {
        "development": args.development_limit,
        "final_gold": args.final_gold_limit,
        "final_asr": args.final_asr_limit,
    }
    return mapping[stage_name]


def build_base_config(args: argparse.Namespace) -> pipeline.PipelineConfig:
    base = pipeline.parse_args([])
    return replace(
        base,
        methods=pipeline.parse_list_argument(args.methods),
        few_shot_k=max(0, args.few_shot_k),
        bootstrap_samples=max(0, args.bootstrap_samples),
        sleep_seconds=args.sleep_seconds,
        language=args.language,
        review_threshold=pipeline.clamp_float(args.review_threshold, 0.0, 1.0),
        high_risk_review_threshold=pipeline.clamp_float(
            args.high_risk_review_threshold, 0.0, 1.0
        ),
        resume_run_dir=None,
        concurrency=max(1, args.concurrency),
        checkpoint_every=max(1, args.checkpoint_every),
    )


def build_stage_config(
    base_config: pipeline.PipelineConfig,
    stage_name: str,
    workflow_root: Path,
    limit: Optional[int],
    stratified: bool = False,
) -> pipeline.PipelineConfig:
    preset = STAGE_PRESETS[stage_name]
    sampled_file_names = sample_stage_file_names(
        recordings_dir=base_config.recordings_dir,
        split=preset["split"],
        stage_name=stage_name,
        limit=limit,
        csv_path=base_config.csv_path if stratified else None,
        stratified=stratified,
    )
    return replace(
        base_config,
        split=preset["split"],
        transcript_source=preset["transcript_source"],
        output_dir=workflow_root / stage_name,
        file_names=sampled_file_names,
        limit=None,
        resume_run_dir=None,
    )


def sample_stage_file_names(
    recordings_dir: Path,
    split: str,
    stage_name: str,
    limit: Optional[int],
    csv_path: Optional[Path] = None,
    stratified: bool = False,
) -> List[str]:
    if limit is None or limit <= 0:
        return []

    split_dir = recordings_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Recording split directory not found: {split_dir}")

    candidates = sorted(path.name for path in split_dir.glob("*.wav"))
    if limit >= len(candidates):
        return candidates

    if stratified and csv_path is not None:
        stratified_sample = _stratified_sample_file_names(
            csv_path=csv_path,
            candidates=candidates,
            stage_name=stage_name,
            split=split,
            limit=limit,
        )
        if stratified_sample is not None:
            return stratified_sample

    rng = random.Random(f"{DEFAULT_STAGE_SAMPLE_SEED}:{stage_name}:{split}:{limit}")
    return sorted(rng.sample(candidates, limit))


def _stratified_sample_file_names(
    csv_path: Path,
    candidates: Sequence[str],
    stage_name: str,
    split: str,
    limit: int,
) -> Optional[List[str]]:
    metadata = pipeline.load_metadata(csv_path)
    by_label: Dict[str, List[str]] = defaultdict(list)
    for file_name in candidates:
        meta = metadata.get(file_name)
        if not meta:
            continue
        label = (meta.get("prompt") or "").strip()
        if not label:
            continue
        by_label[label].append(file_name)

    classes = sorted(by_label)
    if len(classes) < 2 or limit < len(classes):
        return None

    rng = random.Random(
        f"{DEFAULT_STAGE_SAMPLE_SEED}:{stage_name}:{split}:{limit}:strat"
    )
    base_per_class = limit // len(classes)
    residual = limit - base_per_class * len(classes)

    selected: List[str] = []
    leftover: List[str] = []
    sorted_classes = sorted(classes, key=lambda label: (-len(by_label[label]), label))
    for index, label in enumerate(sorted_classes):
        pool = sorted(by_label[label])
        target = base_per_class + (1 if index < residual else 0)
        if target >= len(pool):
            selected.extend(pool)
            continue
        picked = rng.sample(pool, target)
        selected.extend(picked)
        leftover.extend(name for name in pool if name not in set(picked))

    if len(selected) < limit and leftover:
        deficit = limit - len(selected)
        rng.shuffle(leftover)
        selected.extend(leftover[:deficit])

    if len(selected) > limit:
        rng.shuffle(selected)
        selected = selected[:limit]

    return sorted(set(selected))


def load_stage_state(run_dir: Path) -> Dict[str, object]:
    return pipeline.read_json_file(run_dir / pipeline.RUN_STATE_FILE, {})


def write_workflow_summary(path: Path, payload: Dict[str, object]) -> None:
    pipeline.write_json(path, payload)


def workflow_summary_path(workflow_root: Path) -> Path:
    return workflow_root / "workflow_summary.json"


def load_workflow_summary(workflow_root: Path) -> Dict[str, object]:
    return pipeline.read_json_file(workflow_summary_path(workflow_root), {})


def stage_run_dir(
    workflow_root: Path,
    stage_name: str,
    previous_result: Optional[Dict[str, object]] = None,
) -> Path:
    if previous_result and previous_result.get("run_dir"):
        return Path(str(previous_result["run_dir"]))
    return workflow_root / stage_name


def build_stage_result(
    stage_name: str,
    run_dir: Path,
    status: str,
    stop_reason: Optional[object] = None,
) -> Dict[str, object]:
    summary_file = run_dir / pipeline.SUMMARY_JSON_FILE
    return {
        "stage": stage_name,
        "run_dir": str(run_dir),
        "status": status,
        "stop_reason": None if stop_reason is None else str(stop_reason),
        "summary_file": str(summary_file),
        "summary": pipeline.read_json_file(summary_file, {}),
    }


def build_workflow_summary(
    workflow_root: Path,
    plan: str,
    resumed: bool,
    stage_results: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "workflow_root": str(workflow_root),
        "resumed": resumed,
        "stages": list(stage_results),
    }


def is_incomplete_workflow_summary(summary: Dict[str, object]) -> bool:
    stages = summary.get("stages", [])
    if not stages:
        return False
    final_status = stages[-1].get("status")
    return final_status in {"partial_budget_stop", "failed", "running", "unknown"}


def validate_existing_workflow(
    workflow_root: Path,
    summary: Dict[str, object],
    plan: str,
    base_config: pipeline.PipelineConfig,
    args: argparse.Namespace,
) -> None:
    existing_plan = summary.get("plan")
    if existing_plan and existing_plan != plan:
        raise ValueError(
            f"Workflow plan mismatch: previous={existing_plan!r}, current={plan!r}"
        )

    for stage_result in summary.get("stages", []):
        stage_name = stage_result.get("stage")
        run_dir = stage_result.get("run_dir")
        if not stage_name or not run_dir or stage_name not in STAGE_PRESETS:
            continue
        stage_config = build_stage_config(
            base_config=base_config,
            stage_name=stage_name,
            workflow_root=workflow_root,
            limit=limit_for_stage(args, stage_name),
            stratified=getattr(args, "stratified", False),
        )
        stage_state = load_stage_state(Path(run_dir))
        pipeline.ensure_resume_compatible(stage_state, stage_config)


def find_latest_resumable_workflow(
    output_dir: Path,
    plan: str,
    base_config: pipeline.PipelineConfig,
    args: argparse.Namespace,
) -> Optional[Path]:
    if not output_dir.exists():
        return None

    candidates = sorted(
        output_dir.glob("*/workflow_summary.json"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for summary_file in candidates:
        workflow_root = summary_file.parent
        summary = load_workflow_summary(workflow_root)
        if summary.get("plan") != plan:
            continue
        if not is_incomplete_workflow_summary(summary):
            continue
        try:
            validate_existing_workflow(
                workflow_root=workflow_root,
                summary=summary,
                plan=plan,
                base_config=base_config,
                args=args,
            )
        except Exception:
            continue
        return workflow_root
    return None


def resolve_workflow_root(
    args: argparse.Namespace,
    base_config: pipeline.PipelineConfig,
) -> Path:
    if args.resume_workflow_dir:
        workflow_root = Path(args.resume_workflow_dir)
        if not workflow_root.exists():
            raise FileNotFoundError(f"Resume workflow directory not found: {workflow_root}")
        summary = load_workflow_summary(workflow_root)
        validate_existing_workflow(
            workflow_root=workflow_root,
            summary=summary,
            plan=args.plan,
            base_config=base_config,
            args=args,
        )
        print(f"Resuming workflow from {workflow_root}", flush=True)
        return workflow_root

    output_dir = Path(args.output_dir)
    if not args.fresh:
        latest = find_latest_resumable_workflow(
            output_dir=output_dir,
            plan=args.plan,
            base_config=base_config,
            args=args,
        )
        if latest is not None:
            print(f"Resuming latest incomplete workflow from {latest}", flush=True)
            return latest

    workflow_root = output_dir / workflow_timestamp()
    workflow_root.mkdir(parents=True, exist_ok=True)
    return workflow_root


def run_stage(
    stage_name: str,
    stage_config: pipeline.PipelineConfig,
) -> Dict[str, object]:
    print(f"\n=== {stage_name} ===", flush=True)
    print(STAGE_PRESETS[stage_name]["description"], flush=True)
    run_dir = pipeline.run_pipeline(stage_config)
    state = load_stage_state(run_dir)
    return build_stage_result(
        stage_name=stage_name,
        run_dir=run_dir,
        status=str(state.get("status", "unknown")),
        stop_reason=state.get("stop_reason"),
    )


def run_workflow(args: argparse.Namespace) -> Path:
    stage_names = resolve_stage_names(args.plan)
    base_config = build_base_config(args)
    workflow_root = resolve_workflow_root(args, base_config)
    workflow_root.mkdir(parents=True, exist_ok=True)
    existing_summary = load_workflow_summary(workflow_root)
    resumed = bool(existing_summary)
    previous_stage_results = {
        stage_result.get("stage"): stage_result
        for stage_result in existing_summary.get("stages", [])
        if stage_result.get("stage")
    }
    stage_results: List[Dict[str, object]] = []

    for stage_name in stage_names:
        previous_result = previous_stage_results.get(stage_name)
        stage_config = build_stage_config(
            base_config=base_config,
            stage_name=stage_name,
            workflow_root=workflow_root,
            limit=limit_for_stage(args, stage_name),
            stratified=getattr(args, "stratified", False),
        )
        if previous_result and previous_result.get("status") == "completed":
            stage_results.append(previous_result)
            continue
        run_dir = stage_run_dir(workflow_root, stage_name, previous_result)
        run_dir.mkdir(parents=True, exist_ok=True)
        stage_config = replace(
            stage_config,
            resume_run_dir=run_dir,
        )
        stage_results.append(
            build_stage_result(
                stage_name=stage_name,
                run_dir=run_dir,
                status="running",
                stop_reason=previous_result.get("stop_reason") if previous_result else None,
            )
        )
        write_workflow_summary(
            workflow_summary_path(workflow_root),
            build_workflow_summary(
                workflow_root=workflow_root,
                plan=args.plan,
                resumed=resumed,
                stage_results=stage_results,
            ),
        )
        result = run_stage(stage_name, stage_config)
        stage_results[-1] = result
        write_workflow_summary(
            workflow_summary_path(workflow_root),
            build_workflow_summary(
                workflow_root=workflow_root,
                plan=args.plan,
                resumed=resumed,
                stage_results=stage_results,
            ),
        )
        status = result.get("status")
        if status in {"partial_budget_stop", "failed"}:
            break

    workflow_summary = build_workflow_summary(
        workflow_root=workflow_root,
        plan=args.plan,
        resumed=resumed,
        stage_results=stage_results,
    )
    write_workflow_summary(workflow_summary_path(workflow_root), workflow_summary)
    print(f"\nWorkflow summary saved to {workflow_summary_path(workflow_root)}", flush=True)
    return workflow_root


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    workflow_root = run_workflow(args)
    summary = load_workflow_summary(workflow_root)
    stages = summary.get("stages", [])
    if stages:
        final_status = stages[-1].get("status")
        if final_status == "failed":
            return 1
        if final_status == "partial_budget_stop":
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
