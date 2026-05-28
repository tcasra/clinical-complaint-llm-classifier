"""Generate publication-quality figures from one or more workflow runs.

By default the script scans `outputs/workflows/` and picks one coherent
workflow folder with the strongest set of completed stages
(`development`, `final_gold`, `final_asr`). It writes a consolidated set of
figures under `outputs/figures/`.

Usage:
    python3 plot_workflow_results.py
    python3 plot_workflow_results.py --workflow 20260506T062602Z
    python3 plot_workflow_results.py --output-dir outputs/figures-run2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mphy-workflow-mpl-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKFLOWS_DIR = Path("outputs/workflows")
DEFAULT_FIGURE_DIR = Path("outputs/figures")
STAGE_ORDER: Sequence[Tuple[str, str]] = (
    ("development", "Development"),
    ("final_gold", "Final Gold"),
    ("final_asr", "Final ASR"),
)
METHOD_ORDER: Sequence[Tuple[str, str]] = (
    ("zero_shot", "Zero-shot"),
    ("few_shot", "Few-shot"),
)
METHOD_COLORS = {
    "zero_shot": "#1f77b4",
    "few_shot": "#d95f02",
}
CORRECT_COLOR = "#2ca02c"
INCORRECT_COLOR = "#d62728"
MIN_SUBGROUP_COUNT = 10
PER_CLASS_PLOT_LABEL_LIMIT = 12
CONFUSION_FOCUS_LABEL_LIMIT = 10


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def method_display_name(method_key: str) -> str:
    lookup = dict(METHOD_ORDER)
    if method_key in lookup:
        return lookup[method_key]
    return method_key.replace("_", " ").title()


def method_color(method_key: str) -> str:
    palette = list(METHOD_COLORS.values())
    if method_key in METHOD_COLORS:
        return METHOD_COLORS[method_key]
    return palette[hash(method_key) % len(palette)]


def ordered_methods_from_keys(method_keys: Sequence[str]) -> List[Tuple[str, str]]:
    present = set(method_keys)
    ordered = [(key, label) for key, label in METHOD_ORDER if key in present]
    for extra_key in sorted(present - {key for key, _ in METHOD_ORDER}):
        ordered.append((extra_key, method_display_name(extra_key)))
    return ordered


def available_methods(stages: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
    method_keys = set()
    for payload in stages.values():
        summary = payload.get("summary", {})
        method_keys.update((summary.get("method_metrics") or {}).keys())
    return ordered_methods_from_keys(sorted(method_keys))


def stage_methods(stage_payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    summary = stage_payload.get("summary", {})
    method_metrics = summary.get("method_metrics") or {}
    return ordered_methods_from_keys(sorted(method_metrics.keys()))


def preferred_method_key(stage_payload: Dict[str, Any]) -> Optional[str]:
    method_keys = [key for key, _ in stage_methods(stage_payload)]
    if not method_keys:
        return None
    if "few_shot" in method_keys:
        return "few_shot"
    return method_keys[0]


def stage_file_names(
    stage_payload: Dict[str, Any],
    method_key: Optional[str] = None,
) -> List[str]:
    rows = stage_payload.get("results", [])
    if method_key is not None:
        rows = [row for row in rows if row.get("method") == method_key]
    return sorted(
        {
            str(row.get("file_name"))
            for row in rows
            if row.get("file_name")
        }
    )


def nice_percent_ceiling(value: float, minimum: float = 10.0) -> float:
    if value <= 0:
        return minimum
    return max(minimum, 5.0 * math.ceil(value / 5.0))


def nice_percent_floor(value: float, maximum: float = 0.0) -> float:
    return min(maximum, 5.0 * math.floor(value / 5.0))


def quantile_calibration_points(
    confidences: np.ndarray,
    correct: np.ndarray,
    max_bins: int = 6,
    min_bin_size: int = 50,
) -> List[Dict[str, float]]:
    """Build equal-frequency calibration bins so sparse confidence regions do not disappear."""
    if confidences.size == 0:
        return []
    target_bins = max(1, min(max_bins, confidences.size // max(1, min_bin_size)))
    split_points = np.linspace(0, confidences.size, target_bins + 1, dtype=int)
    order = np.argsort(confidences)
    sorted_confidences = confidences[order]
    sorted_correct = correct[order]
    points: List[Dict[str, float]] = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        if end <= start:
            continue
        group_confidences = sorted_confidences[start:end]
        group_correct = sorted_correct[start:end]
        points.append(
            {
                "confidence_min": float(group_confidences[0]),
                "confidence_max": float(group_confidences[-1]),
                "mean_confidence": float(group_confidences.mean()),
                "accuracy": float(group_correct.mean()),
                "count": float(group_confidences.size),
            }
        )
    return points


def load_confusion_matrix(path: Path) -> Tuple[List[str], np.ndarray]:
    if not path.exists():
        return [], np.zeros((0, 0), dtype=int)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], np.zeros((0, 0), dtype=int)
    header = rows[0][1:]
    matrix = np.array([[int(value) for value in row[1:]] for row in rows[1:]], dtype=int)
    return header, matrix


def aggregate_stages(
    workflows_root: Path,
    explicit_workflow: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Pick completed stages from one coherent workflow directory."""

    def load_completed_stages(workflow_dir: Path) -> Dict[str, Dict[str, Any]]:
        stages: Dict[str, Dict[str, Any]] = {}
        for stage_name, _ in STAGE_ORDER:
            stage_dir = workflow_dir / stage_name
            summary_path = stage_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = read_json(summary_path, {})
            if summary.get("run_status") != "completed":
                continue
            stages[stage_name] = {
                "workflow_dir": workflow_dir,
                "stage_dir": stage_dir,
                "summary": summary,
                "errors": read_json(stage_dir / "errors.json", []),
                "results": read_json(stage_dir / "results.json", []),
            }
        return stages

    if explicit_workflow is not None:
        return load_completed_stages(explicit_workflow)

    if not workflows_root.exists():
        return {}

    best_stages: Dict[str, Dict[str, Any]] = {}
    best_score = (-1, "")
    for workflow_dir in sorted(
        (path for path in workflows_root.iterdir() if path.is_dir()),
        reverse=True,
    ):
        stages = load_completed_stages(workflow_dir)
        score = (len(stages), workflow_dir.name)
        if stages and score > best_score:
            best_stages = stages
            best_score = score
    return best_stages


def stage_keys_present(stages: Dict[str, Dict[str, Any]]) -> List[str]:
    return [stage_key for stage_key, _ in STAGE_ORDER if stage_key in stages]


def stage_label(stage_key: str, stage_payload: Dict[str, Any]) -> str:
    display = dict(STAGE_ORDER)[stage_key]
    summary = stage_payload["summary"]
    evaluated = summary.get("evaluated_examples")
    if evaluated is None:
        return display
    return f"{display}\nn={evaluated}"


def metric_value(
    stages: Dict[str, Dict[str, Any]],
    stage_key: str,
    method_key: str,
    metric_key: str,
) -> Optional[float]:
    method_metrics = stages[stage_key]["summary"]["method_metrics"]
    method_payload = method_metrics.get(method_key)
    if not method_payload:
        return None
    value = method_payload.get(metric_key)
    if value is None:
        return None
    return float(value or 0.0)


def metric_ci(
    stages: Dict[str, Dict[str, Any]],
    stage_key: str,
    method_key: str,
    ci_key: str,
) -> Optional[Tuple[float, float]]:
    method_metrics = stages[stage_key]["summary"]["method_metrics"]
    method_payload = method_metrics.get(method_key)
    if not method_payload:
        return None
    ci = method_payload.get(ci_key) or {}
    if not ci:
        return None
    lower = float(ci.get("lower", 0.0) or 0.0)
    upper = float(ci.get("upper", 0.0) or 0.0)
    return lower, upper


def plot_grouped_metric(
    ax: Any,
    stages: Dict[str, Dict[str, Any]],
    metric_key: str,
    ci_key: Optional[str],
    title: str,
    percent: bool = True,
    y_min: float = 0.0,
    y_max: Optional[float] = None,
) -> None:
    stage_keys = stage_keys_present(stages)
    methods = available_methods(stages)
    if not stage_keys or not methods:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    x_positions = list(range(len(stage_keys)))
    group_width = 0.72
    width = group_width / max(1, len(methods))
    multiplier = 100.0 if percent else 1.0
    max_value = 0.0
    for method_index, (method_key, method_label) in enumerate(methods):
        values: List[Optional[float]] = []
        for stage_key in stage_keys:
            value = metric_value(stages, stage_key, method_key, metric_key)
            values.append(None if value is None else value * multiplier)
        finite_values = [value for value in values if value is not None]
        if finite_values:
            max_value = max(max_value, max(finite_values))
        positions = [
            x + (method_index - (len(methods) - 1) / 2) * width for x in x_positions
        ]
        bars = ax.bar(
            positions,
            [0.0 if value is None else value for value in values],
            width=width,
            label=method_label,
            color=method_color(method_key),
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            if value is None:
                bar.set_alpha(0.0)
                bar.set_edgecolor("none")
                continue
            offset = 0.03 * max((y_max or max_value or 1.0) - y_min, 1.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{value:.1f}" if percent else f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#222",
                clip_on=False,
            )
        if ci_key:
            valid_positions = []
            valid_values = []
            error_bars_low = []
            error_bars_high = []
            for stage_key, position, value in zip(stage_keys, positions, values):
                if value is None:
                    continue
                ci = metric_ci(stages, stage_key, method_key, ci_key)
                if ci is None:
                    continue
                lower, upper = ci
                max_value = max(max_value, upper * multiplier)
                valid_positions.append(position)
                valid_values.append(value)
                error_bars_low.append(max(value - (lower * multiplier), 0.0))
                error_bars_high.append(max((upper * multiplier) - value, 0.0))
            if valid_positions:
                ax.errorbar(
                    valid_positions,
                    valid_values,
                    yerr=[error_bars_low, error_bars_high],
                    fmt="none",
                    ecolor="#374151",
                    capsize=4,
                    linewidth=1.2,
                    zorder=4,
                )

    ax.set_title(title, fontsize=11.5, weight="bold", pad=8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [stage_label(stage_key, stages[stage_key]) for stage_key in stage_keys]
    )
    ax.grid(axis="y", alpha=0.25, zorder=0)
    if percent:
        if y_max is None:
            y_max = nice_percent_ceiling(max_value + 6.0, minimum=100.0)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Percent")
    else:
        if y_max is not None:
            ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Score")
    ax.margins(x=0.05)


def make_headline_performance(
    output_dir: Path,
    stages: Dict[str, Dict[str, Any]],
) -> Optional[Path]:
    if not stages:
        return None
    methods = available_methods(stages)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    plot_grouped_metric(
        axes[0, 0], stages, "accuracy", "accuracy_ci_95",
        "Top-1 Accuracy (95% CI)",
    )
    plot_grouped_metric(
        axes[0, 1], stages, "macro_f1", "macro_f1_ci_95",
        "Macro F1 (95% CI)",
    )
    plot_grouped_metric(
        axes[1, 0], stages, "top_3_accuracy", None,
        "Top-3 Accuracy",
    )
    plot_grouped_metric(
        axes[1, 1], stages, "average_confidence", None,
        "Average Self-reported Confidence",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        bbox_to_anchor=(0.5, -0.01), ncol=max(1, len(methods)), frameon=False, fontsize=11,
    )
    fig.suptitle(
        "Clinical Complaint Classifier — Headline Performance",
        fontsize=16, weight="bold", y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.93))
    output_path = output_dir / "01_headline_performance.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_per_class_f1(
    output_dir: Path,
    stages: Dict[str, Dict[str, Any]],
    target_stage: str = "final_gold",
) -> Optional[Path]:
    payload = stages.get(target_stage)
    if payload is None:
        return None
    stage_dir = payload["stage_dir"]
    method_series: Dict[str, List[Dict[str, Any]]] = {}
    for method_key, _ in stage_methods(payload):
        series = read_json(stage_dir / f"per_class_metrics_{method_key}.json", [])
        if series:
            method_series[method_key] = series
    if not method_series:
        return None

    by_method_by_label = {
        method_key: {row["label"]: row for row in rows}
        for method_key, rows in method_series.items()
    }
    label_sets = [set(rows.keys()) for rows in by_method_by_label.values()]
    shared_labels = sorted(set.intersection(*label_sets)) if label_sets else []
    if not shared_labels:
        return None

    ranking_method = "few_shot" if "few_shot" in by_method_by_label else next(iter(by_method_by_label))
    observed_labels = [
        lab for lab in shared_labels
        if by_method_by_label[ranking_method][lab].get("support", 0) > 0
    ]
    if not observed_labels:
        return None
    labels = sorted(
        observed_labels,
        key=lambda lab: (
            by_method_by_label[ranking_method][lab].get("f1", 0.0),
            by_method_by_label[ranking_method][lab].get("support", 0),
            lab,
        ),
    )[:PER_CLASS_PLOT_LABEL_LIMIT]

    plotted_methods = ordered_methods_from_keys(list(method_series.keys()))
    support = [by_method_by_label[ranking_method][lab].get("support", 0) for lab in labels]

    fig_height = max(5.8, 0.55 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10.2, fig_height))
    y = np.arange(len(labels))
    height = 0.78 / max(1, len(plotted_methods))
    for method_index, (method_key, method_label) in enumerate(plotted_methods):
        offsets = y + (method_index - (len(plotted_methods) - 1) / 2) * height
        scores = [
            by_method_by_label[method_key][lab].get("f1", 0.0) * 100
            for lab in labels
        ]
        ax.barh(
            offsets,
            scores,
            height=height,
            label=method_label,
            color=method_color(method_key),
            edgecolor="white",
        )

    yticks = [f"{lab} (n={support[i]})" for i, lab in enumerate(labels)]
    ax.set_yticks(y)
    ax.set_yticklabels(yticks, fontsize=10.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("F1 (%)")
    ax.set_title(
        f"Weakest {len(labels)} Labels by F1 — {dict(STAGE_ORDER)[target_stage]} "
        f"(n={payload['summary'].get('evaluated_examples', '?')})",
        fontsize=13, weight="bold",
    )
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.invert_yaxis()
    fig.tight_layout()
    if target_stage == "final_gold":
        output_path = output_dir / "02_per_class_f1.png"
    else:
        output_path = output_dir / f"02_per_class_f1_{target_stage}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_confusion_matrix(
    output_dir: Path,
    stages: Dict[str, Dict[str, Any]],
    target_stage: str = "final_gold",
    method: str = "few_shot",
) -> Optional[Path]:
    payload = stages.get(target_stage)
    if payload is None:
        return None
    available_method_keys = [key for key, _ in stage_methods(payload)]
    if method not in available_method_keys:
        if not available_method_keys:
            return None
        method = "few_shot" if "few_shot" in available_method_keys else available_method_keys[0]

    stage_dir = payload["stage_dir"]
    labels, matrix = load_confusion_matrix(stage_dir / f"confusion_matrix_{method}.csv")
    if not labels or matrix.size == 0:
        return None

    row_totals = matrix.sum(axis=1, keepdims=True)
    row_totals_safe = np.where(row_totals == 0, 1, row_totals)
    norm = matrix.astype(float) / row_totals_safe

    results = payload.get("results", [])
    method_rows = [row for row in results if row.get("method") == method]
    confusion_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in method_rows:
        actual = row.get("ground_truth_prompt")
        predicted = row.get("predicted_clinical_label")
        if actual and predicted and actual != predicted:
            confusion_counts[(actual, predicted)] += 1

    if not confusion_counts:
        return None

    selected_labels: List[str] = []
    for (actual, predicted), _count in sorted(
        confusion_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        for label in (actual, predicted):
            if label not in selected_labels:
                selected_labels.append(label)
            if len(selected_labels) >= CONFUSION_FOCUS_LABEL_LIMIT:
                break
        if len(selected_labels) >= CONFUSION_FOCUS_LABEL_LIMIT:
            break

    if len(selected_labels) < 2:
        return None

    label_to_index = {label: index for index, label in enumerate(labels)}
    selected_labels = sorted(
        selected_labels,
        key=lambda label: (
            norm[label_to_index[label], label_to_index[label]],
            label,
        ),
    )
    indices = [label_to_index[label] for label in selected_labels]
    focus_matrix = matrix[np.ix_(indices, indices)]
    focus_norm = norm[np.ix_(indices, indices)]

    fig_size = max(7.8, 0.9 * len(selected_labels) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 0.3))
    im = ax.imshow(focus_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(selected_labels)))
    ax.set_yticks(np.arange(len(selected_labels)))
    ax.set_xticklabels(selected_labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(selected_labels, fontsize=10)
    for i in range(focus_matrix.shape[0]):
        for j in range(focus_matrix.shape[1]):
            count = int(focus_matrix[i, j])
            if count == 0:
                continue
            color = "white" if focus_norm[i, j] > 0.55 else "#1f2937"
            ax.text(
                j,
                i,
                str(count),
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Ground-truth label")
    ax.set_title(
        f"Focused Confusion Matrix — {dict(STAGE_ORDER)[target_stage]}, "
        f"{method_display_name(method)} "
        f"(n={payload['summary'].get('evaluated_examples', '?')})",
        fontsize=13, weight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Row-normalized rate", fontsize=10)
    fig.text(
        0.5,
        0.02,
        "Subset of labels involved in the most frequent confusions.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    output_path = output_dir / f"03_confusion_matrix_{target_stage}_{method}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_subgroup_performance(
    output_dir: Path,
    stages: Dict[str, Dict[str, Any]],
    target_stage: str = "final_asr",
) -> Optional[Path]:
    payload = stages.get(target_stage)
    if payload is None:
        return None
    stage_dir = payload["stage_dir"]
    subgroup_series: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method_key, _ in stage_methods(payload):
        subgroup_metrics = read_json(stage_dir / f"subgroup_metrics_{method_key}.json", {})
        if subgroup_metrics:
            subgroup_series[method_key] = subgroup_metrics
    if not subgroup_series:
        return None

    plotted_methods = ordered_methods_from_keys(list(subgroup_series.keys()))
    reference_method = preferred_method_key(payload) or plotted_methods[0][0]
    reference_metrics = subgroup_series[reference_method]
    fields = [
        field for field in sorted(reference_metrics.keys())
        if reference_metrics.get(field)
    ]
    if not fields:
        return None

    plottable_fields: List[Tuple[str, List[str]]] = []
    omitted_groups: List[str] = []
    for field in fields:
        candidate_groups = sorted(reference_metrics.get(field, {}).keys())
        eligible_groups = []
        for group in candidate_groups:
            count = reference_metrics.get(field, {}).get(group, {}).get("count", 0)
            if count >= MIN_SUBGROUP_COUNT:
                eligible_groups.append(group)
            else:
                omitted_groups.append(f"{field.replace('_', ' ')}/{group.replace('_', ' ')} (n={count})")
        if len(eligible_groups) >= 2:
            plottable_fields.append((field, eligible_groups))

    if not plottable_fields:
        return None

    fig, axes = plt.subplots(
        1,
        len(plottable_fields),
        figsize=(5.4 * len(plottable_fields), 4.8),
        sharey=True,
    )
    if len(plottable_fields) == 1:
        axes = [axes]

    for ax, (field, groups) in zip(axes, plottable_fields):
        x = np.arange(len(groups))
        width = 0.78 / max(1, len(plotted_methods))
        counts = [reference_metrics.get(field, {}).get(g, {}).get("count", 0) for g in groups]
        for method_index, (method_key, method_label) in enumerate(plotted_methods):
            offsets = x + (method_index - (len(plotted_methods) - 1) / 2) * width
            values = [
                subgroup_series[method_key].get(field, {}).get(group, {}).get("accuracy", 0.0) * 100
                for group in groups
            ]
            ax.bar(
                offsets,
                values,
                width,
                label=method_label,
                color=method_color(method_key),
                edgecolor="white",
            )
        for i, count in enumerate(counts):
            tallest = max(
                subgroup_series[method_key].get(field, {}).get(group, {}).get("accuracy", 0.0) * 100
                for method_key, _ in plotted_methods
                for group_index, group in enumerate(groups)
                if group_index == i
            )
            ax.text(i, tallest + 2, f"n={count}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace("_", " ") for g in groups],
                            rotation=15, ha="right", fontsize=10)
        ax.set_title(field.replace("_", " ").title(), fontsize=12, weight="bold")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=max(1, len(plotted_methods)),
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        f"Subgroup Performance — {dict(STAGE_ORDER)[target_stage]} "
        f"(n={payload['summary'].get('evaluated_examples', '?')})",
        fontsize=15, weight="bold", y=0.985,
    )
    if omitted_groups:
        fig.text(
            0.5,
            0.01,
            "Omitted sparse groups (n<10): " + "; ".join(omitted_groups),
            ha="center",
            fontsize=8.5,
            color="#4b5563",
        )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.88))
    output_path = output_dir / "04_subgroup_performance.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_gold_vs_asr(
    output_dir: Path,
    stages: Dict[str, Dict[str, Any]],
) -> Optional[Path]:
    if "final_gold" not in stages or "final_asr" not in stages:
        return None
    gold_methods = {key for key, _ in stage_methods(stages["final_gold"])}
    asr_methods = {key for key, _ in stage_methods(stages["final_asr"])}
    methods = ordered_methods_from_keys(sorted(gold_methods & asr_methods))
    if not methods:
        return None
    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("top_3_accuracy", "Top-3 Acc."),
    ]
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    gold_files = set(stage_file_names(stages["final_gold"]))
    asr_files = set(stage_file_names(stages["final_asr"]))
    overlap = len(gold_files & asr_files)
    same_sample = bool(gold_files) and gold_files == asr_files

    for ax, (method_key, method_label) in zip(axes, methods):
        x = np.arange(len(metrics))
        width = 0.35
        gold_vals = [(metric_value(stages, "final_gold", method_key, m) or 0.0) * 100
                     for m, _ in metrics]
        asr_vals = [(metric_value(stages, "final_asr", method_key, m) or 0.0) * 100
                    for m, _ in metrics]
        b1 = ax.bar(x - width / 2, gold_vals, width,
                     label=f"Gold (n={stages['final_gold']['summary']['evaluated_examples']})",
                     color="#5b8def", edgecolor="white")
        b2 = ax.bar(x + width / 2, asr_vals, width,
                     label=f"ASR (n={stages['final_asr']['summary']['evaluated_examples']})",
                     color="#c47fb6", edgecolor="white")
        for bars, vals in [(b1, gold_vals), (b2, asr_vals)]:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        if same_sample:
            for i, (g, a) in enumerate(zip(gold_vals, asr_vals)):
                delta = a - g
                ax.text(i, max(g, a) + 7,
                        f"Δ {delta:+.1f}", ha="center", va="bottom",
                        fontsize=10, color="#444",
                        fontweight="bold" if abs(delta) >= 1 else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics])
        ax.set_title(method_label, fontsize=13, weight="bold")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    axes[0].set_ylabel("Percent")
    title_suffix = "Paired Sample Comparison" if same_sample else "Stage-level Comparison"
    fig.suptitle(
        f"Gold Transcript vs End-to-End ASR — {title_suffix}",
        fontsize=16, weight="bold", y=0.98,
    )
    if same_sample:
        note = "Same files in both stages."
    else:
        note = (
            f"Separate stratified samples; compare levels, not paired deltas "
            f"(file overlap: {overlap}/{min(len(gold_files), len(asr_files))})."
        )
    fig.text(0.5, 0.015, note, ha="center", fontsize=9.5, color="#4b5563")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.90))
    output_path = output_dir / "05_gold_vs_asr.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_confidence_calibration(
    output_dir: Path,
    stages: Dict[str, Dict[str, Any]],
    target_stage: Optional[str] = None,
) -> Optional[Path]:
    if target_stage is None:
        stage_keys = [
            stage_key for stage_key in ("final_gold", "final_asr")
            if stage_key in stages
        ]
    else:
        stage_keys = [target_stage] if target_stage in stages else []
    if not stage_keys:
        return None

    methods_present = set()
    for stage_key in stage_keys:
        methods_present.update(key for key, _ in stage_methods(stages[stage_key]))
    plotted_methods = ordered_methods_from_keys(sorted(methods_present))
    if not plotted_methods:
        return None

    stage_display = {
        "final_gold": "Gold Text",
        "final_asr": "End-to-End ASR",
    }
    n_rows = 2 * len(stage_keys)
    n_cols = len(plotted_methods)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.0 * len(stage_keys) + 3.0),
        sharex=False,
        sharey="row",
        gridspec_kw={"height_ratios": [2.2, 1.3] * len(stage_keys)},
    )
    axes = np.atleast_2d(axes)
    if axes.shape != (n_rows, n_cols):
        axes = axes.reshape(n_rows, n_cols)
    bins = np.linspace(0, 100, 11)

    for stage_index, stage_key in enumerate(stage_keys):
        payload = stages[stage_key]
        rows = payload.get("results", [])
        row_top = 2 * stage_index
        row_bottom = row_top + 1
        for column_index, (method_key, method_label) in enumerate(plotted_methods):
            ax_top = axes[row_top, column_index]
            ax_bottom = axes[row_bottom, column_index]
            method_rows = [r for r in rows if r.get("method") == method_key]
            if not method_rows:
                ax_top.set_axis_off()
                ax_bottom.set_axis_off()
                continue
            confidences = np.array([r.get("confidence") or 0.0 for r in method_rows])
            correct = np.array([bool(r.get("clinical_label_exact_match")) for r in method_rows])
            calibration_points = quantile_calibration_points(confidences, correct)
            mean_confidences = [point["mean_confidence"] * 100 for point in calibration_points]
            observed_accuracies = [point["accuracy"] * 100 for point in calibration_points]

            ax_top.plot(
                [0, 100],
                [0, 100],
                "--",
                color="#9ca3af",
                linewidth=1.8,
                label="Perfect calibration",
            )
            ax_top.plot(
                mean_confidences,
                observed_accuracies,
                "o-",
                color="#1f2937",
                linewidth=2.4,
                markersize=7,
                label="Observed accuracy",
            )
            ax_top.set_xlim(0, 100)
            ax_top.set_ylim(0, 105)
            ax_top.set_title(
                f"{stage_display.get(stage_key, dict(STAGE_ORDER).get(stage_key, stage_key))}\n{method_label}",
                fontsize=12.5,
                weight="bold",
            )
            ax_top.grid(alpha=0.3)
            if column_index == 0:
                ax_top.set_ylabel("Observed accuracy (%)")
                if stage_index == 0:
                    ax_top.legend(loc="upper left", fontsize=8.5, frameon=False)

            correct_conf = confidences[correct] * 100
            wrong_conf = confidences[~correct] * 100
            ax_bottom.hist(
                [correct_conf, wrong_conf],
                bins=bins,
                stacked=True,
                color=[CORRECT_COLOR, INCORRECT_COLOR],
                label=[
                    f"Correct (n={correct.sum()})",
                    f"Incorrect (n={(~correct).sum()})",
                ],
                edgecolor="white",
            )
            ax_bottom.set_xlim(0, 100)
            ax_bottom.set_xlabel("Self-reported confidence (%)")
            if column_index == 0:
                ax_bottom.set_ylabel("Count")
                if stage_index == 0:
                    ax_bottom.legend(loc="upper left", fontsize=8.5, frameon=False)
            ax_bottom.grid(axis="y", alpha=0.3)

    n_values = [
        stages[stage_key]["summary"].get("evaluated_examples", "?")
        for stage_key in stage_keys
    ]
    if len(set(n_values)) == 1:
        sample_note = f"n={n_values[0]} per stage"
    else:
        sample_note = ", ".join(
            f"{stage_display.get(stage_key, stage_key)} n={stages[stage_key]['summary'].get('evaluated_examples', '?')}"
            for stage_key in stage_keys
        )
    fig.suptitle(
        f"Confidence Calibration — {sample_note}",
        fontsize=15, weight="bold", y=1.03,
    )
    fig.text(
        0.5,
        0.01,
        "Top panels use equal-frequency calibration bins to avoid empty-bin artifacts.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    output_path = output_dir / "06_confidence_calibration.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_review_economics(
    output_dir: Path,
    stages: Dict[str, Dict[str, Any]],
) -> Optional[Path]:
    if not stages:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    review_max = nice_percent_ceiling(
        max(
            metric_value(stages, stage_key, method_key, "human_review_rate") or 0.0
            for stage_key in stage_keys_present(stages)
            for method_key, _ in available_methods(stages)
        ) * 100
        + 1.5,
        minimum=10.0,
    )
    high_risk_max = nice_percent_ceiling(
        max(
            metric_value(stages, stage_key, method_key, "high_risk_review_rate") or 0.0
            for stage_key in stage_keys_present(stages)
            for method_key, _ in available_methods(stages)
        ) * 100
        + 1.5,
        minimum=10.0,
    )
    auto_accept_values = [
        (metric_value(stages, stage_key, method_key, "auto_accept_accuracy") or 0.0) * 100
        for stage_key in stage_keys_present(stages)
        for method_key, _ in available_methods(stages)
        if metric_value(stages, stage_key, method_key, "auto_accept_accuracy") is not None
    ]
    auto_accept_min = min(auto_accept_values) if auto_accept_values else 85.0
    plot_grouped_metric(
        axes[0], stages, "human_review_rate", None,
        "Human Review Rate",
        y_max=review_max,
    )
    plot_grouped_metric(
        axes[1], stages, "auto_accept_accuracy", None,
        "Auto-accept Accuracy",
        y_min=max(0.0, nice_percent_floor(auto_accept_min - 2.0, maximum=85.0)),
        y_max=100.0,
    )
    plot_grouped_metric(
        axes[2], stages, "high_risk_review_rate", None,
        "High-risk Review Rate",
        y_max=high_risk_max,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=max(1, len(available_methods(stages))),
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(
        "Governance — Human-in-the-Loop Workload",
        fontsize=16, weight="bold", y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    output_path = output_dir / "07_review_economics.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_asr_quality(
    output_dir: Path,
    stages: Dict[str, Dict[str, Any]],
) -> Optional[Path]:
    payload = stages.get("final_asr")
    if payload is None:
        return None
    stt = payload["summary"].get("stt_summary") or {}
    if not stt:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    metrics = [
        ("Exact Match Rate",
         float(stt.get("transcript_exact_match_rate", 0.0)) * 100, "#4daf4a"),
        ("Avg. Char Similarity",
         float(stt.get("average_transcript_similarity", 0.0)) * 100, "#377eb8"),
        ("Avg. Word Overlap",
         float(stt.get("average_transcript_word_overlap", 0.0)) * 100, "#984ea3"),
    ]
    bars = axes[0].bar([m[0] for m in metrics], [m[1] for m in metrics],
                       color=[m[2] for m in metrics], edgecolor="white")
    for bar, (_, value, _) in zip(bars, metrics):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                      f"{value:.1f}%", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Whisper STT Transcript Fidelity", fontsize=13, weight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    rows = payload.get("results", [])
    similarity_method = preferred_method_key(payload) or "few_shot"
    sims = [r.get("transcript_similarity") for r in rows
            if r.get("method") == similarity_method
            and isinstance(r.get("transcript_similarity"), (int, float))]
    if sims:
        axes[1].hist(sims, bins=20, color="#377eb8", edgecolor="white")
        axes[1].axvline(np.median(sims), color="black", linestyle="--",
                         label=f"median={np.median(sims):.2f}")
        axes[1].set_xlabel("Per-utterance transcript similarity")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of STT Similarity", fontsize=13, weight="bold")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].set_axis_off()

    fig.suptitle(
        f"ASR Pipeline Quality "
        f"(n={stt.get('evaluated_examples', '?')}, "
        f"{stt.get('stt_seconds_billed', 0):.0f}s billed)",
        fontsize=15, weight="bold", y=1.05,
    )
    output_path = output_dir / "08_asr_quality.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_evaluation_summary(
    output_dir: Path,
    stages: Dict[str, Dict[str, Any]],
) -> Path:
    """Write a markdown evaluation digest the user can paste into the field guide."""
    lines: List[str] = ["# Model Performance Evaluation", ""]

    if "final_gold" in stages:
        s = stages["final_gold"]["summary"]
        n = s["evaluated_examples"]
        gold_methods = s["method_metrics"]
        ordered_gold_methods = ordered_methods_from_keys(sorted(gold_methods.keys()))
        lines.append("## Final Gold (held-out test transcripts, gold text input)")
        lines.append(f"- Sample: **n = {n}** stratified across 25 complaint labels (~20 / class)")
        for method_key, method_label in ordered_gold_methods:
            method_metrics = gold_methods[method_key]
            lines.append(
                f"- **{method_label}**: accuracy {method_metrics['accuracy']:.1%} "
                f"(95% CI {method_metrics['accuracy_ci_95']['lower']:.1%}–{method_metrics['accuracy_ci_95']['upper']:.1%}), "
                f"macro-F1 {method_metrics['macro_f1']:.3f} "
                f"(95% CI {method_metrics['macro_f1_ci_95']['lower']:.3f}–{method_metrics['macro_f1_ci_95']['upper']:.3f}), "
                f"top-3 acc {method_metrics['top_3_accuracy']:.1%}"
            )
        if "zero_shot" in gold_methods and "few_shot" in gold_methods:
            zs = gold_methods["zero_shot"]
            fs = gold_methods["few_shot"]
            delta_acc = (fs["accuracy"] - zs["accuracy"]) * 100
            delta_f1 = fs["macro_f1"] - zs["macro_f1"]
            lines.append(
                f"- **Few-shot lift over zero-shot**: +{delta_acc:.1f} pp accuracy, "
                f"+{delta_f1:.3f} macro-F1"
            )
        cost_parts = " + ".join(
            f"{gold_methods[key].get('estimated_chat_cost', 0):.3f} {method_display_name(key)}"
            for key, _ in ordered_gold_methods
        )
        lines.append(
            f"- **Cost**: ~${s['estimated_total_cost_for_this_run']:.3f} ({cost_parts})"
        )
        lines.append("")

    if "final_asr" in stages:
        s = stages["final_asr"]["summary"]
        n = s["evaluated_examples"]
        asr_methods = s["method_metrics"]
        stt = s.get("stt_summary") or {}
        lines.append("## Final ASR (end-to-end audio → Whisper STT → classifier)")
        lines.append(f"- Sample: **n = {n}** stratified across the 25 labels")
        lines.append(
            f"- **STT fidelity** (Whisper large-v3-turbo): "
            f"exact-match {stt.get('transcript_exact_match_rate', 0):.1%}, "
            f"avg similarity {stt.get('average_transcript_similarity', 0):.3f}, "
            f"avg word overlap {stt.get('average_transcript_word_overlap', 0):.3f}"
        )
        for method_key, method_label in ordered_methods_from_keys(sorted(asr_methods.keys())):
            method_metrics = asr_methods[method_key]
            lines.append(
                f"- **{method_label}** (over ASR): accuracy {method_metrics['accuracy']:.1%} "
                f"(95% CI {method_metrics['accuracy_ci_95']['lower']:.1%}–{method_metrics['accuracy_ci_95']['upper']:.1%}), "
                f"macro-F1 {method_metrics['macro_f1']:.3f}"
            )
        if "final_gold" in stages:
            gold_summary = stages["final_gold"]["summary"]["method_metrics"]
            if "few_shot" in gold_summary and "few_shot" in asr_methods:
                gold_fs = gold_summary["few_shot"]
                fs = asr_methods["few_shot"]
                drop = (gold_fs["accuracy"] - fs["accuracy"]) * 100
                gold_files = set(stage_file_names(stages["final_gold"]))
                asr_files = set(stage_file_names(stages["final_asr"]))
                overlap = len(gold_files & asr_files)
                if gold_files == asr_files and gold_files:
                    lines.append(
                        f"- **Gold → ASR drop (few-shot)**: −{drop:.1f} pp accuracy "
                        f"({gold_fs['accuracy']:.1%} → {fs['accuracy']:.1%})"
                    )
                else:
                    lines.append(
                        f"- **Gold vs ASR stage-level gap (few-shot)**: "
                        f"{gold_fs['accuracy']:.1%} vs {fs['accuracy']:.1%} "
                        f"(difference {drop:.1f} pp; separate stratified samples, "
                        f"overlap {overlap}/{min(len(gold_files), len(asr_files))})"
                    )
        lines.append(
            f"- **Cost**: ~${s['estimated_total_cost_for_this_run']:.3f} total "
            f"(STT ${stt.get('estimated_stt_cost', 0):.3f} + chat ${s.get('estimated_total_chat_cost', 0):.3f})"
        )
        lines.append("")

    if "final_gold" in stages:
        preferred_method = preferred_method_key(stages["final_gold"]) or "few_shot"
        per_class = read_json(
            stages["final_gold"]["stage_dir"] / f"per_class_metrics_{preferred_method}.json", []
        )
        worst = sorted(per_class, key=lambda r: r.get("f1", 0.0))[:5]
        lines.append(f"## Weakest labels (final_gold {method_display_name(preferred_method)}, by F1)")
        for row in worst:
            lines.append(
                f"- **{row['label']}** — F1 {row['f1']:.3f} "
                f"(precision {row['precision']:.2f}, recall {row['recall']:.2f}, "
                f"support {row['support']})"
            )
        lines.append("")
        top_conf = read_json(
            stages["final_gold"]["stage_dir"] / f"top_confusions_{preferred_method}.json", []
        )
        if top_conf:
            lines.append(f"## Top confusions (final_gold {method_display_name(preferred_method)})")
            for row in top_conf[:5]:
                lines.append(
                    f"- {row['ground_truth_label']} → {row['predicted_label']} "
                    f"({row['count']} cases)"
                )
            lines.append("")

    if "final_asr" in stages:
        preferred_method = preferred_method_key(stages["final_asr"]) or "few_shot"
        sub = read_json(
            stages["final_asr"]["stage_dir"] / f"subgroup_metrics_{preferred_method}.json", {}
        )
        if sub:
            lines.append(f"## Subgroup performance (final_asr {method_display_name(preferred_method)})")
            for field, groups in sub.items():
                lines.append(f"- **{field.replace('_', ' ')}**:")
                for value, m in groups.items():
                    lines.append(
                        f"  - {value} (n={m['count']}): "
                        f"acc {m['accuracy']:.1%}, F1 {m['macro_f1']:.3f}"
                    )
            lines.append("")

    output_path = output_dir / "evaluation_summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build figures and an evaluation summary from workflow outputs."
    )
    parser.add_argument(
        "--workflow",
        default=None,
        help=(
            "Optional workflow folder name under outputs/workflows/ to use exclusively. "
            "If omitted, the script selects one coherent workflow folder with the "
            "strongest completed stage coverage."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_FIGURE_DIR),
        help=f"Directory to write figures into. Default: {DEFAULT_FIGURE_DIR}",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explicit = WORKFLOWS_DIR / args.workflow if args.workflow else None
    if explicit is not None and not explicit.exists():
        print(f"Workflow folder not found: {explicit}", file=sys.stderr)
        return 1

    stages = aggregate_stages(WORKFLOWS_DIR, explicit)
    if not stages:
        print(
            "No completed stages found. Run `python3 workflow.py --plan recommended` first.",
            file=sys.stderr,
        )
        return 1

    print("Using stages:")
    for stage_key in stage_keys_present(stages):
        payload = stages[stage_key]
        n = payload["summary"].get("evaluated_examples")
        print(f"  - {stage_key}: {payload['workflow_dir'].name}/{stage_key} (n={n})")

    outputs = [
        ("Headline performance", make_headline_performance(output_dir, stages)),
        (
            "Per-class F1 (final_gold)",
            make_per_class_f1(output_dir, stages, target_stage="final_gold"),
        ),
        (
            "Per-class F1 (final_asr)",
            make_per_class_f1(output_dir, stages, target_stage="final_asr"),
        ),
        (
            "Confusion matrix (final_gold, few_shot)",
            make_confusion_matrix(
                output_dir,
                stages,
                target_stage="final_gold",
                method="few_shot",
            ),
        ),
        (
            "Confusion matrix (final_asr, few_shot)",
            make_confusion_matrix(
                output_dir,
                stages,
                target_stage="final_asr",
                method="few_shot",
            ),
        ),
        ("Subgroup performance", make_subgroup_performance(output_dir, stages)),
        ("Gold vs ASR", make_gold_vs_asr(output_dir, stages)),
        ("Confidence calibration", make_confidence_calibration(output_dir, stages)),
        ("Review economics", make_review_economics(output_dir, stages)),
        ("ASR quality", make_asr_quality(output_dir, stages)),
        ("Evaluation summary", write_evaluation_summary(output_dir, stages)),
    ]

    print("\nGenerated:")
    for label, path in outputs:
        if path is None:
            print(f"  - {label}: skipped (missing data)")
        else:
            print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
