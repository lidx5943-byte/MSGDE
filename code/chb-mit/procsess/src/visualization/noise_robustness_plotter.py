"""Plots for noise robustness experiments."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .nature_style import NATURE_COLORS, PALETTE_MULTI, set_nature_style
from ..utils.console import print_info


LEGACY_ARTIFACT_ORDER = {"mild": 1, "moderate": 2, "severe": 3}
# 图例中特征组 S（数据列名仍为 S）显示为 G
_FEATURE_GROUP_PLOT_NAME = {"ALL": "ALL", "S": "G", "T": "T", "D": "D"}
# EMG/EOG 档位横轴用数字 1,2,… 而非 L1, L2,…
_NUMERIC_S_LEVEL_X = frozenset({"emg_burst", "eog_blink"})
PROTOCOL_LABELS = {
    "clean_to_noisy": "Train clean, test noisy",
    "noisy_to_noisy": "Noisy cross-validation",
}


def _set_publication_style() -> None:
    set_nature_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "grid.linewidth": 0.5,
        }
    )


def _parse_snr_condition(condition: str) -> float | None:
    match = re.fullmatch(r"snr_(-?\d+(?:p\d+)?)dB", condition)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def _condition_sort_key(noise_type: str, condition: str) -> tuple[int, float | str]:
    condition = str(condition)
    if condition == "clean":
        return (0, 0.0)
    if noise_type == "awgn":
        snr = _parse_snr_condition(condition)
        if snr is not None:
            return (1, -snr)
    match = re.fullmatch(r"s(\d+)", condition)
    if match:
        return (1, float(match.group(1)))
    if condition in LEGACY_ARTIFACT_ORDER:
        return (1, float(LEGACY_ARTIFACT_ORDER[condition]))
    return (2, condition)


def _condition_order(summary_df: pd.DataFrame, noise_type: str, protocol: str) -> list[str]:
    subset = summary_df[(summary_df["noise_type"] == noise_type) & (summary_df["protocol"] == protocol)]
    conditions = [str(item) for item in subset["condition"].dropna().unique().tolist()]
    return sorted(conditions, key=lambda item: _condition_sort_key(noise_type, item))


def _condition_label(noise_type: str, condition: str) -> str:
    condition = str(condition)
    if condition == "clean":
        return "Clean"
    if noise_type == "awgn":
        snr = _parse_snr_condition(condition)
        if snr is not None:
            return f"{snr:g} dB"
    match = re.fullmatch(r"s0?(\d+)", condition)
    if match:
        level = int(match.group(1))
        if noise_type in _NUMERIC_S_LEVEL_X:
            return str(level)
        return f"L{level}"
    return condition


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3, pad=2)


def _pick_reference_classifier(
    summary_df: pd.DataFrame,
    noise_type: str,
    protocol: str,
    feature_group: str,
) -> str | None:
    rows = summary_df[
        (summary_df["noise_type"] == noise_type)
        & (summary_df["protocol"] == protocol)
        & (summary_df["feature_group"] == feature_group)
        & (summary_df["condition"] == "clean")
    ]
    if rows.empty:
        return None
    best_row = rows.sort_values("Accuracy_mean", ascending=False).iloc[0]
    return str(best_row["classifier"])


def _plot_single_metric(
    summary_df: pd.DataFrame,
    output_dir: Path,
    noise_type: str,
    protocol: str,
    metric: str,
) -> None:
    feature_groups = ["ALL", "S", "T", "D"]
    order = _condition_order(summary_df, noise_type, protocol)
    if not order:
        return
    order_index = {condition: idx for idx, condition in enumerate(order)}
    tick_labels = [_condition_label(noise_type, condition) for condition in order]
    _set_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    metric_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    overall_ax, groups_ax = axes

    all_classifier = _pick_reference_classifier(summary_df, noise_type, protocol, "ALL")
    if all_classifier is not None:
        subset = summary_df[
            (summary_df["noise_type"] == noise_type)
            & (summary_df["protocol"] == protocol)
            & (summary_df["feature_group"] == "ALL")
            & (summary_df["classifier"] == all_classifier)
        ].copy()
        subset["condition"] = pd.Categorical(subset["condition"], categories=order, ordered=True)
        subset = subset.sort_values("condition")
        x = [order_index[str(condition)] for condition in subset["condition"]]
        overall_ax.plot(
            x,
            subset[metric_col],
            marker="o",
            color=NATURE_COLORS["primary"],
            linewidth=1.5,
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=f"ALL ({all_classifier})",
        )
        overall_ax.fill_between(
            x,
            subset[metric_col] - subset[std_col].fillna(0),
            subset[metric_col] + subset[std_col].fillna(0),
            color=NATURE_COLORS["primary"],
            alpha=0.12,
            linewidth=0,
        )
        overall_ax.legend(frameon=False, loc="best")
    overall_ax.set_xticks(list(range(len(order))))
    overall_ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    overall_ax.set_xlim(-0.4, len(order) - 0.6)
    overall_ax.set_title(f"{noise_type} | {PROTOCOL_LABELS.get(protocol, protocol)} | ALL", fontweight="bold")
    overall_ax.set_ylabel(metric)
    overall_ax.set_xlabel("Noise intensity")
    _style_axis(overall_ax)

    for idx, feature_group in enumerate(feature_groups):
        classifier = _pick_reference_classifier(summary_df, noise_type, protocol, feature_group)
        if classifier is None:
            continue
        subset = summary_df[
            (summary_df["noise_type"] == noise_type)
            & (summary_df["protocol"] == protocol)
            & (summary_df["feature_group"] == feature_group)
            & (summary_df["classifier"] == classifier)
        ].copy()
        subset["condition"] = pd.Categorical(subset["condition"], categories=order, ordered=True)
        subset = subset.sort_values("condition")
        x = [order_index[str(condition)] for condition in subset["condition"]]
        plot_name = _FEATURE_GROUP_PLOT_NAME.get(feature_group, feature_group)
        label = f"{plot_name} ({classifier})"
        color = PALETTE_MULTI[idx % len(PALETTE_MULTI)]
        groups_ax.plot(
            x,
            subset[metric_col],
            marker="o",
            linewidth=1.35,
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.4,
            color=color,
            label=label,
        )
        groups_ax.fill_between(
            x,
            subset[metric_col] - subset[std_col].fillna(0),
            subset[metric_col] + subset[std_col].fillna(0),
            color=color,
            alpha=0.10,
            linewidth=0,
        )

    groups_ax.set_title(f"{noise_type} | {PROTOCOL_LABELS.get(protocol, protocol)} | Feature groups", fontweight="bold")
    groups_ax.set_ylabel(metric)
    groups_ax.set_xlabel("Noise intensity")
    groups_ax.set_xticks(list(range(len(order))))
    groups_ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    groups_ax.set_xlim(-0.4, len(order) - 0.6)
    _style_axis(groups_ax)
    groups_ax.legend(frameon=False, loc="best", ncol=1, handlelength=1.6)

    file_stub = f"{noise_type}_{protocol}_{metric.replace('-', '_')}"
    fig.savefig(output_dir / f"{file_stub}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{file_stub}.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_noise_robustness(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate degradation curves for Accuracy and F1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return

    print_info("生成噪声鲁棒性退化曲线")
    metrics = ["Accuracy", "F1-Score"]
    for noise_type in sorted(summary_df["noise_type"].unique()):
        for protocol in sorted(summary_df["protocol"].unique()):
            subset = summary_df[
                (summary_df["noise_type"] == noise_type)
                & (summary_df["protocol"] == protocol)
            ]
            if subset.empty:
                continue
            for metric in metrics:
                metric_col = f"{metric}_mean"
                if metric_col in subset.columns:
                    _plot_single_metric(summary_df, output_dir, noise_type, protocol, metric)
