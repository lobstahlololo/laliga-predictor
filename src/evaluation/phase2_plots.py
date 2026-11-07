"""Charts for baselines, feature experiments, explainability and calibration."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import calibration_table
from src.utils.config import EVAL_DIR, LABEL_ORDER, OUTCOME_LABELS

HOME_COLOR = "#1f7a4d"
DRAW_COLOR = "#c9a227"
AWAY_COLOR = "#b03a2e"
MODEL_COLORS = ["#1f7a4d", "#2c5f8a", "#c9a227", "#7a4fa3", "#b03a2e"]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower() or "chart"


def _save(fig, name: str, out_dir: Path = EVAL_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_baseline_comparison(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path = EVAL_DIR,
) -> Path:
    """Validation and test log-loss for every baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, frame, label in (
        (axes[0], validation, "Validation (25/26)"),
        (axes[1], test, "Test (26/27)"),
    ):
        if frame.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue
        ordered = frame.sort_values("log_loss", ascending=True)
        bars = ax.barh(
            ordered["model"], ordered["log_loss"],
            color=MODEL_COLORS[: len(ordered)],
        )
        ax.invert_yaxis()
        ax.set_xlabel("Log-loss (lower is better)")
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, ordered["log_loss"]):
            ax.text(
                value, bar.get_y() + bar.get_height() / 2,
                f" {value:.3f}", va="center", fontsize=8,
            )
    fig.suptitle("Simple baselines vs the learned models", fontsize=12)
    return _save(fig, "baseline_comparison.png", out_dir)


def plot_feature_experiments(
    table: pd.DataFrame,
    out_dir: Path = EVAL_DIR,
) -> Path:
    """Validation and walk-forward log-loss deltas versus the base feature set."""
    if table.empty:
        return _save(_empty_figure("No feature experiments"), "feature_experiments.png", out_dir)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    part = table.copy()
    part["label"] = part["model"] + " · " + part["feature_set"]
    part = part.sort_values("delta_walk_forward_log_loss")
    y = np.arange(len(part))
    width = 0.36
    ax.barh(
        y - width / 2, part["delta_validation_log_loss"], width,
        label="Validation Δ log-loss", color="#2c5f8a",
    )
    ax.barh(
        y + width / 2, part["delta_walk_forward_log_loss"], width,
        label="Walk-forward Δ log-loss", color="#1f7a4d",
    )
    ax.axvline(0, color="#16202a", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(part["label"], fontsize=7)
    ax.set_xlabel("Δ log-loss vs base 19 features (negative is better)")
    ax.set_title("Feature-group experiments (no test data used)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, "feature_experiments.png", out_dir)


def _empty_figure(message: str):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    return fig


def plot_importance_comparison(
    permutation: pd.DataFrame,
    native: pd.DataFrame,
    shap_table: pd.DataFrame,
    out_dir: Path = EVAL_DIR,
) -> Path:
    """Top features under permutation, impurity and XGBoost TreeSHAP views."""
    panels = [
        ("Permutation (validation)", permutation, "importance_mean", "#2c5f8a"),
        ("Tree impurity (final fit)", native, "importance", "#1f7a4d"),
        ("XGBoost TreeSHAP (validation)", shap_table, "mean_abs_shap", "#c9a227"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (title, frame, value_col, color) in zip(axes, panels):
        if frame is None or frame.empty or value_col not in frame:
            ax.text(0.5, 0.5, "Unavailable", ha="center", va="center")
            ax.axis("off")
            continue
        top = frame.sort_values(value_col, ascending=False).head(10).iloc[::-1]
        ax.barh(top["feature"], top[value_col], color=color)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Feature importance is descriptive, not causal; correlated features share credit.",
        fontsize=10,
    )
    return _save(fig, "feature_importance_comparison.png", out_dir)


def plot_feature_stability(stability: pd.DataFrame, out_dir: Path = EVAL_DIR) -> Path:
    """Mean importance and fold spread for the most stable features."""
    if stability.empty:
        return _save(_empty_figure("No stability data"), "feature_stability.png", out_dir)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for i, ((model, method), part) in enumerate(
        stability.groupby(["model", "method"], sort=False)
    ):
        top = part.sort_values("mean_rank").head(10).iloc[::-1]
        y = np.arange(len(top)) + i * 0.22
        ax.errorbar(
            top["mean_importance"], y,
            xerr=top["std_importance"],
            fmt="o", capsize=2, label=f"{model} · {method}",
            color=MODEL_COLORS[i % len(MODEL_COLORS)], markersize=4,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(top["feature"], fontsize=7)
    ax.set_xlabel("Mean importance across walk-forward folds")
    ax.set_title("Feature stability across folds (error bars = fold spread)")
    ax.legend(fontsize=7)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, "feature_stability.png", out_dir)


def plot_calibration_comparison(
    comparison: pd.DataFrame,
    out_dir: Path = EVAL_DIR,
) -> Path:
    """Uncalibrated vs sigmoid vs isotonic test metrics."""
    if comparison.empty:
        return _save(_empty_figure("No calibration data"), "calibration_comparison.png", out_dir)
    metrics = ["log_loss", "brier_score", "ece"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, metric in zip(axes, metrics):
        values = comparison[metric].astype(float)
        bars = ax.bar(comparison["calibration"], values, color=MODEL_COLORS[: len(comparison)])
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.tick_params(axis="x", labelrotation=12)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value,
                f"{value:.3f}", ha="center", va="bottom", fontsize=8,
            )
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Calibration methods on the untouched test season", fontsize=12)
    return _save(fig, "calibration_comparison.png", out_dir)


def plot_reliability_overlay(
    y_true,
    probabilities: dict[str, np.ndarray],
    classes,
    title: str,
    filename: str,
    out_dir: Path = EVAL_DIR,
) -> Path:
    """Reliability diagram overlaying several probability sets (all classes)."""
    fig, axes = plt.subplots(1, len(classes), figsize=(4.1 * len(classes), 4.0))
    axes = np.atleast_1d(axes)
    colors = [HOME_COLOR, DRAW_COLOR, AWAY_COLOR, "#2c5f8a", "#7a4fa3"]
    for ax, cls in zip(axes, classes):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        for i, (label, proba) in enumerate(probabilities.items()):
            table = calibration_table(y_true, proba, classes)
            part = table[table["class"] == int(cls)]
            if part.empty:
                continue
            ax.plot(
                part["mean_confidence"], part["observed_rate"],
                marker="o", markersize=3, linewidth=1.2,
                color=colors[i % len(colors)], label=label,
            )
        ax.set_title(OUTCOME_LABELS.get(int(cls), str(cls)), fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed frequency")
    for ax in axes:
        ax.set_xlabel("Mean predicted probability")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=11)
    return _save(fig, filename, out_dir)


def plot_experiment_leaderboard(
    experiments: pd.DataFrame,
    out_dir: Path = EVAL_DIR,
) -> Path:
    """Validation log-loss for tracked experiments, with test withheld."""
    if experiments.empty:
        return _save(_empty_figure("No tracked experiments"), "experiment_leaderboard.png", out_dir)
    frame = experiments.copy()
    if "validation_log_loss" not in frame.columns:
        return _save(_empty_figure("No validation metrics"), "experiment_leaderboard.png", out_dir)
    frame = frame.dropna(subset=["validation_log_loss"]).sort_values("validation_log_loss")
    frame = frame.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    labels = frame["experiment_id"].astype(str)
    ax.barh(labels, frame["validation_log_loss"], color="#2c5f8a")
    ax.set_xlabel("Validation log-loss (lower is better)")
    ax.set_title("Experiment leaderboard — test metrics are withheld from selection")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, "experiment_leaderboard.png", out_dir)
