"""Matplotlib charts for model evaluation and monitoring.

Uses the headless Agg backend so the same functions work in scripts, pytest
and Streamlit. All figures are saved to ``models/evaluation/``.
"""

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
MODEL_COLORS = ["#1f7a4d", "#2c5f8a", "#c9a227"]


def _slug(model_name: str) -> str:
    """Filesystem-safe ASCII slug for a model label."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return cleaned or "model"


def _save(fig, name: str, out_dir: Path = EVAL_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    cm: pd.DataFrame | np.ndarray,
    model_name: str,
    accuracy: float | None = None,
    out_dir: Path = EVAL_DIR,
) -> Path:
    """Confusion matrix with raw counts and row-normalised percentages."""
    if isinstance(cm, pd.DataFrame):
        counts = cm.values.astype(int)
        xticklabels = [c.replace("Pred ", "") for c in cm.columns]
        yticklabels = [c.replace("Actual ", "") for c in cm.index]
    else:
        counts = np.asarray(cm, dtype=int)
        xticklabels = yticklabels = [OUTCOME_LABELS.get(c, str(c)) for c in LABEL_ORDER]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(counts, interpolation="nearest", cmap=plt.cm.Greens)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(xticklabels)),
        yticks=range(len(yticklabels)),
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        ylabel="Actual",
        xlabel="Predicted",
        title=f"{model_name} — Confusion Matrix"
        + (f" (acc={accuracy:.3f})" if accuracy is not None else ""),
    )
    row_totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_pct = np.where(row_totals > 0, counts / row_totals * 100.0, 0.0)
    threshold = counts.max() / 2 if counts.max() > 0 else 0.5
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(
                j, i, f"{counts[i, j]}\n({row_pct[i, j]:.0f}%)",
                ha="center", va="center", fontsize=9,
                color="white" if counts[i, j] > threshold else "black",
            )
    return _save(fig, f"confusion_matrix_{_slug(model_name)}.png", out_dir)


def plot_calibration_curves(
    y_true,
    y_proba,
    classes,
    model_name: str,
    n_bins: int = 10,
    out_dir: Path = EVAL_DIR,
) -> Path:
    """One-vs-rest reliability diagram for each outcome class."""
    table = calibration_table(y_true, y_proba, classes, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    colors = {2: HOME_COLOR, 1: DRAW_COLOR, 0: AWAY_COLOR}
    for cls in LABEL_ORDER:
        part = table[table["class"] == cls]
        if part.empty:
            continue
        ax.plot(
            part["mean_confidence"], part["observed_rate"],
            marker="o", linewidth=1.6, color=colors.get(cls),
            label=OUTCOME_LABELS.get(cls, str(cls)),
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed frequency",
        title=f"{model_name} — Calibration",
        xlim=(0, 1), ylim=(0, 1),
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, f"calibration_{_slug(model_name)}.png", out_dir)


def plot_model_comparison(comparison: pd.DataFrame, out_dir: Path = EVAL_DIR) -> Path:
    """Side-by-side accuracy / log-loss / Brier bars for each model."""
    metrics = ["accuracy", "log_loss", "brier_score"]
    titles = ["Accuracy (higher better)", "Log-loss (lower better)", "Brier (lower better)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, metric, title in zip(axes, metrics, titles):
        values = comparison[metric].astype(float)
        bars = ax.bar(
            comparison["model"], values,
            color=MODEL_COLORS[: len(comparison)],
        )
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=12)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value,
                f"{value:.3f}", ha="center", va="bottom", fontsize=8,
            )
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Model comparison on the untouched test season", fontsize=12)
    return _save(fig, "model_comparison.png", out_dir)


def plot_per_class_metrics(per_class: pd.DataFrame, model_name: str, out_dir: Path = EVAL_DIR) -> Path:
    """Grouped precision/recall/F1 bars per outcome class."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(per_class))
    width = 0.26
    for i, metric in enumerate(("precision", "recall", "f1")):
        ax.bar(x + (i - 1) * width, per_class[metric], width, label=metric.title())
    ax.set_xticks(x)
    ax.set_xticklabels(per_class["outcome"])
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name} — Per-class metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, f"per_class_metrics_{_slug(model_name)}.png", out_dir)


def plot_season_performance(season_df: pd.DataFrame, out_dir: Path = EVAL_DIR) -> Path:
    """Log-loss and accuracy by season for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for model, part in season_df.groupby("model"):
        part = part.sort_values("season")
        axes[0].plot(part["season"], part["log_loss"], marker="o", label=model)
        axes[1].plot(part["season"], part["accuracy"], marker="o", label=model)
    axes[0].set_title("Log-loss by season (lower better)")
    axes[1].set_title("Accuracy by season")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend(fontsize=8)
    return _save(fig, "season_performance.png", out_dir)


def plot_walk_forward(wf_df: pd.DataFrame, out_dir: Path = EVAL_DIR) -> Path:
    """Walk-forward log-loss across folds for each model."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for model, part in wf_df.groupby("model"):
        part = part.sort_values("test_season")
        ax.plot(part["test_season"], part["log_loss"], marker="o", label=model)
    ax.set_title("Walk-forward validation log-loss")
    ax.set_xlabel("Test season")
    ax.set_ylabel("Log-loss (lower better)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return _save(fig, "walk_forward_log_loss.png", out_dir)


def plot_prediction_distribution(history: pd.DataFrame, out_dir: Path = EVAL_DIR) -> Path:
    """Share of predicted outcomes over time (model drift indicator)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if history.empty:
        ax.text(0.5, 0.5, "No prediction history", ha="center", va="center")
        ax.axis("off")
    else:
        hist = history.copy()
        hist["predicted_at"] = pd.to_datetime(hist["predicted_at"], errors="coerce")
        hist = hist.dropna(subset=["predicted_at"]).sort_values("predicted_at")
        if hist.empty:
            ax.text(0.5, 0.5, "No dated predictions", ha="center", va="center")
            ax.axis("off")
        else:
            pivot = (
                hist.groupby(hist["predicted_at"].dt.date)["predicted_outcome"]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
                .sort_index()
            )
            bottom = np.zeros(len(pivot))
            for label, color in (("Home Win", HOME_COLOR), ("Draw", DRAW_COLOR), ("Away Win", AWAY_COLOR)):
                if label in pivot.columns:
                    ax.bar(
                        [str(d) for d in pivot.index], pivot[label],
                        bottom=bottom, label=label, color=color,
                    )
                    bottom += pivot[label].to_numpy()
            ax.set_ylim(0, 1)
            ax.set_ylabel("Share of predictions")
            ax.set_title("Prediction distribution over time")
            ax.tick_params(axis="x", labelrotation=30)
            ax.legend(fontsize=8)
    return _save(fig, "prediction_distribution.png", out_dir)
