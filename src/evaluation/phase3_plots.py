"""Charts for Phase 3 Elo, Poisson, ensemble and market diagnostics."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import EVAL_DIR

COLORS = ["#1f7a4d", "#2c5f8a", "#c9a227", "#7a4fa3", "#b03a2e", "#4d7f6a"]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower() or "chart"


def _save(fig, name: str, out_dir: Path = EVAL_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_elo_history(
    history: pd.DataFrame,
    latest: pd.DataFrame,
    out_dir: Path = EVAL_DIR,
    top_n: int = 8,
) -> Path:
    if history.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No Elo history", ha="center", va="center")
        ax.axis("off")
        return _save(fig, "elo_rating_history.png", out_dir)
    teams = latest.head(top_n)["team"].tolist()
    part = history[history["team"].isin(teams)].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (team, group) in enumerate(part.groupby("team")):
        ax.plot(group["date"], group["rating"], label=team, color=COLORS[i % len(COLORS)])
    ax.set_title("Dynamic Elo rating history (top current clubs)")
    ax.set_ylabel("Elo rating")
    ax.set_xlabel("Date")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    return _save(fig, "elo_rating_history.png", out_dir)


def plot_elo_uncertainty(latest: pd.DataFrame, out_dir: Path = EVAL_DIR) -> Path:
    if latest.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No Elo ratings", ha="center", va="center")
        ax.axis("off")
        return _save(fig, "elo_rating_uncertainty.png", out_dir)
    part = latest.sort_values("rating_deviation", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(part["team"], part["rating_deviation"], color="#2c5f8a")
    ax.invert_yaxis()
    ax.set_xlabel("Rating deviation (higher = less certain)")
    ax.set_title("Elo rating uncertainty")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, "elo_rating_uncertainty.png", out_dir)


def plot_poisson_ratings(ratings: pd.DataFrame, out_dir: Path = EVAL_DIR) -> Path:
    if ratings.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No Poisson ratings", ha="center", va="center")
        ax.axis("off")
        return _save(fig, "poisson_attack_defence.png", out_dir)
    part = ratings[ratings["known"]].copy()
    if part.empty:
        part = ratings.copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(part["attack"], part["defence"], color="#1f7a4d", s=35)
    for _, row in part.iterrows():
        ax.annotate(str(row["team"]), (row["attack"], row["defence"]), fontsize=7, alpha=0.85)
    ax.axvline(1.0, color="#8a98a4", linewidth=0.8, linestyle="--")
    ax.axhline(1.0, color="#8a98a4", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Attack multiplier (higher = stronger attack)")
    ax.set_ylabel("Defence multiplier (lower = better defence)")
    ax.set_title("Poisson attack and defence estimates")
    ax.grid(alpha=0.2)
    return _save(fig, "poisson_attack_defence.png", out_dir)


def plot_scoreline_heatmap(
    matrix: pd.DataFrame,
    title: str,
    filename: str = "poisson_scoreline_heatmap.png",
    out_dir: Path = EVAL_DIR,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    values = matrix.to_numpy(float)
    image = ax.imshow(values, cmap="Greens", origin="upper")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels([str(i) for i in range(matrix.shape[1])], fontsize=7)
    ax.set_yticklabels([str(i) for i in range(matrix.shape[0])], fontsize=7)
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Probability")
    return _save(fig, filename, out_dir)


def plot_phase3_model_comparison(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path = EVAL_DIR,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, frame, label in (
        (axes[0], validation, "Validation (25/26)"),
        (axes[1], test, "Test (26/27)"),
    ):
        if frame.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue
        part = frame.sort_values("log_loss", ascending=True).tail(12)
        bars = ax.barh(part["model"], part["log_loss"], color=COLORS[0])
        ax.invert_yaxis()
        ax.set_xlabel("Log-loss (lower is better)")
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, part["log_loss"]):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3f}", va="center", fontsize=7)
    fig.suptitle("Phase 3 model comparison", fontsize=12)
    return _save(fig, "phase3_model_comparison.png", out_dir)


def plot_ensemble_weights(weights: pd.DataFrame, out_dir: Path = EVAL_DIR) -> Path:
    if weights.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No ensemble weights", ha="center", va="center")
        ax.axis("off")
        return _save(fig, "ensemble_weights.png", out_dir)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(weights["component"], weights["weight"], color=COLORS[: len(weights)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation-selected weight")
    ax.set_title("Ensemble weights (selected on validation only)")
    ax.tick_params(axis="x", labelrotation=15)
    for bar, value in zip(bars, weights["weight"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, "ensemble_weights.png", out_dir)


def plot_market_probability_scatter(
    differences: pd.DataFrame,
    model_name: str = "Random Forest",
    out_dir: Path = EVAL_DIR,
) -> Path:
    part = differences[differences["model"] == model_name]
    if part.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No market comparison", ha="center", va="center")
        ax.axis("off")
        return _save(fig, "market_probability_scatter.png", out_dir)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(part["market_home"], part["model_home"], s=18, alpha=0.7, color="#1f7a4d")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Market home-win probability")
    ax.set_ylabel("Model home-win probability")
    ax.set_title(f"Model vs market — {model_name}")
    ax.grid(alpha=0.25)
    return _save(fig, "market_probability_scatter.png", out_dir)


def plot_uncertainty(uncertainty: pd.DataFrame, out_dir: Path = EVAL_DIR) -> Path:
    if uncertainty.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No uncertainty data", ha="center", va="center")
        ax.axis("off")
        return _save(fig, "prediction_uncertainty.png", out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].hist(uncertainty["entropy"], bins=15, color="#2c5f8a", alpha=0.85)
    axes[0].set_title("Normalised outcome entropy")
    axes[0].set_xlabel("Entropy (higher = more uncertain)")
    axes[0].set_ylabel("Fixtures")
    axes[0].grid(axis="y", alpha=0.25)
    if uncertainty["row_log_loss"].notna().any():
        groups = [
            uncertainty.loc[uncertainty["uncertainty_level"] == level, "row_log_loss"].dropna()
            for level in ("low", "medium", "high")
        ]
        labels = ["low", "medium", "high"]
        parts = [group for group in groups if len(group)]
        labels = [label for label, group in zip(labels, groups) if len(group)]
        axes[1].boxplot(parts, tick_labels=labels)
        axes[1].set_title("Realised log-loss by uncertainty level")
        axes[1].set_ylabel("Row log-loss")
        axes[1].grid(axis="y", alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "No settled outcomes", ha="center", va="center")
        axes[1].axis("off")
    fig.suptitle("Prediction uncertainty is descriptive, not a guarantee", fontsize=11)
    return _save(fig, "prediction_uncertainty.png", out_dir)
