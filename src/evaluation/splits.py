"""Chronological splits and walk-forward folds.

No function in this module shuffles or randomly samples rows. Every training
window ends before the corresponding validation/test window starts, which is
the core defence against temporal leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    """One backtest fold: all seasons before ``test_season`` train it."""

    fold: int
    train_seasons: tuple[str, ...]
    test_season: str
    train_index: pd.Index
    test_index: pd.Index

    def describe(self) -> str:
        return (
            f"fold {self.fold}: train {', '.join(self.train_seasons)} "
            f"-> test {self.test_season}"
        )


def _season_order(df: pd.DataFrame) -> list[str]:
    seasons = [s for s in df["season"].dropna().astype(str).unique()]
    # Season labels are "YY/YY"; lexical sort is chronological.
    return sorted(seasons)


def chronological_split(
    meta: pd.DataFrame,
    train_seasons: Iterable[str] | None = None,
    validation_season: str | None = None,
    test_season: str | None = None,
) -> dict:
    """Split rows into earlier-train / middle-validation / latest-test.

    ``train_seasons`` defaults to every season before the validation season.
    The split is validated for chronology and disjointness; a bad
    configuration raises instead of silently producing a leaky split.
    """
    if "season" not in meta.columns:
        raise ValueError("Metadata must contain a 'season' column.")
    seasons = _season_order(meta)
    if not seasons:
        raise ValueError("No seasons found in metadata.")

    if test_season is None:
        test_season = seasons[-1]
    if validation_season is None:
        validation_season = seasons[-2] if len(seasons) > 1 else None

    if train_seasons is None:
        if validation_season is not None:
            train_seasons = [s for s in seasons if s < validation_season]
        else:
            train_seasons = [s for s in seasons if s < test_season]
    train_seasons = list(train_seasons)

    for label, season in (("validation", validation_season), ("test", test_season)):
        if season is not None and season not in seasons:
            raise ValueError(
                f"{label} season {season!r} is not present; found {seasons}."
            )
    if train_seasons and validation_season is not None:
        if max(train_seasons) >= validation_season:
            raise ValueError("Training seasons must precede the validation season.")
    if validation_season is not None and test_season is not None:
        if validation_season >= test_season:
            raise ValueError("Validation season must precede the test season.")

    masks = {"train": meta["season"].isin(train_seasons)}
    if validation_season is not None:
        masks["validation"] = meta["season"] == validation_season
    masks["test"] = meta["season"] == test_season

    result = {
        key: meta[mask].copy()
        for key, mask in masks.items()
    }
    result.update({
        "train_seasons": train_seasons,
        "validation_season": validation_season,
        "test_season": test_season,
        "season_order": seasons,
    })

    # Hard disjointness check: an index appearing twice would leak rows.
    indices = [set(result[k].index) for k in ("train", "validation", "test") if k in result]
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            overlap = indices[i] & indices[j]
            if overlap:
                raise ValueError(f"Split indices overlap ({len(overlap)} rows).")

    for key in ("train", "validation", "test"):
        if key in result and result[key].empty:
            raise ValueError(f"The {key} split is empty.")
    return result


def walk_forward_folds(
    meta: pd.DataFrame,
    min_train_seasons: int = 1,
    max_folds: int | None = None,
) -> list[WalkForwardFold]:
    """Generate rolling-origin folds over seasons.

    With seasons ``[23/24, 24/25, 25/26, 26/27]`` and
    ``min_train_seasons=2`` the folds are:

    * train 23/24+24/25 -> test 25/26
    * train 23/24+24/25+25/26 -> test 26/27

    Every fold trains strictly on earlier seasons.
    """
    if "season" not in meta.columns:
        raise ValueError("Metadata must contain a 'season' column.")
    seasons = _season_order(meta)
    if min_train_seasons < 1:
        raise ValueError("min_train_seasons must be >= 1.")
    if len(seasons) <= min_train_seasons:
        return []

    folds: list[WalkForwardFold] = []
    for fold_no, test_season in enumerate(seasons[min_train_seasons:], start=1):
        train_seasons = tuple(s for s in seasons if s < test_season)
        train_mask = meta["season"].isin(train_seasons)
        test_mask = meta["season"] == test_season
        if not train_mask.any() or not test_mask.any():
            continue
        # Chronology assertion: latest training date must precede the test
        # window's first date.
        if "date" in meta.columns:
            last_train = pd.to_datetime(meta.loc[train_mask, "date"]).max()
            first_test = pd.to_datetime(meta.loc[test_mask, "date"]).min()
            if last_train >= first_test:
                raise ValueError(
                    f"Fold {fold_no} is not chronological: last train {last_train} "
                    f">= first test {first_test}."
                )
        folds.append(
            WalkForwardFold(
                fold=fold_no,
                train_seasons=train_seasons,
                test_season=test_season,
                train_index=meta.index[train_mask],
                test_index=meta.index[test_mask],
            )
        )
        if max_folds is not None and len(folds) >= max_folds:
            break
    return folds


def season_train_test(
    meta: pd.DataFrame, train_seasons: Iterable[str], test_season: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple earlier-seasons -> one-season split with chronology checks."""
    train_seasons = list(train_seasons)
    if max(train_seasons) >= test_season:
        raise ValueError("All training seasons must precede the test season.")
    train = meta[meta["season"].isin(train_seasons)].copy()
    test = meta[meta["season"] == test_season].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty side.")
    if "date" in meta.columns:
        if pd.to_datetime(train["date"]).max() >= pd.to_datetime(test["date"]).min():
            raise ValueError("Train dates must precede test dates.")
    return train, test
