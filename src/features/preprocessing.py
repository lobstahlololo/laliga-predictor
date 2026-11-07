"""Leakage-safe feature preprocessing shared by training and inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

DEFAULT_PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"


@dataclass
class FeaturePreprocessor:
    """Median imputation + standard scaling fitted on training rows only.

    ``transform`` reindexes unknown frames to the training feature order and
    raises a clear error if a required feature is absent. All imputation
    statistics come from the training window, never from validation/test data
    or from the fixtures being predicted.
    """

    feature_names: list[str]
    medians: dict[str, float] = field(default_factory=dict)
    scaler: StandardScaler = field(default_factory=StandardScaler)
    fitted: bool = False

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        frame = X[self.feature_names].apply(pd.to_numeric, errors="coerce").copy()
        if self.fitted:
            frame = frame.fillna(self.medians)
        return frame

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        frame = self._prepare(X)
        # A feature that is entirely missing in training has no median; 0 is
        # the only neutral choice, and we log it loudly.
        medians = frame.median(numeric_only=True)
        for col in self.feature_names:
            value = medians.get(col, np.nan)
            if pd.isna(value):
                logger.warning("Feature %r has no training values; imputing 0.", col)
                value = 0.0
            self.medians[col] = float(value)
        frame = frame.fillna(self.medians)
        self.scaler.fit(frame.values)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform().")
        frame = self._prepare(X).fillna(self.medians)
        scaled = self.scaler.transform(frame.values)
        return pd.DataFrame(scaled, columns=self.feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | str = DEFAULT_PREPROCESSOR_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved preprocessor to %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PREPROCESSOR_PATH) -> "FeaturePreprocessor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {path}. "
                "Run `python -m src.features.build_features` first."
            )
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Unexpected object in {path}: {type(obj)!r}")
        return obj
