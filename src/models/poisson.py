"""Time-decayed Poisson and Dixon-Coles goal models.

The model estimates each club's multiplicative attack and defence strength from
historical goals, with four safeguards against unstable estimates:

* observations are exponentially time-decayed;
* attack/defence estimates are shrunk toward the league average;
* clubs with too few matches fall back to the league average explicitly;
* expected goals and scoreline matrices are clipped and normalised.

A Dixon-Coles low-score correction can be fitted on the training likelihood.
Every fit is restricted to matches on or before an ``as_of`` cutoff. The
pre-match feature builder refits at calendar-month boundaries and predicts each
month from earlier matches only, so it cannot use same-day or future results.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

POISSON_FEATURE_COLUMNS = [
    "poisson_home_xg",
    "poisson_away_xg",
    "poisson_prob_home",
    "poisson_prob_draw",
    "poisson_prob_away",
    "dc_prob_home",
    "dc_prob_draw",
    "dc_prob_away",
    "poisson_home_known",
    "poisson_away_known",
]


@dataclass(frozen=True)
class PoissonConfig:
    """Declared-before-evaluation Poisson/Dixon-Coles settings."""

    max_goals: int = 10
    half_life_days: float = 730.0
    prior_strength: float = 5.0
    min_matches: int = 3
    dixon_coles: bool = True
    rho_min: float = -0.15
    rho_max: float = 0.15
    rho_steps: int = 31
    iterations: int = 80
    minimum_weight: float = 0.01


def _poisson_pmf_vector(max_goals: int, lam: float) -> np.ndarray:
    goals = np.arange(max_goals + 1, dtype=float)
    factorials = np.array(
        [float(math.factorial(int(k))) for k in goals], dtype=float
    )
    if lam <= 0:
        out = np.zeros_like(goals)
        out[0] = 1.0
        return out
    return np.exp(-lam + goals * np.log(lam) - np.log(factorials))


def _dc_tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    rho: float,
) -> np.ndarray:
    tau = np.ones_like(np.asarray(home_goals, dtype=float))
    home_goals = np.asarray(home_goals, dtype=int)
    away_goals = np.asarray(away_goals, dtype=int)
    lam_home = np.asarray(lam_home, dtype=float)
    lam_away = np.asarray(lam_away, dtype=float)
    mask = (home_goals == 0) & (away_goals == 0)
    tau[mask] = 1.0 - lam_home[mask] * lam_away[mask] * rho
    mask = (home_goals == 0) & (away_goals == 1)
    tau[mask] = 1.0 + lam_home[mask] * rho
    mask = (home_goals == 1) & (away_goals == 0)
    tau[mask] = 1.0 + lam_away[mask] * rho
    mask = (home_goals == 1) & (away_goals == 1)
    tau[mask] = 1.0 - rho
    return np.clip(tau, 1e-9, None)


class PoissonModel:
    """A fitted Poisson / Dixon-Coles model for one historical cutoff."""

    def __init__(self, config: PoissonConfig | None = None) -> None:
        self.config = config or PoissonConfig()
        self.home_base: float = 1.35
        self.away_base: float = 1.05
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.matches: dict[str, int] = {}
        self.effective_matches: dict[str, float] = {}
        self.known: dict[str, bool] = {}
        self.rho: float = 0.0
        self.fitted = False
        self.as_of: pd.Timestamp | None = None
        self.n_training_matches: int = 0
        self.rho_table = pd.DataFrame()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, raw: pd.DataFrame, as_of: str | pd.Timestamp | None = None) -> "PoissonModel":
        required = {
            "date", "home_team", "away_team",
            "full_time_home_goals", "full_time_away_goals",
        }
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"Poisson input missing columns: {sorted(missing)}")

        frame = raw.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        if as_of is not None:
            self.as_of = pd.Timestamp(as_of).normalize()
            frame = frame[frame["date"] <= self.as_of]
        else:
            self.as_of = frame["date"].max() if not frame.empty else None
        frame = frame.sort_values(["date", "match_id"] if "match_id" in frame.columns else ["date"])

        self.n_training_matches = int(len(frame))
        if frame.empty:
            self.fitted = True
            return self

        teams = sorted(set(frame["home_team"].astype(str)) | set(frame["away_team"].astype(str)))
        team_index = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)
        home_idx = frame["home_team"].astype(str).map(team_index).to_numpy(int)
        away_idx = frame["away_team"].astype(str).map(team_index).to_numpy(int)
        home_goals = frame["full_time_home_goals"].to_numpy(float)
        away_goals = frame["full_time_away_goals"].to_numpy(float)
        age_days = (self.as_of - frame["date"]).dt.days.to_numpy(float)
        weights = np.clip(
            0.5 ** (np.maximum(age_days, 0.0) / max(self.config.half_life_days, 1.0)),
            self.config.minimum_weight,
            1.0,
        )

        self.home_base = float(np.average(home_goals, weights=weights))
        self.away_base = float(np.average(away_goals, weights=weights))
        if not np.isfinite(self.home_base) or self.home_base <= 0:
            self.home_base = 1.35
        if not np.isfinite(self.away_base) or self.away_base <= 0:
            self.away_base = 1.05

        attack = np.ones(n_teams, dtype=float)
        defence = np.ones(n_teams, dtype=float)
        for _ in range(max(1, self.config.iterations)):
            attack_num = (
                np.bincount(home_idx, weights=weights * home_goals, minlength=n_teams)
                + np.bincount(away_idx, weights=weights * away_goals, minlength=n_teams)
            )
            attack_den = (
                np.bincount(
                    home_idx,
                    weights=weights * self.home_base * defence[away_idx],
                    minlength=n_teams,
                )
                + np.bincount(
                    away_idx,
                    weights=weights * self.away_base * defence[home_idx],
                    minlength=n_teams,
                )
            )
            attack = np.divide(
                attack_num, attack_den, out=np.ones_like(attack), where=attack_den > 0
            )

            defence_num = (
                np.bincount(home_idx, weights=weights * away_goals, minlength=n_teams)
                + np.bincount(away_idx, weights=weights * home_goals, minlength=n_teams)
            )
            defence_den = (
                np.bincount(
                    home_idx,
                    weights=weights * self.away_base * attack[away_idx],
                    minlength=n_teams,
                )
                + np.bincount(
                    away_idx,
                    weights=weights * self.home_base * attack[home_idx],
                    minlength=n_teams,
                )
            )
            defence = np.divide(
                defence_num, defence_den, out=np.ones_like(defence), where=defence_den > 0
            )

            # The multiplicative model is identified only up to a scale. Fix
            # the weighted mean attack to 1 and move the scale into defence so
            # every fitted expected-goal product is unchanged.
            team_weight = (
                np.bincount(home_idx, weights=weights, minlength=n_teams)
                + np.bincount(away_idx, weights=weights, minlength=n_teams)
            )
            mean_attack = np.average(attack, weights=team_weight) if team_weight.sum() > 0 else 1.0
            if np.isfinite(mean_attack) and mean_attack > 0:
                attack = attack / mean_attack
                defence = defence * mean_attack
            attack = np.clip(attack, 0.25, 4.0)
            defence = np.clip(defence, 0.25, 4.0)

        match_counts = (
            np.bincount(home_idx, minlength=n_teams)
            + np.bincount(away_idx, minlength=n_teams)
        )
        effective = (
            np.bincount(home_idx, weights=weights, minlength=n_teams)
            + np.bincount(away_idx, weights=weights, minlength=n_teams)
        )
        shrink = effective / (effective + max(self.config.prior_strength, 1e-6))
        attack = 1.0 + shrink * (attack - 1.0)
        defence = 1.0 + shrink * (defence - 1.0)
        sparse = match_counts < self.config.min_matches
        attack[sparse] = 1.0
        defence[sparse] = 1.0

        self.attack = {team: float(attack[i]) for team, i in team_index.items()}
        self.defence = {team: float(defence[i]) for team, i in team_index.items()}
        self.matches = {team: int(match_counts[i]) for team, i in team_index.items()}
        self.effective_matches = {team: float(effective[i]) for team, i in team_index.items()}
        self.known = {
            team: bool(match_counts[i] >= self.config.min_matches)
            for team, i in team_index.items()
        }
        self.fitted = True

        if self.config.dixon_coles:
            self.rho, self.rho_table = self._fit_rho(
                home_goals, away_goals, home_idx, away_idx, weights
            )
        else:
            self.rho = 0.0
        return self

    def _fit_rho(
        self,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, pd.DataFrame]:
        teams = list(self.attack)
        lam_home = np.array([
            self._expected_goals(teams[i], teams[j])[0]
            for i, j in zip(home_idx, away_idx)
        ])
        lam_away = np.array([
            self._expected_goals(teams[i], teams[j])[1]
            for i, j in zip(home_idx, away_idx)
        ])
        home_pmf = np.array([
            _poisson_pmf_vector(self.config.max_goals, value)[int(goal)]
            for value, goal in zip(lam_home, home_goals)
        ])
        away_pmf = np.array([
            _poisson_pmf_vector(self.config.max_goals, value)[int(goal)]
            for value, goal in zip(lam_away, away_goals)
        ])
        rows = []
        for rho in np.linspace(self.config.rho_min, self.config.rho_max, self.config.rho_steps):
            tau = _dc_tau(home_goals, away_goals, lam_home, lam_away, float(rho))
            probabilities = np.clip(home_pmf * away_pmf * tau, 1e-12, None)
            log_likelihood = float(np.sum(weights * np.log(probabilities)))
            rows.append({"rho": float(rho), "weighted_log_likelihood": log_likelihood})
        table = pd.DataFrame(rows).sort_values(
            "weighted_log_likelihood", ascending=False
        ).reset_index(drop=True)
        return float(table.iloc[0]["rho"]), table

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        attack_home = self.attack.get(home_team, 1.0)
        attack_away = self.attack.get(away_team, 1.0)
        defence_home = self.defence.get(home_team, 1.0)
        defence_away = self.defence.get(away_team, 1.0)
        lam_home = self.home_base * attack_home * defence_away
        lam_away = self.away_base * attack_away * defence_home
        return (
            float(np.clip(lam_home, 0.05, 6.0)),
            float(np.clip(lam_away, 0.05, 6.0)),
        )

    def scoreline_matrix(
        self,
        home_team: str,
        away_team: str,
        *,
        use_dixon_coles: bool | None = None,
    ) -> pd.DataFrame:
        lam_home, lam_away = self._expected_goals(str(home_team), str(away_team))
        home_pmf = _poisson_pmf_vector(self.config.max_goals, lam_home)
        away_pmf = _poisson_pmf_vector(self.config.max_goals, lam_away)
        matrix = np.outer(home_pmf, away_pmf)
        use_dc = self.config.dixon_coles if use_dixon_coles is None else use_dixon_coles
        if use_dc and abs(self.rho) > 0:
            goals = np.arange(self.config.max_goals + 1)
            hg, ag = np.meshgrid(goals, goals, indexing="ij")
            matrix = matrix * _dc_tau(
                hg.ravel(), ag.ravel(), np.full(hg.size, lam_home),
                np.full(hg.size, lam_away), self.rho,
            ).reshape(matrix.shape)
        matrix = np.clip(matrix, 0.0, None)
        total = float(matrix.sum())
        if not np.isfinite(total) or total <= 0:
            matrix = np.ones_like(matrix) / matrix.size
        else:
            matrix = matrix / total
        goals = np.arange(self.config.max_goals + 1)
        return pd.DataFrame(
            matrix,
            index=[f"Home {g}" for g in goals],
            columns=[f"Away {g}" for g in goals],
        )

    def scoreline_table(
        self,
        home_team: str,
        away_team: str,
        *,
        use_dixon_coles: bool | None = None,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        matrix = self.scoreline_matrix(
            home_team, away_team, use_dixon_coles=use_dixon_coles
        )
        rows = []
        for i, home_label in enumerate(matrix.index):
            for j, away_label in enumerate(matrix.columns):
                rows.append({
                    "home_goals": i,
                    "away_goals": j,
                    "scoreline": f"{i}-{j}",
                    "probability": float(matrix.iloc[i, j]),
                })
        table = pd.DataFrame(rows).sort_values(
            "probability", ascending=False
        ).reset_index(drop=True)
        return table.head(top_n) if top_n else table

    def outcome_probabilities(
        self,
        home_team: str,
        away_team: str,
        *,
        use_dixon_coles: bool | None = None,
    ) -> np.ndarray:
        matrix = self.scoreline_matrix(
            home_team, away_team, use_dixon_coles=use_dixon_coles
        ).to_numpy(float)
        home = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away = float(np.triu(matrix, 1).sum())
        out = np.array([home, draw, away], dtype=float)
        out = np.clip(out, 1e-12, None)
        return out / out.sum()

    def predict_fixture(self, home_team: str, away_team: str) -> dict:
        home = str(home_team)
        away = str(away_team)
        lam_home, lam_away = self._expected_goals(home, away)
        probs = self.outcome_probabilities(home, away)
        top = self.scoreline_table(home, away, top_n=1).iloc[0]
        return {
            "home_team": home,
            "away_team": away,
            "lambda_home": lam_home,
            "lambda_away": lam_away,
            "prob_home": float(probs[0]),
            "prob_draw": float(probs[1]),
            "prob_away": float(probs[2]),
            "most_likely_score": str(top["scoreline"]),
            "most_likely_score_probability": float(top["probability"]),
            "home_known": bool(self.known.get(home, False)),
            "away_known": bool(self.known.get(away, False)),
            "home_matches": int(self.matches.get(home, 0)),
            "away_matches": int(self.matches.get(away, 0)),
            "rho": float(self.rho),
        }

    def predict_frame(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        required = {"home_team", "away_team"}
        missing = required - set(fixtures.columns)
        if missing:
            raise ValueError(f"Fixture frame missing: {sorted(missing)}")
        rows = [
            self.predict_fixture(row.home_team, row.away_team)
            for row in fixtures.itertuples(index=False)
        ]
        return pd.DataFrame(rows)

    def team_ratings(self) -> pd.DataFrame:
        rows = []
        for team in sorted(set(self.attack) | set(self.defence)):
            rows.append({
                "team": team,
                "attack": float(self.attack.get(team, 1.0)),
                "defence": float(self.defence.get(team, 1.0)),
                "matches": int(self.matches.get(team, 0)),
                "effective_matches": float(self.effective_matches.get(team, 0.0)),
                "known": bool(self.known.get(team, False)),
                "home_base": self.home_base,
                "away_base": self.away_base,
                "rho": self.rho,
            })
        return pd.DataFrame(rows).sort_values(["attack", "team"], ascending=[False, True]).reset_index(drop=True)

    def summary(self) -> dict:
        return {
            "config": asdict(self.config),
            "as_of": str(self.as_of.date()) if self.as_of is not None else None,
            "n_training_matches": self.n_training_matches,
            "home_base": self.home_base,
            "away_base": self.away_base,
            "home_advantage_goals": self.home_base - self.away_base,
            "rho": self.rho,
            "known_teams": int(sum(bool(v) for v in self.known.values())),
            "teams": int(len(self.known)),
        }


def _fallback_poisson_probabilities(
    home_base: float = 1.35,
    away_base: float = 1.05,
    max_goals: int = 10,
) -> np.ndarray:
    matrix = np.outer(
        _poisson_pmf_vector(max_goals, home_base),
        _poisson_pmf_vector(max_goals, away_base),
    )
    matrix = matrix / matrix.sum()
    home = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away = np.triu(matrix, 1).sum()
    out = np.clip(np.array([home, draw, away], dtype=float), 1e-12, None)
    return out / out.sum()


def build_pre_match_poisson_features(
    raw: pd.DataFrame,
    config: PoissonConfig | None = None,
) -> pd.DataFrame:
    """Build pre-match Poisson features with calendar-month refits.

    Each month is predicted by a model fitted only on earlier matches. The
    first month therefore uses the league-average fallback, and no fixture is
    described using a model that has seen that fixture or a later one.
    """
    config = config or PoissonConfig()
    required = {
        "match_id", "date", "home_team", "away_team",
        "full_time_home_goals", "full_time_away_goals",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Poisson feature input missing: {sorted(missing)}")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values(["date", "match_id"]).reset_index(drop=True)
    periods = sorted(frame["date"].dt.to_period("M").unique())
    rows: list[dict] = []

    for period in periods:
        start = period.start_time.normalize()
        history = frame[frame["date"] < start]
        model = PoissonModel(config).fit(history, as_of=start - pd.Timedelta(days=1))
        month = frame[frame["date"].dt.to_period("M") == period]
        for row in month.itertuples(index=False):
            home = str(row.home_team)
            away = str(row.away_team)
            if model.n_training_matches == 0:
                independent = _fallback_poisson_probabilities(max_goals=config.max_goals)
                dc = independent.copy()
                lam_home, lam_away = 1.35, 1.05
                known_home = known_away = False
            else:
                lam_home, lam_away = model._expected_goals(home, away)
                independent = model.outcome_probabilities(
                    home, away, use_dixon_coles=False
                )
                dc = model.outcome_probabilities(home, away, use_dixon_coles=True)
                known_home = bool(model.known.get(home, False))
                known_away = bool(model.known.get(away, False))
            rows.append({
                "match_id": str(row.match_id),
                "poisson_home_xg": float(lam_home),
                "poisson_away_xg": float(lam_away),
                "poisson_prob_home": float(independent[0]),
                "poisson_prob_draw": float(independent[1]),
                "poisson_prob_away": float(independent[2]),
                "dc_prob_home": float(dc[0]),
                "dc_prob_draw": float(dc[1]),
                "dc_prob_away": float(dc[2]),
                "poisson_home_known": int(known_home),
                "poisson_away_known": int(known_away),
            })

    return pd.DataFrame(rows).set_index("match_id")[POISSON_FEATURE_COLUMNS]
