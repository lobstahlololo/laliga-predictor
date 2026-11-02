"""Production inference dashboard for upcoming La Liga fixtures.

Loads the serialized best model (``models/saved_models/best_model.pkl``) and
the fitted StandardScaler (``data/processed/feature_scaler.joblib``), reads a
list of upcoming fixtures, engineers the current feature vector (rolling
form, momentum, goal differentials, rest advantage, market probabilities)
for each fixture, prints a formatted probabilistic forecast with value-bet
detection, and appends an audit row per fixture to
``data/processed/prediction_history.csv``.

Fixture input (first match wins):

1. A file passed as argv[1], or ``fixtures.txt`` in the project root, with one
   fixture per line::

       Real Madrid vs Valencia
       Barcelona vs Sevilla
       Girona vs Betis | 2.10 3.40 3.60     # optional decimal odds

   Lines starting with ``#`` are comments. Optional trailing odds (``h d a``)
   replace the neutral market default for that fixture.

2. If no file exists and stdin is interactive, fixtures are prompted manually
   (empty line finishes input).

Team names are matched case-insensitively with accent folding, an alias table
(Atletico Madrid -> Ath Madrid, ...), substring matching and fuzzy
suggestions; unknown teams are reported with close matches and skipped.

Usage:
    python predict_upcoming.py [fixtures_file]
"""

from __future__ import annotations

import difflib
import re
import sys
import unicodedata
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the dashboard's
# box-drawing glyphs; force UTF-8 for this process.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "laliga_matches.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "saved_models" / "best_model.pkl"
SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "feature_scaler.joblib"
HEADER_PATH = PROJECT_ROOT / "data" / "processed" / "X_train_scaled.csv"
DEFAULT_FIXTURES_PATH = PROJECT_ROOT / "fixtures.txt"

# Import the shared history builder so inference rolling metrics are
# byte-for-byte identical to training features.
from src.features.build_features import (  # noqa: E402
    ROLLING_WINDOW,
    SHORT_WINDOW,
    build_team_history,
    rest_days_as_of,
    short_rest_flag,
)

# outcome code -> display label (2 = Home Win, 1 = Draw, 0 = Away Win)
OUTCOME_LABELS = {2: "home", 1: "draw", 0: "away"}

# Team aliases: common full names -> Football-Data short names.
TEAM_ALIASES = {
    "atletico madrid": "Ath Madrid",
    "atletico": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao",
    "athletic club": "Ath Bilbao",
    "athletic": "Ath Bilbao",
    "real sociedad": "Sociedad",
    "rayo vallecano": "Vallecano",
    "rayo": "Vallecano",
    "espanyol": "Espanol",
    "celta vigo": "Celta",
    "real betis": "Betis",
    "deportivo alaves": "Alaves",
    "real valladolid": "Valladolid",
    "las palmas": "Las Palmas",
    "ud las palmas": "Las Palmas",
}

# Last-resort neutral market split (implied probability = 1 / typical odds),
# used when the raw data has no odds to average and none were supplied.
FALLBACK_MARKET = {
    "implied_prob_home": 1 / 2.20,
    "implied_prob_draw": 1 / 3.40,
    "implied_prob_away": 1 / 3.80,
}

# Value bet: model probability must exceed market implied probability by
# more than 5.0 absolute percentage points.
VALUE_BET_THRESHOLD = 0.05

# Audit ledger: one appended row per predicted fixture, blank actual_outcome
# for downstream result tracking.
LEDGER_PATH = PROJECT_ROOT / "data" / "processed" / "prediction_history.csv"
LEDGER_COLUMNS = [
    "date_predicted",
    "home_team",
    "away_team",
    "model_prob_home",
    "model_prob_draw",
    "model_prob_away",
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",
    "actual_outcome",
]


class TeamNotFoundError(ValueError):
    """Raised when a fixture team cannot be resolved to a dataset team."""


# ---------------------------------------------------------------------------
# Loading production artifacts
# ---------------------------------------------------------------------------

def load_artifacts():
    """Load model, scaler and the exact training feature headers."""
    for path, hint in (
        (MODEL_PATH, "Run `python src/models/train_model.py` first."),
        (SCALER_PATH, "Run `python src/features/build_features.py` first."),
        (HEADER_PATH, "Run `python src/features/build_features.py` first."),
        (RAW_PATH, "Run `python src/data/make_dataset.py` first."),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. {hint}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_headers = list(pd.read_csv(HEADER_PATH, nrows=0).columns)
    return model, scaler, feature_headers


def load_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw dataset and its long-format team history."""
    raw = pd.read_csv(RAW_PATH, parse_dates=["date"])
    history = build_team_history(raw)
    return raw, history


# ---------------------------------------------------------------------------
# Fixture parsing and team resolution
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Casefold, strip accents and collapse whitespace."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", folded).strip().lower()


def resolve_team(query: str, available: list[str]) -> str:
    """Resolve a user-typed team name to its dataset spelling.

    Order: exact (accent/case-insensitive) -> alias table -> substring ->
    fuzzy suggestions. Raises TeamNotFoundError with suggestions on failure.
    """
    q = _normalize(query)
    if not q:
        raise TeamNotFoundError("Empty team name.")

    avail_norm = {_normalize(t): t for t in available}

    if q in avail_norm:
        return avail_norm[q]
    if q in TEAM_ALIASES:
        return TEAM_ALIASES[q]

    # Substring match (e.g. "atletico" -> "Ath Madrid" handled by alias,
    # "sevill" -> "Sevilla" via containment).
    contains = [orig for norm, orig in avail_norm.items() if q in norm or norm in q]
    if len(contains) == 1:
        return contains[0]

    suggestions = difflib.get_close_matches(q, list(avail_norm), n=3, cutoff=0.5)
    pretty = [avail_norm[s] for s in suggestions]
    hint = f" Did you mean: {', '.join(pretty)}?" if pretty else ""
    available_list = ", ".join(sorted(available))
    raise TeamNotFoundError(
        f"Unknown team '{query}'.{hint} Available teams: {available_list}"
    )


def parse_fixture_line(line: str) -> tuple[str, str, tuple[float, float, float] | None]:
    """Parse ``Home vs Away [| h d a]`` into its parts."""
    odds = None
    body = line
    if "|" in line:
        body, odds_part = line.split("|", 1)
        try:
            values = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", odds_part)]
            if len(values) == 3 and all(v > 1.0 for v in values):
                odds = tuple(values)  # type: ignore[assignment]
        except ValueError:
            odds = None

    if " vs " not in body.lower():
        raise ValueError(
            f"Malformed fixture line (expected 'Home vs Away'): {line!r}"
        )
    home, away = re.split(r"\s+vs\s+", body, maxsplit=1, flags=re.IGNORECASE)
    return home.strip(), away.strip(), odds


def read_fixtures(argv: list[str]) -> list[str]:
    """Collect fixture lines from argv file, fixtures.txt, or stdin prompt."""
    candidate = Path(argv[1]) if len(argv) > 1 else DEFAULT_FIXTURES_PATH
    if candidate.exists():
        lines = candidate.read_text(encoding="utf-8").splitlines()
        fixtures = [
            ln.strip() for ln in lines
            if ln.strip() and not ln.strip().startswith("#")
        ]
        print(f"Loaded {len(fixtures)} fixture(s) from {candidate}")
        return fixtures

    if not sys.stdin.isatty():
        raise FileNotFoundError(
            f"No fixture file at {candidate} and stdin is not interactive. "
            "Create fixtures.txt (one 'Home vs Away' per line) or pass a path."
        )

    print("Enter upcoming fixtures (empty line to finish):")
    fixtures = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        fixtures.append(line)
    return fixtures


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def team_form(history: pd.DataFrame, team: str) -> dict[str, float]:
    """Rolling form for a team over their most recent matches, *including*
    their latest played match (the upcoming fixture comes after it).

    Mirrors training semantics: the feature row for the next match summarizes
    the ROLLING_WINDOW / SHORT_WINDOW matches played up to and including the
    most recent row for this team.
    """
    rows = history[history["team"] == team].sort_values("date")
    if rows.empty:
        raise TeamNotFoundError(f"No historical matches found for '{team}'.")

    last5 = rows.tail(ROLLING_WINDOW)
    last3 = rows.tail(SHORT_WINDOW)
    form = {
        "form_goals_scored": float(last5["goals_scored"].mean()),
        "form_goals_conceded": float(last5["goals_conceded"].mean()),
        "form_total_shots": float(last5["total_shots"].mean()),
        "form_shots_on_target": float(last5["shots_on_target"].mean()),
        "short_goals_scored": float(last3["goals_scored"].mean()),
        "short_goals_conceded": float(last3["goals_conceded"].mean()),
    }
    # Explicit 5-match goal differential (same definition as training:
    # rolling means of scored/conceded subtracted).
    form["goal_diff_5m"] = (
        form["form_goals_scored"] - form["form_goals_conceded"]
    )
    if any(np.isnan(v) for v in form.values()):
        raise TeamNotFoundError(
            f"Incomplete statistics for '{team}' "
            f"(last match: {rows['date'].iloc[-1]:%Y-%m-%d})."
        )
    return form


def neutral_market(raw: pd.DataFrame) -> dict[str, float]:
    """Average implied probabilities from the raw data, else a neutral default."""
    probs = {}
    for col in FALLBACK_MARKET:
        if col in raw.columns and raw[col].notna().any():
            value = float(raw[col].mean())
            if not np.isnan(value):
                probs[col] = value
    for col, fallback in FALLBACK_MARKET.items():
        probs.setdefault(col, fallback)
    return probs


def build_feature_row(
    home_form: dict[str, float],
    away_form: dict[str, float],
    market: dict[str, float],
    headers: list[str],
    short_rest: int = 0,
) -> pd.DataFrame:
    """Assemble one feature row in the exact training column order."""
    row: dict[str, float] = {}
    for col in headers:
        if col == "home_advantage":
            row[col] = 1.0
        elif col == "short_rest_advantage":
            row[col] = float(short_rest)
        elif col.startswith("implied_prob_"):
            row[col] = market[col]
        elif col.startswith("home_"):
            row[col] = home_form[col[len("home_"):]]
        elif col.startswith("away_"):
            row[col] = away_form[col[len("away_"):]]
        else:
            raise KeyError(f"Unrecognized feature column in header: {col!r}")
    return pd.DataFrame([row], columns=headers)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def value_bet_outcomes(
    probs: dict[str, float], market: dict[str, float]
) -> list[tuple[str, float, float, float]]:
    """Outcomes where the model beats market implied probability by more than
    VALUE_BET_THRESHOLD (absolute probability points).

    Returns (outcome, model_prob, market_prob, gap) tuples.
    """
    hits = []
    for outcome in ("home", "draw", "away"):
        model_p = probs.get(outcome, 0.0)
        market_p = market.get(f"implied_prob_{outcome}", 0.0)
        gap = model_p - market_p
        if gap > VALUE_BET_THRESHOLD:
            hits.append((outcome, model_p, market_p, gap))
    return hits


def print_dashboard(
    home: str,
    away: str,
    probs: dict[str, float],
    market_note: str,
    market: dict[str, float],
) -> None:
    """Pretty-print the probabilistic forecast for one fixture."""
    home_pct = 100 * probs.get("home", 0.0)
    draw_pct = 100 * probs.get("draw", 0.0)
    away_pct = 100 * probs.get("away", 0.0)

    width = 68
    print("┌" + "─" * width + "┐")
    fixture = f"{home}  vs  {away}"
    print(f"│ {fixture.upper():<{width}} │")
    print("├" + "─" * width + "┤")
    line = f"  {home} Win: {home_pct:.1f}%  |  Draw: {draw_pct:.1f}%  |  {away} Win: {away_pct:.1f}%"
    print(f"│{line:<{width + 1}}│")

    # Stacked probability bar (50 chars).
    bar_len = 50
    h_blocks = round(home_pct / 100 * bar_len)
    d_blocks = round(draw_pct / 100 * bar_len)
    a_blocks = max(bar_len - h_blocks - d_blocks, 0)
    bar = "█" * h_blocks + "▒" * d_blocks + "░" * a_blocks
    print(f"│  [{bar}]  {'':<{max(width - bar_len - 6, 0)}}│")

    legend = f"  {home[:18]}: {home_pct:.1f}%   Draw: {draw_pct:.1f}%   {away[:18]}: {away_pct:.1f}%"
    print(f"│{legend:<{width + 1}}│")
    print(f"│  Market: {market_note:<{width - 10}} │")

    # Value bet warning: model probability beats market by > 5.0pp.
    labels = {"home": f"{home} Win", "draw": "Draw", "away": f"{away} Win"}
    for outcome, model_p, market_p, gap in value_bet_outcomes(probs, market):
        vb = (
            f"  ⭐ [VALUE BET DETECTED] {labels[outcome]}: "
            f"model {100 * model_p:.1f}% vs market {100 * market_p:.1f}% "
            f"(+{100 * gap:.1f}pp)"
        )
        print(f"│{vb:<{width + 1}}│")
    print("└" + "─" * width + "┘")


def append_ledger(rows: list[dict]) -> None:
    """Append prediction rows to the audit ledger (header on first write)."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LEDGER_PATH.exists()
    frame = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    frame.to_csv(
        LEDGER_PATH,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
    )
    print(f"Logged {len(rows)} prediction(s) to {LEDGER_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    model, scaler, headers = load_artifacts()
    raw, history = load_history()
    available = sorted(set(raw["home_team"]) | set(raw["away_team"]))
    default_market = neutral_market(raw)

    fixtures = read_fixtures(argv)
    if not fixtures:
        print("No fixtures provided; nothing to predict.")
        return 1

    print()
    print("=" * 70)
    print("  LA LIGA MATCH OUTCOME PREDICTOR — UPCOMING FIXTURES")
    print(f"  Model: {type(model).__name__}  |  Features: {len(headers)}  |  "
          f"History rows: {len(raw)}")
    print("=" * 70)

    failures = 0
    # Rest-advantage is evaluated as of the prediction date (upcoming fixture
    # dates are not yet in the dataset).
    as_of = pd.Timestamp.today().normalize()
    date_predicted = as_of.strftime("%Y-%m-%d")
    # Batch all resolvable fixtures into one matrix for a single transform.
    rows, meta = [], []
    for line in fixtures:
        try:
            home_q, away_q, odds = parse_fixture_line(line)
            home = resolve_team(home_q, available)
            away = resolve_team(away_q, available)
            if home == away:
                raise TeamNotFoundError(
                    f"Home and away team resolve to the same team ('{home}')."
                )
            market = dict(default_market)
            note = "neutral default (avg market)"
            if odds is not None:
                market = {
                    "implied_prob_home": 1 / odds[0],
                    "implied_prob_draw": 1 / odds[1],
                    "implied_prob_away": 1 / odds[2],
                }
                note = f"given odds {odds[0]:.2f}/{odds[1]:.2f}/{odds[2]:.2f}"
            home_rest = rest_days_as_of(history, home, as_of)
            away_rest = rest_days_as_of(history, away, as_of)
            row = build_feature_row(
                team_form(history, home),
                team_form(history, away),
                market,
                headers,
                short_rest=short_rest_flag(home_rest, away_rest),
            )
            rows.append(row)
            meta.append((home, away, note, market))
        except (ValueError, KeyError, TeamNotFoundError) as exc:
            failures += 1
            print(f"\n  !! Skipped {line!r}: {exc}")

    if not rows:
        print("\nNo valid fixtures to predict.")
        return 1

    matrix = pd.concat(rows, ignore_index=True)
    # Scaler was fitted on raw values (no feature names) -> transform the
    # numpy array; the model was fitted with feature names -> hand it back a
    # DataFrame with the exact training headers.
    scaled = pd.DataFrame(
        scaler.transform(matrix.values), columns=headers
    )
    proba = model.predict_proba(scaled)
    class_index = {int(c): i for i, c in enumerate(model.classes_)}

    print()
    ledger_rows = []
    for (home, away, note, market), probs_row in zip(meta, proba):
        probs = {
            OUTCOME_LABELS[int(c)]: float(probs_row[class_index[int(c)]])
            for c in model.classes_
        }
        print_dashboard(home, away, probs, note, market)
        ledger_rows.append({
            "date_predicted": date_predicted,
            "home_team": home,
            "away_team": away,
            "model_prob_home": f"{probs['home']:.6f}",
            "model_prob_draw": f"{probs['draw']:.6f}",
            "model_prob_away": f"{probs['away']:.6f}",
            "market_prob_home": f"{market['implied_prob_home']:.6f}",
            "market_prob_draw": f"{market['implied_prob_draw']:.6f}",
            "market_prob_away": f"{market['implied_prob_away']:.6f}",
            "actual_outcome": "",
        })
    append_ledger(ledger_rows)

    if failures:
        print(f"\n{failures} fixture(s) skipped due to errors (see above).")
    print(
        "Note: rolling form includes each team's most recent match; market "
        "probabilities are implied (1/odds) from Bet365 when available."
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
