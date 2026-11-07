"""CLI for predicting upcoming La Liga fixtures.

Thin presentation wrapper around :mod:`src.predictions.predictor`; the
dashboard uses the same inference engine. Every valid prediction is appended
to the versioned, append-only history ledger.

Fixture input (first match wins):

1. A file passed as ``argv[1]`` or ``fixtures.txt``. Legacy format::

       Real Madrid vs Valencia
       Barcelona vs Sevilla | 1.55 4.10 6.50 | 2026-11-08

2. If neither exists and stdin is interactive, type fixtures manually.

Usage:
    python predict_upcoming.py [fixtures_file]
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictions.fixtures import (  # noqa: E402
    load_upcoming_fixtures,
    parse_fixture_line,
    read_fixture_file,
)
from src.predictions.history import (  # noqa: E402
    CANONICAL_COLUMNS,
    append_predictions,
)
from src.predictions.predictor import (  # noqa: E402
    OUTCOME_LABELS,
    load_artifacts,
    load_raw,
    predict_fixtures,
    predictions_to_records,
    value_bet_outcomes,
)
from src.utils.config import LEDGER_PATH, VALUE_BET_THRESHOLD  # noqa: E402
from src.utils.teams import TeamNotFoundError, resolve_team  # noqa: E402

# Backwards-compatible exports used by backfill_results.py and older tests.
LEDGER_COLUMNS = CANONICAL_COLUMNS
__all__ = [
    "LEDGER_COLUMNS", "LEDGER_PATH", "VALUE_BET_THRESHOLD",
    "OUTCOME_LABELS", "resolve_team", "TeamNotFoundError",
    "value_bet_outcomes", "main",
]


def print_dashboard(prediction: dict) -> None:
    """Pretty-print one probabilistic forecast."""
    home = prediction["home_team"]
    away = prediction["away_team"]
    probs = prediction["probs"]
    market = prediction["market"]
    home_pct, draw_pct, away_pct = (
        100 * probs["home"], 100 * probs["draw"], 100 * probs["away"]
    )
    width = 72
    print("┌" + "─" * width + "┐")
    date_note = prediction["match_date"] or "date TBD"
    header = f"{home}  vs  {away}   ({date_note})"
    print(f"│ {header:<{width}} │")
    print("├" + "─" * width + "┤")
    line = (
        f"  {home} Win: {home_pct:.1f}%  |  Draw: {draw_pct:.1f}%  |  "
        f"{away} Win: {away_pct:.1f}%"
    )
    print(f"│{line[:width + 1]:<{width + 1}}│")

    bar_len = 50
    h_blocks = round(home_pct / 100 * bar_len)
    d_blocks = round(draw_pct / 100 * bar_len)
    a_blocks = max(bar_len - h_blocks - d_blocks, 0)
    bar = "█" * h_blocks + "▒" * d_blocks + "░" * a_blocks
    print(f"│  [{bar}]  {'':<{max(width - bar_len - 6, 0)}}│")
    pick = (
        f"  Pick: {prediction['predicted_outcome']} "
        f"(confidence {100 * prediction['confidence']:.1f}%)"
    )
    print(f"│{pick:<{width + 1}}│")
    print(f"│  Market: {prediction['market_note']:<{width - 10}} │")

    labels = {"home": f"{home} Win", "draw": "Draw", "away": f"{away} Win"}
    for outcome, model_p, market_p, gap in prediction["value_bets"]:
        vb = (
            f"  ⭐ [VALUE BET] {labels[outcome]}: model {100 * model_p:.1f}% "
            f"vs market {100 * market_p:.1f}% (+{100 * gap:.1f}pp)"
        )
        print(f"│{vb[:width + 1]:<{width + 1}}│")
    print("└" + "─" * width + "┘")


def _interactive_fixtures() -> list[dict]:
    print("Enter upcoming fixtures (empty line to finish):")
    fixtures = []
    while True:
        try:
            line = input("  > ").strip()
        except EOFError:
            break
        if not line:
            break
        try:
            fixtures.append(parse_fixture_line(line))
        except ValueError as exc:
            print(f"  !! {exc}")
    return fixtures


def main(argv: list[str]) -> int:
    try:
        model, preprocessor, metadata = load_artifacts()
    except Exception as exc:  # noqa: BLE001 - actionable artifact error
        print(f"Cannot load model artifacts: {exc}")
        return 1
    raw = load_raw()

    if len(argv) > 1:
        fixtures = read_fixture_file(argv[1])
        source = argv[1]
    else:
        fixtures, source = load_upcoming_fixtures()
        if not fixtures and sys.stdin.isatty():
            fixtures = _interactive_fixtures()
            source = "stdin"

    if not fixtures:
        print(
            f"No fixtures found (looked at {source}). Add lines to fixtures.txt "
            "or pass a fixture file."
        )
        return 1

    print()
    print("=" * 74)
    print("  LA LIGA MATCH OUTCOME PREDICTOR — UPCOMING FIXTURES")
    print(
        f"  Model: {metadata.get('model_name', type(model).__name__)} "
        f"({metadata.get('model_version', 'unversioned')})  |  "
        f"Features: {len(preprocessor.feature_names)}  |  History rows: {len(raw)}"
    )
    print("=" * 74)
    print(f"  Fixture source: {source}")

    predictions, errors = predict_fixtures(
        fixtures, model=model, preprocessor=preprocessor, metadata=metadata, raw=raw
    )
    for error in errors:
        print(f"\n  !! Skipped {error['fixture']}: {error['error']}")

    if not predictions:
        print("\nNo valid fixtures to predict.")
        return 1

    for prediction in predictions:
        print_dashboard(prediction)
        if prediction.get("expected_home_goals") is not None:
            print(
                f"    Expected goals: {prediction['expected_home_goals']:.2f} - "
                f"{prediction['expected_away_goals']:.2f}  |  Most likely "
                f"scoreline: {prediction['most_likely_scoreline']} "
                f"({100 * prediction['scoreline_probability']:.1f}%)"
            )

    records = predictions_to_records(predictions)
    appended, _ = append_predictions(records)
    print(f"\nLogged {appended} new prediction(s) to {LEDGER_PATH}")
    if errors:
        print(f"{len(errors)} fixture(s) skipped due to errors.")
    print(
        "Note: features use only matches played before each kickoff; "
        "market probabilities are user odds or historical averages."
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
