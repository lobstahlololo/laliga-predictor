"""Post-match result reconciliation and flat-stake performance audit.

Settles pending predictions from ``data/processed/prediction_history.csv``
using only real results in ``data/raw/laliga_matches.csv``. Settled rows are
immutable; only blank result fields are filled. It then re-derives value bets
with the exact inference logic and reports a flat-stake P&L.

Usage:
    python backfill_results.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictions.history import (  # noqa: E402
    CANONICAL_COLUMNS,
    MARKET_COLUMNS,
    PROB_COLUMNS,
    load_history,
    normalize_history,
    reconcile_results,
    save_history,
    settled_only,
)
from src.predictions.predictor import load_raw, value_bet_outcomes  # noqa: E402
from src.utils.config import FLAT_STAKE, LEDGER_PATH  # noqa: E402
from src.utils.teams import canonical_team  # noqa: E402

STAKE = FLAT_STAKE
MAX_DETAIL_LINES = 30
OUTCOME_TO_CODE = {"away": 0, "draw": 1, "home": 2}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _as_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


# Backwards-compatible wrapper around the shared reconciliation logic.
def settle_pending(ledger: pd.DataFrame, raw: pd.DataFrame | None):
    return reconcile_results(normalize_history(ledger), raw)


def _historical_odds(row: pd.Series, raw: pd.DataFrame | None) -> dict[str, float] | None:
    """Find the played fixture's decimal odds in the raw dataset."""
    if raw is None or raw.empty:
        return None
    matches = raw
    fixture_id = str(row.get("fixture_id", "")).strip()
    if fixture_id and "match_id" in matches.columns:
        exact = matches[matches["match_id"].astype(str) == fixture_id]
    else:
        exact = matches.iloc[0:0]

    if exact.empty:
        home = canonical_team(row.get("home_team", ""))
        away = canonical_team(row.get("away_team", ""))
        exact = matches[
            (matches["home_team"].astype(str) == home)
            & (matches["away_team"].astype(str) == away)
        ]
        match_date = pd.to_datetime(row.get("match_date"), errors="coerce")
        if pd.notna(match_date):
            dates = pd.to_datetime(exact["date"], errors="coerce")
            exact = exact[dates.dt.normalize() == match_date.normalize()]
    if exact.empty:
        return None

    match = exact.sort_values("date").iloc[0]
    odds: dict[str, float] = {}
    for key, col in (("home", "odds_home"), ("draw", "odds_draw"), ("away", "odds_away")):
        value = _as_float(match.get(col))
        if value is not None and value > 1.0:
            odds[key] = value
    return odds or None


def audit_ledger(history: pd.DataFrame, raw: pd.DataFrame | None = None) -> dict:
    """Flat-stake audit of every settled prediction row."""
    res: dict = {
        "bets": 0, "wins": 0, "losses": 0,
        "staked": 0.0, "pnl": 0.0,
        "settled_rows": 0, "pending_rows": 0,
        "detail": [],
    }
    if history.empty:
        return res

    settled = settled_only(history)
    res["settled_rows"] = len(settled)
    res["pending_rows"] = len(history) - len(settled)

    for _, row in settled.iterrows():
        actual = _as_float(row.get("actual_outcome"))
        if actual is None:
            res["pending_rows"] += 1
            res["settled_rows"] -= 1
            continue
        probs = {key: _as_float(row.get(f"model_prob_{key}")) for key in ("home", "draw", "away")}
        market = {
            f"implied_prob_{key}": _as_float(row.get(f"market_prob_{key}"))
            for key in ("home", "draw", "away")
        }
        if any(v is None for v in probs.values()) or any(v is None for v in market.values()):
            logger.warning(
                "Skipping %s vs %s: missing probabilities.",
                row.get("home_team"), row.get("away_team"),
            )
            continue

        hist_odds = _historical_odds(row, raw)
        for outcome, _model_p, market_p, _gap in value_bet_outcomes(probs, market):
            if market_p <= 0:
                continue
            odds = (hist_odds or {}).get(outcome) or (1.0 / market_p)
            won = OUTCOME_TO_CODE[outcome] == int(actual)
            pnl = (STAKE * odds - STAKE) if won else -STAKE
            res["bets"] += 1
            res["wins" if won else "losses"] += 1
            res["staked"] += STAKE
            res["pnl"] += pnl
            res["detail"].append({
                "date": str(row.get("predicted_at", "")),
                "fixture": f"{row.get('home_team')} vs {row.get('away_team')}",
                "outcome": outcome.capitalize(),
                "odds": odds,
                "won": won,
                "pnl": pnl,
            })
    return res


def print_audit(res: dict) -> None:
    width = 68
    bets, wins, losses = res["bets"], res["wins"], res["losses"]
    staked, pnl = res["staked"], res["pnl"]
    win_rate = f"{100.0 * wins / bets:.1f}%" if bets else None
    yield_pct = f"{100.0 * pnl / staked:+.1f}%" if staked else None

    print()
    print("=" * width)
    print(f"  FINANCIAL PERFORMANCE AUDIT - FLAT STAKE ${STAKE:,.2f} / VALUE BET")
    print("=" * width)
    print(f"  Settled predictions:       {res['settled_rows']}")
    print(f"  Awaiting kickoff:          {res['pending_rows']}")
    print(f"  Total Value Bets Placed:   {bets}   ({wins}W / {losses}L)")
    print(f"  Win Rate:                  {win_rate or 'n/a - no settled bets yet'}")
    print(f"  Total Profit/Loss:         {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}")
    print(f"  Total Capital Risked:      ${staked:,.2f}")
    print(f"  Yield (Net / Capital):     {yield_pct or 'n/a'}")
    print("=" * width)
    if res["detail"]:
        print("  Settled bets:")
        for bet in res["detail"][:MAX_DETAIL_LINES]:
            sign = "+" if bet["pnl"] >= 0 else "-"
            mark = "WIN " if bet["won"] else "LOSS"
            print(
                f"    {bet['date'][:10]}  {bet['fixture']:<32}"
                f" BET {bet['outcome']:<5} @ {bet['odds']:.2f}"
                f"  {mark}  {sign}${abs(bet['pnl']):.2f}"
            )
        extra = len(res["detail"]) - MAX_DETAIL_LINES
        if extra > 0:
            print(f"    ... and {extra} more")
    else:
        print("  No settled value bets yet - results settle as fixtures are")
        print("  played and the raw dataset is refreshed.")
    print()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    history, changed = load_history()
    if history.empty:
        print(f"No ledger found at {LEDGER_PATH}. Run predict_upcoming.py first.")
        return 0

    try:
        raw = load_raw()
    except Exception as exc:  # noqa: BLE001 - audit can still run on settled rows
        logger.warning("Raw results unavailable (%s); auditing settled rows only.", exc)
        raw = None

    updated, stats = reconcile_results(history, raw)
    logger.info(
        "Settlement: %d newly settled, %d pending, %d already settled.",
        stats["settled"], stats["pending"], stats["already_settled"],
    )
    if stats["settled"] or changed:
        save_history(updated)
        logger.info("Saved updated ledger to %s", LEDGER_PATH)

    print_audit(audit_ledger(updated, raw))

    # Keep the persistent report files in sync with the ledger.
    try:
        from src.predictions.reports import write_reports

        written = write_reports(updated)
        logger.info("Wrote %d report file(s) to reports/.", len(written))
    except Exception as exc:  # noqa: BLE001 - reporting must not fail settlement
        logger.warning("Could not write reports: %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
