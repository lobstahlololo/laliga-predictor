"""Post-match settlement and financial performance audit for the ledger.

Opens ``data/processed/prediction_history.csv``, finds every row whose
``actual_outcome`` is still blank, looks the fixture up in
``data/raw/laliga_matches.csv`` (case-insensitive, accent-insensitive team
matching reusing ``predict_upcoming.resolve_team``), and backfills the real
result (0 = Away Win, 1 = Draw, 2 = Home Win) once the match has been played.

It then runs a flat-stake Financial Performance Audit over all settled rows:
every outcome flagged as a value bet by ``predict_upcoming.value_bet_outcomes``
(model probability beating the market's implied probability by more than
``VALUE_BET_THRESHOLD``) is assumed to carry a $10 stake:

    win  -> profit = stake * decimal_odds - stake
    loss -> profit = -stake

Decimal odds are read from the settled fixture's historical odds in the raw
dataset (``odds_home`` / ``odds_draw`` / ``odds_away``), falling back to the
odds implied by the stored market probability (``1 / market prob``) when the
raw row or its odds are unavailable.

Summary dashboard: total value bets placed, win rate %, total profit/loss and
overall yield % (net profit / total capital risked).

The ledger is only rewritten when new results were backfilled; missing files,
unplayed fixtures and unresolvable team names are handled gracefully.

Usage:
    python backfill_results.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from predict_upcoming import (
    LEDGER_COLUMNS,
    LEDGER_PATH,
    VALUE_BET_THRESHOLD,
    resolve_team,
    value_bet_outcomes,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "laliga_matches.csv"

# Flat-stake strategy: risk this amount on every flagged value bet.
STAKE = 10.0

# value_bet_outcomes() outcome names -> numerical match_outcome codes
# (mirrors FTR_TO_OUTCOME in make_dataset: H=2, D=1, A=0).
OUTCOME_TO_CODE = {"away": 0, "draw": 1, "home": 2}

# Fallback mapping if a raw row only carries the raw FTR letter.
FTR_TO_OUTCOME = {"H": 2, "D": 1, "A": 0}

MAX_DETAIL_LINES = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ledger() -> pd.DataFrame | None:
    """Load the audit ledger as raw strings so values round-trip unchanged.

    Reading with ``dtype=str`` / ``keep_default_na=False`` keeps blank
    ``actual_outcome`` cells as empty strings and prevents float
    re-formatting when the file is written back.
    """
    if not LEDGER_PATH.exists():
        logger.warning(
            "Ledger %s not found - run predict_upcoming.py first.", LEDGER_PATH
        )
        return None
    ledger = pd.read_csv(LEDGER_PATH, dtype=str, keep_default_na=False)
    for col in LEDGER_COLUMNS:
        if col not in ledger.columns:
            logger.warning("Ledger missing column %r - adding it blank.", col)
            ledger[col] = ""
    return ledger[LEDGER_COLUMNS]


def load_raw() -> pd.DataFrame | None:
    """Load the raw match dataset (or None if it has not been fetched yet)."""
    if not RAW_PATH.exists():
        logger.warning(
            "Raw results %s not found - run src/data/make_dataset.py first.",
            RAW_PATH,
        )
        return None
    try:
        return pd.read_csv(RAW_PATH, parse_dates=["date"])
    except Exception as exc:  # noqa: BLE001 - degrade, never crash the audit
        logger.warning("Could not read raw results: %s", exc)
        return None


def _as_float(value: object) -> float | None:
    """Parse a ledger cell to float; None for blanks / garbage."""
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None  # reject NaN


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

def settle_pending(
    ledger: pd.DataFrame, raw: pd.DataFrame | None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Backfill blank ``actual_outcome`` cells from played matches.

    A ledger row is settled only when the same fixture (same home/away
    pairing) appears in the raw data with a match date on or after
    ``date_predicted`` - earlier season meetings are never used, so an
    upcoming fixture stays pending until it is actually played.

    Returns the updated ledger plus counters: settled / pending /
    unresolved. Never raises on missing rows or unknown teams.
    """
    stats = {"settled": 0, "pending": 0, "unresolved": 0}

    blank = ledger["actual_outcome"].astype(str).str.strip() == ""
    n_blank = int(blank.sum())
    if n_blank == 0:
        return ledger, stats

    if raw is None or raw.empty:
        stats["pending"] = n_blank
        return ledger, stats

    available = sorted(set(raw["home_team"].astype(str)) | set(raw["away_team"].astype(str)))
    today = pd.Timestamp.today().normalize()

    for idx in ledger.index[blank]:
        home_q = str(ledger.at[idx, "home_team"]).strip()
        away_q = str(ledger.at[idx, "away_team"]).strip()

        # Case/accent-insensitive resolution with alias table (shared with
        # the inference dashboard).
        try:
            home = resolve_team(home_q, available)
            away = resolve_team(away_q, available)
        except Exception as exc:  # noqa: BLE001 - TeamNotFoundError etc.
            logger.warning(
                "Cannot resolve fixture %r vs %r: %s", home_q, away_q, exc
            )
            stats["unresolved"] += 1
            continue

        candidates = raw[
            (raw["home_team"].astype(str) == home)
            & (raw["away_team"].astype(str) == away)
        ]

        # Only matches played on/after the prediction date - and on or
        # before today - count; a future-dated row is not played yet.
        pred = pd.to_datetime(
            str(ledger.at[idx, "date_predicted"]).strip(), errors="coerce"
        )
        if pd.isna(pred):
            logger.warning(
                "Unparseable date_predicted %r for %s vs %s - left pending.",
                ledger.at[idx, "date_predicted"], home, away,
            )
            stats["pending"] += 1
            continue
        dates = pd.to_datetime(candidates["date"], errors="coerce")
        candidates = candidates[(dates >= pred) & (dates <= today)]

        if candidates.empty:
            # Fixture pairing known but not yet played (or not yet in the
            # refreshed raw data) - graceful, stays pending.
            stats["pending"] += 1
            continue

        row = candidates.sort_values("date").iloc[0]
        outcome = row.get("match_outcome")
        if pd.isna(outcome):
            outcome = FTR_TO_OUTCOME.get(str(row.get("full_time_result", "")).strip().upper())
        if outcome is None or pd.isna(outcome):
            logger.warning(
                "Played match %s vs %s has no usable result - left pending.",
                home, away,
            )
            stats["pending"] += 1
            continue

        ledger.at[idx, "actual_outcome"] = str(int(outcome))
        stats["settled"] += 1
        logger.info(
            "Settled %s vs %s (played %s) -> %s",
            home, away, row.get("date"), int(outcome),
        )

    return ledger, stats


# ---------------------------------------------------------------------------
# Financial performance audit
# ---------------------------------------------------------------------------

def _historical_odds(
    pred: pd.Timestamp | None, home_q: str, away_q: str, raw: pd.DataFrame | None
) -> dict[str, float] | None:
    """Decimal odds (home/draw/away) of the played fixture in the raw data.

    Reuses the same case/accent-insensitive resolution and played-window
    filters as ``settle_pending`` so payouts price exactly the match that
    settled the ledger row. Returns None when the fixture or its odds are
    unavailable; the caller then falls back to odds implied by the ledger's
    market probability.
    """
    if raw is None or raw.empty or pred is None or pd.isna(pred):
        return None
    available = sorted(
        set(raw["home_team"].astype(str)) | set(raw["away_team"].astype(str))
    )
    try:
        home = resolve_team(home_q, available)
        away = resolve_team(away_q, available)
    except Exception as exc:  # noqa: BLE001 - fall back to raw name matching
        logger.debug("Odds lookup for %r vs %r: %s", home_q, away_q, exc)
        home, away = home_q, away_q

    candidates = raw[
        (raw["home_team"].astype(str) == home)
        & (raw["away_team"].astype(str) == away)
    ]
    dates = pd.to_datetime(candidates["date"], errors="coerce")
    candidates = candidates[
        (dates >= pred) & (dates <= pd.Timestamp.today().normalize())
    ]
    if candidates.empty:
        return None

    match = candidates.sort_values("date").iloc[0]
    odds: dict[str, float] = {}
    for key, col in (("home", "odds_home"), ("draw", "odds_draw"), ("away", "odds_away")):
        if col in match.index:
            value = _as_float(match[col])
            if value is not None and value > 1.0:
                odds[key] = value
    return odds or None


def audit_ledger(
    ledger: pd.DataFrame, raw: pd.DataFrame | None = None
) -> dict:
    """Flat-stake audit of every settled ledger row.

    Value bets are re-derived with the exact ``value_bet_outcomes`` logic
    used by the dashboard, so the audit always matches what was displayed
    at prediction time. Each flagged outcome is one $10 bet; payouts use the
    match's historical decimal odds from the raw dataset, falling back to
    odds implied by the stored market probability (odds = 1 / prob).
    """
    res: dict = {
        "bets": 0, "wins": 0, "losses": 0,
        "staked": 0.0, "pnl": 0.0,
        "settled_rows": 0, "pending_rows": 0,
        "detail": [],
    }

    for _, row in ledger.iterrows():
        actual_raw = str(row.get("actual_outcome", "")).strip()
        if not actual_raw:
            res["pending_rows"] += 1
            continue
        try:
            actual = int(float(actual_raw))
        except ValueError:
            logger.warning(
                "Unparseable actual_outcome %r for %s vs %s - treated as pending.",
                actual_raw, row.get("home_team"), row.get("away_team"),
            )
            res["pending_rows"] += 1
            continue
        res["settled_rows"] += 1

        probs = {
            key: _as_float(row.get(f"model_prob_{key}"))
            for key in ("home", "draw", "away")
        }
        market = {
            f"implied_prob_{key}": _as_float(row.get(f"market_prob_{key}"))
            for key in ("home", "draw", "away")
        }
        if any(v is None for v in probs.values()) or any(
            v is None for v in market.values()
        ):
            logger.warning(
                "Skipping %s vs %s: missing/invalid probabilities in ledger.",
                row.get("home_team"), row.get("away_team"),
            )
            continue

        pred = pd.to_datetime(
            str(row.get("date_predicted", "")).strip(), errors="coerce"
        )
        hist_odds = _historical_odds(
            pred if pd.notna(pred) else None,
            str(row.get("home_team", "")).strip(),
            str(row.get("away_team", "")).strip(),
            raw,
        )

        hits = value_bet_outcomes(probs, market)  # type: ignore[arg-type]
        for outcome, _model_p, market_p, _gap in hits:
            if market_p <= 0:
                logger.warning(
                    "Skipping unpriceable market probability for %s vs %s (%s).",
                    row.get("home_team"), row.get("away_team"), outcome,
                )
                continue
            odds = (hist_odds or {}).get(outcome)
            if odds is None:
                odds = 1.0 / market_p
            won = OUTCOME_TO_CODE[outcome] == actual
            pnl = (STAKE * odds - STAKE) if won else -STAKE

            res["bets"] += 1
            res["wins" if won else "losses"] += 1
            res["staked"] += STAKE
            res["pnl"] += pnl
            res["detail"].append(
                {
                    "date": str(row.get("date_predicted", "")),
                    "fixture": f"{row.get('home_team')} vs {row.get('away_team')}",
                    "outcome": outcome.capitalize(),
                    "odds": odds,
                    "won": won,
                    "pnl": pnl,
                }
            )
    return res


def print_audit(res: dict) -> None:
    """Print the Financial Performance Audit dashboard."""
    width = 68
    bets, wins, losses = res["bets"], res["wins"], res["losses"]
    staked, pnl = res["staked"], res["pnl"]
    win_rate = f"{100.0 * wins / bets:.1f}%" if bets else None
    yield_pct = f"{100.0 * pnl / staked:+.1f}%" if staked else None

    print()
    print("=" * width)
    print(f"  FINANCIAL PERFORMANCE AUDIT - FLAT STAKE ${STAKE:,.2f} / VALUE BET")
    print(
        f"  Value bet = model prob > market implied + "
        f"{VALUE_BET_THRESHOLD * 100:.1f}pp  "
        f"(payout @ historical odds, fallback 1 / market prob)"
    )
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
                f"    {bet['date']}  {bet['fixture']:<32}"
                f" BET {bet['outcome']:<5} @ {bet['odds']:.2f}"
                f"  {mark}  {sign}${abs(bet['pnl']):.2f}"
            )
        extra = len(res["detail"]) - MAX_DETAIL_LINES
        if extra > 0:
            print(f"    ... and {extra} more")
    else:
        print("  No settled value bets yet - predictions will settle as")
        print("  fixtures are played and the raw data is refreshed.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - console may not support reconfiguration
        pass

    ledger = load_ledger()
    if ledger is None:
        print(f"No ledger found at {LEDGER_PATH}. Run predict_upcoming.py first.")
        return 0

    raw = load_raw()
    if raw is None:
        logger.warning(
            "Skipping settlement (raw results unavailable); auditing settled rows only."
        )

    ledger, stats = settle_pending(ledger, raw)
    logger.info(
        "Settlement: %d backfilled, %d awaiting kickoff, %d unresolved team name(s).",
        stats["settled"], stats["pending"], stats["unresolved"],
    )

    if stats["settled"]:
        ledger.to_csv(LEDGER_PATH, index=False)
        logger.info("Saved updated ledger to %s", LEDGER_PATH)

    print_audit(audit_ledger(ledger, raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
