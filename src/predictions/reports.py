"""Generate persistent performance reports from the prediction ledger.

Writes CSV tables plus a Markdown summary to ``reports/``. The dashboard uses
the same functions, so the files and UI cannot disagree.

Usage:
    python -m src.predictions.reports
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictions import history as hist  # noqa: E402
from src.utils.config import LEDGER_PATH, REPORTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)


def _safe_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free Markdown table renderer."""
    if df.empty:
        return "No settled data."
    columns = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_reports(history: pd.DataFrame) -> dict:
    """Build all report tables. Returns a dict of ``name -> DataFrame``."""
    summary = hist.performance_summary(history)
    return {
        "summary": summary,
        "by_season": hist.performance_by(history, "season"),
        "by_team": hist.performance_by(history, "team"),
        "by_outcome": hist.performance_by(history, "outcome"),
        "by_model": hist.performance_by(history, "model"),
        "over_time": hist.performance_over_time(history),
    }


def write_reports(history: pd.DataFrame, out_dir: Path = REPORTS_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = build_reports(history)
    written: list[Path] = []

    names = {
        "by_season": "performance_by_season.csv",
        "by_team": "performance_by_team.csv",
        "by_outcome": "performance_by_outcome.csv",
        "by_model": "performance_by_model_version.csv",
        "over_time": "performance_over_time.csv",
    }
    for key, filename in names.items():
        path = out_dir / filename
        _safe_table(tables[key], path)
        written.append(path)

    summary_path = out_dir / "overall_performance.json"
    summary_path.write_text(
        json.dumps(tables["summary"], indent=2, default=str), encoding="utf-8"
    )
    written.append(summary_path)

    summary = tables["summary"]
    def fmt_pct(value):
        try:
            return "n/a" if pd.isna(value) else f"{100 * float(value):.1f}%"
        except (TypeError, ValueError):
            return "n/a"

    lines = [
        "# Prediction Ledger Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Ledger: `{LEDGER_PATH}`",
        "",
        "## Overall",
        "",
        f"- Predictions: **{summary['predictions']}**",
        f"- Settled: **{summary['settled']}**",
        f"- Pending: **{summary['pending']}**",
        f"- Accuracy: **{fmt_pct(summary['accuracy'])}**",
        f"- Log-loss: **{summary['log_loss']:.4f}**" if pd.notna(summary["log_loss"]) else "- Log-loss: n/a",
        f"- Brier score: **{summary['brier_score']:.4f}**" if pd.notna(summary["brier_score"]) else "- Brier score: n/a",
        f"- Draw recall: **{fmt_pct(summary['draw_recall'])}**",
        "",
        "Accuracy is reported only for settled rows. Pending rows are not scored.",
        "",
    ]
    for key, title in (
        ("by_season", "By season"),
        ("by_outcome", "By predicted outcome"),
        ("by_model", "By model version"),
    ):
        table = tables[key]
        lines.extend([f"## {title}", ""])
        lines.append(_markdown_table(table))
        lines.append("")
    (out_dir / "prediction_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    written.append(out_dir / "prediction_report.md")
    return written


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    history, _ = hist.load_history()
    if history.empty:
        print(f"No ledger at {LEDGER_PATH}. Run predict_upcoming.py first.")
        return 0
    written = write_reports(history)
    print(f"Wrote {len(written)} report files to {REPORTS_DIR}:")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
