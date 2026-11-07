"""Shared visual components and styling for the dashboard."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

GREEN = "#1f7a4d"
GOLD = "#c9a227"
RED = "#b03a2e"
INK = "#16202a"
MUTED = "#6b7a87"
SURFACE = "#ffffff"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }

/* Hero */
.ll-hero {
    background: linear-gradient(120deg, #10241c 0%, #1f7a4d 55%, #2f9e67 100%);
    border-radius: 18px;
    padding: 28px 32px;
    color: #ffffff;
    margin-bottom: 22px;
    box-shadow: 0 12px 30px rgba(16, 36, 28, 0.18);
}
.ll-hero h1 { color: #ffffff; font-size: 2.1rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.02em; }
.ll-hero p { color: rgba(255,255,255,0.86); margin: 0; font-size: 1rem; max-width: 900px; }
.ll-badge {
    display: inline-block; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
    color: #fff; padding: 5px 11px; border-radius: 999px; font-size: 0.74rem;
    font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 12px;
}

/* Metric cards */
.ll-card {
    background: #ffffff; border: 1px solid #e4e9ee; border-radius: 14px;
    padding: 18px 20px; height: 100%;
    box-shadow: 0 2px 10px rgba(22, 32, 42, 0.04);
}
.ll-card .label { color: #6b7a87; font-size: 0.73rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
.ll-card .value { color: #16202a; font-size: 1.7rem; font-weight: 800; margin-top: 4px; letter-spacing: -0.02em; }
.ll-card .help { color: #6b7a87; font-size: 0.78rem; margin-top: 4px; }
.ll-card.green { border-top: 4px solid #1f7a4d; }
.ll-card.gold { border-top: 4px solid #c9a227; }
.ll-card.red { border-top: 4px solid #b03a2e; }
.ll-card.blue { border-top: 4px solid #2c5f8a; }

/* Section headings */
.ll-section { margin: 26px 0 10px 0; }
.ll-section h2 { font-size: 1.18rem; font-weight: 750; color: #16202a; margin: 0; letter-spacing: -0.01em; }
.ll-section p { color: #6b7a87; font-size: 0.86rem; margin: 3px 0 0 0; }

/* Probability bar */
.ll-prob-wrap { margin: 10px 0 6px 0; }
.ll-prob-bar { display: flex; height: 16px; border-radius: 999px; overflow: hidden; background: #eef1f4; }
.ll-prob-seg { height: 100%; }
.ll-prob-labels { display: flex; justify-content: space-between; margin-top: 8px; gap: 10px; }
.ll-prob-label { font-size: 0.82rem; color: #16202a; }
.ll-prob-label b { font-size: 0.95rem; }
.ll-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }

/* Fixture card */
.ll-fixture {
    background: #ffffff; border: 1px solid #e4e9ee; border-radius: 14px;
    padding: 18px 20px; margin-bottom: 14px;
}
.ll-fixture .teams { font-size: 1.05rem; font-weight: 750; color: #16202a; }
.ll-fixture .meta { color: #6b7a87; font-size: 0.8rem; margin-top: 2px; }
.ll-pick {
    display: inline-block; background: #e9f5ef; color: #1f7a4d; border: 1px solid #bfe3d1;
    border-radius: 999px; padding: 4px 11px; font-size: 0.78rem; font-weight: 700;
}
.ll-value {
    display: inline-block; background: #fdf7e3; color: #8a6d12; border: 1px solid #f0e0a8;
    border-radius: 8px; padding: 3px 9px; font-size: 0.75rem; font-weight: 600; margin: 3px 4px 0 0;
}
.ll-xg { color: #16202a; font-size: 0.84rem; margin: 8px 0 2px 0; }
.ll-xg b { font-weight: 750; }
.ll-scores { margin-top: 4px; }
.ll-score {
    display: inline-block; background: #eef4f8; color: #2c5f8a; border: 1px solid #cfe0ec;
    border-radius: 8px; padding: 2px 8px; font-size: 0.73rem; font-weight: 600;
    margin: 2px 4px 0 0;
}
.ll-level {
    display: inline-block; background: #f1eef8; color: #5a4b8a; border: 1px solid #d9d1ee;
    border-radius: 999px; padding: 2px 9px; font-size: 0.72rem; font-weight: 700;
    margin-top: 4px;
}
.ll-chip-ok { background:#e9f5ef;color:#1f7a4d;border:1px solid #bfe3d1;border-radius:999px;padding:2px 9px;font-size:0.74rem;font-weight:700; }
.ll-chip-warn { background:#fdf7e3;color:#8a6d12;border:1px solid #f0e0a8;border-radius:999px;padding:2px 9px;font-size:0.74rem;font-weight:700; }
.ll-chip-error { background:#fbeae8;color:#b03a2e;border:1px solid #f2c9c3;border-radius:999px;padding:2px 9px;font-size:0.74rem;font-weight:700; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10241c 0%, #163a2b 100%);
}
[data-testid="stSidebar"] * { color: #e8f1ec !important; }
[data-testid="stSidebar"] a { color: #d6eadf !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #b9cec2 !important; }

/* Tables */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.ll-note { color: #6b7a87; font-size: 0.8rem; line-height: 1.5; }
.ll-footer { margin-top: 36px; padding-top: 14px; border-top: 1px solid #e4e9ee; color: #8a98a4; font-size: 0.76rem; }
</style>
"""


def configure_page(title: str, icon: str = "⚽") -> None:
    st.set_page_config(
        page_title=f"{title} · La Liga Predictor",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, badge: str = "La Liga Analytics") -> None:
    st.markdown(
        f"""
        <div class="ll-hero">
            <div class="ll-badge">{html.escape(badge)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    caption_html = f"<p>{html.escape(caption)}</p>" if caption else ""
    st.markdown(
        f'<div class="ll-section"><h2>{html.escape(title)}</h2>{caption_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str = "", accent: str = "green") -> None:
    st.markdown(
        f"""
        <div class="ll-card {accent}">
            <div class="label">{html.escape(label)}</div>
            <div class="value">{html.escape(value)}</div>
            <div class="help">{html.escape(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(cards: list[dict]) -> None:
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            metric_card(
                card["label"], card["value"], card.get("help", ""),
                card.get("accent", "green"),
            )


def pct(value, digits: int = 1, missing: str = "n/a") -> str:
    try:
        if value is None or pd.isna(value):
            return missing
        return f"{100 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return missing


def num(value, digits: int = 3, missing: str = "n/a") -> str:
    try:
        if value is None or pd.isna(value):
            return missing
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return missing


def outcome_badge(label: str) -> str:
    color = {"Home Win": GREEN, "Draw": GOLD, "Away Win": RED}.get(label, MUTED)
    return (
        f'<span style="background:{color}18;color:{color};border:1px solid {color}55;'
        f'border-radius:999px;padding:2px 9px;font-size:0.74rem;font-weight:700;">'
        f"{html.escape(str(label))}</span>"
    )


def probability_bar(probs: dict, home: str, away: str) -> None:
    """Segmented 100% bar with legend, guaranteed to render as percentages."""
    home_p = max(float(probs.get("home", 0)), 0)
    draw_p = max(float(probs.get("draw", 0)), 0)
    away_p = max(float(probs.get("away", 0)), 0)
    total = home_p + draw_p + away_p or 1.0
    home_p, draw_p, away_p = home_p / total, draw_p / total, away_p / total
    st.markdown(
        f"""
        <div class="ll-prob-wrap">
            <div class="ll-prob-bar">
                <div class="ll-prob-seg" style="width:{home_p*100:.2f}%;background:{GREEN};"></div>
                <div class="ll-prob-seg" style="width:{draw_p*100:.2f}%;background:{GOLD};"></div>
                <div class="ll-prob-seg" style="width:{away_p*100:.2f}%;background:{RED};"></div>
            </div>
            <div class="ll-prob-labels">
                <span class="ll-prob-label"><span class="ll-dot" style="background:{GREEN}"></span>{html.escape(home)} <b>{home_p*100:.1f}%</b></span>
                <span class="ll-prob-label"><span class="ll-dot" style="background:{GOLD}"></span>Draw <b>{draw_p*100:.1f}%</b></span>
                <span class="ll-prob-label"><span class="ll-dot" style="background:{RED}"></span>{html.escape(away)} <b>{away_p*100:.1f}%</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fixture_card(prediction: dict, home_form: str = "", away_form: str = "") -> None:
    probs = prediction["probs"]
    value_bets = prediction.get("value_bets", [])
    values_html = "".join(
        f'<span class="ll-value">⭐ {html.escape(outcome.title())} edge '
        f"+{gap*100:.1f}pp</span>"
        for outcome, _model_p, _market_p, gap in value_bets
    )
    form_html = ""
    if home_form or away_form:
        form_html = (
            f'<div class="meta" style="margin-top:8px;">Recent form — '
            f"{html.escape(prediction['home_team'])}: <b>{html.escape(home_form or 'n/a')}</b> &nbsp;·&nbsp; "
            f"{html.escape(prediction['away_team'])}: <b>{html.escape(away_form or 'n/a')}</b></div>"
        )
    st.markdown(
        f"""
        <div class="ll-fixture">
            <div class="teams">{html.escape(prediction['home_team'])} <span style="color:#8a98a4;font-weight:500;">vs</span> {html.escape(prediction['away_team'])}</div>
            <div class="meta">{html.escape(prediction.get('match_date') or 'Date TBD')} &nbsp;·&nbsp; fixture {html.escape(prediction.get('fixture_id',''))} &nbsp;·&nbsp; {html.escape(prediction.get('market_note',''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    probability_bar(probs, prediction["home_team"], prediction["away_team"])
    if prediction.get("expected_home_goals") is not None:
        scores_html = "".join(
            f'<span class="ll-score">{html.escape(str(item["scoreline"]))} · '
            f'{100 * float(item["probability"]):.1f}%</span>'
            for item in (prediction.get("top_scorelines") or [])[:3]
        )
        st.markdown(
            f'<div class="ll-xg">Expected goals '
            f'<b>{prediction["expected_home_goals"]:.2f}</b> – '
            f'<b>{prediction["expected_away_goals"]:.2f}</b> &nbsp;·&nbsp; '
            f'Most likely <b>{html.escape(str(prediction.get("most_likely_scoreline", "")))}</b> '
            f'({100 * float(prediction.get("scoreline_probability", 0.0)):.1f}%)</div>'
            f'<div class="ll-scores">{scores_html}</div>',
            unsafe_allow_html=True,
        )
    uncertainty = prediction.get("uncertainty") or {}
    if uncertainty.get("level"):
        st.markdown(
            f'<span class="ll-level">Uncertainty: {html.escape(uncertainty["level"])}'
            f' · agreement {uncertainty.get("agreement", 0):.2f}</span>',
            unsafe_allow_html=True,
        )
    pick_col, conf_col = st.columns([2, 3])
    with pick_col:
        st.markdown(
            f'<span class="ll-pick">Pick: {html.escape(prediction["predicted_outcome"])}</span>',
            unsafe_allow_html=True,
        )
    with conf_col:
        st.caption(f"Confidence {prediction['confidence']*100:.1f}% of probability mass")
    if values_html:
        st.markdown(values_html, unsafe_allow_html=True)
    if form_html:
        st.markdown(form_html, unsafe_allow_html=True)


def severity_chip(severity: str) -> str:
    css = {
        "ok": "ll-chip-ok", "info": "ll-chip-ok",
        "warning": "ll-chip-warn", "error": "ll-chip-error",
    }.get(severity, "ll-chip-warn")
    label = {"ok": "OK", "info": "INFO", "warning": "WARN", "error": "ERROR"}.get(
        severity, severity.upper()
    )
    return f'<span class="{css}">{label}</span>'


def footer() -> None:
    st.markdown(
        '<div class="ll-footer">La Liga Predictor · Statistical analytics project. '
        "Predictions are probabilistic estimates, not guarantees or betting advice. "
        "Metrics shown are measured on the indicated windows only.</div>",
        unsafe_allow_html=True,
    )


def dataframe(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def empty_state(message: str, command: str = "") -> None:
    st.info(message)
    if command:
        st.code(command, language="bash")
