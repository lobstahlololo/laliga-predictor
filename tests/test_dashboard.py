"""Streamlit smoke tests: pages start and render an empty state when needed."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _run(page: str) -> AppTest:
    app = AppTest.from_file(str(ROOT / "dashboard" / page), default_timeout=90)
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    return app


def test_home_page_renders():
    _run("app.py")


def test_model_operations_page_renders():
    _run("pages/7_Model_Operations.py")


def test_match_predictions_page_renders():
    _run("pages/1_Match_Predictions.py")
