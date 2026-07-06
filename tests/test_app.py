from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("chainlit", MagicMock())

from app import _parse_viz  # noqa: E402


def test_and_separator_with_of():
    chart, cols = _parse_viz("Scatter of gender and SeniorCitizen")
    assert chart == "scatter"
    assert cols == ["gender", "SeniorCitizen"]


def test_and_separator_without_of():
    chart, cols = _parse_viz("scatter gender and SeniorCitizen")
    assert chart == "scatter"
    assert cols == ["gender", "SeniorCitizen"]


def test_vs_separator_with_of():
    chart, cols = _parse_viz("scatter of age vs salary")
    assert chart == "scatter"
    assert cols == ["age", "salary"]


def test_vs_separator_without_of():
    chart, cols = _parse_viz("scatter age vs salary")
    assert chart == "scatter"
    assert cols == ["age", "salary"]


def test_single_column_with_of():
    chart, cols = _parse_viz("histogram of age")
    assert chart == "histogram"
    assert cols == ["age"]


def test_single_column_without_of():
    chart, cols = _parse_viz("histogram age")
    assert chart == "histogram"
    assert cols == ["age"]


def test_unrecognised_prefix_returns_none():
    chart, cols = _parse_viz("pie of age")
    assert chart is None
    assert cols == []


def test_case_insensitive_prefix():
    chart, cols = _parse_viz("Box of salary")
    assert chart == "box"
    assert cols == ["salary"]


def test_chart_filler_word_stripped():
    chart, cols = _parse_viz("bar chart gender")
    assert chart == "bar"
    assert cols == ["gender"]


def test_chart_of_filler_stripped():
    chart, cols = _parse_viz("scatter chart of age vs salary")
    assert chart == "scatter"
    assert cols == ["age", "salary"]
