from __future__ import annotations

import pandas as pd

from agent.nodes.correlations import run
from agent.state import initial_state


def _binary_state() -> dict:
    """State with binary outcome (nunique=2) and two numeric predictors."""
    df = pd.DataFrame({
        "outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "strong":  [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "weak":    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    })
    return {**initial_state(), "df": df, "outcome_col": "outcome"}


def _continuous_state() -> dict:
    """State with continuous outcome and one low-skew + one high-skew predictor."""
    df = pd.DataFrame({
        "outcome":   list(range(1, 11)),
        "low_skew":  list(range(2, 12)),               # skew ≈ 0 → Pearson
        "high_skew": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100],  # skew ≈ 2.67 → Spearman
    })
    return {**initial_state(), "df": df, "outcome_col": "outcome", "discrete_cols": []}


async def test_table_includes_all_numeric_predictors():
    result = await run(_binary_state())
    msg = "\n".join(result["messages"])
    assert "strong" in msg
    assert "weak" in msg
    # outcome itself should not appear as a predictor row
    predictor_names = [
        row.split("|")[1].strip()
        for row in msg.splitlines()
        if row.startswith("| ") and "---" not in row and "Predictor" not in row
    ]
    assert "outcome" not in predictor_names


async def test_sorted_by_absolute_correlation_descending():
    result = await run(_binary_state())
    msg = "\n".join(result["messages"])
    # strong has |r|≈1.0, weak has lower |r|; strong must appear first
    assert msg.index("strong") < msg.index("weak")


async def test_point_biserial_used_for_binary_outcome():
    result = await run(_binary_state())
    msg = "\n".join(result["messages"])
    assert "Point-Biserial" in msg


async def test_pearson_used_for_low_skew_continuous_predictor():
    result = await run(_continuous_state())
    msg = "\n".join(result["messages"])
    assert "Pearson" in msg


async def test_spearman_used_for_high_skew_predictor():
    result = await run(_continuous_state())
    msg = "\n".join(result["messages"])
    assert "Spearman" in msg


async def test_unknown_outcome_col_returns_error_no_session_log():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    state = {**initial_state(), "df": df, "outcome_col": "nonexistent"}
    result = await run(state)
    steps = [e["step"] for e in result["session_log"]]
    assert "correlations" not in steps
    msg = "\n".join(result["messages"]).lower()
    assert "nonexistent" in msg or "not found" in msg or "outcome" in msg


async def test_none_outcome_col_returns_error():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    state = {**initial_state(), "df": df, "outcome_col": None}
    result = await run(state)
    steps = [e["step"] for e in result["session_log"]]
    assert "correlations" not in steps


async def test_session_log_updated():
    result = await run(_binary_state())
    steps = [e["step"] for e in result["session_log"]]
    assert "correlations" in steps


async def test_top_correlations_are_bold():
    result = await run(_binary_state())
    msg = "\n".join(result["messages"])
    assert "**" in msg
