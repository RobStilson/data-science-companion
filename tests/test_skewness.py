from __future__ import annotations

import pandas as pd

from agent.nodes.skewness import run
from agent.state import initial_state


def _state(df: pd.DataFrame):
    return {**initial_state(), "df": df}


def _skewed_df():
    return pd.DataFrame({
        "right_skewed": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100],  # strong right skew ~2.67
        "normal":       list(range(1, 11)),                   # near-symmetric
    })


async def test_skewness_table_includes_numeric_columns():
    result = await run(_state(_skewed_df()))
    msg = "\n".join(result["messages"])
    assert "right_skewed" in msg
    assert "normal" in msg


async def test_skewness_sorted_by_absolute_value_descending():
    result = await run(_state(_skewed_df()))
    msg = "\n".join(result["messages"])
    # right_skewed has higher |skew| — its row should appear before normal's row
    pos_right = msg.index("right_skewed")
    pos_normal = msg.index("normal")
    assert pos_right < pos_normal


async def test_skew_labels_present():
    result = await run(_state(_skewed_df()))
    msg = "\n".join(result["messages"])
    assert "High" in msg       # right_skewed ≈ 2.67
    assert "Symmetric" in msg  # normal ≈ 0


async def test_high_skew_gets_transformation_suggestion():
    result = await run(_state(_skewed_df()))
    msg = "\n".join(result["messages"])
    assert any(kw in msg for kw in ("log", "sqrt", "Box-Cox"))


async def test_session_log_updated():
    result = await run(_state(_skewed_df()))
    steps = [e["step"] for e in result["session_log"]]
    assert "skewness" in steps
