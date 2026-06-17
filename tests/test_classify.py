from __future__ import annotations

import pandas as pd

from agent.nodes.classify import run
from agent.state import initial_state


def _state(df: pd.DataFrame):
    return {**initial_state(), "df": df}


async def test_classify_object_columns_are_categorical(sample_df):
    result = await run(_state(sample_df))
    assert "department" in result["cat_cols"]
    assert "education" in result["cat_cols"]


async def test_classify_numeric_cols_populated(sample_df):
    result = await run(_state(sample_df))
    for col in ("age", "income", "is_promoted", "years_exp", "score", "salary"):
        assert col in result["num_cols"]


async def test_classify_discrete_below_10_unique(sample_df):
    # age=9, income=8, is_promoted=2, years_exp=8, score=8 unique → discrete
    result = await run(_state(sample_df))
    for col in ("age", "income", "is_promoted", "years_exp", "score"):
        assert col in result["discrete_cols"], f"{col} should be discrete"


async def test_classify_continuous_10_or_more_unique(sample_df):
    # salary has 10 unique values → continuous
    result = await run(_state(sample_df))
    assert "salary" in result["continuous_cols"]
    assert "salary" not in result["discrete_cols"]


async def test_classify_boundary_at_10_unique():
    # 9 unique → discrete; 10 unique → continuous
    df = pd.DataFrame({
        "nine": list(range(9)) + [0],   # 9 unique values
        "ten":  list(range(10)),        # 10 unique values
    })
    result = await run(_state(df))
    assert "nine" in result["discrete_cols"]
    assert "nine" not in result["continuous_cols"]
    assert "ten" in result["continuous_cols"]
    assert "ten" not in result["discrete_cols"]


async def test_classify_boundary_at_20_unique_for_categorical():
    # 20 unique numeric → categorical; 21 unique numeric → NOT categorical
    df = pd.DataFrame({
        "twenty":     list(range(20)) * 2,        # 20 unique
        "twenty_one": list(range(21)) + [0] * 19, # 21 unique
    })
    result = await run(_state(df))
    assert "twenty" in result["cat_cols"]
    assert "twenty_one" not in result["cat_cols"]


async def test_classify_session_log_updated(sample_df):
    result = await run(_state(sample_df))
    steps = [entry["step"] for entry in result["session_log"]]
    assert "classify" in steps
