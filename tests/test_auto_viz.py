from __future__ import annotations

import pandas as pd
import pytest

from agent.nodes.auto_viz import run
from agent.state import initial_state


def _state(df: pd.DataFrame) -> dict:
    return {**initial_state(), "df": df}


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """DataFrame with continuous numeric, discrete numeric, categorical, and high-cardinality cols."""
    return pd.DataFrame({
        "age": list(range(100)),                          # numeric, > 10 unique → histogram
        "score": [float(i) * 1.5 for i in range(100)],  # numeric, > 10 unique → histogram
        "gender": (["Male", "Female"] * 50),              # categorical, 2 unique → bar
        "grade": (["A", "B", "C", "D"] * 25),            # categorical, 4 unique → bar
        "is_active": ([0, 1] * 50),                      # numeric, 2 unique (≤ 10) → bar
        "free_text": [f"note_{i}" for i in range(100)],  # 100 unique → skip
    })


async def test_histograms_generated_for_continuous_cols(mixed_df):
    result = await run(_state(mixed_df))
    all_cols = [c for e in result["session_log"] if e.get("auto") for c in e["cols"]]
    assert "age" in all_cols
    assert "score" in all_cols
    age_entry = next(e for e in result["session_log"] if e.get("auto") and "age" in e["cols"])
    assert age_entry["viz"] == "histogram"


async def test_bar_charts_generated_for_categorical_cols(mixed_df):
    result = await run(_state(mixed_df))
    all_cols = [c for e in result["session_log"] if e.get("auto") for c in e["cols"]]
    assert "gender" in all_cols
    assert "grade" in all_cols


async def test_discrete_numeric_gets_bar_not_histogram(mixed_df):
    result = await run(_state(mixed_df))
    entries = {c: e["viz"] for e in result["session_log"] if e.get("auto") for c in e["cols"]}
    assert entries.get("is_active") == "bar"


async def test_high_cardinality_col_skipped(mixed_df):
    result = await run(_state(mixed_df))
    all_cols = [c for e in result["session_log"] if e.get("auto") for c in e["cols"]]
    assert "free_text" not in all_cols


async def test_heatmap_generated_when_two_plus_numeric_cols(mixed_df):
    result = await run(_state(mixed_df))
    heatmap_entries = [e for e in result["session_log"] if e.get("auto") and e["viz"] == "heatmap"]
    assert len(heatmap_entries) == 1


async def test_no_heatmap_when_fewer_than_two_numeric_cols():
    df = pd.DataFrame({"label": ["a", "b", "c"], "value": [1, 2, 3]})
    result = await run(_state(df))
    heatmap_entries = [e for e in result["session_log"] if e.get("auto") and e["viz"] == "heatmap"]
    assert len(heatmap_entries) == 0


async def test_figures_stored_as_dicts(mixed_df):
    result = await run(_state(mixed_df))
    for entry in result["session_log"]:
        if entry.get("auto"):
            assert isinstance(entry["figure"], dict)


async def test_discrete_numeric_bar_has_pct_text(mixed_df):
    result = await run(_state(mixed_df))
    entry = next(e for e in result["session_log"] if e.get("auto") and "is_active" in e["cols"])
    bar_text = entry["figure"]["data"][0].get("text", [])
    assert any("%" in str(t) for t in bar_text)


async def test_categorical_bar_has_no_pct_text(mixed_df):
    result = await run(_state(mixed_df))
    entry = next(e for e in result["session_log"] if e.get("auto") and "gender" in e["cols"])
    bar_text = entry["figure"]["data"][0].get("text", [])
    assert not any("%" in str(t) for t in bar_text)


async def test_summary_message_emitted(mixed_df):
    result = await run(_state(mixed_df))
    msgs = " ".join(result["messages"])
    assert "auto-generated" in msgs.lower() or "chart" in msgs.lower()
