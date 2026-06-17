from __future__ import annotations

import pandas as pd

from agent.nodes.missing import run
from agent.state import initial_state


def _state(df: pd.DataFrame):
    return {**initial_state(), "df": df}


async def test_missing_table_includes_only_null_columns(sample_df):
    result = await run(_state(sample_df))
    msg = "\n".join(result["messages"])
    # age, income, score have nulls — should appear
    assert "age" in msg
    assert "income" in msg
    assert "score" in msg
    # salary has no nulls — should NOT appear in the table rows
    lines = msg.splitlines()
    data_lines = [
        row for row in lines
        if row.startswith("| ") and "---" not in row and "Column" not in row
    ]
    col_names_in_table = [row.split("|")[1].strip() for row in data_lines]
    assert "salary" not in col_names_in_table


async def test_missing_completeness_percentage(sample_df):
    result = await run(_state(sample_df))
    msg = "\n".join(result["messages"])
    # 80 total cells, 5 nulls → 93.75%
    assert "93.75" in msg or "93.8" in msg


async def test_missing_severity_labels_correct(sample_df):
    result = await run(_state(sample_df))
    msg = "\n".join(result["messages"])
    # age=10%, income=20%, score=20% → all Medium
    assert "Medium" in msg


async def test_missing_session_log_updated(sample_df):
    result = await run(_state(sample_df))
    steps = [entry["step"] for entry in result["session_log"]]
    assert "missing_data" in steps


async def test_missing_all_null_column():
    df = pd.DataFrame({
        "empty_col": [None, None, None],
        "full_col": [1, 2, 3],
    })
    result = await run(_state(df))
    msg = "\n".join(result["messages"])
    assert "empty_col" in msg
    assert "High" in msg       # 100% nulls → High severity
    assert "full_col" not in "\n".join(
        row for row in msg.splitlines()
        if row.startswith("| ") and "---" not in row and "Column" not in row
    )
