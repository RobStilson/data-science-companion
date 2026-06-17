from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from agent.nodes.visualize import render, suggest
from agent.state import initial_state


def _state(df: pd.DataFrame) -> dict:
    return {**initial_state(), "df": df}


def _mock_llm(suggestions: list[str]) -> AsyncMock:
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(suggestions)
    llm = AsyncMock()
    llm.ainvoke.return_value = mock_resp
    return llm


# ── suggest ───────────────────────────────────────────────────────────────────

async def test_suggest_populates_viz_suggestions(sample_df):
    expected = ["histogram of age", "scatter of income vs salary"]
    with patch("agent.nodes.visualize.get_llm", return_value=_mock_llm(expected)):
        result = await suggest(_state(sample_df))
    assert result["viz_suggestions"] == expected


async def test_suggest_llm_called_once(sample_df):
    llm = _mock_llm(["one suggestion"])
    with patch("agent.nodes.visualize.get_llm", return_value=llm):
        await suggest(_state(sample_df))
    assert llm.ainvoke.call_count == 1


# ── render ────────────────────────────────────────────────────────────────────

async def test_render_histogram_session_log(sample_df):
    result = await render(_state(sample_df), "histogram", ["age"])
    log = next((e for e in result["session_log"] if "viz" in e), None)
    assert log is not None
    assert log["viz"] == "histogram"
    assert "age" in log["cols"]


async def test_render_scatter_session_log(sample_df):
    result = await render(_state(sample_df), "scatter", ["age", "salary"])
    log = next(e for e in result["session_log"] if "viz" in e)
    assert log["viz"] == "scatter"


async def test_render_figure_stored_as_dict(sample_df):
    result = await render(_state(sample_df), "histogram", ["age"])
    log = next(e for e in result["session_log"] if "viz" in e)
    assert "figure" in log
    assert isinstance(log["figure"], dict)


async def test_render_unsupported_type_returns_error_no_log(sample_df):
    result = await render(_state(sample_df), "pie", ["age"])
    viz_logs = [e for e in result["session_log"] if "viz" in e]
    assert len(viz_logs) == 0
    msg = "\n".join(result["messages"]).lower()
    assert "unsupported" in msg or "pie" in msg


async def test_render_unknown_column_returns_error_no_log(sample_df):
    result = await render(_state(sample_df), "histogram", ["nonexistent_col"])
    viz_logs = [e for e in result["session_log"] if "viz" in e]
    assert len(viz_logs) == 0
    msg = "\n".join(result["messages"]).lower()
    assert "nonexistent_col" in msg or "unknown" in msg


async def test_render_adds_message_on_success(sample_df):
    result = await render(_state(sample_df), "box", ["salary"])
    msg = "\n".join(result["messages"])
    assert "box" in msg.lower() or "salary" in msg
