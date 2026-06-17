from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from agent.nodes.data_dict import run
from agent.state import initial_state


def _state(df: pd.DataFrame):
    return {**initial_state(), "df": df}


def _mock_llm(df: pd.DataFrame) -> AsyncMock:
    """Return a mock LLM whose response is valid JSON with one sentence per column."""
    descriptions = {col: f"This column represents {col}." for col in df.columns}
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(descriptions)
    llm = AsyncMock()
    llm.ainvoke.return_value = mock_resp
    return llm


async def test_table_includes_all_columns(sample_df):
    with patch("agent.nodes.data_dict.get_llm", return_value=_mock_llm(sample_df)):
        result = await run(_state(sample_df))
    msg = "\n".join(result["messages"])
    for col in sample_df.columns:
        assert col in msg, f"Column '{col}' not found in output"


async def test_table_headers_present(sample_df):
    with patch("agent.nodes.data_dict.get_llm", return_value=_mock_llm(sample_df)):
        result = await run(_state(sample_df))
    msg = "\n".join(result["messages"])
    for header in ("Column", "Type", "Samples", "Nulls", "Description"):
        assert header in msg, f"Header '{header}' missing from output"


async def test_descriptions_are_single_sentences(sample_df):
    with patch("agent.nodes.data_dict.get_llm", return_value=_mock_llm(sample_df)):
        result = await run(_state(sample_df))
    log_entry = next(e for e in result["session_log"] if e["step"] == "data_dict")
    for col, desc in log_entry["descriptions"].items():
        assert "\n" not in desc, f"Description for '{col}' contains a newline"
        assert len(desc) > 0, f"Description for '{col}' is empty"


async def test_session_log_updated(sample_df):
    with patch("agent.nodes.data_dict.get_llm", return_value=_mock_llm(sample_df)):
        result = await run(_state(sample_df))
    steps = [e["step"] for e in result["session_log"]]
    assert "data_dict" in steps


async def test_llm_called_exactly_once(sample_df):
    llm = _mock_llm(sample_df)
    with patch("agent.nodes.data_dict.get_llm", return_value=llm):
        await run(_state(sample_df))
    assert llm.ainvoke.call_count == 1, "LLM should be called exactly once (batched)"


async def test_json_with_markdown_fences_parsed_correctly(sample_df):
    """LLMs sometimes wrap JSON in ```json ... ``` fences — the node must strip them."""
    descriptions = {col: f"This column represents {col}." for col in sample_df.columns}
    fenced = f"```json\n{json.dumps(descriptions)}\n```"
    mock_resp = MagicMock()
    mock_resp.content = fenced
    llm = AsyncMock()
    llm.ainvoke.return_value = mock_resp
    with patch("agent.nodes.data_dict.get_llm", return_value=llm):
        result = await run(_state(sample_df))
    msg = "\n".join(result["messages"])
    for col in sample_df.columns:
        assert col in msg
