from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from agent.nodes.descriptive import run
from agent.state import initial_state


def _state(df: pd.DataFrame):
    return {**initial_state(), "df": df}


def _mock_llm(text: str = "Mocked callout paragraph."):
    mock_resp = MagicMock()
    mock_resp.content = text
    llm = AsyncMock()
    llm.ainvoke.return_value = mock_resp
    return llm


async def test_describe_includes_all_numeric_columns(sample_df):
    with patch("agent.nodes.descriptive.get_llm", return_value=_mock_llm()):
        result = await run(_state(sample_df))
    msg = "\n".join(result["messages"])
    for col in ("age", "income", "is_promoted", "years_exp", "score", "salary"):
        assert col in msg


async def test_callout_section_always_emitted(sample_df):
    sentinel = "Mocked callout paragraph."
    with patch("agent.nodes.descriptive.get_llm", return_value=_mock_llm(sentinel)):
        result = await run(_state(sample_df))
    msg = "\n".join(result["messages"])
    assert "Callout" in msg
    assert sentinel in msg


async def test_session_log_updated(sample_df):
    with patch("agent.nodes.descriptive.get_llm", return_value=_mock_llm()):
        result = await run(_state(sample_df))
    steps = [e["step"] for e in result["session_log"]]
    assert "descriptive" in steps


async def test_llm_called_with_findings(sample_df):
    llm = _mock_llm()
    with patch("agent.nodes.descriptive.get_llm", return_value=llm):
        await run(_state(sample_df))
    assert llm.ainvoke.called


async def test_high_cv_flag_detected():
    # mean=0.2, std≈0.45 → std > mean → high_cv flag
    df = pd.DataFrame({"flag_col": [0, 0, 0, 0, 1], "normal": [10, 20, 30, 40, 50]})
    llm = _mock_llm()
    with patch("agent.nodes.descriptive.get_llm", return_value=llm):
        await run(_state(df))
    call_args = llm.ainvoke.call_args[0][0]
    prompt_text = str(call_args)
    assert "high_cv" in prompt_text or "flag_col" in prompt_text
