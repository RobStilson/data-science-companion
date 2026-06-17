from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from agent.graph import build_graph
from agent.state import initial_state


def _mock_llm(content: str) -> AsyncMock:
    resp = MagicMock()
    resp.content = content
    llm = AsyncMock()
    llm.ainvoke.return_value = resp
    return llm


def _state_with_df() -> dict:
    df = pd.DataFrame({
        "age":    [25, 30, 35, 40, 45],
        "salary": [50000, 55000, 60000, 65000, 70000],
    })
    return {**initial_state(), "df": df}


# ── graph structure ────────────────────────────────────────────────────────────

def test_build_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_compiled_graph_has_ainvoke():
    graph = build_graph()
    assert hasattr(graph, "ainvoke")


# ── graph execution ────────────────────────────────────────────────────────────

async def test_graph_populates_cat_and_num_cols():
    desc_mock = _mock_llm("Some callout text.")
    dict_mock = _mock_llm('{"age": "Person age in years", "salary": "Annual salary"}')
    graph = build_graph()
    with (
        patch("agent.nodes.descriptive.get_llm", return_value=desc_mock),
        patch("agent.nodes.data_dict.get_llm", return_value=dict_mock),
    ):
        result = await graph.ainvoke(_state_with_df())
    assert "num_cols" in result
    assert "age" in result["num_cols"]
    assert "salary" in result["num_cols"]


async def test_graph_appends_session_log_entries():
    desc_mock = _mock_llm("Some callout text.")
    dict_mock = _mock_llm('{"age": "Person age in years", "salary": "Annual salary"}')
    graph = build_graph()
    with (
        patch("agent.nodes.descriptive.get_llm", return_value=desc_mock),
        patch("agent.nodes.data_dict.get_llm", return_value=dict_mock),
    ):
        result = await graph.ainvoke(_state_with_df())
    steps = [e.get("step") for e in result["session_log"]]
    assert "missing_data" in steps
    assert "classify" in steps
    assert "descriptive" in steps
    assert "skewness" in steps
    assert "data_dict" in steps


async def test_graph_accumulates_messages():
    desc_mock = _mock_llm("Some callout text.")
    dict_mock = _mock_llm('{"age": "Person age in years", "salary": "Annual salary"}')
    graph = build_graph()
    with (
        patch("agent.nodes.descriptive.get_llm", return_value=desc_mock),
        patch("agent.nodes.data_dict.get_llm", return_value=dict_mock),
    ):
        result = await graph.ainvoke(_state_with_df())
    assert len(result["messages"]) >= 5
