"""
Integration tests: multiple components exercised together in realistic flows.
Also covers remaining uncovered branches in visualize.py and code_gen.py.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

from agent.graph import build_graph
from agent.nodes import correlations, export, ingest
from agent.nodes import visualize as viz_node
from agent.state import initial_state
from utils.code_gen import generate_python, generate_r


def _mock_llm(content: str) -> AsyncMock:
    resp = MagicMock()
    resp.content = content
    llm = AsyncMock()
    llm.ainvoke.return_value = resp
    return llm


def _write_csv(tmp_path) -> str:
    path = os.path.join(str(tmp_path), "test.csv")
    with open(path, "w") as f:
        f.write("age,salary,outcome\n25,50000,0\n30,55000,1\n35,60000,0\n40,65000,1\n45,70000,0\n")
    return path


# ── full end-to-end pipeline ───────────────────────────────────────────────────

async def test_full_pipeline_csv_to_export(tmp_path):
    csv_path = _write_csv(tmp_path)

    # Ingest a real file from disk
    state = await ingest.run(initial_state(), csv_path, "test.csv")
    assert state["df"] is not None
    assert state["filename"] == "test.csv"

    # Auto-EDA sequence through the graph (LLMs mocked)
    desc_mock = _mock_llm("The data looks reasonable with no major outliers.")
    dict_mock = _mock_llm('{"age": "Age in years", "salary": "Annual salary", "outcome": "Binary target"}')
    graph = build_graph()
    with (
        patch("agent.nodes.descriptive.get_llm", return_value=desc_mock),
        patch("agent.nodes.data_dict.get_llm", return_value=dict_mock),
    ):
        state = await graph.ainvoke({**state, "messages": []})

    # Graph populated classification + session log
    assert "age" in state["num_cols"]
    steps = {e.get("step") for e in state["session_log"]}
    assert {"missing_data", "classify", "descriptive", "skewness", "data_dict"} <= steps

    # User-triggered: correlations
    state = {**state, "outcome_col": "outcome", "messages": []}
    state = await correlations.run(state)
    assert any(e.get("step") == "correlations" for e in state["session_log"])
    assert any("age" in m or "salary" in m for m in state["messages"])

    # User-triggered: export
    state = {**state, "messages": []}
    state = await export.run(state)
    combined = "\n".join(state["messages"])
    assert "import pandas" in combined
    assert "library(" in combined
    assert any(e.get("step") == "export" for e in state["session_log"])


# ── code_gen: Excel filename branch ───────────────────────────────────────────

def test_code_gen_excel_uses_read_excel():
    log = [{"step": "missing_data", "null_cols": [], "completeness_pct": 100.0}]
    py_code = generate_python(log, "data.xlsx")
    r_code = generate_r(log, "data.xlsx")
    assert "read_excel" in py_code
    assert "read_excel" in r_code


# ── code_gen: descriptive + skewness sections ─────────────────────────────────

def test_code_gen_descriptive_and_skewness_python():
    log = [
        {"step": "descriptive", "findings": {}},
        {"step": "skewness", "skewness": {"age": 0.5}},
    ]
    code = generate_python(log, "data.csv")
    assert "describe()" in code
    assert "skew" in code.lower()


def test_code_gen_descriptive_and_skewness_r():
    log = [
        {"step": "descriptive", "findings": {}},
        {"step": "skewness", "skewness": {"age": 0.5}},
    ]
    code = generate_r(log, "data.csv")
    assert "summary(" in code
    assert "skewness" in code.lower()


def test_code_gen_viz_scatter_in_python():
    log = [{"viz": "scatter", "cols": ["age", "salary"], "figure": {}}]
    code = generate_python(log, "data.csv")
    assert "scatter" in code


def test_code_gen_viz_box_in_python():
    log = [{"viz": "box", "cols": ["salary"], "figure": {}}]
    code = generate_python(log, "data.csv")
    assert "box" in code


def test_code_gen_viz_fallback_in_python():
    log = [{"viz": "heatmap", "cols": ["age", "salary"], "figure": {}}]
    code = generate_python(log, "data.csv")
    assert "heatmap" in code.lower()


# ── visualize: uncovered chart-type branches ──────────────────────────────────

async def test_render_box_two_cols(sample_df):
    state = {**initial_state(), "df": sample_df}
    result = await viz_node.render(state, "box", ["salary", "department"])
    log = next(e for e in result["session_log"] if "viz" in e)
    assert log["viz"] == "box"
    assert isinstance(log["figure"], dict)


async def test_render_bar_two_cols(sample_df):
    state = {**initial_state(), "df": sample_df}
    result = await viz_node.render(state, "bar", ["department", "salary"])
    log = next(e for e in result["session_log"] if "viz" in e)
    assert log["viz"] == "bar"


async def test_render_line_two_cols(sample_df):
    state = {**initial_state(), "df": sample_df}
    result = await viz_node.render(state, "line", ["age", "salary"])
    log = next(e for e in result["session_log"] if "viz" in e)
    assert log["viz"] == "line"


async def test_render_heatmap(sample_df):
    state = {**initial_state(), "df": sample_df}
    result = await viz_node.render(state, "heatmap", ["age", "salary"])
    log = next(e for e in result["session_log"] if "viz" in e)
    assert log["viz"] == "heatmap"


async def test_render_pair_plot(sample_df):
    state = {**initial_state(), "df": sample_df}
    result = await viz_node.render(state, "pair plot", ["age", "salary"])
    log = next(e for e in result["session_log"] if "viz" in e)
    assert log["viz"] == "pair plot"
