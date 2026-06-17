from __future__ import annotations

from agent.nodes.export import run
from agent.state import initial_state
from utils.code_gen import generate_python, generate_r

_SAMPLE_LOG = [
    {"step": "missing_data", "null_cols": ["age", "income"], "completeness_pct": 87.5},
    {
        "step": "correlations",
        "outcome_col": "salary",
        "results": [{"predictor": "age", "r": 0.72, "method": "Pearson"}],
    },
    {"viz": "histogram", "cols": ["age"], "figure": {}},
]


# ── generate_python ────────────────────────────────────────────────────────────

def test_generate_python_is_string():
    assert isinstance(generate_python(_SAMPLE_LOG, "data.csv"), str)


def test_generate_python_loads_file_by_name():
    result = generate_python(_SAMPLE_LOG, "data.csv")
    assert "data.csv" in result


def test_generate_python_includes_missing_section():
    result = generate_python(_SAMPLE_LOG, "data.csv")
    assert "isnull" in result or "isna" in result


def test_generate_python_includes_correlation_section():
    result = generate_python(_SAMPLE_LOG, "data.csv")
    assert "salary" in result
    assert "age" in result


# ── generate_r ─────────────────────────────────────────────────────────────────

def test_generate_r_is_string():
    assert isinstance(generate_r(_SAMPLE_LOG, "data.csv"), str)


def test_generate_r_loads_file_by_name():
    result = generate_r(_SAMPLE_LOG, "data.csv")
    assert "data.csv" in result


# ── export node ────────────────────────────────────────────────────────────────

async def test_export_node_adds_python_and_r_to_messages():
    state = {**initial_state(), "filename": "data.csv", "session_log": _SAMPLE_LOG}
    result = await run(state)
    combined = "\n".join(result["messages"])
    assert "import pandas" in combined
    assert "library(" in combined


async def test_export_node_appends_export_step_to_log():
    state = {**initial_state(), "filename": "data.csv", "session_log": _SAMPLE_LOG}
    result = await run(state)
    steps = [e.get("step") for e in result["session_log"]]
    assert "export" in steps
