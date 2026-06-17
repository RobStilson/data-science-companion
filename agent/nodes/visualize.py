from __future__ import annotations

import json
import re
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from llm.factory import get_llm

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "viz_suggest.md"

_SUPPORTED_TYPES = {"histogram", "box", "scatter", "bar", "heatmap", "line", "pair plot"}
_MIN_COLS = {"scatter": 2, "heatmap": 2, "pair plot": 2}


def _build_figure(df, chart_type: str, cols: list[str]) -> go.Figure:
    if chart_type == "histogram":
        return px.histogram(df, x=cols[0])
    if chart_type == "box":
        if len(cols) == 2:
            return px.box(df, x=cols[1], y=cols[0])
        return px.box(df, y=cols[0])
    if chart_type == "scatter":
        return px.scatter(df, x=cols[0], y=cols[1])
    if chart_type == "bar":
        if len(cols) == 2:
            return px.bar(df, x=cols[0], y=cols[1])
        counts = df[cols[0]].value_counts().reset_index()
        counts.columns = [cols[0], "count"]
        return px.bar(counts, x=cols[0], y="count")
    if chart_type == "heatmap":
        num_df = df[cols].select_dtypes(include="number")
        corr = num_df.corr()
        return px.imshow(corr, text_auto=True)
    if chart_type == "line":
        return px.line(df, x=cols[0], y=cols[1] if len(cols) > 1 else None)
    if chart_type == "pair plot":
        return px.scatter_matrix(df, dimensions=cols)
    raise ValueError(f"Unsupported chart type: {chart_type}")


async def suggest(state: AgentState) -> AgentState:
    """Ask the LLM for visualisation suggestions given the dataset's column summary."""
    df = state["df"]

    col_lines: list[str] = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        col_lines.append(f"{col} ({dtype}, {n_unique} unique values)")
    column_summary = "\n".join(col_lines)

    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{{column_summary}}", column_summary)

    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    raw = response.content.strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip("`\n ")
    try:
        suggestions: list[str] = json.loads(raw)
    except json.JSONDecodeError:
        suggestions = []

    return {**state, "viz_suggestions": suggestions}


async def render(state: AgentState, chart_type: str, cols: list[str]) -> AgentState:
    """Render a Plotly figure and store it in the session log."""
    df = state["df"]

    if chart_type not in _SUPPORTED_TYPES:
        err = f"Unsupported chart type '{chart_type}'. Choose from: {', '.join(sorted(_SUPPORTED_TYPES))}."
        return {**state, "messages": state["messages"] + [err]}

    min_required = _MIN_COLS.get(chart_type, 1)
    if len(cols) < min_required:
        err = f"'{chart_type}' requires at least {min_required} column(s); got {len(cols)}."
        return {**state, "messages": state["messages"] + [err]}

    unknown = [c for c in cols if c not in df.columns]
    if unknown:
        err = f"Unknown column(s): {', '.join(unknown)}. Available: {', '.join(df.columns)}."
        return {**state, "messages": state["messages"] + [err]}

    fig = _build_figure(df, chart_type, cols)
    log_entry = {"viz": chart_type, "cols": cols, "figure": fig.to_dict()}
    msg = f"#### {chart_type.title()} — {', '.join(cols)}"

    return {
        **state,
        "messages": state["messages"] + [msg],
        "session_log": state["session_log"] + [log_entry],
    }
