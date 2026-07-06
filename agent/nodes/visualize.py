from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from llm.factory import get_llm

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "viz_suggest.md"

_SUPPORTED_TYPES = {"histogram", "box", "scatter", "bar", "heatmap", "line", "pair plot"}
# Minimum number of columns required for each chart type.
# heatmap=0 means no columns need to be specified — it always uses all numeric columns.
_MIN_COLS = {"scatter": 2, "heatmap": 0, "pair plot": 2, "line": 2}


def _build_figure(df: pd.DataFrame, chart_type: str, cols: list[str]) -> go.Figure:
    # Each branch handles one chart type. px (plotly.express) is the high-level
    # API that creates figures from DataFrames with minimal code.

    if chart_type == "histogram":
        # Histogram: show the distribution of a single numeric column.
        return px.histogram(df, x=cols[0])

    if chart_type == "box":
        # Box plot shows median, IQR, and outlier spread.
        # With two columns, the second is used as a grouping variable (x-axis categories).
        if len(cols) == 2:
            return px.box(df, x=cols[1], y=cols[0])
        return px.box(df, y=cols[0])

    if chart_type == "scatter":
        # Scatter plot: reveals the relationship between two numeric columns.
        return px.scatter(df, x=cols[0], y=cols[1])

    if chart_type == "bar":
        if len(cols) == 2:
            col_x, col_y = cols[0], cols[1]
            if pd.api.types.is_numeric_dtype(df[col_y]):
                # Numeric y column: show the mean of y for each category of x.
                agg = df.groupby(col_x, observed=True)[[col_y]].mean().reset_index()
                return px.bar(agg, x=col_x, y=col_y)
            else:
                # Both columns are categorical: stacked bar showing co-occurrence counts.
                agg = (
                    df.groupby([col_x, col_y], observed=True)
                    .size()
                    .reset_index(name="count")
                )
                return px.bar(agg, x=col_x, y="count", color=col_y, barmode="stack")
        # Single column: show value counts.
        # Cap at 30 bars — numeric columns with many values would produce an unreadable chart.
        counts = df[cols[0]].value_counts().head(30).reset_index()
        counts.columns = [cols[0], "count"]
        if pd.api.types.is_numeric_dtype(df[cols[0]]):
            # For numeric columns, annotate each bar with its percentage of total.
            total = counts["count"].sum()
            counts["pct"] = (counts["count"] / total * 100).round(1).astype(str) + "%"
            fig = px.bar(counts, x=cols[0], y="count", text="pct")
            fig.update_traces(textposition="outside")
            # Add 15% headroom so text labels don't get clipped at the top.
            fig.update_layout(yaxis_range=[0, counts["count"].max() * 1.15])
            return fig
        return px.bar(counts, x=cols[0], y="count")

    if chart_type == "heatmap":
        # Correlation heatmap — always computed from all numeric columns.
        # The user doesn't need to specify columns for this chart type.
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for a heatmap.")
        # .corr() computes the pairwise Pearson correlation matrix.
        # text_auto=True annotates each cell with its r value.
        return px.imshow(num_df.corr(), text_auto=True, aspect="auto")

    if chart_type == "line":
        # Sort by x before plotting so the line connects points in order
        # rather than zigzagging across unsorted row indices.
        sorted_df = df.sort_values(cols[0])
        return px.line(sorted_df, x=cols[0], y=cols[1])

    if chart_type == "pair plot":
        # Scatter matrix: every numeric column vs every other numeric column.
        # Filter to only numeric columns in case the user included a categorical one.
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 2:
            raise ValueError(
                f"Pair plot needs at least 2 numeric columns; got: {', '.join(cols)}."
            )
        return px.scatter_matrix(df, dimensions=num_cols)

    raise ValueError(f"Unsupported chart type: {chart_type}")


async def suggest(state: AgentState) -> AgentState:
    """Ask the LLM for visualisation suggestions given the dataset's column summary."""
    df = state["df"]

    # Build a compact column summary for the prompt — dtype and cardinality help
    # the LLM recommend appropriate chart types (e.g. histogram vs bar).
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

    # Strip markdown fences and parse the JSON list of suggestion strings.
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

    # Validate chart type before doing any work.
    if chart_type not in _SUPPORTED_TYPES:
        err = f"Unsupported chart type '{chart_type}'. Choose from: {', '.join(sorted(_SUPPORTED_TYPES))}."
        return {**state, "messages": state["messages"] + [err]}

    # Check the user provided enough columns for the chosen chart type.
    min_required = _MIN_COLS.get(chart_type, 1)
    if len(cols) < min_required:
        err = f"'{chart_type}' requires at least {min_required} column(s); got {len(cols)}."
        return {**state, "messages": state["messages"] + [err]}

    # Check every named column actually exists in the DataFrame.
    unknown = [c for c in cols if c not in df.columns]
    if unknown:
        err = f"Unknown column(s): {', '.join(unknown)}. Available: {', '.join(df.columns)}."
        return {**state, "messages": state["messages"] + [err]}

    try:
        fig = _build_figure(df, chart_type, cols)
    except (ValueError, KeyError) as exc:
        # _build_figure raises ValueError for data-driven failures (e.g. wrong column types).
        return {**state, "messages": state["messages"] + [str(exc)]}

    # Store the figure as a plain dict in the session log so it can be
    # serialised and re-hydrated as a go.Figure in app.py.
    log_entry = {"viz": chart_type, "cols": cols, "figure": fig.to_dict()}
    msg = f"#### {chart_type.title()} — {', '.join(cols) or 'all numeric'}"

    return {
        **state,
        "messages": state["messages"] + [msg],
        "session_log": state["session_log"] + [log_entry],
    }
