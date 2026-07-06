from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from agent.state import AgentState

# Numeric columns with more than this many unique values are treated as
# continuous and get a histogram; fewer unique values get a bar chart.
_CONTINUOUS_THRESHOLD = 10
# Categorical columns with more unique values than this are skipped
# (e.g. free-text, IDs).
_MAX_CATEGORIES = 20


def _histogram(df: pd.DataFrame, col: str) -> go.Figure:
    return px.histogram(df, x=col, title=f"Distribution — {col}")


def _bar(df: pd.DataFrame, col: str, show_pct: bool = False) -> go.Figure:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    if show_pct:
        total = counts["count"].sum()
        counts["pct"] = (counts["count"] / total * 100).round(1).astype(str) + "%"
        fig = px.bar(counts, x=col, y="count", text="pct", title=f"Value counts — {col}")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_range=[0, counts["count"].max() * 1.15])
        return fig
    return px.bar(counts, x=col, y="count", title=f"Value counts — {col}")


def _heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.select_dtypes(include="number").corr()
    return px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap")


async def run(state: AgentState) -> AgentState:
    df = state["df"]
    new_entries: list[dict] = []

    for col in df.columns:
        n_unique = df[col].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        if is_numeric and n_unique > _CONTINUOUS_THRESHOLD:
            fig = _histogram(df, col)
            new_entries.append({"viz": "histogram", "cols": [col], "figure": fig.to_dict(), "auto": True})
        elif n_unique <= _MAX_CATEGORIES:
            fig = _bar(df, col, show_pct=is_numeric)
            new_entries.append({"viz": "bar", "cols": [col], "figure": fig.to_dict(), "auto": True})
        # else: skip high-cardinality categorical (e.g. customer IDs, free-text)

    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] >= 2:
        fig = _heatmap(df)
        new_entries.append({"viz": "heatmap", "cols": [], "figure": fig.to_dict(), "auto": True})

    n = len(new_entries)
    msg = f"**Auto-generated {n} chart{'s' if n != 1 else ''}** — scroll up to view them."

    return {
        **state,
        "session_log": state["session_log"] + new_entries,
        "messages": state["messages"] + [msg],
    }
