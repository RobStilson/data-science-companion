from __future__ import annotations

from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    df: Any  # pandas DataFrame, None until file is loaded
    filename: str
    file_size_mb: float
    session_log: list[dict[str, Any]]
    cat_cols: list[str]
    num_cols: list[str]
    discrete_cols: list[str]
    continuous_cols: list[str]
    outcome_col: Optional[str]
    messages: list[str]
    viz_suggestions: list[str]


def initial_state() -> AgentState:
    return AgentState(
        df=None,
        filename="",
        file_size_mb=0.0,
        session_log=[],
        cat_cols=[],
        num_cols=[],
        discrete_cols=[],
        continuous_cols=[],
        outcome_col=None,
        messages=[],
        viz_suggestions=[],
    )
