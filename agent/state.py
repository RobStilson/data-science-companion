from __future__ import annotations

# TypedDict lets us define a dictionary with a fixed set of typed keys.
# LangGraph passes this dict between every node in the graph — each node
# reads from it and returns a new copy with its fields updated.
from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    # The pandas DataFrame loaded from the uploaded file. None until ingest runs.
    df: Any
    # Original filename (e.g. "titanic.csv") — used in export script headers.
    filename: str
    # Size of the uploaded file in megabytes — used for size-limit warnings.
    file_size_mb: float
    # Every analysis step appends a dict here so export.py can replay the session.
    session_log: list[dict[str, Any]]
    # Columns classified as categorical (string/bool/low-cardinality numeric).
    cat_cols: list[str]
    # All numeric columns regardless of cardinality.
    num_cols: list[str]
    # Numeric columns with very few unique values (treated like categories).
    discrete_cols: list[str]
    # Numeric columns with many unique values (true continuous measurements).
    continuous_cols: list[str]
    # The column the user wants to correlate all others against. Set by the user.
    outcome_col: Optional[str]
    # Queue of markdown strings waiting to be sent to the Chainlit UI.
    # Each node appends here; app.py drains and clears this after every step.
    messages: list[str]
    # LLM-generated chart suggestions from the visualize.suggest() call.
    viz_suggestions: list[str]


def initial_state() -> AgentState:
    # Returns a blank state dict with safe defaults for every field.
    # Called once per chat session when no file has been uploaded yet.
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
