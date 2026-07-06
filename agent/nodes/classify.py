from __future__ import annotations

import pandas as pd

from agent.state import AgentState
from utils.formatting import make_markdown_table

# A numeric column with 20 or fewer unique values is probably a category code
# (e.g. 0/1 flags, star ratings, survey scales) rather than a true measurement.
_CAT_UNIQUE_THRESHOLD = 20
# Within numeric columns, fewer than 10 unique values means "discrete"
# (countable values like number of children); 10+ means "continuous".
_DISCRETE_THRESHOLD = 10


async def run(state: AgentState) -> AgentState:
    """Classify DataFrame columns into categorical / numeric / discrete / continuous."""
    df = state["df"]

    cat_cols: list[str] = []
    num_cols: list[str] = []
    discrete_cols: list[str] = []
    continuous_cols: list[str] = []
    col_unique: dict[str, int] = {}

    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        col_unique[col] = n_unique

        if pd.api.types.is_numeric_dtype(dtype):
            num_cols.append(col)
            # Split numeric columns into discrete vs continuous by unique-value count.
            if n_unique < _DISCRETE_THRESHOLD:
                discrete_cols.append(col)
            else:
                continuous_cols.append(col)
            # Low-cardinality numeric columns also appear in cat_cols because
            # they're often better visualised as bar charts than histograms.
            if n_unique <= _CAT_UNIQUE_THRESHOLD:
                cat_cols.append(col)
        else:
            # object / category / bool / datetime columns are always categorical.
            cat_cols.append(col)

    # Build a classification table for display — each column gets all its tags.
    headers = ["Column", "Dtype", "Unique", "Classification"]
    rows = []
    for col in df.columns:
        tags = []
        if col in cat_cols:
            tags.append("categorical")
        if col in discrete_cols:
            tags.append("discrete")
        if col in continuous_cols:
            tags.append("continuous")
        rows.append([col, str(df[col].dtype), str(col_unique[col]), ", ".join(tags)])

    table = make_markdown_table(headers, rows)

    msg_lines = [
        "### Column Classification",
        f"**Categorical** ({len(cat_cols)}): {', '.join(cat_cols)}",
        f"**Numeric** ({len(num_cols)}): {', '.join(num_cols)}",
        f"  — Discrete (<{_DISCRETE_THRESHOLD} unique): {', '.join(discrete_cols)}",
        f"  — Continuous (≥{_DISCRETE_THRESHOLD} unique): {', '.join(continuous_cols)}",
        "",
        table,
    ]

    log_entry = {
        "step": "classify",
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "discrete_cols": discrete_cols,
        "continuous_cols": continuous_cols,
    }

    # Write the column lists back into state so later nodes (correlations,
    # auto_viz) can use them without re-computing classification.
    return {
        **state,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "discrete_cols": discrete_cols,
        "continuous_cols": continuous_cols,
        "messages": state["messages"] + ["\n".join(msg_lines)],
        "session_log": state["session_log"] + [log_entry],
    }
