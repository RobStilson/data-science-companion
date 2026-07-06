from __future__ import annotations

from scipy import stats

from agent.state import AgentState
from utils.formatting import make_markdown_table, skew_label

# Suggested transformations depend on which direction the tail points.
# Right-skewed (positive skew): tail on the right, long run of large values.
# Left-skewed (negative skew): tail on the left, long run of small values.
_TRANSFORM: dict[str, str] = {
    "right": "log, sqrt, or Box-Cox",
    "left": "square, exp, or reflect + log",
}


async def run(state: AgentState) -> AgentState:
    """Compute skewness per numeric column, label severity, and suggest transforms."""
    df = state["df"]
    num_df = df.select_dtypes(include="number")

    # scipy.stats.skew measures the asymmetry of the distribution.
    # Positive = right tail, Negative = left tail, 0 = perfectly symmetric.
    # dropna() is required because skew() doesn't ignore NaN by default.
    skew_values: dict[str, float] = {
        col: float(stats.skew(num_df[col].dropna()))
        for col in num_df.columns
    }

    # Sort by absolute skew value descending so the most skewed columns appear first.
    sorted_cols = sorted(skew_values, key=lambda c: abs(skew_values[c]), reverse=True)

    headers = ["Column", "Skewness", "Label", "Suggestion"]
    rows = []
    for col in sorted_cols:
        sk = skew_values[col]
        label = skew_label(sk)
        suggestion = ""
        # Only suggest a transformation when the skew is severe enough to matter.
        if label == "High":
            direction = "right" if sk > 0 else "left"
            suggestion = _TRANSFORM[direction]
        rows.append([col, f"{sk:.3f}", label, suggestion])

    table = make_markdown_table(headers, rows)

    msg = "\n".join(["### Skewness Analysis", table])

    log_entry = {
        "step": "skewness",
        # Round to 4 decimal places for a compact but precise log entry.
        "skewness": {col: round(skew_values[col], 4) for col in sorted_cols},
    }

    return {
        **state,
        "messages": state["messages"] + [msg],
        "session_log": state["session_log"] + [log_entry],
    }
