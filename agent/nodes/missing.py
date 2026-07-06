from __future__ import annotations

from agent.state import AgentState
from utils.formatting import make_markdown_table, severity_label


async def run(state: AgentState) -> AgentState:
    """Analyse missing data in the loaded DataFrame and emit a summary."""
    df = state["df"]

    # Total number of cells in the entire dataset (rows × columns).
    total_cells = df.shape[0] * df.shape[1]

    # isnull() returns a boolean DataFrame; .sum() counts True values per column.
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df) * 100).round(2)

    # Split columns into two groups for different treatment in the output.
    null_cols = null_counts[null_counts > 0].index.tolist()
    complete_cols = null_counts[null_counts == 0].index.tolist()

    # Overall data quality metric: what fraction of all cells are non-null?
    completeness_pct = round((total_cells - null_counts.sum()) / total_cells * 100, 2)

    # Build the detail table — only show columns that actually have missing values.
    headers = ["Column", "Null Count", "Null %", "Severity"]
    rows = [
        [col, str(null_counts[col]), f"{null_pcts[col]}%", severity_label(null_pcts[col])]
        for col in null_cols
    ]
    table = make_markdown_table(headers, rows)

    lines = [
        "### Missing Data Analysis",
        f"**Overall completeness: {completeness_pct}%** "
        f"({total_cells - null_counts.sum():,}/{total_cells:,} cells non-null)",
        "",
        table,
    ]
    if complete_cols:
        # List fully complete columns in a footnote so the user knows they're fine.
        names = ", ".join(complete_cols)
        lines.append(f"\n*{len(complete_cols)} column(s) fully complete: {names}.*")

    # The session_log entry is a compact summary used by export.py to regenerate
    # the missing-data section in the downloadable script.
    log_entry = {
        "step": "missing_data",
        "null_cols": null_cols,
        "completeness_pct": completeness_pct,
    }

    return {
        **state,
        "messages": state["messages"] + ["\n".join(lines)],
        "session_log": state["session_log"] + [log_entry],
    }
