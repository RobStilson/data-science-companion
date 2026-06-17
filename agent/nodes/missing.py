from __future__ import annotations

from agent.state import AgentState
from utils.formatting import make_markdown_table, severity_label


async def run(state: AgentState) -> AgentState:
    """Analyse missing data in the loaded DataFrame and emit a summary."""
    df = state["df"]

    total_cells = df.shape[0] * df.shape[1]
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df) * 100).round(2)

    null_cols = null_counts[null_counts > 0].index.tolist()
    complete_cols = null_counts[null_counts == 0].index.tolist()

    completeness_pct = round((total_cells - null_counts.sum()) / total_cells * 100, 2)

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
        names = ", ".join(complete_cols)
        lines.append(f"\n*{len(complete_cols)} column(s) fully complete: {names}.*")

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
