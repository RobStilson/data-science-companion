from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.state import AgentState
from llm.factory import get_llm
from utils.formatting import make_markdown_table

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "callouts.md"


async def run(state: AgentState) -> AgentState:
    """Compute descriptive statistics, detect callout flags, and summarise via LLM."""
    df = state["df"]
    num_df = df.select_dtypes(include="number")

    desc = num_df.describe()

    # IQR outlier detection
    q1 = num_df.quantile(0.25)
    q3 = num_df.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (num_df < q1 - 1.5 * iqr) | (num_df > q3 + 1.5 * iqr)
    outlier_counts: dict[str, int] = outlier_mask.sum().to_dict()

    # Per-column flags
    flags: dict[str, list[str]] = {}
    for col in num_df.columns:
        col_flags: list[str] = []
        mean_val = desc.loc["mean", col]
        std_val = desc.loc["std", col]
        if abs(mean_val) > 1e-9 and std_val > abs(mean_val):
            col_flags.append("high_cv")
        if desc.loc["min", col] == desc.loc["max", col]:
            col_flags.append("zero_variance")
        if col_flags:
            flags[col] = col_flags

    findings = {"outlier_counts": outlier_counts, "flags": flags}

    # Build describe table (rows = columns, columns = stats + outliers)
    stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    headers = ["Column"] + stats + ["IQR Outliers"]
    rows = []
    for col in num_df.columns:
        row = [col] + [f"{desc.loc[s, col]:.2f}" for s in stats] + [str(outlier_counts.get(col, 0))]
        rows.append(row)
    desc_table = make_markdown_table(headers, rows)

    # LLM callout
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{{findings}}", json.dumps(findings, indent=2))
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    callout_text = response.content

    msg = "\n".join([
        "### Descriptive Statistics",
        desc_table,
        "",
        "#### Callouts",
        callout_text,
    ])

    log_entry = {"step": "descriptive", "findings": findings}

    return {
        **state,
        "messages": state["messages"] + [msg],
        "session_log": state["session_log"] + [log_entry],
    }
