from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.state import AgentState
from llm.factory import get_llm
from utils.formatting import make_markdown_table

# Path is resolved relative to this file so it works regardless of where the
# process is launched from. The prompt file lives at prompts/callouts.md.
_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "callouts.md"


async def run(state: AgentState) -> AgentState:
    """Compute descriptive statistics, detect callout flags, and summarise via LLM."""
    df = state["df"]
    # Only compute stats on numeric columns — describe() on strings isn't useful here.
    num_df = df.select_dtypes(include="number")

    # pandas describe() gives count, mean, std, min, 25%, 50%, 75%, max per column.
    desc = num_df.describe()

    # ── IQR outlier detection ─────────────────────────────────────────────────
    # The Interquartile Range (IQR) is the distance between the 25th and 75th
    # percentiles. Values more than 1.5×IQR beyond either fence are flagged as
    # potential outliers (this is the standard "Tukey fence" rule).
    q1 = num_df.quantile(0.25)
    q3 = num_df.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (num_df < q1 - 1.5 * iqr) | (num_df > q3 + 1.5 * iqr)
    outlier_counts: dict[str, int] = outlier_mask.sum().to_dict()

    # ── Per-column anomaly flags ──────────────────────────────────────────────
    flags: dict[str, list[str]] = {}
    for col in num_df.columns:
        col_flags: list[str] = []
        mean_val = desc.loc["mean", col]
        std_val = desc.loc["std", col]
        # High coefficient of variation: std > |mean| means extreme spread.
        # The 1e-9 guard prevents a false positive when the mean is near zero.
        if abs(mean_val) > 1e-9 and std_val > abs(mean_val):
            col_flags.append("high_cv")
        # Zero variance: every row has the same value — usually a data issue.
        if desc.loc["min", col] == desc.loc["max", col]:
            col_flags.append("zero_variance")
        if col_flags:
            flags[col] = col_flags

    findings = {"outlier_counts": outlier_counts, "flags": flags}

    # ── Describe table ────────────────────────────────────────────────────────
    stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    headers = ["Column"] + stats + ["IQR Outliers"]
    rows = []
    for col in num_df.columns:
        row = [col] + [f"{desc.loc[s, col]:.2f}" for s in stats] + [str(outlier_counts.get(col, 0))]
        rows.append(row)
    desc_table = make_markdown_table(headers, rows)

    # ── LLM callout ───────────────────────────────────────────────────────────
    # We pass the raw findings JSON to the LLM so it can write a plain-English
    # summary of anomalies. The prompt template lives in prompts/callouts.md.
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{{findings}}", json.dumps(findings, indent=2))
    llm = get_llm()
    # ainvoke is the async version of invoke — required because this function is async.
    # HumanMessage wraps the prompt text in the format LangChain expects.
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
