from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.state import AgentState
from llm.factory import get_llm
from utils.formatting import make_markdown_table

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "data_dict.md"


async def run(state: AgentState) -> AgentState:
    """Generate a data dictionary via a single batched LLM call."""
    df = state["df"]

    # Build per-column summary for the prompt
    col_lines: list[str] = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        samples = [str(v) for v in df[col].dropna().unique()[:3]]
        samples_str = ", ".join(samples)
        null_count = int(df[col].isnull().sum())
        null_pct = round(null_count / len(df) * 100, 1)
        col_lines.append(
            f"Column: {col} | Type: {dtype} | Samples: {samples_str} | Nulls: {null_count} ({null_pct}%)"
        )
    column_info = "\n".join(col_lines)

    # Single batched LLM call
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{{column_info}}", column_info)
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    # Parse JSON — strip markdown fences if the LLM added them
    raw = response.content.strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip("`\n ")
    descriptions: dict[str, str] = json.loads(raw)

    # Build table
    headers = ["Column", "Type", "Samples (3)", "Nulls", "Null%", "Description"]
    rows: list[list[str]] = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        samples = [str(v) for v in df[col].dropna().unique()[:3]]
        null_count = int(df[col].isnull().sum())
        null_pct = f"{round(null_count / len(df) * 100, 1)}%"
        desc = descriptions.get(col, "—")
        rows.append([col, dtype, ", ".join(samples), str(null_count), null_pct, desc])

    table = make_markdown_table(headers, rows)
    msg = "\n".join(["### Data Dictionary", table])

    log_entry = {
        "step": "data_dict",
        "columns": list(df.columns),
        "descriptions": descriptions,
    }

    return {
        **state,
        "messages": state["messages"] + [msg],
        "session_log": state["session_log"] + [log_entry],
    }
