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

    # ── Collect metadata for every column in one pass ─────────────────────────
    # We do this before calling the LLM so we can reuse the metadata both for
    # building the prompt and for rendering the final table (no double-pass).
    col_meta: list[tuple] = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        # Show up to 3 non-null example values so the LLM understands the content.
        samples = [str(v) for v in df[col].dropna().unique()[:3]]
        null_count = int(df[col].isnull().sum())
        null_pct = round(null_count / len(df) * 100, 1)
        col_meta.append((col, dtype, samples, null_count, null_pct))

    # Format all column metadata into a compact text block for the LLM prompt.
    col_lines = [
        f"Column: {col} | Type: {dtype} | Samples: {', '.join(samples)} | Nulls: {null_count} ({null_pct}%)"
        for col, dtype, samples, null_count, null_pct in col_meta
    ]
    column_info = "\n".join(col_lines)

    # ── Single batched LLM call ───────────────────────────────────────────────
    # All columns are described in one request rather than one-per-column,
    # which would be much slower and more expensive.
    # The prompt asks the LLM to return a JSON object: {"col_name": "description", ...}
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{{column_info}}", column_info)
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    # ── Parse LLM response ────────────────────────────────────────────────────
    raw = response.content.strip()
    # LLMs sometimes wrap JSON in markdown code fences (```json ... ```).
    # This regex strips those fences so json.loads can parse the content.
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip("`\n ")
    try:
        descriptions: dict[str, str] = json.loads(raw)
    except json.JSONDecodeError:
        # If the LLM returns malformed JSON, fall back to a dash placeholder
        # so the rest of the table still renders correctly.
        descriptions = {col: "—" for col, *_ in col_meta}

    # ── Build the display table ───────────────────────────────────────────────
    # Reuse col_meta collected above — no need to re-query the DataFrame.
    headers = ["Column", "Type", "Samples (3)", "Nulls", "Null%", "Description"]
    rows = [
        [col, dtype, ", ".join(samples), str(null_count), f"{null_pct}%", descriptions.get(col, "—")]
        for col, dtype, samples, null_count, null_pct in col_meta
    ]

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
