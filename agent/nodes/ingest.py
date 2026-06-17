from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from agent.state import AgentState
from utils.formatting import make_markdown_table

_SUPPORTED = {".csv", ".xlsx", ".xls"}
_SIZE_LIMIT_MB = 100.0


async def run(state: AgentState, file_path: str, filename: str) -> AgentState:
    """Load a CSV or Excel file into state and emit a preview message."""
    ext = Path(filename).suffix.lower()

    if ext not in _SUPPORTED:
        return {
            **state,
            "messages": state["messages"] + [
                f"Error: unsupported file type '{ext}'. Please upload a .csv or .xlsx file."
            ],
        }

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > _SIZE_LIMIT_MB:
        return {
            **state,
            "messages": state["messages"] + [
                f"**{filename}** ({size_mb:.1f} MB) exceeds the {_SIZE_LIMIT_MB:.0f} MB limit "
                "and was not loaded. Please upload a smaller file."
            ],
        }

    if ext in {".xlsx", ".xls"}:
        xl = pd.ExcelFile(file_path)
        if len(xl.sheet_names) > 1:
            sheet_list = ", ".join(xl.sheet_names)
            return {
                **state,
                "messages": state["messages"] + [
                    f"This Excel file has multiple sheets: **{sheet_list}**. "
                    "Please type the sheet name you want to load."
                ],
            }
        df = pd.read_excel(file_path, sheet_name=xl.sheet_names[0])
    else:
        df = pd.read_csv(file_path)

    shape_msg = (
        f"Loaded **{filename}** — {df.shape[0]:,} rows × {df.shape[1]} columns."
    )
    preview = _head_table(df)

    return {
        **state,
        "df": df,
        "filename": filename,
        "file_size_mb": round(size_mb, 3),
        "messages": state["messages"] + [shape_msg, preview],
    }


def _head_table(df: pd.DataFrame, n: int = 5) -> str:
    sample = df.head(n)
    headers = list(sample.columns.astype(str))
    rows = [[str(v) for v in row] for row in sample.itertuples(index=False)]
    return make_markdown_table(headers, rows)
