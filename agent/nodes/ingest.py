from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from agent.state import AgentState
from utils.formatting import make_markdown_table

# File types we're willing to load. Anything else is rejected with a clear error.
_SUPPORTED = {".csv", ".xlsx", ".xls"}
# Files larger than this trigger a hard stop rather than attempting to load.
_SIZE_LIMIT_MB = 100.0


def _to_snake(name: str) -> str:
    # Converts any column name to lowercase_with_underscores so downstream code
    # can reference columns consistently regardless of how the file was saved.
    # Example: "SeniorCitizen" → "senior_citizen", "Monthly Charges" → "monthly_charges"
    s = str(name)
    # Step 1: Insert underscore between a run of caps and a cap+lowercase (e.g. "ABCDef" → "ABC_Def")
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    # Step 2: Insert underscore between lowercase/digit and uppercase (e.g. "camelCase" → "camel_Case")
    s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
    # Step 3: Replace spaces, hyphens, and dots with underscores
    s = re.sub(r'[\s\-\.]+', '_', s)
    # Step 4: Collapse double-underscores, lowercase everything, strip leading/trailing underscores
    return re.sub(r'_+', '_', s).lower().strip('_')


async def run(state: AgentState, file_path: str, filename: str) -> AgentState:
    """Load a CSV or Excel file into state and emit a preview message."""
    ext = Path(filename).suffix.lower()

    # Reject unsupported formats immediately — better than a confusing pandas error.
    if ext not in _SUPPORTED:
        return {
            **state,
            "messages": state["messages"] + [
                f"Error: unsupported file type '{ext}'. Please upload a .csv or .xlsx file."
            ],
        }

    # Check file size before reading — pandas would happily try to load a 5 GB file into RAM.
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > _SIZE_LIMIT_MB:
        return {
            **state,
            "messages": state["messages"] + [
                f"**{filename}** ({size_mb:.1f} MB) exceeds the {_SIZE_LIMIT_MB:.0f} MB limit "
                "and was not loaded. Please upload a smaller file."
            ],
        }

    extra_sheet_note = ""
    if ext in {".xlsx", ".xls"}:
        # Excel files can have multiple sheets; we always load the first one.
        # We tell the user which sheet we used and which were ignored.
        xl = pd.ExcelFile(file_path)
        sheet = xl.sheet_names[0]
        extra_sheets = xl.sheet_names[1:]
        df = pd.read_excel(file_path, sheet_name=sheet)
        if extra_sheets:
            extra_sheet_note = (
                f" (loaded sheet **{sheet}**; "
                f"ignored: {', '.join(extra_sheets)})"
            )
    else:
        df = pd.read_csv(file_path)

    # Normalise all column names to snake_case so nodes don't need to handle
    # mixed naming conventions (CamelCase, "Space Separated", etc.).
    df.columns = [_to_snake(c) for c in df.columns]

    shape_msg = (
        f"Loaded **{filename}**{extra_sheet_note}"
        f" — {df.shape[0]:,} rows × {df.shape[1]} columns."
    )
    preview = _head_table(df)

    # Return a new state dict — never mutate the existing state in place.
    # The {**state, ...} pattern spreads all existing fields and overrides specific ones.
    return {
        **state,
        "df": df,
        "filename": filename,
        "file_size_mb": round(size_mb, 3),
        "messages": state["messages"] + [shape_msg, preview],
    }


def _head_table(df: pd.DataFrame, n: int = 5) -> str:
    # Show the first 5 rows as a Markdown table so the user can sanity-check
    # that the data loaded correctly before EDA begins.
    sample = df.head(n)
    headers = list(sample.columns.astype(str))
    rows = [[str(v) for v in row] for row in sample.itertuples(index=False)]
    return make_markdown_table(headers, rows)
