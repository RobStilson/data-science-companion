from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agent.nodes.ingest import run
from agent.state import initial_state

FIXTURES = Path(__file__).parent / "fixtures"
CSV_PATH = FIXTURES / "sample.csv"
XLSX_PATH = FIXTURES / "sample.xlsx"


async def test_csv_load_happy_path():
    state = initial_state()
    result = await run(state, str(CSV_PATH), "sample.csv")
    assert isinstance(result["df"], pd.DataFrame)
    assert result["df"].shape[0] > 0
    assert result["filename"] == "sample.csv"
    assert result["file_size_mb"] > 0
    assert any("sample.csv" in m for m in result["messages"])


async def test_excel_load_happy_path():
    state = initial_state()
    result = await run(state, str(XLSX_PATH), "sample.xlsx")
    assert isinstance(result["df"], pd.DataFrame)
    assert result["df"].shape[0] > 0
    assert result["filename"] == "sample.xlsx"
    assert result["file_size_mb"] > 0


async def test_unsupported_extension_leaves_state_unchanged():
    state = initial_state()
    result = await run(state, "/some/file.json", "data.json")
    assert result["df"] is None
    assert result["filename"] == ""
    combined = " ".join(result["messages"]).lower()
    assert "unsupported" in combined or "error" in combined


async def test_large_file_warns_and_leaves_state_unchanged(tmp_path):
    big_csv = tmp_path / "big.csv"
    big_csv.write_text("a,b\n1,2\n")
    state = initial_state()
    with patch("os.path.getsize", return_value=110 * 1024 * 1024):
        result = await run(state, str(big_csv), "big.csv")
    assert result["df"] is None
    combined = " ".join(result["messages"]).lower()
    assert "warning" in combined or "100" in combined or "mb" in combined


async def test_multisheet_excel_prompts_for_sheet_and_leaves_state_unchanged(tmp_path):
    xl_path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(str(xl_path), engine="openpyxl") as writer:
        pd.DataFrame({"x": [1, 2]}).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame({"y": [3, 4]}).to_excel(writer, sheet_name="Sheet2", index=False)
    state = initial_state()
    result = await run(state, str(xl_path), "multi.xlsx")
    assert result["df"] is None
    combined = " ".join(result["messages"]).lower()
    assert "sheet1" in combined and "sheet2" in combined
