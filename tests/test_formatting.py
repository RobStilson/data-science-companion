from utils.formatting import make_markdown_table, severity_label, skew_label


def test_make_markdown_table_structure():
    headers = ["Name", "Value"]
    rows = [["Alice", "1"], ["Bob", "2"]]
    result = make_markdown_table(headers, rows)
    lines = result.split("\n")
    assert len(lines) == 4
    assert lines[0] == "| Name | Value |"
    assert "---" in lines[1]
    assert lines[2] == "| Alice | 1 |"
    assert lines[3] == "| Bob | 2 |"


def test_make_markdown_table_single_row():
    result = make_markdown_table(["A"], [["x"]])
    assert "| A |" in result
    assert "| x |" in result


def test_severity_label_low():
    assert severity_label(0.0) == "Low"
    assert severity_label(4.9) == "Low"


def test_severity_label_medium():
    assert severity_label(5.0) == "Medium"
    assert severity_label(20.0) == "Medium"


def test_severity_label_high():
    assert severity_label(20.1) == "High"
    assert severity_label(100.0) == "High"


def test_skew_label_symmetric():
    assert skew_label(0.0) == "Symmetric"
    assert skew_label(-0.4) == "Symmetric"
    assert skew_label(0.49) == "Symmetric"


def test_skew_label_moderate():
    assert skew_label(0.5) == "Moderate"
    assert skew_label(-0.8) == "Moderate"
    assert skew_label(1.0) == "Moderate"


def test_skew_label_high():
    assert skew_label(1.1) == "High"
    assert skew_label(-2.5) == "High"
