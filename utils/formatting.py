def make_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_row, separator] + data_rows)


def severity_label(pct: float) -> str:
    if pct < 5.0:
        return "Low"
    elif pct <= 20.0:
        return "Medium"
    return "High"


def skew_label(skew: float) -> str:
    abs_skew = abs(skew)
    if abs_skew < 0.5:
        return "Symmetric"
    elif abs_skew <= 1.0:
        return "Moderate"
    return "High"
