# Shared helpers for turning Python data structures into formatted Markdown.
# Centralised here so all nodes produce consistent output without duplicating logic.


def make_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    # GitHub-Flavoured Markdown table format:
    #   | Col A | Col B |
    #   | --- | --- |
    #   | val | val |
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_row, separator] + data_rows)


def severity_label(pct: float) -> str:
    # Classifies the proportion of null values in a column into three tiers.
    # Thresholds are a common industry convention for data quality assessment.
    if pct < 5.0:
        return "Low"
    elif pct <= 20.0:
        return "Medium"
    return "High"


def skew_label(skew: float) -> str:
    # Standard statistics thresholds for skewness magnitude:
    #   |skew| < 0.5  → distribution is roughly bell-shaped (symmetric)
    #   0.5–1.0       → noticeable tail on one side (moderate)
    #   > 1.0         → strong tail; mean is pulled far from median (high)
    abs_skew = abs(skew)
    if abs_skew < 0.5:
        return "Symmetric"
    elif abs_skew <= 1.0:
        return "Moderate"
    return "High"
