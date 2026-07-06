from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from agent.state import AgentState
from utils.formatting import make_markdown_table

# Thresholds for labelling correlation strength by absolute r value.
# These match standard interpretations used in social science and data analysis.
_STRENGTH = [
    (0.7, "Very Strong"),
    (0.5, "Strong"),
    (0.3, "Moderate"),
    (0.1, "Weak"),
    (0.0, "Negligible"),
]


def _strength_label(r: float) -> str:
    # Walk thresholds from highest to lowest; return the first label that fits.
    abs_r = abs(r)
    for threshold, label in _STRENGTH:
        if abs_r >= threshold:
            return label
    return "Negligible"


def _pval_str(p: float) -> str:
    # Format very small p-values as "< 0.001" to avoid scientific notation in the table.
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def _correlate(
    x: np.ndarray,
    y: np.ndarray,
    outcome_binary: bool,
    predictor_col: str,
    discrete_cols: list[str],
) -> tuple[str, float, float]:
    """Return (method, r, p) using the most appropriate correlation test.

    Method selection rules:
    - Binary outcome (exactly 2 unique values) → Point-Biserial, which is
      mathematically equivalent to Pearson but designed for 0/1 outcomes.
    - Highly skewed predictor (|skew| > 1) or discrete column → Spearman,
      which is rank-based and doesn't assume a normal distribution.
    - Everything else → Pearson, which assumes both variables are continuous
      and approximately normally distributed.
    """
    if outcome_binary:
        res = stats.pointbiserialr(y, x)
        return "Point-Biserial", float(res[0]), float(res[1])

    skew_val = abs(float(stats.skew(x)))
    if skew_val > 1.0 or predictor_col in discrete_cols:
        res = stats.spearmanr(x, y)
        return "Spearman", float(res[0]), float(res[1])

    res = stats.pearsonr(x, y)
    return "Pearson", float(res[0]), float(res[1])


async def run(state: AgentState) -> AgentState:
    """Compute predictor–outcome correlations with auto-selected method."""
    df = state["df"]
    outcome_col = state["outcome_col"]

    # Guard: the outcome column must exist in the dataset.
    if not outcome_col or outcome_col not in df.columns:
        err = (
            f"Cannot compute correlations: outcome column '{outcome_col}' "
            "not found in the dataset. Please specify a valid column name."
        )
        return {**state, "messages": state["messages"] + [err]}

    outcome = df[outcome_col]
    # Guard: the outcome must be numeric — we can't compute r with a string column.
    if not pd.api.types.is_numeric_dtype(outcome.dtype):
        err = (
            f"Outcome column '{outcome_col}' must be numeric. "
            f"Found dtype: {outcome.dtype}. "
            "Encode categorical targets as integers before running correlations."
        )
        return {**state, "messages": state["messages"] + [err]}

    num_df = df.select_dtypes(include="number")
    # Exclude the outcome column from the predictor list.
    predictors = [c for c in num_df.columns if c != outcome_col]

    if not predictors:
        err = "No numeric predictor columns found to correlate against the outcome."
        return {**state, "messages": state["messages"] + [err]}

    # A binary outcome (exactly 2 distinct values) triggers Point-Biserial.
    outcome_binary = outcome.nunique() == 2
    discrete_cols: list[str] = state["discrete_cols"]

    records = []
    for col in predictors:
        # Drop rows where either the predictor or the outcome is null before computing.
        mask = df[col].notna() & outcome.notna()
        x = df[col][mask].to_numpy(dtype=float)
        y = outcome[mask].to_numpy(dtype=float)
        method, r, p = _correlate(x, y, outcome_binary, col, discrete_cols)
        records.append({
            "predictor": col,
            "r": r,
            "method": method,
            "p": p,
            "strength": _strength_label(r),
        })

    # Sort by |r| descending so the strongest relationships appear at the top.
    records.sort(key=lambda d: abs(d["r"]), reverse=True)

    # Bold-highlight the top 5 positive and top 5 negative correlations
    # so the user's eye is drawn to the most important predictors.
    positives = sorted([d for d in records if d["r"] >= 0], key=lambda d: d["r"], reverse=True)
    negatives = sorted([d for d in records if d["r"] < 0], key=lambda d: d["r"])
    bold = {d["predictor"] for d in positives[:5]} | {d["predictor"] for d in negatives[:5]}

    headers = ["Predictor", "Correlation", "Method", "p-value", "Strength"]
    rows = []
    for d in records:
        corr_str = f"{d['r']:.4f}"
        if d["predictor"] in bold:
            corr_str = f"**{corr_str}**"
        rows.append([d["predictor"], corr_str, d["method"], _pval_str(d["p"]), d["strength"]])

    table = make_markdown_table(headers, rows)
    outcome_type = "binary" if outcome_binary else "continuous"
    msg = "\n".join([
        f"### Correlation Analysis (outcome: **{outcome_col}**, {outcome_type})",
        table,
    ])

    log_entry = {
        "step": "correlations",
        "outcome_col": outcome_col,
        # Store a compact version of results (omit p-value) for the export script.
        "results": [
            {"predictor": d["predictor"], "r": round(d["r"], 4), "method": d["method"]}
            for d in records
        ],
    }

    return {
        **state,
        "messages": state["messages"] + [msg],
        "session_log": state["session_log"] + [log_entry],
    }
