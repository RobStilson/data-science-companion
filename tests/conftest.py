import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic DataFrame for testing individual nodes."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, None, 45, 28, 52, 33, 41, 29],
            "income": [30000, 50000, None, 70000, 90000, 45000, 110000, 62000, None, 38000],
            "department": [
                "HR", "Sales", "IT", "HR", "Finance",
                "IT", "Sales", "HR", "IT", "Finance",
            ],
            "education": [
                "Bachelor", "Master", "PhD", "Bachelor", "Master",
                "Bachelor", "PhD", "Master", "Bachelor", "Master",
            ],
            "is_promoted": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            "years_exp": [2, 5, 3, 8, 1, 4, 9, 6, 3, 2],
            "score": [72.5, 85.0, None, 91.0, 68.5, 77.0, 95.5, 88.0, None, 71.5],
            "salary": [35000, 55000, 48000, 72000, 95000, 49000, 115000, 67000, 58000, 41000],
        }
    )
