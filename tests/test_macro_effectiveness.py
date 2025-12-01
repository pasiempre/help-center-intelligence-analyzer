"""Tests for macro effectiveness module."""

import pandas as pd
import pytest

from src.macro_effectiveness import (
    calculate_baseline_metrics,
    compute_macro_scores,
)


def test_calculate_baseline_metrics():
    """Test baseline metric calculation."""
    tickets_df = pd.DataFrame({
        "csat_score": [4, 3, 5, 2, 4],
        "total_handle_time_minutes": [30, 45, 20, 60, 35],
        "reopens_count": [0, 1, 0, 0, 0],
        "contact_driver": ["billing", "billing", "technical", "technical", "account"],
    })

    baselines = calculate_baseline_metrics(tickets_df)

    assert "overall_avg_csat" in baselines
    assert "overall_avg_handle_time" in baselines
    assert "overall_reopen_rate" in baselines
    assert baselines["overall_avg_csat"] == 3.6
