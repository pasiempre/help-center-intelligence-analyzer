"""Tests for time series analysis module."""

import pandas as pd
import pytest


def test_compute_daily_metrics():
    """Test daily metrics computation."""
    from src.time_series_analysis import compute_daily_metrics

    tickets = pd.DataFrame({
        "ticket_id": ["T1", "T2", "T3", "T4"],
        "created_at": pd.date_range("2024-01-01", periods=4, freq="D"),
        "csat_score": [4.0, 5.0, 3.0, 4.5],
        "total_handle_time_minutes": [10, 15, 20, 12],
        "num_macros_used": [2, 0, 1, 3],
        "reopens_count": [0, 0, 1, 0],
    })

    daily = compute_daily_metrics(tickets)

    assert len(daily) == 4
    assert "total_tickets" in daily.columns
    assert "avg_csat" in daily.columns
    assert "macro_adoption_rate" in daily.columns


def test_compute_macro_usage_trends():
    """Test macro usage trends computation."""
    from src.time_series_analysis import compute_macro_usage_trends

    macro_usage = pd.DataFrame({
        "ticket_id": ["T1", "T2", "T3", "T4"],
        "macro_id": ["M1", "M1", "M2", "M1"],
        "applied_at": pd.date_range("2024-01-01", periods=4, freq="W"),
    })

    macro_clusters = pd.DataFrame({
        "macro_id": ["M1", "M2"],
        "macro_name": ["Refund", "Shipping"],
        "category": ["Refund", "Shipping"],
        "macro_effectiveness_index": [75.0, 65.0],
    })

    trends = compute_macro_usage_trends(macro_usage, macro_clusters)

    assert "macro_id" in trends.columns
    assert "trend_direction" in trends.columns


def test_compute_category_trends():
    """Test category trends computation."""
    from src.time_series_analysis import compute_category_trends

    macro_usage = pd.DataFrame({
        "ticket_id": ["T1", "T2", "T3"],
        "macro_id": ["M1", "M2", "M1"],
        "applied_at": pd.date_range("2024-01-01", periods=3, freq="W"),
    })

    macro_clusters = pd.DataFrame({
        "macro_id": ["M1", "M2"],
        "category": ["Refund", "Shipping"],
    })

    trends = compute_category_trends(macro_usage, macro_clusters)

    assert "week" in trends.columns
    assert "category" in trends.columns
    assert "usage_count" in trends.columns


def test_detect_anomalies():
    """Test anomaly detection."""
    from src.time_series_analysis import detect_anomalies

    daily_metrics = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "avg_csat": [4.0, 4.1, 4.0, 4.2, 4.0, 4.1, 4.0, 1.0, 4.0, 4.1],  # Day 8 is anomaly
        "avg_handle_time": [15, 14, 16, 15, 14, 15, 16, 15, 14, 15],
        "macro_adoption_rate": [0.7, 0.72, 0.71, 0.73, 0.70, 0.71, 0.72, 0.70, 0.71, 0.72],
    })

    result = detect_anomalies(daily_metrics)

    assert "any_anomaly" in result.columns
    assert "avg_csat_anomaly" in result.columns
    # Day 8 (index 7) should be flagged
    assert result.iloc[7]["avg_csat_anomaly"] == True
