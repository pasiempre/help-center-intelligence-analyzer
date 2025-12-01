"""Tests for agent analysis module."""

import pandas as pd
import pytest


def test_compute_agent_metrics():
    """Test agent metrics computation."""
    from src.agent_analysis import compute_agent_metrics

    # Create test data - matches what compute_agent_metrics expects
    tickets = pd.DataFrame({
        "ticket_id": ["T1", "T2", "T3", "T4"],
        "agent_id": ["A1", "A1", "A2", "A2"],
        "csat_score": [4.5, 5.0, 3.0, 4.0],
        "total_handle_time_minutes": [10, 15, 20, 12],
        "reopens_count": [0, 0, 1, 0],
        "first_response_time_minutes": [5, 8, 10, 6],
        "status": ["solved", "solved", "open", "solved"],
        "num_macros_used": [2, 1, 0, 3],
    })

    metrics = compute_agent_metrics(tickets)

    assert len(metrics) == 2  # Two agents
    assert "agent_id" in metrics.columns
    assert "avg_csat" in metrics.columns
    assert "total_tickets" in metrics.columns
    assert "macro_adoption_rate" in metrics.columns


def test_compute_agent_macro_usage():
    """Test agent macro usage analysis."""
    from src.agent_analysis import compute_agent_macro_usage

    tickets = pd.DataFrame({
        "ticket_id": ["T1", "T2", "T3", "T4"],
        "agent_id": ["A1", "A1", "A2", "A2"],
        "csat_score": [4.5, 5.0, 3.0, 4.0],
        "total_handle_time_minutes": [10, 15, 20, 12],
        "reopens_count": [0, 0, 1, 0],
    })

    macro_usage = pd.DataFrame({
        "ticket_id": ["T1", "T1", "T2", "T4"],
        "macro_id": ["M1", "M2", "M1", "M3"],
        "applied_at": pd.date_range("2024-01-01", periods=4),
    })

    macro_clusters = pd.DataFrame({
        "macro_id": ["M1", "M2", "M3"],
        "macro_name": ["Refund", "Shipping", "Billing"],
        "category": ["Refund", "Shipping", "Billing"],
        "macro_effectiveness_index": [75.0, 65.0, 80.0],
    })

    usage = compute_agent_macro_usage(tickets, macro_usage, macro_clusters)

    assert "agent_id" in usage.columns
    assert "macro_id" in usage.columns
    assert "usage_count" in usage.columns


def test_identify_best_practice_agents():
    """Test identifying best practice agents."""
    from src.agent_analysis import identify_best_practice_agents

    agent_metrics = pd.DataFrame({
        "agent_id": ["A1", "A2", "A3", "A4", "A5"],
        "avg_csat": [4.8, 4.5, 3.0, 4.9, 2.5],
        "avg_handle_time": [10, 15, 30, 8, 40],
        "agent_effectiveness_score": [90, 80, 50, 95, 30],
    })

    agent_macro_usage = pd.DataFrame({
        "agent_id": ["A1", "A1", "A2", "A3", "A4", "A5"],
        "macro_id": ["M1", "M2", "M1", "M2", "M1", "M2"],
        "usage_count": [10, 5, 8, 3, 12, 2],
        "macro_effectiveness_index": [80, 70, 80, 70, 80, 70],
    })

    result = identify_best_practice_agents(agent_metrics, agent_macro_usage, top_n=2)

    assert "top_performers" in result
    assert "best_macro_selectors" in result
    assert "macro_selection_correlation" in result
    assert len(result["top_performers"]) <= 2


def test_get_agent_recommendations():
    """Test agent recommendations."""
    from src.agent_analysis import get_agent_recommendations

    agent_macro_usage = pd.DataFrame({
        "agent_id": ["A1", "A1", "A2"],
        "macro_id": ["M1", "M2", "M1"],
    })

    macro_clusters = pd.DataFrame({
        "macro_id": ["M1", "M2", "M3", "M4"],
        "macro_name": ["Macro 1", "Macro 2", "Macro 3", "Macro 4"],
        "category": ["Cat1", "Cat1", "Cat2", "Cat2"],
        "macro_effectiveness_index": [75.0, 65.0, 80.0, 90.0],
        "has_sufficient_usage": [True, True, True, True],
        "usage_count": [100, 50, 80, 120],
    })

    recommendations = get_agent_recommendations("A1", agent_macro_usage, macro_clusters)

    # A1 uses M1 and M2, so recommendations should include M3, M4
    assert len(recommendations) <= 5
    assert "macro_id" in recommendations.columns
