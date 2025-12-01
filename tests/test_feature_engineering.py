"""Tests for feature engineering module."""

import pandas as pd
import pytest

from src.feature_engineering import (
    create_ticket_features,
    create_macro_features,
)


def test_create_ticket_features():
    """Test ticket feature creation."""
    # Mock data
    tickets_df = pd.DataFrame({
        "ticket_id": ["TKT-001", "TKT-002"],
        "csat_score": [4, 3],
        "total_handle_time_minutes": [30, 45],
        "reopens_count": [0, 1],
    })

    macro_usage_df = pd.DataFrame({
        "ticket_id": ["TKT-001", "TKT-001"],
        "macro_id": ["MACRO-1", "MACRO-2"],
        "position_in_thread": [1, 2],
        "agent_id": ["AGENT_001", "AGENT_001"],
    })

    result = create_ticket_features(tickets_df, macro_usage_df)

    assert "num_macros_used" in result.columns
    assert result[result["ticket_id"] == "TKT-001"]["num_macros_used"].iloc[0] == 2
    assert result[result["ticket_id"] == "TKT-002"]["num_macros_used"].iloc[0] == 0


def test_create_macro_features():
    """Test macro feature creation."""
    # Mock data
    macro_usage_df = pd.DataFrame({
        "macro_id": ["MACRO-1", "MACRO-1", "MACRO-2"],
        "ticket_id": ["TKT-001", "TKT-002", "TKT-003"],
        "agent_id": ["AGENT_001", "AGENT_002", "AGENT_001"],
    })

    tickets_df = pd.DataFrame({
        "ticket_id": ["TKT-001", "TKT-002", "TKT-003"],
        "csat_score": [4, 3, 5],
        "total_handle_time_minutes": [30, 45, 20],
        "reopens_count": [0, 1, 0],
        "contact_driver": ["billing", "technical", "account"],
        "channel": ["email", "chat", "phone"],
        "priority": ["normal", "high", "low"],
    })

    macros_df = pd.DataFrame({
        "macro_id": ["MACRO-1", "MACRO-2"],
        "macro_name": ["Macro 1", "Macro 2"],
        "category": ["billing", "technical"],
    })

    result = create_macro_features(macro_usage_df, tickets_df, macros_df)

    assert len(result) == 2
    assert "usage_count" in result.columns
    assert "avg_csat" in result.columns
    assert result[result["macro_id"] == "MACRO-1"]["usage_count"].iloc[0] == 2
