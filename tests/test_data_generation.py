"""Tests for data generation module."""

import pandas as pd
import pytest

from src.data_generation import (
    generate_all_data,
    generate_macros,
    generate_tickets,
)


def test_generate_macros():
    """Test macro generation."""
    macros_df = generate_macros(num_macros=50)

    assert len(macros_df) == 50
    assert "macro_id" in macros_df.columns
    assert "macro_body" in macros_df.columns
    assert "category" in macros_df.columns


def test_generate_tickets():
    """Test ticket generation."""
    tickets_df = generate_tickets(num_tickets=100)

    assert len(tickets_df) == 100
    assert "ticket_id" in tickets_df.columns
    assert "csat_score" in tickets_df.columns
    assert "total_handle_time_minutes" in tickets_df.columns


def test_generate_all_data():
    """Test full data generation pipeline."""
    macros_df, tickets_df, macro_usage_df = generate_all_data(
        num_tickets=100,
        num_macros=20,
        save=False,
    )

    assert len(macros_df) == 20
    assert len(tickets_df) == 100
    assert len(macro_usage_df) > 0
