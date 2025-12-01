"""Tests for Pydantic schema validation."""

import pandas as pd
import pytest
from datetime import datetime

from src.schemas import (
    MacroSchema,
    TicketSchema,
    MacroUsageSchema,
    validate_tickets_df,
    validate_macros_df,
    validate_macro_usage_df,
)


class TestTicketSchema:
    """Tests for TicketSchema validation."""

    def test_valid_ticket(self):
        """Test valid ticket passes validation."""
        ticket = TicketSchema(
            ticket_id="TKT-000001",
            created_at=datetime(2025, 9, 1, 10, 30),
            resolved_at=datetime(2025, 9, 1, 11, 45),
            status="solved",
            channel="email",
            priority="normal",
            contact_driver="billing_issue",
            first_response_time_minutes=15.5,
            total_handle_time_minutes=45.0,
            replies_count=3,
            reopens_count=0,
            csat_score=4,
            agent_id="AGENT_001",
        )
        assert ticket.ticket_id == "TKT-000001"
        assert ticket.status == "solved"

    def test_invalid_status(self):
        """Test invalid status raises error."""
        with pytest.raises(ValueError, match="status must be one of"):
            TicketSchema(
                ticket_id="TKT-000001",
                created_at=datetime(2025, 9, 1, 10, 30),
                status="invalid_status",
                channel="email",
                priority="normal",
                contact_driver="billing_issue",
                first_response_time_minutes=15.5,
                total_handle_time_minutes=45.0,
                replies_count=3,
                reopens_count=0,
                agent_id="AGENT_001",
            )

    def test_invalid_csat_score(self):
        """Test CSAT score outside valid range raises error."""
        with pytest.raises(ValueError):
            TicketSchema(
                ticket_id="TKT-000001",
                created_at=datetime(2025, 9, 1, 10, 30),
                status="solved",
                channel="email",
                priority="normal",
                contact_driver="billing_issue",
                first_response_time_minutes=15.5,
                total_handle_time_minutes=45.0,
                replies_count=3,
                reopens_count=0,
                csat_score=10,  # Invalid - should be 1-5
                agent_id="AGENT_001",
            )

    def test_negative_handle_time(self):
        """Test negative handle time raises error."""
        with pytest.raises(ValueError):
            TicketSchema(
                ticket_id="TKT-000001",
                created_at=datetime(2025, 9, 1, 10, 30),
                status="solved",
                channel="email",
                priority="normal",
                contact_driver="billing_issue",
                first_response_time_minutes=15.5,
                total_handle_time_minutes=-10.0,  # Invalid
                replies_count=3,
                reopens_count=0,
                agent_id="AGENT_001",
            )


class TestMacroSchema:
    """Tests for MacroSchema validation."""

    def test_valid_macro(self):
        """Test valid macro passes validation."""
        macro = MacroSchema(
            macro_id="BILL_001",
            macro_name="Billing Response",
            category="billing",
            macro_body="Thank you for contacting us...",
            created_at=datetime(2025, 1, 15),
            updated_at=datetime(2025, 6, 20),
            owner_team="billing",
        )
        assert macro.macro_id == "BILL_001"
        assert macro.is_active is True  # Default

    def test_invalid_category(self):
        """Test invalid category raises error."""
        with pytest.raises(ValueError, match="category must be one of"):
            MacroSchema(
                macro_id="BILL_001",
                macro_name="Billing Response",
                category="invalid_category",
                macro_body="Thank you...",
                created_at=datetime(2025, 1, 15),
                updated_at=datetime(2025, 6, 20),
                owner_team="billing",
            )


class TestDataFrameValidation:
    """Tests for DataFrame validation functions."""

    def test_valid_tickets_df(self):
        """Test valid tickets DataFrame passes validation."""
        df = pd.DataFrame({
            "ticket_id": ["TKT-001", "TKT-002"],
            "created_at": ["2025-09-01", "2025-09-02"],
            "status": ["solved", "open"],
            "channel": ["email", "chat"],
            "priority": ["normal", "high"],
            "contact_driver": ["billing", "technical"],
            "total_handle_time_minutes": [30, 45],
            "csat_score": [4, 5],
        })
        errors = validate_tickets_df(df)
        assert len(errors) == 0

    def test_missing_columns_tickets(self):
        """Test missing columns are detected."""
        df = pd.DataFrame({
            "ticket_id": ["TKT-001"],
            # Missing other required columns
        })
        errors = validate_tickets_df(df)
        assert len(errors) > 0
        assert "Missing required columns" in errors[0]

    def test_duplicate_ticket_ids(self):
        """Test duplicate ticket IDs are detected."""
        df = pd.DataFrame({
            "ticket_id": ["TKT-001", "TKT-001"],  # Duplicate
            "created_at": ["2025-09-01", "2025-09-02"],
            "status": ["solved", "open"],
            "channel": ["email", "chat"],
            "priority": ["normal", "high"],
            "contact_driver": ["billing", "technical"],
            "total_handle_time_minutes": [30, 45],
            "csat_score": [4, 5],
        })
        errors = validate_tickets_df(df)
        assert any("Duplicate" in e for e in errors)

    def test_invalid_csat_in_df(self):
        """Test invalid CSAT scores are detected in DataFrame."""
        df = pd.DataFrame({
            "ticket_id": ["TKT-001", "TKT-002"],
            "created_at": ["2025-09-01", "2025-09-02"],
            "status": ["solved", "open"],
            "channel": ["email", "chat"],
            "priority": ["normal", "high"],
            "contact_driver": ["billing", "technical"],
            "total_handle_time_minutes": [30, 45],
            "csat_score": [4, 10],  # 10 is invalid
        })
        errors = validate_tickets_df(df)
        assert any("csat_score" in e for e in errors)

    def test_valid_macros_df(self):
        """Test valid macros DataFrame passes validation."""
        df = pd.DataFrame({
            "macro_id": ["BILL_001", "TECH_001"],
            "macro_name": ["Billing 1", "Tech 1"],
            "category": ["billing", "technical"],
            "macro_body": ["Thank you...", "I understand..."],
            "is_active": [True, True],
        })
        errors = validate_macros_df(df)
        assert len(errors) == 0

    def test_empty_macro_body(self):
        """Test empty macro body is detected."""
        df = pd.DataFrame({
            "macro_id": ["BILL_001"],
            "macro_name": ["Billing 1"],
            "category": ["billing"],
            "macro_body": [""],  # Empty
            "is_active": [True],
        })
        errors = validate_macros_df(df)
        assert any("macro_body" in e for e in errors)

    def test_valid_macro_usage_df(self):
        """Test valid macro usage DataFrame passes validation."""
        df = pd.DataFrame({
            "ticket_id": ["TKT-001", "TKT-002"],
            "macro_id": ["BILL_001", "TECH_001"],
            "applied_at": ["2025-09-01 10:30:00", "2025-09-01 11:00:00"],
            "position_in_thread": [1, 2],
        })
        errors = validate_macro_usage_df(df)
        assert len(errors) == 0

    def test_invalid_position_in_thread(self):
        """Test position_in_thread < 1 is detected."""
        df = pd.DataFrame({
            "ticket_id": ["TKT-001"],
            "macro_id": ["BILL_001"],
            "applied_at": ["2025-09-01 10:30:00"],
            "position_in_thread": [0],  # Invalid
        })
        errors = validate_macro_usage_df(df)
        assert any("position_in_thread" in e for e in errors)
