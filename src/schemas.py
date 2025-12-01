"""
Pydantic data models for schema validation.

These models define the expected schema for tickets, macros, and macro usage data.
Used for validation at data ingestion and transformation stages.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class TicketSchema(BaseModel):
    """Schema for a support ticket."""

    ticket_id: str = Field(..., description="Unique ticket identifier")
    created_at: datetime = Field(..., description="Ticket creation timestamp")
    resolved_at: Optional[datetime] = Field(None, description="Ticket resolution timestamp")
    status: str = Field(..., description="Ticket status")
    channel: str = Field(..., description="Communication channel")
    priority: str = Field(..., description="Ticket priority")
    contact_driver: str = Field(..., description="Reason for contact")
    first_response_time_minutes: float = Field(..., ge=0, description="Time to first response")
    total_handle_time_minutes: float = Field(..., ge=0, description="Total handle time")
    replies_count: int = Field(..., ge=0, description="Number of replies")
    reopens_count: int = Field(..., ge=0, description="Number of reopens")
    csat_score: Optional[int] = Field(None, ge=1, le=5, description="CSAT score 1-5")
    csat_response: Optional[str] = Field(None, description="Optional CSAT comment")
    agent_id: str = Field(..., description="Agent identifier")
    macro_sequence: Optional[str] = Field(None, description="Comma-separated macro IDs")
    final_macro_id: Optional[str] = Field(None, description="Last macro used")
    language: str = Field(default="en", description="Ticket language")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid_statuses = {"open", "solved", "closed", "pending"}
        if v.lower() not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        return v.lower()

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        valid_priorities = {"low", "normal", "high", "urgent"}
        if v.lower() not in valid_priorities:
            raise ValueError(f"priority must be one of {valid_priorities}")
        return v.lower()

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        valid_channels = {"email", "chat", "phone", "webform"}
        if v.lower() not in valid_channels:
            raise ValueError(f"channel must be one of {valid_channels}")
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "TKT-000001",
                "created_at": "2025-09-01T10:30:00",
                "resolved_at": "2025-09-01T11:45:00",
                "status": "solved",
                "channel": "email",
                "priority": "normal",
                "contact_driver": "billing_issue",
                "first_response_time_minutes": 15.5,
                "total_handle_time_minutes": 45.0,
                "replies_count": 3,
                "reopens_count": 0,
                "csat_score": 4,
                "agent_id": "AGENT_001",
                "language": "en",
            }
        }


class MacroSchema(BaseModel):
    """Schema for a support macro/template."""

    macro_id: str = Field(..., description="Unique macro identifier")
    macro_name: str = Field(..., min_length=1, description="Macro display name")
    category: str = Field(..., description="Macro category")
    macro_body: str = Field(..., min_length=1, description="Macro text content")
    created_at: datetime = Field(..., description="Macro creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    owner_team: str = Field(..., description="Team owning the macro")
    is_active: bool = Field(default=True, description="Whether macro is active")
    is_internal_only: bool = Field(default=False, description="Internal use only flag")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        valid_categories = {"billing", "technical", "account", "policy", "product", "escalation"}
        if v.lower() not in valid_categories:
            raise ValueError(f"category must be one of {valid_categories}")
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "macro_id": "BILL_001",
                "macro_name": "Billing Inquiry Response",
                "category": "billing",
                "macro_body": "Thank you for contacting us about your billing...",
                "created_at": "2025-01-15T09:00:00",
                "updated_at": "2025-06-20T14:30:00",
                "owner_team": "billing",
                "is_active": True,
                "is_internal_only": False,
            }
        }


class MacroUsageSchema(BaseModel):
    """Schema for a macro usage event."""

    ticket_id: str = Field(..., description="Associated ticket ID")
    macro_id: str = Field(..., description="Macro that was used")
    applied_at: datetime = Field(..., description="When macro was applied")
    position_in_thread: int = Field(..., ge=1, description="Position in conversation")
    agent_id: str = Field(..., description="Agent who applied macro")
    channel: str = Field(..., description="Channel of the ticket")

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "TKT-000001",
                "macro_id": "BILL_001",
                "applied_at": "2025-09-01T10:35:00",
                "position_in_thread": 1,
                "agent_id": "AGENT_001",
                "channel": "email",
            }
        }


class MacroScoreSchema(BaseModel):
    """Schema for computed macro effectiveness scores."""

    macro_id: str = Field(..., description="Macro identifier")
    macro_name: str = Field(..., description="Macro display name")
    category: str = Field(..., description="Macro category")
    usage_count: int = Field(..., ge=0, description="Number of times used")
    avg_csat: float = Field(..., ge=0, le=5, description="Average CSAT when used")
    avg_handle_time: float = Field(..., ge=0, description="Average handle time")
    reopen_rate: float = Field(..., ge=0, le=1, description="Rate of ticket reopens")
    csat_component: float = Field(..., ge=0, le=100, description="CSAT score component")
    handle_time_component: float = Field(..., ge=0, le=100, description="Handle time component")
    reopen_component: float = Field(..., ge=0, le=100, description="Reopen rate component")
    macro_effectiveness_index: float = Field(..., ge=0, le=100, description="Composite score")
    macro_category: str = Field(..., description="Effectiveness category label")
    has_sufficient_usage: bool = Field(..., description="Whether usage meets threshold")

    class Config:
        json_schema_extra = {
            "example": {
                "macro_id": "BILL_001",
                "macro_name": "Billing Inquiry Response",
                "category": "billing",
                "usage_count": 150,
                "avg_csat": 4.2,
                "avg_handle_time": 28.5,
                "reopen_rate": 0.08,
                "csat_component": 72.5,
                "handle_time_component": 68.0,
                "reopen_component": 85.0,
                "macro_effectiveness_index": 74.9,
                "macro_category": "High Impact (Popular)",
                "has_sufficient_usage": True,
            }
        }


class ClusterSummarySchema(BaseModel):
    """Schema for topic cluster summary."""

    cluster_id: int = Field(..., ge=0, description="Cluster identifier")
    cluster_label: str = Field(..., description="Human-readable cluster label")
    num_macros: int = Field(..., ge=0, description="Number of macros in cluster")
    avg_effectiveness: float = Field(..., ge=0, le=100, description="Avg effectiveness")
    median_effectiveness: float = Field(..., ge=0, le=100, description="Median effectiveness")
    total_usage: int = Field(..., ge=0, description="Total usage across cluster")
    avg_usage: float = Field(..., ge=0, description="Average usage per macro")
    num_underused_gems: int = Field(..., ge=0, description="Count of underused gems")
    num_low_effectiveness: int = Field(..., ge=0, description="Count of low performers")
    consolidation_candidate: bool = Field(..., description="Flagged for consolidation")

    class Config:
        json_schema_extra = {
            "example": {
                "cluster_id": 0,
                "cluster_label": "Billing / Payment / Refund",
                "num_macros": 12,
                "avg_effectiveness": 65.2,
                "median_effectiveness": 68.0,
                "total_usage": 1250,
                "avg_usage": 104.2,
                "num_underused_gems": 2,
                "num_low_effectiveness": 1,
                "consolidation_candidate": False,
            }
        }


# Validation functions for DataFrames

def validate_tickets_df(df) -> List[str]:
    """
    Validate a tickets DataFrame against the schema.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    required_cols = [
        "ticket_id", "created_at", "status", "channel", "priority",
        "contact_driver", "total_handle_time_minutes", "csat_score"
    ]
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors  # Return early if missing columns
    
    if df["ticket_id"].duplicated().any():
        errors.append("Duplicate ticket_id values found")
    
    if (df["csat_score"].dropna() < 1).any() or (df["csat_score"].dropna() > 5).any():
        errors.append("csat_score values outside valid range 1-5")
    
    if (df["total_handle_time_minutes"] < 0).any():
        errors.append("Negative total_handle_time_minutes values found")
    
    return errors


def validate_macros_df(df) -> List[str]:
    """
    Validate a macros DataFrame against the schema.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    required_cols = ["macro_id", "macro_name", "category", "macro_body", "is_active"]
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    if df["macro_id"].duplicated().any():
        errors.append("Duplicate macro_id values found")
    
    if df["macro_body"].isna().any() or (df["macro_body"].str.len() == 0).any():
        errors.append("Empty macro_body values found")
    
    return errors


def validate_macro_usage_df(df) -> List[str]:
    """
    Validate a macro_usage DataFrame against the schema.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    required_cols = ["ticket_id", "macro_id", "applied_at", "position_in_thread"]
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    if (df["position_in_thread"] < 1).any():
        errors.append("position_in_thread values less than 1 found")
    
    return errors
