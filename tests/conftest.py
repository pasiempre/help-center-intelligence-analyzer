"""Shared test fixtures for the Macro Help-Center Intelligence Analyzer tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_tickets_df():
    """Create a sample tickets DataFrame for testing."""
    np.random.seed(42)
    n_tickets = 100
    base_date = datetime(2025, 1, 1)
    
    tickets = []
    for i in range(n_tickets):
        created = base_date + timedelta(days=np.random.randint(0, 180))
        resolved = created + timedelta(hours=np.random.randint(1, 48))
        tickets.append({
            "ticket_id": f"TKT-{i:06d}",
            "created_at": created,
            "resolved_at": resolved,
            "status": np.random.choice(["solved", "closed", "open"], p=[0.6, 0.3, 0.1]),
            "channel": np.random.choice(["email", "chat", "phone", "social"]),
            "priority": np.random.choice(["low", "normal", "high", "urgent"]),
            "contact_driver": np.random.choice(["billing", "technical", "account", "feature_request"]),
            "first_response_time_minutes": np.random.uniform(5, 120),
            "total_handle_time_minutes": np.random.uniform(10, 180),
            "replies_count": np.random.randint(1, 10),
            "reopens_count": np.random.randint(0, 3),
            "csat_score": np.random.choice([None, 1, 2, 3, 4, 5]),
            "agent_id": f"AGENT_{np.random.randint(1, 20):03d}",
        })
    return pd.DataFrame(tickets)


@pytest.fixture
def sample_macros_df():
    """Create a sample macros DataFrame for testing."""
    categories = ["billing", "technical", "account", "general"]
    macros = []
    
    for i, cat in enumerate(categories):
        for j in range(5):
            macros.append({
                "macro_id": f"{cat.upper()[:4]}_{j:03d}",
                "macro_name": f"{cat.title()} Response {j+1}",
                "category": cat,
                "macro_body": f"Thank you for contacting us about your {cat} issue. We appreciate your patience.",
                "created_at": datetime(2024, 1, 1) + timedelta(days=i*30 + j*7),
                "updated_at": datetime(2025, 1, 1),
                "owner_team": cat,
                "is_active": True,
            })
    return pd.DataFrame(macros)


@pytest.fixture
def sample_macro_usage_df(sample_tickets_df, sample_macros_df):
    """Create a sample macro usage DataFrame for testing."""
    np.random.seed(42)
    
    usage = []
    ticket_ids = sample_tickets_df["ticket_id"].tolist()
    macro_ids = sample_macros_df["macro_id"].tolist()
    
    for ticket_id in ticket_ids[:80]:  # 80% of tickets use macros
        n_macros = np.random.randint(1, 4)
        used_macros = np.random.choice(macro_ids, size=n_macros, replace=False)
        for pos, macro_id in enumerate(used_macros, 1):
            usage.append({
                "ticket_id": ticket_id,
                "macro_id": macro_id,
                "applied_at": datetime(2025, 1, 15) + timedelta(hours=np.random.randint(0, 100)),
                "position_in_thread": pos,
            })
    return pd.DataFrame(usage)


@pytest.fixture
def sample_macro_features_df(sample_macros_df):
    """Create a sample macro features DataFrame for effectiveness scoring."""
    np.random.seed(42)
    n_macros = len(sample_macros_df)
    
    features = sample_macros_df.copy()
    features["usage_count"] = np.random.randint(10, 500, n_macros)
    features["avg_csat"] = np.random.uniform(2.5, 5.0, n_macros)
    features["avg_handle_time_minutes"] = np.random.uniform(15, 120, n_macros)
    features["reopen_rate"] = np.random.uniform(0.0, 0.3, n_macros)
    features["tickets_with_macro"] = np.random.randint(10, 200, n_macros)
    
    return features


@pytest.fixture
def high_impact_macro_features():
    """Create macro features for a high-impact macro."""
    return pd.DataFrame([{
        "macro_id": "BILL_001",
        "macro_name": "Billing Best",
        "category": "billing",
        "usage_count": 300,
        "avg_csat": 4.8,
        "avg_handle_time_minutes": 20.0,
        "reopen_rate": 0.02,
        "tickets_with_macro": 150,
    }])


@pytest.fixture
def underperformer_macro_features():
    """Create macro features for an underperforming macro."""
    return pd.DataFrame([{
        "macro_id": "TECH_001",
        "macro_name": "Tech Bad",
        "category": "technical",
        "usage_count": 200,
        "avg_csat": 2.1,
        "avg_handle_time_minutes": 120.0,
        "reopen_rate": 0.45,
        "tickets_with_macro": 100,
    }])


@pytest.fixture
def empty_df():
    """Create an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def minimal_tickets_df():
    """Create minimal tickets DataFrame with required columns only."""
    return pd.DataFrame({
        "ticket_id": ["TKT-001"],
        "created_at": [datetime(2025, 1, 1)],
        "status": ["solved"],
        "channel": ["email"],
        "priority": ["normal"],
        "contact_driver": ["billing"],
        "total_handle_time_minutes": [30.0],
        "csat_score": [4],
    })
