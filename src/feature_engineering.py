"""
Feature engineering for tickets and macros.

Creates two main feature sets:
1. Ticket-level features: macro usage patterns per ticket
2. Macro-level features: aggregated performance metrics per macro
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

from src.config import (
    INTERIM_MACROS_FILE,
    INTERIM_MACRO_USAGE_FILE,
    INTERIM_TICKETS_FILE,
    MACRO_FEATURES_BASE_FILE,
    TICKETS_FEATURES_FILE,
)


def create_ticket_features(
    tickets_df: pd.DataFrame,
    macro_usage_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create ticket-level features related to macro usage.

    Args:
        tickets_df: Cleaned tickets DataFrame
        macro_usage_df: Cleaned macro usage DataFrame

    Returns:
        Tickets DataFrame with additional macro usage features
    """
    df = tickets_df.copy()

    # Count macros per ticket
    macro_counts = (
        macro_usage_df.groupby("ticket_id")
        .agg(
            num_macros_used=("macro_id", "count"),
            macro_count_unique=("macro_id", "nunique"),
        )
        .reset_index()
    )

    df = df.merge(macro_counts, on="ticket_id", how="left")
    df["num_macros_used"] = df["num_macros_used"].fillna(0).astype(int)
    df["macro_count_unique"] = df["macro_count_unique"].fillna(0).astype(int)

    # First and last macro used
    first_macros = (
        macro_usage_df[macro_usage_df["position_in_thread"] == 1]
        .groupby("ticket_id")["macro_id"]
        .first()
        .reset_index()
        .rename(columns={"macro_id": "macro_position_first"})
    )

    last_macros = (
        macro_usage_df.sort_values("position_in_thread", ascending=False)
        .groupby("ticket_id")["macro_id"]
        .first()
        .reset_index()
        .rename(columns={"macro_id": "macro_position_last"})
    )

    df = df.merge(first_macros, on="ticket_id", how="left")
    df = df.merge(last_macros, on="ticket_id", how="left")

    # Fill NaN for tickets without macros
    df["macro_position_first"] = df["macro_position_first"].fillna("")
    df["macro_position_last"] = df["macro_position_last"].fillna("")

    # Flag for macro usage
    df["is_macro_used"] = df["num_macros_used"] > 0

    return df


def create_macro_features(
    macro_usage_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    macros_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create macro-level aggregate features.

    Args:
        macro_usage_df: Cleaned macro usage DataFrame
        tickets_df: Tickets DataFrame with features
        macros_df: Cleaned macros DataFrame

    Returns:
        DataFrame with one row per macro and aggregate metrics
    """
    # Merge macro usage with ticket outcomes
    usage_with_outcomes = macro_usage_df.merge(
        tickets_df[
            [
                "ticket_id",
                "csat_score",
                "total_handle_time_minutes",
                "reopens_count",
                "contact_driver",
                "channel",
                "priority",
            ]
        ],
        on="ticket_id",
        how="left",
    )

    # Aggregate by macro_id
    macro_agg = (
        usage_with_outcomes.groupby("macro_id")
        .agg(
            usage_count=("ticket_id", "count"),
            avg_csat=("csat_score", "mean"),
            median_csat=("csat_score", "median"),
            avg_handle_time=("total_handle_time_minutes", "mean"),
            median_handle_time=("total_handle_time_minutes", "median"),
            reopen_rate=("reopens_count", lambda x: (x > 0).mean()),
            unique_agents_used=("agent_id", "nunique"),
            unique_contact_drivers=("contact_driver", "nunique"),
        )
        .reset_index()
    )

    # Merge with macro metadata
    macro_features = macros_df.merge(macro_agg, on="macro_id", how="left")

    # Fill NaN for unused macros
    macro_features["usage_count"] = macro_features["usage_count"].fillna(0).astype(int)
    macro_features["avg_csat"] = macro_features["avg_csat"].fillna(0)
    macro_features["median_csat"] = macro_features["median_csat"].fillna(0)
    macro_features["avg_handle_time"] = macro_features["avg_handle_time"].fillna(0)
    macro_features["median_handle_time"] = macro_features["median_handle_time"].fillna(0)
    macro_features["reopen_rate"] = macro_features["reopen_rate"].fillna(0)
    macro_features["unique_agents_used"] = macro_features["unique_agents_used"].fillna(0).astype(int)
    macro_features["unique_contact_drivers"] = (
        macro_features["unique_contact_drivers"].fillna(0).astype(int)
    )

    return macro_features


def engineer_all_features(
    tickets_path: str = INTERIM_TICKETS_FILE,
    macros_path: str = INTERIM_MACROS_FILE,
    macro_usage_path: str = INTERIM_MACRO_USAGE_FILE,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cleaned data and engineer all features.

    Args:
        tickets_path: Path to cleaned tickets CSV
        macros_path: Path to cleaned macros CSV
        macro_usage_path: Path to cleaned macro usage CSV
        save: Whether to save feature sets to processed/

    Returns:
        Tuple of (tickets_features_df, macro_features_df)
    """
    logger.info("Loading cleaned data...")
    tickets_df = pd.read_csv(tickets_path, parse_dates=["created_at", "resolved_at"])
    macros_df = pd.read_csv(macros_path, parse_dates=["created_at", "updated_at"])
    macro_usage_df = pd.read_csv(macro_usage_path, parse_dates=["applied_at"])

    logger.info("Creating ticket-level features...")
    tickets_features = create_ticket_features(tickets_df, macro_usage_df)

    logger.info("Creating macro-level features...")
    macro_features = create_macro_features(macro_usage_df, tickets_features, macros_df)

    if save:
        logger.info(f"Saving feature sets to {TICKETS_FEATURES_FILE.parent}...")
        tickets_features.to_csv(TICKETS_FEATURES_FILE, index=False)
        macro_features.to_csv(MACRO_FEATURES_BASE_FILE, index=False)
        logger.info("✓ Feature engineering complete!")

    logger.info(f"  Ticket features shape: {tickets_features.shape}")
    logger.info(f"  Macro features shape: {macro_features.shape}")

    return tickets_features, macro_features


if __name__ == "__main__":
    engineer_all_features()
