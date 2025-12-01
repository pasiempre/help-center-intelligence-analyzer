"""
Clean and preprocess raw ticket, macro, and macro usage data.

Responsibilities:
- Parse datetimes
- Ensure ID consistency across tables
- Handle missing values
- Create derived fields (resolution_time, is_reopened, etc.)
- Merge and explode macro sequences
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

from src.config import (
    INTERIM_MACROS_FILE,
    INTERIM_MACRO_USAGE_FILE,
    INTERIM_TICKETS_FILE,
    RAW_MACROS_FILE,
    RAW_MACRO_USAGE_FILE,
    RAW_TICKETS_FILE,
)


def clean_tickets(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich tickets data.

    Args:
        tickets_df: Raw tickets DataFrame

    Returns:
        Cleaned tickets DataFrame with derived fields
    """
    df = tickets_df.copy()

    # Parse datetimes
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["resolved_at"] = pd.to_datetime(df["resolved_at"])

    # Derived fields
    df["resolution_time_minutes"] = (
        df["resolved_at"] - df["created_at"]
    ).dt.total_seconds() / 60

    df["is_reopened"] = df["reopens_count"] > 0
    df["is_high_priority"] = df["priority"].isin(["high", "urgent"])
    df["resolved_same_day"] = (
        df["created_at"].dt.date == df["resolved_at"].dt.date
    )

    # Handle missing macro sequences
    df["macro_sequence"] = df["macro_sequence"].fillna("")
    df["final_macro_id"] = df["final_macro_id"].fillna("")

    # Add macro usage flag
    df["has_macro"] = df["macro_sequence"] != ""

    return df


def clean_macros(macros_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean macros data.

    Args:
        macros_df: Raw macros DataFrame

    Returns:
        Cleaned macros DataFrame
    """
    df = macros_df.copy()

    # Parse datetimes
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["updated_at"] = pd.to_datetime(df["updated_at"])

    # Ensure boolean fields
    df["is_active"] = df["is_active"].astype(bool)
    df["is_internal_only"] = df["is_internal_only"].astype(bool)

    # Clean text fields
    df["macro_body"] = df["macro_body"].str.strip()
    df["macro_name"] = df["macro_name"].str.strip()

    return df


def clean_macro_usage(macro_usage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean macro usage data.

    Args:
        macro_usage_df: Raw macro usage DataFrame

    Returns:
        Cleaned macro usage DataFrame
    """
    df = macro_usage_df.copy()

    # Parse datetime
    df["applied_at"] = pd.to_datetime(df["applied_at"])

    # Ensure position is integer
    df["position_in_thread"] = df["position_in_thread"].astype(int)

    return df


def merge_and_validate(
    tickets_df: pd.DataFrame,
    macros_df: pd.DataFrame,
    macro_usage_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate relationships between tables and ensure referential integrity.

    Args:
        tickets_df: Cleaned tickets DataFrame
        macros_df: Cleaned macros DataFrame
        macro_usage_df: Cleaned macro usage DataFrame

    Returns:
        Tuple of validated DataFrames
    """
    # Get valid IDs
    valid_ticket_ids = set(tickets_df["ticket_id"])
    valid_macro_ids = set(macros_df["macro_id"])

    # Filter macro_usage to only valid references
    initial_usage_count = len(macro_usage_df)
    macro_usage_df = macro_usage_df[
        macro_usage_df["ticket_id"].isin(valid_ticket_ids)
        & macro_usage_df["macro_id"].isin(valid_macro_ids)
    ]

    removed_count = initial_usage_count - len(macro_usage_df)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} macro_usage records with invalid references")

    return tickets_df, macros_df, macro_usage_df


def clean_all_data(
    tickets_path: str = RAW_TICKETS_FILE,
    macros_path: str = RAW_MACROS_FILE,
    macro_usage_path: str = RAW_MACRO_USAGE_FILE,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load, clean, and optionally save all data.

    Args:
        tickets_path: Path to raw tickets CSV
        macros_path: Path to raw macros CSV
        macro_usage_path: Path to raw macro usage CSV
        save: Whether to save cleaned data to interim/

    Returns:
        Tuple of (tickets_df, macros_df, macro_usage_df)
    """
    logger.info("Loading raw data...")
    tickets_df = pd.read_csv(tickets_path)
    macros_df = pd.read_csv(macros_path)
    macro_usage_df = pd.read_csv(macro_usage_path)

    logger.info("Cleaning tickets...")
    tickets_df = clean_tickets(tickets_df)

    logger.info("Cleaning macros...")
    macros_df = clean_macros(macros_df)

    logger.info("Cleaning macro usage...")
    macro_usage_df = clean_macro_usage(macro_usage_df)

    logger.info("Validating relationships...")
    tickets_df, macros_df, macro_usage_df = merge_and_validate(
        tickets_df, macros_df, macro_usage_df
    )

    if save:
        logger.info(f"Saving to {INTERIM_TICKETS_FILE.parent}...")
        tickets_df.to_csv(INTERIM_TICKETS_FILE, index=False)
        macros_df.to_csv(INTERIM_MACROS_FILE, index=False)
        macro_usage_df.to_csv(INTERIM_MACRO_USAGE_FILE, index=False)
        logger.info("✓ Data cleaning complete!")

    logger.info(f"  Tickets: {len(tickets_df):,}")
    logger.info(f"  Macros: {len(macros_df):,}")
    logger.info(f"  Macro Usage: {len(macro_usage_df):,}")

    return tickets_df, macros_df, macro_usage_df


if __name__ == "__main__":
    clean_all_data()
