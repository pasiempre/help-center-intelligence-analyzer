"""
Time series analysis module for macro usage trends.

Analyzes how macro usage patterns change over time.
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

from src.config import (
    INTERIM_DIR,
    MACRO_CLUSTERS_FILE,
    PROCESSED_DIR,
    TICKETS_FEATURES_FILE,
)

USAGE_TRENDS_FILE = PROCESSED_DIR / "usage_trends.csv"
DAILY_METRICS_FILE = PROCESSED_DIR / "daily_metrics.csv"


def compute_daily_metrics(
    tickets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily aggregate metrics.

    Args:
        tickets_df: Tickets DataFrame

    Returns:
        DataFrame with daily metrics
    """
    # Ensure datetime
    tickets_df = tickets_df.copy()
    tickets_df["created_at"] = pd.to_datetime(tickets_df["created_at"])
    tickets_df["date"] = tickets_df["created_at"].dt.date

    daily = tickets_df.groupby("date").agg(
        total_tickets=("ticket_id", "count"),
        avg_csat=("csat_score", "mean"),
        avg_handle_time=("total_handle_time_minutes", "mean"),
        tickets_with_macros=("num_macros_used", lambda x: (x > 0).sum()),
        total_macros_used=("num_macros_used", "sum"),
        reopen_rate=("reopens_count", lambda x: (x > 0).mean()),
    ).reset_index()

    daily["macro_adoption_rate"] = daily["tickets_with_macros"] / daily["total_tickets"]
    daily["macros_per_ticket"] = daily["total_macros_used"] / daily["total_tickets"]

    # Rolling averages
    daily["csat_7d_avg"] = daily["avg_csat"].rolling(7, min_periods=1).mean()
    daily["handle_time_7d_avg"] = daily["avg_handle_time"].rolling(7, min_periods=1).mean()
    daily["macro_adoption_7d_avg"] = daily["macro_adoption_rate"].rolling(7, min_periods=1).mean()

    return daily


def compute_macro_usage_trends(
    macro_usage_df: pd.DataFrame,
    macro_clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute weekly usage trends for each macro.

    Args:
        macro_usage_df: Macro usage DataFrame
        macro_clusters_df: Macro clusters DataFrame

    Returns:
        DataFrame with macro usage trends
    """
    usage = macro_usage_df.copy()
    usage["applied_at"] = pd.to_datetime(usage["applied_at"])
    usage["week"] = usage["applied_at"].dt.to_period("W").astype(str)

    # Weekly usage by macro
    weekly_usage = usage.groupby(["week", "macro_id"]).agg(
        usage_count=("ticket_id", "count"),
    ).reset_index()

    # Pivot to wide format
    usage_pivot = weekly_usage.pivot(
        index="macro_id",
        columns="week",
        values="usage_count",
    ).fillna(0)

    # Add macro metadata
    macro_info = macro_clusters_df[["macro_id", "macro_name", "category", "macro_effectiveness_index"]]
    usage_pivot = usage_pivot.reset_index()
    usage_pivot = usage_pivot.merge(macro_info, on="macro_id", how="left")

    # Calculate trend direction
    week_cols = [c for c in usage_pivot.columns if c not in ["macro_id", "macro_name", "category", "macro_effectiveness_index"]]
    if len(week_cols) >= 4:
        # Compare last 2 weeks to previous 2 weeks
        recent = usage_pivot[week_cols[-2:]].sum(axis=1)
        previous = usage_pivot[week_cols[-4:-2]].sum(axis=1)
        usage_pivot["trend"] = (recent - previous) / (previous + 1)
        usage_pivot["trend_direction"] = usage_pivot["trend"].apply(
            lambda x: "📈 Rising" if x > 0.1 else ("📉 Declining" if x < -0.1 else "➡️ Stable")
        )
    else:
        usage_pivot["trend"] = 0
        usage_pivot["trend_direction"] = "➡️ Stable"

    return usage_pivot


def compute_category_trends(
    macro_usage_df: pd.DataFrame,
    macro_clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute weekly usage trends by category.

    Args:
        macro_usage_df: Macro usage DataFrame
        macro_clusters_df: Macro clusters DataFrame

    Returns:
        DataFrame with category-level trends
    """
    # Add category to usage
    usage = macro_usage_df.merge(
        macro_clusters_df[["macro_id", "category"]],
        on="macro_id",
        how="left",
    )

    usage["applied_at"] = pd.to_datetime(usage["applied_at"])
    usage["week"] = usage["applied_at"].dt.to_period("W").astype(str)

    # Weekly usage by category
    category_trends = usage.groupby(["week", "category"]).agg(
        usage_count=("ticket_id", "count"),
        unique_macros=("macro_id", "nunique"),
    ).reset_index()

    return category_trends


def detect_anomalies(
    daily_metrics: pd.DataFrame,
    std_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Detect anomalous days in metrics.

    Args:
        daily_metrics: Daily metrics DataFrame
        std_threshold: Number of standard deviations for anomaly

    Returns:
        DataFrame with anomaly flags
    """
    metrics = daily_metrics.copy()

    for col in ["avg_csat", "avg_handle_time", "macro_adoption_rate"]:
        mean = metrics[col].mean()
        std = metrics[col].std()
        metrics[f"{col}_anomaly"] = (
            (metrics[col] < mean - std_threshold * std)
            | (metrics[col] > mean + std_threshold * std)
        )

    metrics["any_anomaly"] = (
        metrics["avg_csat_anomaly"]
        | metrics["avg_handle_time_anomaly"]
        | metrics["macro_adoption_rate_anomaly"]
    )

    return metrics


def analyze_trends(
    tickets_path: str = TICKETS_FEATURES_FILE,
    macro_usage_path: str = None,
    macro_clusters_path: str = MACRO_CLUSTERS_FILE,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full time series analysis pipeline.

    Args:
        tickets_path: Path to tickets features CSV
        macro_usage_path: Path to macro usage CSV
        macro_clusters_path: Path to macro clusters CSV
        save: Whether to save results

    Returns:
        Tuple of (daily_metrics, usage_trends)
    """
    logger.info("Loading data...")
    tickets_df = pd.read_csv(tickets_path)
    macro_clusters_df = pd.read_csv(macro_clusters_path)

    if macro_usage_path is None:
        macro_usage_path = INTERIM_DIR / "macro_usage_cleaned.csv"
    macro_usage_df = pd.read_csv(macro_usage_path)

    logger.info("Computing daily metrics...")
    daily_metrics = compute_daily_metrics(tickets_df)
    daily_metrics = detect_anomalies(daily_metrics)

    logger.info("Computing macro usage trends...")
    usage_trends = compute_macro_usage_trends(macro_usage_df, macro_clusters_df)

    anomaly_days = daily_metrics["any_anomaly"].sum()
    logger.info(f"Found {anomaly_days} anomalous days")

    if save:
        logger.info(f"Saving daily metrics to {DAILY_METRICS_FILE}...")
        daily_metrics.to_csv(DAILY_METRICS_FILE, index=False)

        logger.info(f"Saving usage trends to {USAGE_TRENDS_FILE}...")
        usage_trends.to_csv(USAGE_TRENDS_FILE, index=False)

    return daily_metrics, usage_trends


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyze_trends()
