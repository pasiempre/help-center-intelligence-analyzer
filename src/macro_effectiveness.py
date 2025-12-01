"""
Compute macro effectiveness scores.

This is the core scoring module that calculates a composite Macro Effectiveness Index
based on CSAT impact, handle time impact, and reopen rate.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from src.config import (
    EFFECTIVENESS_WEIGHTS,
    MACRO_FEATURES_BASE_FILE,
    MACRO_SCORES_FILE,
    MIN_USAGE_FOR_SCORING,
    TICKETS_FEATURES_FILE,
)
from src.utils import normalize_score, safe_divide


def calculate_baseline_metrics(tickets_df: pd.DataFrame) -> dict:
    """
    Calculate global baseline metrics for comparison.

    Args:
        tickets_df: Tickets DataFrame with features

    Returns:
        Dictionary of baseline metrics
    """
    # Overall baselines
    baseline = {
        "overall_avg_csat": tickets_df["csat_score"].mean(),
        "overall_avg_handle_time": tickets_df["total_handle_time_minutes"].mean(),
        "overall_reopen_rate": (tickets_df["reopens_count"] > 0).mean(),
    }

    # Baselines by contact driver
    driver_baselines = (
        tickets_df.groupby("contact_driver")
        .agg(
            avg_csat=("csat_score", "mean"),
            avg_handle_time=("total_handle_time_minutes", "mean"),
            reopen_rate=("reopens_count", lambda x: (x > 0).mean()),
        )
        .to_dict("index")
    )

    baseline["by_contact_driver"] = driver_baselines

    return baseline


def compute_macro_scores(
    macro_features_df: pd.DataFrame,
    baseline_metrics: dict,
) -> pd.DataFrame:
    """
    Compute macro effectiveness scores.

    Args:
        macro_features_df: DataFrame with macro features
        baseline_metrics: Dictionary of baseline metrics

    Returns:
        DataFrame with effectiveness scores
    """
    df = macro_features_df.copy()

    # Filter to macros with sufficient usage
    df["has_sufficient_usage"] = df["usage_count"] >= MIN_USAGE_FOR_SCORING

    # Get global baselines
    baseline_csat = baseline_metrics["overall_avg_csat"]
    baseline_handle_time = baseline_metrics["overall_avg_handle_time"]
    baseline_reopen_rate = baseline_metrics["overall_reopen_rate"]

    # Calculate lifts/deltas
    df["csat_lift"] = df["avg_csat"] - baseline_csat
    df["handle_time_delta"] = baseline_handle_time - df["avg_handle_time"]  # Positive = faster
    df["reopen_rate_delta"] = baseline_reopen_rate - df["reopen_rate"]  # Positive = fewer reopens

    # Normalize components to 0-100 scale
    # CSAT component (higher is better)
    df["csat_component"] = df.apply(
        lambda row: normalize_score(
            row["avg_csat"],
            df["avg_csat"].min(),
            df["avg_csat"].max(),
            reverse=False,
        )
        if row["has_sufficient_usage"]
        else 50.0,
        axis=1,
    )

    # Handle time component (lower is better, so reverse=True)
    df["handle_time_component"] = df.apply(
        lambda row: normalize_score(
            row["avg_handle_time"],
            df["avg_handle_time"].min(),
            df["avg_handle_time"].max(),
            reverse=True,
        )
        if row["has_sufficient_usage"]
        else 50.0,
        axis=1,
    )

    # Reopen component (lower is better, so reverse=True)
    df["reopen_component"] = df.apply(
        lambda row: normalize_score(
            row["reopen_rate"],
            df["reopen_rate"].min(),
            df["reopen_rate"].max(),
            reverse=True,
        )
        if row["has_sufficient_usage"]
        else 50.0,
        axis=1,
    )

    # Weighted composite score
    df["macro_effectiveness_index"] = (
        EFFECTIVENESS_WEIGHTS["csat"] * df["csat_component"]
        + EFFECTIVENESS_WEIGHTS["handle_time"] * df["handle_time_component"]
        + EFFECTIVENESS_WEIGHTS["reopen_rate"] * df["reopen_component"]
    )

    # Set score to 0 for macros without sufficient usage
    df.loc[~df["has_sufficient_usage"], "macro_effectiveness_index"] = 0.0

    # Add percentile ranks
    df["effectiveness_percentile"] = df["macro_effectiveness_index"].rank(pct=True) * 100

    return df


def categorize_macros(macro_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize macros based on effectiveness and usage.

    Args:
        macro_scores_df: DataFrame with macro scores

    Returns:
        DataFrame with category labels
    """
    df = macro_scores_df.copy()

    # Define categories
    def categorize(row):
        if not row["has_sufficient_usage"]:
            return "Unused/Low Usage"
        elif row["effectiveness_percentile"] >= 75:
            if row["usage_count"] >= df["usage_count"].quantile(0.75):
                return "High Impact (Popular)"
            else:
                return "Underused Gem"
        elif row["effectiveness_percentile"] >= 50:
            return "Moderate Effectiveness"
        elif row["effectiveness_percentile"] >= 25:
            return "Below Average"
        else:
            return "Low Effectiveness"

    df["macro_category"] = df.apply(categorize, axis=1)

    return df


def score_all_macros(
    macro_features_path: str = MACRO_FEATURES_BASE_FILE,
    tickets_features_path: str = TICKETS_FEATURES_FILE,
    save: bool = True,
) -> pd.DataFrame:
    """
    Load features, calculate baselines, and score all macros.

    Args:
        macro_features_path: Path to macro features CSV
        tickets_features_path: Path to ticket features CSV
        save: Whether to save scores to processed/

    Returns:
        DataFrame with macro scores
    """
    logger.info("Loading feature data...")
    macro_features = pd.read_csv(macro_features_path)
    tickets_features = pd.read_csv(tickets_features_path)

    logger.info("Calculating baseline metrics...")
    baselines = calculate_baseline_metrics(tickets_features)
    logger.info(f"  Overall avg CSAT: {baselines['overall_avg_csat']:.2f}")
    logger.info(f"  Overall avg handle time: {baselines['overall_avg_handle_time']:.1f} min")
    logger.info(f"  Overall reopen rate: {baselines['overall_reopen_rate']:.1%}")

    logger.info("Computing macro effectiveness scores...")
    macro_scores = compute_macro_scores(macro_features, baselines)

    logger.info("Categorizing macros...")
    macro_scores = categorize_macros(macro_scores)

    if save:
        logger.info(f"Saving macro scores to {MACRO_SCORES_FILE}...")
        macro_scores.to_csv(MACRO_SCORES_FILE, index=False)
        logger.info("✓ Macro scoring complete!")

    # Log summary
    logger.info("Macro Effectiveness Summary:")
    for category, count in macro_scores["macro_category"].value_counts().items():
        logger.info(f"  {category}: {count}")
    
    top_5 = macro_scores.nlargest(5, "macro_effectiveness_index")[
        ["macro_id", "macro_name", "macro_effectiveness_index", "usage_count"]
    ]
    logger.info("Top 5 macros by effectiveness:")
    for _, row in top_5.iterrows():
        logger.info(f"  {row['macro_id']}: {row['macro_effectiveness_index']:.1f} (uses: {row['usage_count']})")

    return macro_scores


if __name__ == "__main__":
    score_all_macros()
