"""
Agent-level analysis module.

Analyzes how individual agents use macros and correlates with their performance.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from src.config import (
    INTERIM_DIR,
    MACRO_CLUSTERS_FILE,
    PROCESSED_DIR,
    RAW_MACRO_USAGE_FILE,
    TICKETS_FEATURES_FILE,
)

AGENT_PERFORMANCE_FILE = PROCESSED_DIR / "agent_performance.csv"
AGENT_MACRO_USAGE_FILE = PROCESSED_DIR / "agent_macro_usage.csv"


def compute_agent_metrics(
    tickets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-agent performance metrics.

    Args:
        tickets_df: Tickets DataFrame with features

    Returns:
        DataFrame with agent-level metrics
    """
    agent_metrics = tickets_df.groupby("agent_id").agg(
        total_tickets=("ticket_id", "count"),
        avg_csat=("csat_score", "mean"),
        avg_handle_time=("total_handle_time_minutes", "mean"),
        median_handle_time=("total_handle_time_minutes", "median"),
        reopen_rate=("reopens_count", lambda x: (x > 0).mean()),
        avg_first_response_time=("first_response_time_minutes", "mean"),
        resolution_rate=("status", lambda x: (x == "solved").mean()),
        tickets_with_macros=("num_macros_used", lambda x: (x > 0).sum()),
        avg_macros_per_ticket=("num_macros_used", "mean"),
    ).reset_index()

    # Calculate macro adoption rate
    agent_metrics["macro_adoption_rate"] = (
        agent_metrics["tickets_with_macros"] / agent_metrics["total_tickets"]
    )

    # Rank agents
    agent_metrics["csat_rank"] = agent_metrics["avg_csat"].rank(ascending=False)
    agent_metrics["handle_time_rank"] = agent_metrics["avg_handle_time"].rank(ascending=True)

    # Composite score (normalized)
    agent_metrics["agent_effectiveness_score"] = (
        0.4 * agent_metrics["avg_csat"].rank(pct=True)
        + 0.3 * (1 - agent_metrics["avg_handle_time"].rank(pct=True))
        + 0.2 * (1 - agent_metrics["reopen_rate"].rank(pct=True))
        + 0.1 * agent_metrics["resolution_rate"].rank(pct=True)
    ) * 100

    return agent_metrics


def compute_agent_macro_usage(
    tickets_df: pd.DataFrame,
    macro_usage_df: pd.DataFrame,
    macro_clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute which macros each agent uses and their effectiveness with each.

    Args:
        tickets_df: Tickets DataFrame
        macro_usage_df: Macro usage DataFrame
        macro_clusters_df: Macro clusters DataFrame with scores

    Returns:
        DataFrame with agent-macro usage patterns
    """
    # Join tickets with macro usage
    ticket_macro = macro_usage_df.merge(
        tickets_df[["ticket_id", "agent_id", "csat_score", "total_handle_time_minutes", "reopens_count"]],
        on="ticket_id",
        how="left",
    )

    # Group by agent and macro
    agent_macro = ticket_macro.groupby(["agent_id", "macro_id"]).agg(
        usage_count=("ticket_id", "count"),
        avg_csat_with_macro=("csat_score", "mean"),
        avg_handle_time_with_macro=("total_handle_time_minutes", "mean"),
        reopen_rate_with_macro=("reopens_count", lambda x: (x > 0).mean()),
    ).reset_index()

    # Add macro metadata
    macro_info = macro_clusters_df[["macro_id", "macro_name", "category", "macro_effectiveness_index"]]
    agent_macro = agent_macro.merge(macro_info, on="macro_id", how="left")

    return agent_macro


def identify_best_practice_agents(
    agent_metrics: pd.DataFrame,
    agent_macro_usage: pd.DataFrame,
    top_n: int = 10,
) -> Dict:
    """
    Identify agents who consistently use high-effectiveness macros.

    Args:
        agent_metrics: Agent performance DataFrame
        agent_macro_usage: Agent-macro usage DataFrame
        top_n: Number of top agents to identify

    Returns:
        Dictionary with best practice agent analysis
    """
    # Average macro effectiveness used by each agent
    agent_macro_eff = agent_macro_usage.groupby("agent_id").agg(
        avg_macro_effectiveness=("macro_effectiveness_index", "mean"),
        unique_macros_used=("macro_id", "nunique"),
        total_macro_uses=("usage_count", "sum"),
    ).reset_index()

    # Join with agent metrics
    agent_combined = agent_metrics.merge(agent_macro_eff, on="agent_id", how="left")

    # Top performers by effectiveness score
    top_performers = agent_combined.nlargest(top_n, "agent_effectiveness_score")

    # Agents with best macro selection
    best_macro_selectors = agent_combined.nlargest(top_n, "avg_macro_effectiveness")

    # Correlation between macro effectiveness and agent performance
    correlation = agent_combined["avg_macro_effectiveness"].corr(
        agent_combined["agent_effectiveness_score"]
    )

    return {
        "top_performers": top_performers,
        "best_macro_selectors": best_macro_selectors,
        "macro_selection_correlation": correlation,
        "agent_combined": agent_combined,
    }


def get_agent_recommendations(
    agent_id: str,
    agent_macro_usage: pd.DataFrame,
    macro_clusters_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Get macro recommendations for a specific agent based on their usage patterns.

    Args:
        agent_id: Agent ID
        agent_macro_usage: Agent-macro usage DataFrame
        macro_clusters_df: Macro clusters DataFrame
        top_n: Number of recommendations

    Returns:
        DataFrame with recommended macros
    """
    # Macros this agent already uses
    used_macros = set(
        agent_macro_usage[agent_macro_usage["agent_id"] == agent_id]["macro_id"]
    )

    # High-effectiveness macros they don't use
    recommendations = macro_clusters_df[
        (~macro_clusters_df["macro_id"].isin(used_macros))
        & (macro_clusters_df["macro_effectiveness_index"] >= 70)
        & (macro_clusters_df["has_sufficient_usage"])
    ].nlargest(top_n, "macro_effectiveness_index")

    return recommendations[["macro_id", "macro_name", "category", "macro_effectiveness_index", "usage_count"]]


def analyze_agents(
    tickets_path: str = TICKETS_FEATURES_FILE,
    macro_usage_path: str = None,
    macro_clusters_path: str = MACRO_CLUSTERS_FILE,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run full agent analysis pipeline.

    Args:
        tickets_path: Path to tickets features CSV
        macro_usage_path: Path to macro usage CSV
        macro_clusters_path: Path to macro clusters CSV
        save: Whether to save results

    Returns:
        Tuple of (agent_metrics, agent_macro_usage, best_practice_analysis)
    """
    logger.info("Loading data...")
    tickets_df = pd.read_csv(tickets_path)
    macro_clusters_df = pd.read_csv(macro_clusters_path)

    # Load macro usage from interim if not specified
    if macro_usage_path is None:
        macro_usage_path = INTERIM_DIR / "macro_usage_cleaned.csv"
    macro_usage_df = pd.read_csv(macro_usage_path)

    logger.info("Computing agent metrics...")
    agent_metrics = compute_agent_metrics(tickets_df)

    logger.info("Computing agent-macro usage patterns...")
    agent_macro_usage = compute_agent_macro_usage(
        tickets_df, macro_usage_df, macro_clusters_df
    )

    logger.info("Identifying best practice agents...")
    best_practice = identify_best_practice_agents(agent_metrics, agent_macro_usage)

    logger.info(f"Macro selection correlation with performance: {best_practice['macro_selection_correlation']:.3f}")

    if save:
        logger.info(f"Saving agent performance to {AGENT_PERFORMANCE_FILE}...")
        agent_metrics.to_csv(AGENT_PERFORMANCE_FILE, index=False)

        logger.info(f"Saving agent macro usage to {AGENT_MACRO_USAGE_FILE}...")
        agent_macro_usage.to_csv(AGENT_MACRO_USAGE_FILE, index=False)

    return agent_metrics, agent_macro_usage, best_practice


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyze_agents()
