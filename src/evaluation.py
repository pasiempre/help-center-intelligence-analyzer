"""
Evaluation and reporting module.

Generates human-readable insights and recommendations about macro effectiveness.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from src.config import (
    CLUSTER_SUMMARY_FILE,
    EVALUATION_REPORT_FILE,
    MACRO_CLUSTERS_FILE,
)
from src.utils import format_percentage


def generate_evaluation_report(
    macro_clusters_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
) -> str:
    """
    Generate a comprehensive text report on macro effectiveness.

    Args:
        macro_clusters_df: DataFrame with macro clusters and scores
        cluster_summary_df: DataFrame with cluster summaries

    Returns:
        Report text string
    """
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("MACRO HELP-CENTER INTELLIGENCE ANALYZER")
    report_lines.append("Evaluation Report")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Overall landscape
    total_macros = len(macro_clusters_df)
    active_macros = len(macro_clusters_df[macro_clusters_df["is_active"]])
    unused_macros = len(macro_clusters_df[macro_clusters_df["usage_count"] == 0])
    low_usage_macros = len(
        macro_clusters_df[
            (macro_clusters_df["usage_count"] > 0)
            & (macro_clusters_df["usage_count"] < 5)
        ]
    )

    report_lines.append("📊 MACRO LANDSCAPE OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Total macros: {total_macros}")
    report_lines.append(f"Active macros: {active_macros}")
    report_lines.append(f"Unused macros: {unused_macros} ({format_percentage(unused_macros / total_macros)})")
    report_lines.append(f"Low usage macros (<5 uses): {low_usage_macros}")
    report_lines.append("")

    # Effectiveness distribution
    avg_effectiveness = macro_clusters_df["macro_effectiveness_index"].mean()
    high_impact = len(
        macro_clusters_df[macro_clusters_df["macro_effectiveness_index"] >= 75]
    )
    low_effectiveness = len(
        macro_clusters_df[macro_clusters_df["macro_effectiveness_index"] < 25]
    )

    report_lines.append("🎯 EFFECTIVENESS DISTRIBUTION")
    report_lines.append("-" * 80)
    report_lines.append(f"Average effectiveness index: {avg_effectiveness:.1f}/100")
    report_lines.append(f"High impact macros (≥75): {high_impact}")
    report_lines.append(f"Low effectiveness macros (<25): {low_effectiveness}")
    report_lines.append("")

    # Top performing macros
    report_lines.append("⭐ TOP 10 MACROS BY EFFECTIVENESS")
    report_lines.append("-" * 80)
    top_10 = macro_clusters_df.nlargest(10, "macro_effectiveness_index")[
        ["macro_id", "macro_name", "macro_effectiveness_index", "usage_count", "avg_csat"]
    ]
    for _, row in top_10.iterrows():
        report_lines.append(
            f"  {row['macro_id']}: {row['macro_name'][:50]} "
            f"(Score: {row['macro_effectiveness_index']:.1f}, Uses: {row['usage_count']}, CSAT: {row['avg_csat']:.2f})"
        )
    report_lines.append("")

    # Bottom performing macros
    report_lines.append("⚠️  BOTTOM 10 MACROS BY EFFECTIVENESS")
    report_lines.append("-" * 80)
    bottom_10 = macro_clusters_df[macro_clusters_df["has_sufficient_usage"]].nsmallest(
        10, "macro_effectiveness_index"
    )[["macro_id", "macro_name", "macro_effectiveness_index", "usage_count", "avg_csat"]]
    for _, row in bottom_10.iterrows():
        report_lines.append(
            f"  {row['macro_id']}: {row['macro_name'][:50]} "
            f"(Score: {row['macro_effectiveness_index']:.1f}, Uses: {row['usage_count']}, CSAT: {row['avg_csat']:.2f})"
        )
    report_lines.append("")

    # Underused gems
    report_lines.append("💎 UNDERUSED GEMS (High effectiveness, low usage)")
    report_lines.append("-" * 80)
    underused_gems = macro_clusters_df[
        macro_clusters_df["macro_category"] == "Underused Gem"
    ].nlargest(10, "macro_effectiveness_index")[
        ["macro_id", "macro_name", "macro_effectiveness_index", "usage_count"]
    ]
    if len(underused_gems) > 0:
        for _, row in underused_gems.iterrows():
            report_lines.append(
                f"  {row['macro_id']}: {row['macro_name'][:50]} "
                f"(Score: {row['macro_effectiveness_index']:.1f}, Uses: {row['usage_count']})"
            )
        report_lines.append("")
        report_lines.append(
            f"💡 RECOMMENDATION: Promote these {len(underused_gems)} high-performing macros "
            "to increase their usage."
        )
    else:
        report_lines.append("  None identified.")
    report_lines.append("")

    # Cluster analysis
    report_lines.append("🔍 TOPIC CLUSTER ANALYSIS")
    report_lines.append("-" * 80)
    for _, cluster in cluster_summary_df.iterrows():
        report_lines.append(
            f"  Cluster: {cluster['cluster_label']} | "
            f"Macros: {cluster['num_macros']} | "
            f"Avg Effectiveness: {cluster['avg_effectiveness']:.1f} | "
            f"Total Usage: {cluster['total_usage']}"
        )
    report_lines.append("")

    # Consolidation candidates
    consolidation_clusters = cluster_summary_df[
        cluster_summary_df["consolidation_candidate"]
    ]
    if len(consolidation_clusters) > 0:
        report_lines.append("🔧 CONSOLIDATION OPPORTUNITIES")
        report_lines.append("-" * 80)
        for _, cluster in consolidation_clusters.iterrows():
            report_lines.append(
                f"  Cluster '{cluster['cluster_label']}' has {cluster['num_macros']} macros "
                f"with avg effectiveness {cluster['avg_effectiveness']:.1f}. "
                f"Consider consolidating or rewriting."
            )
        report_lines.append("")

    # Recommendations
    report_lines.append("📋 KEY RECOMMENDATIONS")
    report_lines.append("-" * 80)

    if unused_macros > 0:
        report_lines.append(
            f"1. Archive or deprecate {unused_macros} unused macros to reduce clutter."
        )

    if len(underused_gems) > 0:
        report_lines.append(
            f"2. Promote {len(underused_gems)} underused but high-performing macros "
            "through agent training."
        )

    if low_effectiveness > 0:
        report_lines.append(
            f"3. Review and rewrite {low_effectiveness} low-effectiveness macros."
        )

    if len(consolidation_clusters) > 0:
        report_lines.append(
            f"4. Consolidate {len(consolidation_clusters)} topic clusters with many low-performing macros."
        )

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    return "\n".join(report_lines)


def evaluate_all(
    macro_clusters_path: str = MACRO_CLUSTERS_FILE,
    cluster_summary_path: str = CLUSTER_SUMMARY_FILE,
    save: bool = True,
) -> str:
    """
    Load clustering results and generate evaluation report.

    Args:
        macro_clusters_path: Path to macro clusters CSV
        cluster_summary_path: Path to cluster summary CSV
        save: Whether to save report to file

    Returns:
        Report text string
    """
    logger.info("Loading clustering results...")
    macro_clusters = pd.read_csv(macro_clusters_path)
    cluster_summary = pd.read_csv(cluster_summary_path)

    logger.info("Generating evaluation report...")
    report = generate_evaluation_report(macro_clusters, cluster_summary)

    if save:
        logger.info(f"Saving report to {EVALUATION_REPORT_FILE}...")
        with open(EVALUATION_REPORT_FILE, "w") as f:
            f.write(report)
        logger.info("✓ Evaluation complete!")

    # Log key findings
    total_macros = len(macro_clusters)
    unused = (macro_clusters["usage_count"] == 0).sum()
    logger.info(f"Report generated: {total_macros} total macros, {unused} unused")

    return report


if __name__ == "__main__":
    evaluate_all()
