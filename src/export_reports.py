"""
Export and reporting module for generating reports.

Generates Excel exports, PDF reports, and summary documents.
"""

import io
import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

from src.config import (
    MACRO_CLUSTERS_FILE,
    PROCESSED_DIR,
    REPORTS_DIR,
    TICKETS_FEATURES_FILE,
)


def create_excel_export(
    macro_clusters_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    daily_metrics: Optional[pd.DataFrame] = None,
    ab_results: Optional[pd.DataFrame] = None,
    agent_metrics: Optional[pd.DataFrame] = None,
) -> bytes:
    """
    Create comprehensive Excel export with multiple sheets.

    Args:
        macro_clusters_df: Macro clusters DataFrame
        tickets_df: Tickets DataFrame
        daily_metrics: Optional daily metrics DataFrame
        ab_results: Optional A/B test results DataFrame
        agent_metrics: Optional agent metrics DataFrame

    Returns:
        Excel file as bytes
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Macro summary sheet
        macro_summary = macro_clusters_df[[
            "macro_id", "macro_name", "category", "topic_cluster",
            "macro_effectiveness_index", "usage_count", "effectiveness_category",
        ]].copy()
        macro_summary.to_excel(writer, sheet_name="Macro Summary", index=False)

        # Top performers
        top_performers = macro_summary.nlargest(20, "macro_effectiveness_index")
        top_performers.to_excel(writer, sheet_name="Top Performers", index=False)

        # Low performers (action needed)
        low_performers = macro_summary.nsmallest(20, "macro_effectiveness_index")
        low_performers.to_excel(writer, sheet_name="Needs Improvement", index=False)

        # Ticket summary
        ticket_summary = tickets_df.groupby("ticket_type").agg(
            count=("ticket_id", "count"),
            avg_csat=("csat_score", "mean"),
            avg_handle_time=("total_handle_time_minutes", "mean"),
            avg_macros_used=("num_macros_used", "mean"),
        ).reset_index()
        ticket_summary.to_excel(writer, sheet_name="Ticket Summary", index=False)

        # Category breakdown
        category_stats = macro_summary.groupby("category").agg(
            macro_count=("macro_id", "count"),
            avg_effectiveness=("macro_effectiveness_index", "mean"),
            total_usage=("usage_count", "sum"),
        ).reset_index()
        category_stats.to_excel(writer, sheet_name="Category Stats", index=False)

        # Optional sheets
        if daily_metrics is not None:
            daily_metrics.to_excel(writer, sheet_name="Daily Metrics", index=False)

        if ab_results is not None and len(ab_results) > 0:
            ab_results.to_excel(writer, sheet_name="A-B Test Results", index=False)

        if agent_metrics is not None:
            agent_metrics.to_excel(writer, sheet_name="Agent Performance", index=False)

    return output.getvalue()


def generate_markdown_report(
    macro_clusters_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    redundant_pairs: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate markdown summary report.

    Args:
        macro_clusters_df: Macro clusters DataFrame
        tickets_df: Tickets DataFrame
        redundant_pairs: Optional redundant pairs DataFrame

    Returns:
        Markdown report string
    """
    report = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report.append(f"# Macro Effectiveness Report")
    report.append(f"*Generated: {now}*\n")

    # Executive Summary
    report.append("## Executive Summary\n")

    total_macros = len(macro_clusters_df)
    avg_effectiveness = macro_clusters_df["macro_effectiveness_index"].mean()
    total_usage = macro_clusters_df["usage_count"].sum()

    report.append(f"- **Total Macros Analyzed**: {total_macros}")
    report.append(f"- **Average Effectiveness Index**: {avg_effectiveness:.3f}")
    report.append(f"- **Total Macro Applications**: {total_usage:,}")
    report.append(f"- **Tickets Analyzed**: {len(tickets_df):,}\n")

    # Effectiveness Distribution
    report.append("## Effectiveness Distribution\n")
    eff_counts = macro_clusters_df["effectiveness_category"].value_counts()
    for cat, count in eff_counts.items():
        pct = count / total_macros * 100
        report.append(f"- **{cat}**: {count} macros ({pct:.1f}%)")
    report.append("")

    # Top Performers
    report.append("## Top 10 Most Effective Macros\n")
    report.append("| Rank | Macro Name | Category | Effectiveness | Usage |")
    report.append("|------|------------|----------|---------------|-------|")

    top10 = macro_clusters_df.nlargest(10, "macro_effectiveness_index")
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        report.append(
            f"| {i} | {row['macro_name'][:30]} | {row['category']} | "
            f"{row['macro_effectiveness_index']:.3f} | {row['usage_count']} |"
        )
    report.append("")

    # Low Performers
    report.append("## Macros Needing Attention\n")
    report.append("| Macro Name | Category | Effectiveness | Issue |")
    report.append("|------------|----------|---------------|-------|")

    bottom10 = macro_clusters_df.nsmallest(10, "macro_effectiveness_index")
    for _, row in bottom10.iterrows():
        issue = "Low usage" if row["usage_count"] < 10 else "Poor performance"
        report.append(
            f"| {row['macro_name'][:30]} | {row['category']} | "
            f"{row['macro_effectiveness_index']:.3f} | {issue} |"
        )
    report.append("")

    # Redundancy
    if redundant_pairs is not None and len(redundant_pairs) > 0:
        report.append("## Potential Redundancies\n")
        report.append("The following macro pairs have high content similarity and could potentially be merged:\n")

        for _, row in redundant_pairs.head(10).iterrows():
            report.append(
                f"- **{row['macro_a_name']}** ↔ **{row['macro_b_name']}** "
                f"(Similarity: {row['similarity']:.1%})"
            )
        report.append("")

    # Recommendations
    report.append("## Recommendations\n")

    # Calculate recommendations
    low_usage = macro_clusters_df[macro_clusters_df["usage_count"] < 5]
    high_eff_low_use = macro_clusters_df[
        (macro_clusters_df["macro_effectiveness_index"] > 0.7)
        & (macro_clusters_df["usage_count"] < macro_clusters_df["usage_count"].median())
    ]

    report.append("### Quick Wins")
    if len(high_eff_low_use) > 0:
        report.append(f"- Promote {len(high_eff_low_use)} high-performing but underutilized macros")
    report.append("")

    report.append("### Cleanup Opportunities")
    if len(low_usage) > 0:
        report.append(f"- Review {len(low_usage)} rarely-used macros for potential retirement")
    if redundant_pairs is not None and len(redundant_pairs) > 0:
        report.append(f"- Consider merging {len(redundant_pairs)} potentially redundant macro pairs")
    report.append("")

    return "\n".join(report)


def save_report(
    report_content: str,
    filename: str = "macro_effectiveness_report.md",
) -> str:
    """
    Save markdown report to file.

    Args:
        report_content: Markdown report string
        filename: Output filename

    Returns:
        Path to saved file
    """
    output_path = REPORTS_DIR / filename
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(report_content)

    logger.info(f"Report saved to {output_path}")
    return str(output_path)


def generate_csv_exports(
    macro_clusters_df: pd.DataFrame,
    export_prefix: str = "export",
) -> Dict[str, str]:
    """
    Generate individual CSV exports.

    Args:
        macro_clusters_df: Macro clusters DataFrame
        export_prefix: Prefix for filenames

    Returns:
        Dict mapping export name to file path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exports = {}

    # Full data
    full_path = REPORTS_DIR / f"{export_prefix}_full_{timestamp}.csv"
    macro_clusters_df.to_csv(full_path, index=False)
    exports["full"] = str(full_path)

    # Summary by category
    category_summary = macro_clusters_df.groupby("category").agg(
        macro_count=("macro_id", "count"),
        avg_effectiveness=("macro_effectiveness_index", "mean"),
        total_usage=("usage_count", "sum"),
    ).reset_index()
    category_path = REPORTS_DIR / f"{export_prefix}_by_category_{timestamp}.csv"
    category_summary.to_csv(category_path, index=False)
    exports["by_category"] = str(category_path)

    # Action items
    action_items = macro_clusters_df[
        (macro_clusters_df["effectiveness_category"] == "Low")
        | (macro_clusters_df["usage_count"] < 5)
    ][["macro_id", "macro_name", "category", "macro_effectiveness_index", "usage_count"]]
    action_path = REPORTS_DIR / f"{export_prefix}_action_items_{timestamp}.csv"
    action_items.to_csv(action_path, index=False)
    exports["action_items"] = str(action_path)

    logger.info(f"Generated {len(exports)} CSV exports")
    return exports


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Export module loaded.")
    print("Use create_excel_export() for Excel files.")
    print("Use generate_markdown_report() for markdown reports.")
