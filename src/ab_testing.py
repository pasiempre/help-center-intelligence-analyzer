"""
A/B Testing framework for macro comparisons.

Provides statistical tools for comparing macro effectiveness.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

from src.config import MACRO_CLUSTERS_FILE, PROCESSED_DIR

AB_RESULTS_FILE = PROCESSED_DIR / "ab_test_results.csv"


def calculate_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Calculate required sample size for A/B test.

    Args:
        baseline_rate: Current conversion/success rate
        minimum_detectable_effect: Minimum effect size to detect
        alpha: Significance level
        power: Statistical power

    Returns:
        Required sample size per variant
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)

    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Pooled variance
    p_bar = (p1 + p2) / 2
    variance = p1 * (1 - p1) + p2 * (1 - p2)

    n = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(variance)) ** 2 / (p1 - p2) ** 2

    return int(np.ceil(n))


def compare_macros(
    macro_a_data: Dict[str, float],
    macro_b_data: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Compare two macros using statistical tests.

    Args:
        macro_a_data: Dict with 'csat_mean', 'csat_std', 'n', 'handle_time_mean', 'handle_time_std'
        macro_b_data: Same structure for macro B
        alpha: Significance level

    Returns:
        Dict with comparison results
    """
    results = {}

    # CSAT comparison (two-sample t-test)
    csat_t, csat_p = stats.ttest_ind_from_stats(
        macro_a_data["csat_mean"],
        macro_a_data["csat_std"],
        macro_a_data["n"],
        macro_b_data["csat_mean"],
        macro_b_data["csat_std"],
        macro_b_data["n"],
    )

    results["csat_t_stat"] = csat_t
    results["csat_p_value"] = csat_p
    results["csat_significant"] = csat_p < alpha
    results["csat_winner"] = (
        "A" if macro_a_data["csat_mean"] > macro_b_data["csat_mean"] and csat_p < alpha
        else ("B" if macro_b_data["csat_mean"] > macro_a_data["csat_mean"] and csat_p < alpha else "Tie")
    )

    # Handle time comparison
    ht_t, ht_p = stats.ttest_ind_from_stats(
        macro_a_data["handle_time_mean"],
        macro_a_data["handle_time_std"],
        macro_a_data["n"],
        macro_b_data["handle_time_mean"],
        macro_b_data["handle_time_std"],
        macro_b_data["n"],
    )

    results["handle_time_t_stat"] = ht_t
    results["handle_time_p_value"] = ht_p
    results["handle_time_significant"] = ht_p < alpha
    # Lower handle time is better
    results["handle_time_winner"] = (
        "A" if macro_a_data["handle_time_mean"] < macro_b_data["handle_time_mean"] and ht_p < alpha
        else ("B" if macro_b_data["handle_time_mean"] < macro_a_data["handle_time_mean"] and ht_p < alpha else "Tie")
    )

    # Effect sizes (Cohen's d)
    pooled_std_csat = np.sqrt(
        ((macro_a_data["n"] - 1) * macro_a_data["csat_std"] ** 2
         + (macro_b_data["n"] - 1) * macro_b_data["csat_std"] ** 2)
        / (macro_a_data["n"] + macro_b_data["n"] - 2)
    )
    results["csat_effect_size"] = (
        (macro_a_data["csat_mean"] - macro_b_data["csat_mean"]) / pooled_std_csat
        if pooled_std_csat > 0 else 0
    )

    pooled_std_ht = np.sqrt(
        ((macro_a_data["n"] - 1) * macro_a_data["handle_time_std"] ** 2
         + (macro_b_data["n"] - 1) * macro_b_data["handle_time_std"] ** 2)
        / (macro_a_data["n"] + macro_b_data["n"] - 2)
    )
    results["handle_time_effect_size"] = (
        (macro_a_data["handle_time_mean"] - macro_b_data["handle_time_mean"]) / pooled_std_ht
        if pooled_std_ht > 0 else 0
    )

    return results


def find_similar_macro_pairs(
    macro_clusters_df: pd.DataFrame,
    same_category: bool = True,
    effectiveness_diff_threshold: float = 0.3,
) -> List[Tuple[str, str]]:
    """
    Find pairs of macros suitable for comparison.

    Args:
        macro_clusters_df: DataFrame with macro clusters
        same_category: Only compare within same category
        effectiveness_diff_threshold: Max effectiveness difference

    Returns:
        List of macro ID pairs
    """
    pairs = []

    if same_category:
        for category in macro_clusters_df["category"].unique():
            cat_macros = macro_clusters_df[macro_clusters_df["category"] == category]
            for i in range(len(cat_macros)):
                for j in range(i + 1, len(cat_macros)):
                    row_i = cat_macros.iloc[i]
                    row_j = cat_macros.iloc[j]

                    eff_diff = abs(
                        row_i["macro_effectiveness_index"]
                        - row_j["macro_effectiveness_index"]
                    )
                    if eff_diff <= effectiveness_diff_threshold:
                        pairs.append((row_i["macro_id"], row_j["macro_id"]))
    else:
        for i in range(len(macro_clusters_df)):
            for j in range(i + 1, len(macro_clusters_df)):
                row_i = macro_clusters_df.iloc[i]
                row_j = macro_clusters_df.iloc[j]

                eff_diff = abs(
                    row_i["macro_effectiveness_index"]
                    - row_j["macro_effectiveness_index"]
                )
                if eff_diff <= effectiveness_diff_threshold:
                    pairs.append((row_i["macro_id"], row_j["macro_id"]))

    return pairs[:100]  # Limit to 100 pairs


def run_batch_comparisons(
    macro_usage_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    macro_clusters_df: pd.DataFrame,
    min_samples: int = 30,
) -> pd.DataFrame:
    """
    Run batch A/B comparisons for similar macros.

    Args:
        macro_usage_df: Macro usage DataFrame
        tickets_df: Tickets DataFrame
        macro_clusters_df: Macro clusters DataFrame
        min_samples: Minimum samples per macro

    Returns:
        DataFrame with comparison results
    """
    # Compute macro-level stats from usage
    usage_with_outcomes = macro_usage_df.merge(
        tickets_df[["ticket_id", "csat_score", "total_handle_time_minutes"]],
        on="ticket_id",
        how="left",
    )

    macro_stats = usage_with_outcomes.groupby("macro_id").agg(
        n=("ticket_id", "count"),
        csat_mean=("csat_score", "mean"),
        csat_std=("csat_score", "std"),
        handle_time_mean=("total_handle_time_minutes", "mean"),
        handle_time_std=("total_handle_time_minutes", "std"),
    ).reset_index()

    macro_stats["csat_std"] = macro_stats["csat_std"].fillna(0.1)
    macro_stats["handle_time_std"] = macro_stats["handle_time_std"].fillna(1)

    # Find comparable pairs
    pairs = find_similar_macro_pairs(macro_clusters_df)

    results = []
    for macro_a, macro_b in pairs:
        stats_a = macro_stats[macro_stats["macro_id"] == macro_a]
        stats_b = macro_stats[macro_stats["macro_id"] == macro_b]

        if len(stats_a) == 0 or len(stats_b) == 0:
            continue
        if stats_a.iloc[0]["n"] < min_samples or stats_b.iloc[0]["n"] < min_samples:
            continue

        comparison = compare_macros(
            stats_a.iloc[0].to_dict(),
            stats_b.iloc[0].to_dict(),
        )
        comparison["macro_a"] = macro_a
        comparison["macro_b"] = macro_b
        comparison["n_a"] = stats_a.iloc[0]["n"]
        comparison["n_b"] = stats_b.iloc[0]["n"]

        results.append(comparison)

    return pd.DataFrame(results)


def generate_ab_report(
    comparison_results: pd.DataFrame,
    macro_clusters_df: pd.DataFrame,
) -> str:
    """
    Generate human-readable A/B test report.

    Args:
        comparison_results: Results from run_batch_comparisons
        macro_clusters_df: Macro clusters DataFrame

    Returns:
        Markdown report string
    """
    report = ["# Macro A/B Test Report\n"]

    if len(comparison_results) == 0:
        return "# Macro A/B Test Report\n\nNo valid comparisons found."

    # Summary
    significant_csat = comparison_results["csat_significant"].sum()
    significant_ht = comparison_results["handle_time_significant"].sum()

    report.append(f"## Summary")
    report.append(f"- Total comparisons: {len(comparison_results)}")
    report.append(f"- Significant CSAT differences: {significant_csat}")
    report.append(f"- Significant Handle Time differences: {significant_ht}\n")

    # Top findings
    report.append("## Notable Findings\n")

    significant_results = comparison_results[
        comparison_results["csat_significant"]
        | comparison_results["handle_time_significant"]
    ].head(10)

    for _, row in significant_results.iterrows():
        macro_a_name = macro_clusters_df[
            macro_clusters_df["macro_id"] == row["macro_a"]
        ]["macro_name"].values[0] if len(macro_clusters_df[
            macro_clusters_df["macro_id"] == row["macro_a"]
        ]) > 0 else row["macro_a"]

        macro_b_name = macro_clusters_df[
            macro_clusters_df["macro_id"] == row["macro_b"]
        ]["macro_name"].values[0] if len(macro_clusters_df[
            macro_clusters_df["macro_id"] == row["macro_b"]
        ]) > 0 else row["macro_b"]

        report.append(f"### {macro_a_name} vs {macro_b_name}")

        if row["csat_significant"]:
            winner = macro_a_name if row["csat_winner"] == "A" else macro_b_name
            report.append(f"- **CSAT Winner**: {winner} (p={row['csat_p_value']:.4f})")

        if row["handle_time_significant"]:
            winner = macro_a_name if row["handle_time_winner"] == "A" else macro_b_name
            report.append(f"- **Handle Time Winner**: {winner} (p={row['handle_time_p_value']:.4f})")

        report.append("")

    return "\n".join(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage would go here
    print("A/B Testing module loaded. Use run_batch_comparisons() to analyze.")
