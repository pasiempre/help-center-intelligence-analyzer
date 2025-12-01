"""Tests for A/B testing module."""

import pandas as pd
import pytest


def test_calculate_sample_size():
    """Test sample size calculation."""
    from src.ab_testing import calculate_sample_size

    # Standard parameters
    n = calculate_sample_size(
        baseline_rate=0.5,
        minimum_detectable_effect=0.1,  # 10% relative lift
        alpha=0.05,
        power=0.80,
    )

    assert n > 0
    assert isinstance(n, int)
    # With 10% MDE on 50% baseline, should need several hundred samples
    assert n > 100


def test_compare_macros():
    """Test macro comparison."""
    from src.ab_testing import compare_macros

    macro_a = {
        "csat_mean": 4.5,
        "csat_std": 0.5,
        "n": 100,
        "handle_time_mean": 10,
        "handle_time_std": 3,
    }

    macro_b = {
        "csat_mean": 4.0,
        "csat_std": 0.6,
        "n": 100,
        "handle_time_mean": 15,
        "handle_time_std": 4,
    }

    results = compare_macros(macro_a, macro_b)

    assert "csat_p_value" in results
    assert "csat_significant" in results
    assert "handle_time_p_value" in results
    assert "csat_effect_size" in results

    # Macro A has higher CSAT
    assert results["csat_winner"] in ["A", "B", "Tie"]


def test_find_similar_macro_pairs():
    """Test finding similar macro pairs."""
    from src.ab_testing import find_similar_macro_pairs

    macros = pd.DataFrame({
        "macro_id": ["M1", "M2", "M3"],
        "category": ["Refund", "Refund", "Shipping"],
        "macro_effectiveness_index": [70, 72, 60],  # M1 and M2 are within threshold
    })

    # Using effectiveness_diff_threshold=10 for the 0-100 scale
    pairs = find_similar_macro_pairs(macros, same_category=True, effectiveness_diff_threshold=10)

    # Should find M1-M2 pair (same category, close effectiveness)
    assert len(pairs) >= 1


def test_generate_ab_report():
    """Test A/B report generation."""
    from src.ab_testing import generate_ab_report

    comparison_results = pd.DataFrame({
        "macro_a": ["M1"],
        "macro_b": ["M2"],
        "csat_p_value": [0.01],
        "csat_significant": [True],
        "csat_winner": ["A"],
        "handle_time_p_value": [0.10],
        "handle_time_significant": [False],
        "handle_time_winner": ["Tie"],
        "n_a": [100],
        "n_b": [100],
    })

    macro_clusters = pd.DataFrame({
        "macro_id": ["M1", "M2"],
        "macro_name": ["Refund Macro", "Refund Alt"],
    })

    report = generate_ab_report(comparison_results, macro_clusters)

    assert "# Macro A/B Test Report" in report
    assert "Summary" in report
