"""Tests for similarity analysis module."""

import pandas as pd
import pytest


def test_compute_similarity_matrix():
    """Test similarity matrix computation."""
    from src.similarity_analysis import compute_similarity_matrix

    # Create test macros with known similarity
    macros = pd.DataFrame({
        "macro_id": ["M1", "M2", "M3"],
        "macro_name": ["Refund Request", "Refund Process", "Shipping Delay"],
        "macro_body": [
            "We apologize for the inconvenience. We will process your refund within 3-5 business days.",
            "We are sorry for the trouble. Your refund will be processed in 3-5 business days.",
            "We apologize for the delay in shipping. Your package is on its way.",
        ],
    })

    similarity_df, similarity_matrix = compute_similarity_matrix(macros)

    # Check shape
    assert similarity_matrix.shape == (3, 3)
    assert len(similarity_df) == 3

    # Diagonal should be 1 (self-similarity)
    assert abs(similarity_matrix[0, 0] - 1.0) < 0.01
    assert abs(similarity_matrix[1, 1] - 1.0) < 0.01

    # M1 and M2 should be more similar to each other than to M3
    assert similarity_matrix[0, 1] > similarity_matrix[0, 2]


def test_find_redundant_pairs():
    """Test finding redundant pairs."""
    from src.similarity_analysis import compute_similarity_matrix, find_redundant_pairs

    macros = pd.DataFrame({
        "macro_id": ["M1", "M2", "M3"],
        "macro_name": ["Refund Request", "Refund Process", "Shipping"],
        "macro_body": [
            "We will process your refund request",
            "We will process your refund request",  # Duplicate
            "Your package is shipping today",
        ],
        "category": ["Refund", "Refund", "Shipping"],
        "macro_effectiveness_index": [75.0, 70.0, 60.0],
        "usage_count": [100, 50, 80],
    })

    similarity_df, _ = compute_similarity_matrix(macros)
    pairs = find_redundant_pairs(similarity_df, threshold=0.9, macro_df=macros)

    # M1 and M2 are identical, should be found
    assert len(pairs) >= 1
    assert "macro_a" in pairs.columns
    assert "similarity" in pairs.columns


def test_get_similarity_stats():
    """Test getting similarity statistics."""
    import numpy as np
    from src.similarity_analysis import get_similarity_stats

    # Create a simple similarity matrix
    similarity_matrix = np.array([
        [1.0, 0.8, 0.3],
        [0.8, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ])

    stats = get_similarity_stats(similarity_matrix)

    assert "mean_similarity" in stats
    assert "max_similarity" in stats
    assert "pairs_above_80" in stats
    assert stats["max_similarity"] == 0.8


def test_analyze_similarity_pipeline():
    """Test full similarity analysis pipeline requires files, so just test imports."""
    from src.similarity_analysis import analyze_similarity

    # Just verify it can be imported
    assert callable(analyze_similarity)
