"""Tests for NLP clustering module."""

import numpy as np
import pandas as pd
import pytest

from src.nlp_clustering import (
    cluster_macros,
    prepare_macro_texts,
    vectorize_macros,
)


@pytest.fixture
def sample_macro_scores():
    """Create sample macro scores DataFrame for testing."""
    return pd.DataFrame({
        "macro_id": ["BILL_001", "BILL_002", "TECH_001", "TECH_002"],
        "macro_name": [
            "Billing Response 1",
            "Billing Response 2",
            "Technical Support 1",
            "Technical Support 2",
        ],
        "category": ["billing", "billing", "technical", "technical"],
        "macro_body": [
            "Thank you for contacting us about your billing inquiry. We have reviewed your account.",
            "I understand your concern about the billing issue. Your refund has been processed.",
            "I'm sorry you're experiencing technical difficulties. Please try restarting.",
            "Thank you for reporting this technical error. Our team has identified the issue.",
        ],
        "usage_count": [100, 50, 75, 25],
        "macro_effectiveness_index": [72.5, 65.0, 80.0, 55.0],
        "has_sufficient_usage": [True, True, True, True],
    })


def test_prepare_macro_texts(sample_macro_scores):
    """Test text preparation for clustering."""
    result = prepare_macro_texts(sample_macro_scores)
    
    assert "combined_text" in result.columns
    assert "cleaned_text" in result.columns
    assert len(result) == 4
    
    # Check text was combined
    assert "billing" in result.iloc[0]["combined_text"].lower()


def test_vectorize_macros(sample_macro_scores):
    """Test TF-IDF vectorization."""
    prepared = prepare_macro_texts(sample_macro_scores)
    vectors, vectorizer = vectorize_macros(prepared["cleaned_text"])
    
    assert vectors.shape[0] == 4  # 4 macros
    assert vectors.shape[1] > 0  # Some features extracted
    assert vectorizer is not None


def test_cluster_macros(sample_macro_scores):
    """Test KMeans clustering."""
    prepared = prepare_macro_texts(sample_macro_scores)
    vectors, _ = vectorize_macros(prepared["cleaned_text"])
    
    # Cluster into 2 groups
    labels, model = cluster_macros(vectors, n_clusters=2)
    
    assert len(labels) == 4
    assert set(labels).issubset({0, 1})
    assert model is not None


def test_vectorize_with_existing_vectorizer(sample_macro_scores):
    """Test vectorization with pre-fitted vectorizer."""
    prepared = prepare_macro_texts(sample_macro_scores)
    vectors, vectorizer = vectorize_macros(prepared["cleaned_text"])
    
    # Use same vectorizer on new data
    new_texts = pd.Series(["billing payment refund issue"])
    new_vectors, same_vectorizer = vectorize_macros(new_texts, vectorizer=vectorizer)
    
    assert new_vectors.shape[1] == vectors.shape[1]  # Same feature space
    assert same_vectorizer is vectorizer
