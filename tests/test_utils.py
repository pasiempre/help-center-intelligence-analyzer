"""Tests for utility functions."""

import pytest

from src.utils import (
    clean_text,
    format_percentage,
    normalize_score,
    safe_divide,
)


def test_clean_text():
    """Test text cleaning."""
    text = "  Hello   World  "
    result = clean_text(text)
    assert result == "hello world"


def test_safe_divide():
    """Test safe division."""
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(10, 0) == 0.0
    assert safe_divide(10, 0, default=1.0) == 1.0


def test_normalize_score():
    """Test score normalization."""
    result = normalize_score(5, 0, 10)
    assert result == 50.0

    result = normalize_score(10, 0, 10)
    assert result == 100.0

    result = normalize_score(0, 0, 10, reverse=True)
    assert result == 100.0


def test_format_percentage():
    """Test percentage formatting."""
    assert format_percentage(0.234) == "23.4%"
    assert format_percentage(0.5, decimals=0) == "50%"
