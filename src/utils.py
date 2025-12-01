"""
Shared utility functions for the Macro Help-Center Intelligence Analyzer.

Functions for random seeding, text cleaning, validation, and common helpers.
"""

import random
import re
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seed for reproducibility across numpy and random.

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, special characters, etc.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def remove_boilerplate(text: str, boilerplate_patterns: Optional[List[str]] = None) -> str:
    """
    Remove common boilerplate phrases from macro text.

    Args:
        text: Macro text
        boilerplate_patterns: List of regex patterns to remove

    Returns:
        Text with boilerplate removed
    """
    if boilerplate_patterns is None:
        boilerplate_patterns = [
            r"dear customer,?",
            r"thank you for contacting us,?",
            r"we appreciate your patience,?",
            r"best regards,?",
            r"sincerely,?",
            r"thanks,?",
        ]

    text = text.lower()
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame"
) -> None:
    """
    Validate that a DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name of the DataFrame (for error messages)

    Raises:
        ValueError: If required columns are missing
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{name} is missing required columns: {missing_columns}"
        )


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator is zero

    Returns:
        Result of division or default value
    """
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def normalize_score(
    value: float,
    min_val: float,
    max_val: float,
    reverse: bool = False
) -> float:
    """
    Normalize a value to 0-100 scale.

    Args:
        value: Value to normalize
        min_val: Minimum value in range
        max_val: Maximum value in range
        reverse: If True, reverse the scale (lower is better)

    Returns:
        Normalized score 0-100
    """
    if pd.isna(value):
        return 50.0  # Neutral score for missing values

    if max_val == min_val:
        return 50.0

    normalized = (value - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)

    if reverse:
        normalized = 1 - normalized

    return normalized * 100


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal as a percentage string.

    Args:
        value: Decimal value (e.g., 0.234)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string (e.g., "23.4%")
    """
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_minutes(minutes: float) -> str:
    """
    Format minutes as a readable time string.

    Args:
        minutes: Time in minutes

    Returns:
        Formatted string (e.g., "1h 23m" or "45m")
    """
    if pd.isna(minutes):
        return "N/A"

    hours = int(minutes // 60)
    mins = int(minutes % 60)

    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def get_percentile_label(percentile: float) -> str:
    """
    Convert a percentile to a label.

    Args:
        percentile: Percentile value 0-100

    Returns:
        Label string (e.g., "Top 10%", "Bottom 25%")
    """
    if percentile >= 90:
        return "Top 10%"
    elif percentile >= 75:
        return "Top 25%"
    elif percentile >= 50:
        return "Above Average"
    elif percentile >= 25:
        return "Below Average"
    else:
        return "Bottom 25%"
