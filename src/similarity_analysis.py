"""
Similarity analysis module for detecting redundant macros.

Uses cosine similarity to identify macro pairs that may be candidates for consolidation.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

from src.config import (
    MACRO_CLUSTERS_FILE,
    PROCESSED_DIR,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from src.utils import clean_text, remove_boilerplate

SIMILARITY_MATRIX_FILE = PROCESSED_DIR / "similarity_matrix.csv"
REDUNDANT_PAIRS_FILE = PROCESSED_DIR / "redundant_pairs.csv"


def compute_similarity_matrix(
    macro_df: pd.DataFrame,
    text_column: str = "macro_body",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute pairwise cosine similarity between all macros.

    Args:
        macro_df: DataFrame with macro information
        text_column: Column containing macro text

    Returns:
        Tuple of (similarity DataFrame, raw similarity matrix)
    """
    # Clean texts
    texts = macro_df[text_column].fillna("").apply(
        lambda x: remove_boilerplate(clean_text(x))
    )

    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Create labeled DataFrame
    macro_ids = macro_df["macro_id"].tolist()
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=macro_ids,
        columns=macro_ids,
    )

    return similarity_df, similarity_matrix


def find_redundant_pairs(
    similarity_df: pd.DataFrame,
    threshold: float = 0.8,
    macro_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Find macro pairs with similarity above threshold.

    Args:
        similarity_df: Pairwise similarity DataFrame
        threshold: Minimum similarity to flag as redundant
        macro_df: Optional DataFrame with macro metadata

    Returns:
        DataFrame with redundant pair information
    """
    pairs = []
    macro_ids = similarity_df.index.tolist()

    for i, macro_a in enumerate(macro_ids):
        for j, macro_b in enumerate(macro_ids):
            if i >= j:  # Skip diagonal and duplicates
                continue

            similarity = similarity_df.loc[macro_a, macro_b]
            if similarity >= threshold:
                pair_info = {
                    "macro_a": macro_a,
                    "macro_b": macro_b,
                    "similarity": similarity,
                }

                # Add metadata if available
                if macro_df is not None:
                    macro_a_row = macro_df[macro_df["macro_id"] == macro_a].iloc[0]
                    macro_b_row = macro_df[macro_df["macro_id"] == macro_b].iloc[0]

                    pair_info["name_a"] = macro_a_row.get("macro_name", "")
                    pair_info["name_b"] = macro_b_row.get("macro_name", "")
                    pair_info["category_a"] = macro_a_row.get("category", "")
                    pair_info["category_b"] = macro_b_row.get("category", "")
                    pair_info["usage_a"] = macro_a_row.get("usage_count", 0)
                    pair_info["usage_b"] = macro_b_row.get("usage_count", 0)
                    pair_info["effectiveness_a"] = macro_a_row.get("macro_effectiveness_index", 0)
                    pair_info["effectiveness_b"] = macro_b_row.get("macro_effectiveness_index", 0)

                    # Recommendation: keep the one with higher usage or effectiveness
                    if pair_info["usage_a"] > pair_info["usage_b"]:
                        pair_info["recommendation"] = f"Keep {macro_a}, archive {macro_b}"
                    elif pair_info["usage_b"] > pair_info["usage_a"]:
                        pair_info["recommendation"] = f"Keep {macro_b}, archive {macro_a}"
                    elif pair_info["effectiveness_a"] > pair_info["effectiveness_b"]:
                        pair_info["recommendation"] = f"Keep {macro_a}, archive {macro_b}"
                    else:
                        pair_info["recommendation"] = f"Keep {macro_b}, archive {macro_a}"

                pairs.append(pair_info)

    return pd.DataFrame(pairs).sort_values("similarity", ascending=False)


def get_similarity_stats(similarity_matrix: np.ndarray) -> dict:
    """
    Compute summary statistics for similarity distribution.

    Args:
        similarity_matrix: Raw similarity matrix

    Returns:
        Dictionary of statistics
    """
    # Get upper triangle (excluding diagonal)
    upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

    return {
        "mean_similarity": float(np.mean(upper_tri)),
        "median_similarity": float(np.median(upper_tri)),
        "max_similarity": float(np.max(upper_tri)),
        "min_similarity": float(np.min(upper_tri)),
        "std_similarity": float(np.std(upper_tri)),
        "pairs_above_80": int(np.sum(upper_tri >= 0.8)),
        "pairs_above_60": int(np.sum(upper_tri >= 0.6)),
        "pairs_above_40": int(np.sum(upper_tri >= 0.4)),
    }


def analyze_similarity(
    macro_clusters_path: str = MACRO_CLUSTERS_FILE,
    threshold: float = 0.8,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run full similarity analysis pipeline.

    Args:
        macro_clusters_path: Path to macro clusters CSV
        threshold: Similarity threshold for redundancy
        save: Whether to save results

    Returns:
        Tuple of (similarity_df, redundant_pairs_df, stats)
    """
    logger.info("Loading macro data...")
    macro_df = pd.read_csv(macro_clusters_path)

    logger.info("Computing similarity matrix...")
    similarity_df, similarity_matrix = compute_similarity_matrix(macro_df)

    logger.info(f"Finding redundant pairs (threshold={threshold})...")
    redundant_pairs = find_redundant_pairs(similarity_df, threshold, macro_df)

    stats = get_similarity_stats(similarity_matrix)
    logger.info(f"Found {len(redundant_pairs)} redundant pairs")
    logger.info(f"Mean similarity: {stats['mean_similarity']:.3f}")

    if save:
        logger.info(f"Saving similarity matrix to {SIMILARITY_MATRIX_FILE}...")
        similarity_df.to_csv(SIMILARITY_MATRIX_FILE)

        logger.info(f"Saving redundant pairs to {REDUNDANT_PAIRS_FILE}...")
        redundant_pairs.to_csv(REDUNDANT_PAIRS_FILE, index=False)

    return similarity_df, redundant_pairs, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyze_similarity()
