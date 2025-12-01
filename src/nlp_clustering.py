"""
NLP clustering module for grouping macros by topic and detecting overlaps.

Uses TF-IDF vectorization and KMeans clustering to identify macro topics.
"""

import logging
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

from src.config import (
    CLUSTER_SUMMARY_FILE,
    KMEANS_MODEL_FILE,
    MACRO_CLUSTERS_FILE,
    MACRO_SCORES_FILE,
    NUM_CLUSTERS,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_VECTORIZER_FILE,
)
from src.utils import clean_text, remove_boilerplate


def prepare_macro_texts(macro_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare macro texts for vectorization and clustering.

    Args:
        macro_scores_df: DataFrame with macro scores

    Returns:
        DataFrame with cleaned text field
    """
    df = macro_scores_df.copy()

    # Combine macro name, category, and body for richer context
    df["combined_text"] = (
        df["macro_name"].fillna("")
        + " "
        + df["category"].fillna("")
        + " "
        + df["macro_body"].fillna("")
    )

    # Clean and remove boilerplate
    df["cleaned_text"] = df["combined_text"].apply(
        lambda x: remove_boilerplate(clean_text(x))
    )

    return df


def vectorize_macros(
    texts: pd.Series,
    vectorizer: TfidfVectorizer = None,
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Vectorize macro texts using TF-IDF.

    Args:
        texts: Series of text strings
        vectorizer: Optional pre-fitted vectorizer (for inference)

    Returns:
        Tuple of (feature matrix, vectorizer)
    """
    if vectorizer is None:
        # Fit new vectorizer
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
        )
        vectors = vectorizer.fit_transform(texts)
    else:
        # Use existing vectorizer
        vectors = vectorizer.transform(texts)

    return vectors.toarray(), vectorizer


def cluster_macros(
    vectors: np.ndarray,
    n_clusters: int = NUM_CLUSTERS,
    model: KMeans = None,
) -> Tuple[np.ndarray, KMeans]:
    """
    Cluster macro vectors using KMeans.

    Args:
        vectors: Feature matrix
        n_clusters: Number of clusters
        model: Optional pre-fitted model (for inference)

    Returns:
        Tuple of (cluster labels, model)
    """
    if model is None:
        # Fit new model
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(vectors)
    else:
        # Use existing model
        labels = model.predict(vectors)

    return labels, model


def generate_cluster_labels(
    macro_clusters_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
) -> dict:
    """
    Generate human-readable labels for each cluster based on top keywords.

    Args:
        macro_clusters_df: DataFrame with cluster assignments
        vectorizer: Fitted TF-IDF vectorizer

    Returns:
        Dictionary mapping cluster_id to label
    """
    feature_names = vectorizer.get_feature_names_out()
    cluster_labels = {}

    for cluster_id in macro_clusters_df["cluster_id"].unique():
        cluster_texts = macro_clusters_df[
            macro_clusters_df["cluster_id"] == cluster_id
        ]["cleaned_text"]

        # Re-vectorize cluster texts
        cluster_vectors = vectorizer.transform(cluster_texts).toarray()

        # Get mean vector for cluster
        mean_vector = cluster_vectors.mean(axis=0)

        # Top 3 features
        top_indices = mean_vector.argsort()[-3:][::-1]
        top_words = [feature_names[i] for i in top_indices]

        cluster_labels[cluster_id] = " / ".join(top_words).title()

    return cluster_labels


def compute_cluster_summary(macro_clusters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for each cluster.

    Args:
        macro_clusters_df: DataFrame with cluster assignments and scores

    Returns:
        DataFrame with cluster-level summary
    """
    summary = (
        macro_clusters_df.groupby(["cluster_id", "cluster_label"])
        .agg(
            num_macros=("macro_id", "count"),
            avg_effectiveness=("macro_effectiveness_index", "mean"),
            median_effectiveness=("macro_effectiveness_index", "median"),
            total_usage=("usage_count", "sum"),
            avg_usage=("usage_count", "mean"),
            num_underused_gems=(
                "macro_category",
                lambda x: (x == "Underused Gem").sum(),
            ),
            num_low_effectiveness=(
                "macro_category",
                lambda x: (x == "Low Effectiveness").sum(),
            ),
        )
        .reset_index()
    )

    # Flag clusters with consolidation opportunities
    summary["consolidation_candidate"] = (
        (summary["num_macros"] >= 5)
        & (summary["avg_effectiveness"] < 50)
    )

    return summary


def cluster_all_macros(
    macro_scores_path: str = MACRO_SCORES_FILE,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load macro scores, cluster by topic, and generate summary.

    Args:
        macro_scores_path: Path to macro scores CSV
        save: Whether to save results and models

    Returns:
        Tuple of (macro_clusters_df, cluster_summary_df)
    """
    logger.info("Loading macro scores...")
    macro_scores = pd.read_csv(macro_scores_path)

    logger.info("Preparing texts for clustering...")
    macro_scores = prepare_macro_texts(macro_scores)

    logger.info("Vectorizing macro texts...")
    vectors, vectorizer = vectorize_macros(macro_scores["cleaned_text"])

    logger.info(f"Clustering into {NUM_CLUSTERS} topics...")
    labels, model = cluster_macros(vectors, n_clusters=NUM_CLUSTERS)

    # Add cluster assignments
    macro_scores["cluster_id"] = labels

    logger.info("Generating cluster labels...")
    cluster_label_map = generate_cluster_labels(macro_scores, vectorizer)
    macro_scores["cluster_label"] = macro_scores["cluster_id"].map(cluster_label_map)

    logger.info("Computing cluster summary...")
    cluster_summary = compute_cluster_summary(macro_scores)

    if save:
        logger.info(f"Saving macro clusters to {MACRO_CLUSTERS_FILE}...")
        macro_scores.to_csv(MACRO_CLUSTERS_FILE, index=False)

        logger.info(f"Saving cluster summary to {CLUSTER_SUMMARY_FILE}...")
        cluster_summary.to_csv(CLUSTER_SUMMARY_FILE, index=False)

        logger.info(f"Saving models to {TFIDF_VECTORIZER_FILE.parent}...")
        with open(TFIDF_VECTORIZER_FILE, "wb") as f:
            pickle.dump(vectorizer, f)
        with open(KMEANS_MODEL_FILE, "wb") as f:
            pickle.dump(model, f)

        logger.info("✓ Clustering complete!")

    # Log summary
    logger.info("Cluster Summary:")
    for _, row in cluster_summary.iterrows():
        logger.info(f"  {row['cluster_label']}: {row['num_macros']} macros, avg eff: {row['avg_effectiveness']:.1f}")

    consolidation_candidates = cluster_summary[cluster_summary["consolidation_candidate"]]
    if len(consolidation_candidates) > 0:
        logger.warning(f"{len(consolidation_candidates)} clusters identified as consolidation candidates")

    return macro_scores, cluster_summary


if __name__ == "__main__":
    cluster_all_macros()
