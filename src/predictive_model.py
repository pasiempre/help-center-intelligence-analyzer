"""
Predictive model for macro recommendations.

Uses historical data to recommend macros for new tickets.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

from src.config import MACRO_CLUSTERS_FILE, MODELS_DIR, PROCESSED_DIR

PREDICTION_MODEL_FILE = MODELS_DIR / "macro_predictor.pkl"
RECOMMENDATIONS_FILE = PROCESSED_DIR / "ticket_recommendations.csv"


class MacroPredictor:
    """Predicts which macro(s) would be most effective for a ticket."""

    def __init__(self, max_features: int = 1000, n_estimators: int = 100):
        """
        Initialize the predictor.

        Args:
            max_features: Max features for TF-IDF
            n_estimators: Number of trees in random forest
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(
        self,
        tickets_df: pd.DataFrame,
        macro_usage_df: pd.DataFrame,
        macro_clusters_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Train the predictor.

        Args:
            tickets_df: Tickets DataFrame
            macro_usage_df: Macro usage DataFrame
            macro_clusters_df: Macro clusters DataFrame

        Returns:
            Dict with training metrics
        """
        # Get only high-performing macros (top 50%)
        top_macros = macro_clusters_df[
            macro_clusters_df["macro_effectiveness_index"]
            >= macro_clusters_df["macro_effectiveness_index"].median()
        ]["macro_id"].tolist()

        # Filter usage to good macros only
        good_usage = macro_usage_df[macro_usage_df["macro_id"].isin(top_macros)]

        # Get first macro used per ticket (primary recommendation)
        first_macro = good_usage.sort_values("applied_at").groupby("ticket_id").first().reset_index()

        # Merge with ticket text
        train_data = first_macro.merge(
            tickets_df[["ticket_id", "text_combined"]],
            on="ticket_id",
            how="inner",
        )

        if len(train_data) < 100:
            logger.warning("Not enough training data")
            return {"error": "Insufficient training data"}

        # Prepare features
        X = self.vectorizer.fit_transform(train_data["text_combined"].fillna(""))
        y = self.label_encoder.fit_transform(train_data["macro_id"])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        logger.info("Training macro predictor...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Top-3 accuracy
        y_proba = self.model.predict_proba(X_test)
        top3_correct = sum(
            y_test[i] in np.argsort(y_proba[i])[-3:]
            for i in range(len(y_test))
        )
        top3_accuracy = top3_correct / len(y_test)

        metrics = {
            "accuracy": accuracy,
            "top3_accuracy": top3_accuracy,
            "n_classes": len(self.label_encoder.classes_),
            "n_samples": len(train_data),
        }

        logger.info(f"Training complete: accuracy={accuracy:.3f}, top3_accuracy={top3_accuracy:.3f}")
        return metrics

    def predict(
        self,
        ticket_text: str,
        top_n: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Predict best macros for a ticket.

        Args:
            ticket_text: Ticket text content
            top_n: Number of recommendations

        Returns:
            List of (macro_id, confidence) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.vectorizer.transform([ticket_text])
        proba = self.model.predict_proba(X)[0]

        # Get top N
        top_indices = np.argsort(proba)[-top_n:][::-1]
        results = [
            (self.label_encoder.inverse_transform([i])[0], proba[i])
            for i in top_indices
        ]

        return results

    def predict_batch(
        self,
        tickets_df: pd.DataFrame,
        text_column: str = "text_combined",
        top_n: int = 3,
    ) -> pd.DataFrame:
        """
        Predict for multiple tickets.

        Args:
            tickets_df: Tickets DataFrame
            text_column: Column with text
            top_n: Number of recommendations per ticket

        Returns:
            DataFrame with recommendations
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.vectorizer.transform(tickets_df[text_column].fillna(""))
        proba = self.model.predict_proba(X)

        results = []
        for i, ticket_id in enumerate(tickets_df["ticket_id"]):
            top_indices = np.argsort(proba[i])[-top_n:][::-1]
            for rank, idx in enumerate(top_indices, 1):
                results.append({
                    "ticket_id": ticket_id,
                    "rank": rank,
                    "recommended_macro": self.label_encoder.inverse_transform([idx])[0],
                    "confidence": proba[i][idx],
                })

        return pd.DataFrame(results)


def compute_macro_category_affinity(
    tickets_df: pd.DataFrame,
    macro_usage_df: pd.DataFrame,
    macro_clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute which macro categories work best for which ticket types.

    Args:
        tickets_df: Tickets DataFrame
        macro_usage_df: Macro usage DataFrame
        macro_clusters_df: Macro clusters DataFrame

    Returns:
        DataFrame with category affinity scores
    """
    # Add macro category to usage
    usage_with_cat = macro_usage_df.merge(
        macro_clusters_df[["macro_id", "category"]],
        on="macro_id",
        how="left",
    )

    # Add ticket type
    usage_with_cat = usage_with_cat.merge(
        tickets_df[["ticket_id", "ticket_type", "csat_score"]],
        on="ticket_id",
        how="left",
    )

    # Compute average CSAT by ticket_type x macro_category
    affinity = usage_with_cat.groupby(["ticket_type", "category"]).agg(
        avg_csat=("csat_score", "mean"),
        usage_count=("ticket_id", "count"),
    ).reset_index()

    # Pivot to matrix format
    affinity_matrix = affinity.pivot(
        index="ticket_type",
        columns="category",
        values="avg_csat",
    ).fillna(0)

    return affinity_matrix


def get_rule_based_recommendations(
    ticket_type: str,
    ticket_priority: str,
    affinity_matrix: pd.DataFrame,
    macro_clusters_df: pd.DataFrame,
    top_n: int = 3,
) -> List[Dict]:
    """
    Get rule-based macro recommendations.

    Args:
        ticket_type: Type of ticket
        ticket_priority: Priority level
        affinity_matrix: Category affinity matrix
        macro_clusters_df: Macro clusters DataFrame
        top_n: Number of recommendations

    Returns:
        List of recommendation dicts
    """
    recommendations = []

    # Get best categories for this ticket type
    if ticket_type in affinity_matrix.index:
        best_categories = affinity_matrix.loc[ticket_type].nlargest(2).index.tolist()
    else:
        best_categories = macro_clusters_df["category"].unique()[:2].tolist()

    # Get top macros from best categories
    for category in best_categories:
        cat_macros = macro_clusters_df[macro_clusters_df["category"] == category]
        cat_macros = cat_macros.nlargest(2, "macro_effectiveness_index")

        for _, macro in cat_macros.iterrows():
            recommendations.append({
                "macro_id": macro["macro_id"],
                "macro_name": macro["macro_name"],
                "category": category,
                "effectiveness": macro["macro_effectiveness_index"],
                "reason": f"High-performing macro in recommended category '{category}'",
            })

    return recommendations[:top_n]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Predictive model module loaded.")
    print("Use MacroPredictor class for ML-based recommendations.")
    print("Use get_rule_based_recommendations() for simpler heuristic approach.")
