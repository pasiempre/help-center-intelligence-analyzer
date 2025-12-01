"""
Central configuration for the Macro Help-Center Intelligence Analyzer project.

This module contains all project paths, constants, and default parameters.
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for directory in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# RANDOM SEED
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# DATA GENERATION DEFAULTS
# ============================================================================

DEFAULT_NUM_TICKETS = 10000
DEFAULT_NUM_MACROS = 150
DEFAULT_NUM_AGENTS = 50
DEFAULT_TIME_RANGE_DAYS = 90

# Contact driver distribution (weighted)
CONTACT_DRIVERS = {
    "billing_issue": 0.25,
    "login_problem": 0.15,
    "device_error": 0.12,
    "account_management": 0.10,
    "payment_failed": 0.08,
    "feature_request": 0.07,
    "refund_request": 0.08,
    "technical_error": 0.10,
    "general_inquiry": 0.05,
}

# Macro categories
MACRO_CATEGORIES = [
    "billing",
    "technical",
    "account",
    "policy",
    "product",
    "escalation",
]

# Owner teams
OWNER_TEAMS = ["support", "billing", "tier2", "technical", "retention"]

# Channels
CHANNELS = ["email", "chat", "phone", "webform"]

# Priority levels
PRIORITIES = ["low", "normal", "high", "urgent"]

# Ticket statuses
STATUSES = ["open", "solved", "closed", "pending"]

# Languages
LANGUAGES = ["en", "es"]

# ============================================================================
# CSAT SCORING
# ============================================================================

CSAT_SCALE_MIN = 1
CSAT_SCALE_MAX = 5
CSAT_THRESHOLD_GOOD = 4  # CSAT >= 4 is considered good

# ============================================================================
# MACRO EFFECTIVENESS WEIGHTS
# ============================================================================

# Weights for computing macro_effectiveness_index
EFFECTIVENESS_WEIGHTS = {
    "csat": 0.4,
    "handle_time": 0.3,
    "reopen_rate": 0.3,
}

# Minimum usage count for a macro to be scored (avoid noise from low-n samples)
MIN_USAGE_FOR_SCORING = 5

# ============================================================================
# NLP / CLUSTERING CONFIGURATION
# ============================================================================

# Number of clusters for macro topic grouping
NUM_CLUSTERS = 12

# Embedding model placeholder (for future enhancement)
EMBEDDING_MODEL = "TF-IDF"  # Options: "TF-IDF", "sentence-transformers", etc.

# TF-IDF parameters
TFIDF_MAX_FEATURES = 500
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.8
TFIDF_NGRAM_RANGE = (1, 2)

# ============================================================================
# HANDLE TIME THRESHOLDS (minutes)
# ============================================================================

HANDLE_TIME_EXCELLENT = 15
HANDLE_TIME_GOOD = 30
HANDLE_TIME_ACCEPTABLE = 60
HANDLE_TIME_POOR = 120

# ============================================================================
# REOPEN RATE THRESHOLDS
# ============================================================================

REOPEN_RATE_EXCELLENT = 0.05
REOPEN_RATE_GOOD = 0.10
REOPEN_RATE_ACCEPTABLE = 0.20
REOPEN_RATE_POOR = 0.30

# ============================================================================
# FILE NAMES
# ============================================================================

# Raw data files
RAW_TICKETS_FILE = RAW_DIR / "tickets.csv"
RAW_MACROS_FILE = RAW_DIR / "macros.csv"
RAW_MACRO_USAGE_FILE = RAW_DIR / "macro_usage.csv"

# Interim data files
INTERIM_TICKETS_FILE = INTERIM_DIR / "tickets_cleaned.csv"
INTERIM_MACROS_FILE = INTERIM_DIR / "macros_cleaned.csv"
INTERIM_MACRO_USAGE_FILE = INTERIM_DIR / "macro_usage_cleaned.csv"

# Processed data files
TICKETS_FEATURES_FILE = PROCESSED_DIR / "tickets_features.csv"
MACRO_FEATURES_BASE_FILE = PROCESSED_DIR / "macro_features_base.csv"
MACRO_SCORES_FILE = PROCESSED_DIR / "macro_scores.csv"
MACRO_CLUSTERS_FILE = PROCESSED_DIR / "macro_clusters.csv"
CLUSTER_SUMMARY_FILE = PROCESSED_DIR / "cluster_summary.csv"
EVALUATION_REPORT_FILE = PROCESSED_DIR / "macro_evaluation_report.txt"

# Model files
TFIDF_VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.pkl"
KMEANS_MODEL_FILE = MODELS_DIR / "kmeans_model.pkl"

# ============================================================================
# STREAMLIT CONFIG
# ============================================================================

STREAMLIT_PAGE_TITLE = "Macro Intelligence Analyzer"
STREAMLIT_PAGE_ICON = "📊"
STREAMLIT_LAYOUT = "wide"
