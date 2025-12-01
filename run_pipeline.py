#!/usr/bin/env python3
"""
Run the complete Macro Help-Center Intelligence Analyzer pipeline.

This script executes all pipeline stages in sequence:
1. Data Generation - Create synthetic tickets, macros, and usage data
2. Data Cleaning - Preprocess and validate data
3. Feature Engineering - Create ticket and macro features
4. Macro Effectiveness - Score macros on effectiveness index
5. NLP Clustering - Group macros by topic
6. Evaluation - Generate recommendations report

Usage:
    python run_pipeline.py                  # Run full pipeline
    python run_pipeline.py --step generate  # Run specific step
    python run_pipeline.py --skip-generate  # Skip data generation
    python run_pipeline.py --quick          # Quick mode (fewer records)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_NUM_MACROS,
    DEFAULT_NUM_TICKETS,
    RAW_MACROS_FILE,
    RAW_TICKETS_FILE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_step(step_name: str, func, *args, **kwargs):
    """Run a pipeline step with timing and logging."""
    logger.info(f"{'='*60}")
    logger.info(f"STARTING: {step_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"✅ COMPLETED: {step_name} ({elapsed:.2f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ FAILED: {step_name} ({elapsed:.2f}s)")
        logger.error(f"Error: {e}")
        raise


def run_data_generation(num_tickets: int, num_macros: int):
    """Run data generation step."""
    from src.data_generation import generate_all_data
    
    return generate_all_data(
        num_tickets=num_tickets,
        num_macros=num_macros,
        save=True,
    )


def run_data_cleaning():
    """Run data cleaning step."""
    from src.data_cleaning import clean_all_data
    
    return clean_all_data(save=True)


def run_feature_engineering():
    """Run feature engineering step."""
    from src.feature_engineering import engineer_all_features
    
    return engineer_all_features(save=True)


def run_macro_effectiveness():
    """Run macro effectiveness scoring step."""
    from src.macro_effectiveness import score_all_macros
    
    return score_all_macros(save=True)


def run_nlp_clustering():
    """Run NLP clustering step."""
    from src.nlp_clustering import cluster_all_macros
    
    return cluster_all_macros(save=True)


def run_evaluation():
    """Run evaluation and report generation step."""
    from src.evaluation import evaluate_all
    
    return evaluate_all(save=True)


def check_data_exists() -> bool:
    """Check if raw data already exists."""
    return RAW_TICKETS_FILE.exists() and RAW_MACROS_FILE.exists()


def run_full_pipeline(
    skip_generate: bool = False,
    num_tickets: int = DEFAULT_NUM_TICKETS,
    num_macros: int = DEFAULT_NUM_MACROS,
):
    """
    Run the complete pipeline end-to-end.
    
    Args:
        skip_generate: Skip data generation if True
        num_tickets: Number of tickets to generate
        num_macros: Number of macros to generate
    """
    total_start = time.time()
    
    logger.info("🚀 Starting Macro Help-Center Intelligence Analyzer Pipeline")
    logger.info(f"   Tickets: {num_tickets:,} | Macros: {num_macros}")
    
    # Step 1: Data Generation
    if not skip_generate:
        run_step(
            "Data Generation",
            run_data_generation,
            num_tickets,
            num_macros,
        )
    else:
        if not check_data_exists():
            logger.error("Cannot skip generation - no data exists!")
            sys.exit(1)
        logger.info("⏭️  Skipping data generation (using existing data)")
    
    # Step 2: Data Cleaning
    run_step("Data Cleaning", run_data_cleaning)
    
    # Step 3: Feature Engineering
    run_step("Feature Engineering", run_feature_engineering)
    
    # Step 4: Macro Effectiveness Scoring
    run_step("Macro Effectiveness Scoring", run_macro_effectiveness)
    
    # Step 5: NLP Clustering
    run_step("NLP Clustering", run_nlp_clustering)
    
    # Step 6: Evaluation & Reporting
    run_step("Evaluation & Reporting", run_evaluation)
    
    # Summary
    total_elapsed = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"🎉 PIPELINE COMPLETE! Total time: {total_elapsed:.2f}s")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Launch dashboard:  streamlit run app/streamlit_app.py")
    logger.info("  2. View report:       cat data/processed/macro_evaluation_report.txt")
    logger.info("  3. Explore data:      jupyter notebook notebooks/")


def main():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the Macro Help-Center Intelligence Analyzer pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Full pipeline with default settings
  python run_pipeline.py --quick            # Quick run with 1000 tickets
  python run_pipeline.py --skip-generate    # Skip data generation
  python run_pipeline.py --step clean       # Run only cleaning step
  python run_pipeline.py --tickets 50000    # Generate 50k tickets
        """,
    )
    
    parser.add_argument(
        "--step",
        choices=["generate", "clean", "features", "score", "cluster", "evaluate"],
        help="Run only a specific step",
    )
    
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip data generation (use existing raw data)",
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: generate only 1000 tickets and 50 macros",
    )
    
    parser.add_argument(
        "--tickets",
        type=int,
        default=DEFAULT_NUM_TICKETS,
        help=f"Number of tickets to generate (default: {DEFAULT_NUM_TICKETS})",
    )
    
    parser.add_argument(
        "--macros",
        type=int,
        default=DEFAULT_NUM_MACROS,
        help=f"Number of macros to generate (default: {DEFAULT_NUM_MACROS})",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Quick mode overrides
    num_tickets = 1000 if args.quick else args.tickets
    num_macros = 50 if args.quick else args.macros
    
    # Run specific step or full pipeline
    if args.step:
        step_map = {
            "generate": lambda: run_step(
                "Data Generation",
                run_data_generation,
                num_tickets,
                num_macros,
            ),
            "clean": lambda: run_step("Data Cleaning", run_data_cleaning),
            "features": lambda: run_step("Feature Engineering", run_feature_engineering),
            "score": lambda: run_step("Macro Effectiveness Scoring", run_macro_effectiveness),
            "cluster": lambda: run_step("NLP Clustering", run_nlp_clustering),
            "evaluate": lambda: run_step("Evaluation & Reporting", run_evaluation),
        }
        step_map[args.step]()
    else:
        run_full_pipeline(
            skip_generate=args.skip_generate,
            num_tickets=num_tickets,
            num_macros=num_macros,
        )


if __name__ == "__main__":
    main()
