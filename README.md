# 📊 Macro Help-Center Intelligence Analyzer

A data-driven system for analyzing customer support macros and help-center content to identify effectiveness, redundancies, and optimization opportunities.

---

## 🎯 Project Overview

Support organizations accumulate hundreds of macros and templates over time, but lack systematic ways to measure their impact. This project provides:

- **Quantitative effectiveness scoring** for each macro based on CSAT, handle time, and reopen rates
- **Topic clustering** to identify redundant or overlapping macros
- **Usage analysis** to surface high-performing but underused "hidden gems"
- **Actionable recommendations** for macro consolidation, deprecation, and promotion

---

## 🏗️ Architecture

```
macro-helpcenter-intelligence/
├── app/
│   └── streamlit_app.py          # Interactive dashboard
├── data/
│   ├── raw/                      # Synthetic base CSVs
│   ├── interim/                  # Cleaned data
│   └── processed/                # Feature sets, scores, clusters
├── models/                       # Saved ML models (TF-IDF, KMeans)
├── notebooks/                    # Exploration notebooks
├── reports/
│   └── figures/                  # Exported visualizations
├── src/
│   ├── config.py                 # Central configuration
│   ├── data_generation.py        # Synthetic data generation
│   ├── data_cleaning.py          # Data preprocessing
│   ├── feature_engineering.py    # Ticket & macro features
│   ├── macro_effectiveness.py    # Effectiveness scoring
│   ├── nlp_clustering.py         # Topic modeling
│   ├── evaluation.py             # Reporting & recommendations
│   └── utils.py                  # Shared utilities
├── tests/                        # Unit tests
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 📊 Data Model

### Tables

1. **tickets** (fact table)
   - One row per support ticket
   - Fields: ticket_id, created_at, resolved_at, csat_score, handle_time, contact_driver, etc.

2. **macros** (dimension table)
   - One row per macro/template
   - Fields: macro_id, macro_name, category, macro_body, owner_team, is_active

3. **macro_usage** (bridge table)
   - Many-to-many relationship between tickets and macros
   - Tracks which macros were used in each ticket

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd "Macro Help-Center Intelligence Analyzer"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -e ".[dev]"
```

### 2. Run the Data Pipeline

```bash
# Generate synthetic data
python -m src.data_generation

# Clean and preprocess
python -m src.data_cleaning

# Engineer features
python -m src.feature_engineering

# Score macro effectiveness
python -m src.macro_effectiveness

# Cluster macros by topic
python -m src.nlp_clustering

# Generate evaluation report
python -m src.evaluation
```

### 3. Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📈 Key Metrics

### Macro Effectiveness Index (0-100)

Composite score combining:
- **CSAT Impact** (40%): Average customer satisfaction when macro is used
- **Handle Time Efficiency** (30%): Time savings vs. baseline
- **Reopen Rate** (30%): How often tickets reopen after macro use

### Macro Categories

- **High Impact (Popular)**: Top quartile effectiveness, high usage
- **Underused Gem**: Top quartile effectiveness, low usage → promote these!
- **Moderate Effectiveness**: Middle 50%
- **Low Effectiveness**: Bottom quartile → candidates for rewrite
- **Unused/Low Usage**: <5 uses → candidates for deprecation

---

## 🎨 Dashboard Features

### Tab 1: Overview
- KPI cards (total macros, unused %, avg effectiveness)
- Effectiveness distribution histogram
- Usage vs. effectiveness scatter plot

### Tab 2: Macro Explorer
- Filter by category, topic cluster, min usage
- Sortable table of all macros
- Detail pane showing macro body and metrics

### Tab 3: Topic Clusters
- Bar chart of cluster effectiveness
- Cluster summary table
- Consolidation candidate identification

### Tab 4: Recommendations
- Full evaluation report
- Expandable action items:
  - Macros to promote
  - Macros to rewrite
  - Macros to archive

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_utils.py
```

---

## 🔧 Configuration

Edit `src/config.py` to customize:

- **Dataset sizes**: `DEFAULT_NUM_TICKETS`, `DEFAULT_NUM_MACROS`
- **Effectiveness weights**: `EFFECTIVENESS_WEIGHTS`
- **Clustering params**: `NUM_CLUSTERS`, `TFIDF_MAX_FEATURES`
- **File paths**: All output locations

---

## 📚 Core Concepts

### Effectiveness Scoring Logic

For each macro:
1. Calculate aggregate metrics (avg CSAT, handle time, reopen rate)
2. Compare to baseline for same contact driver
3. Normalize each component to 0-100 scale
4. Apply weighted combination
5. Assign only if usage ≥ minimum threshold (default: 5 uses)

### NLP Clustering

1. Combine macro name + category + body text
2. Clean and remove boilerplate phrases
3. TF-IDF vectorization (500 features, 1-2 grams)
4. KMeans clustering (default: 12 clusters)
5. Generate cluster labels from top keywords
6. Identify clusters with many low-performing macros → consolidation targets

---

## 🛣️ Roadmap / Future Enhancements

### V2 Features
- **Agent normalization**: Control for agent skill in effectiveness scoring
- **Propensity matching**: Account for ticket difficulty confounders
- **Time series analysis**: Track macro effectiveness over time
- **A/B test recommendations**: Suggest macros to test against each other

### V3 Features
- **LLM embeddings**: Upgrade from TF-IDF to sentence-transformers
- **Semantic search**: Find similar macros using embeddings
- **Auto-generated macro suggestions**: Use GPT to draft improved macro text
- **Integration with Zendesk/Freshdesk APIs**: Real data ingestion

---

## 📊 Sample Outputs

After running the pipeline, you'll find:

- `data/processed/macro_scores.csv` - All macros with effectiveness scores
- `data/processed/macro_clusters.csv` - Macro-to-cluster assignments
- `data/processed/cluster_summary.csv` - Cluster-level aggregates
- `data/processed/macro_evaluation_report.txt` - Human-readable insights

---

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome!

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📝 License

MIT License - feel free to use this project as inspiration for your own analytics work.

---

## 👤 Author

Created as part of a data analytics portfolio showcasing:
- End-to-end data pipeline development
- Feature engineering and metric design
- NLP/ML for practical business problems
- Interactive dashboards with Streamlit
- Clean, production-style Python code

---

## 🙏 Acknowledgments

- Inspired by real-world CX analytics challenges
- Built with: pandas, scikit-learn, Streamlit, plotly
- Follows best practices from ModivCare project structure

---

**Questions?** Open an issue or reach out!
