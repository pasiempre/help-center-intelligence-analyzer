"""
Enhanced Streamlit dashboard for Macro Help-Center Intelligence Analyzer.

Multi-tab interface with advanced visualizations, agent analysis,
redundancy detection, time series, and export capabilities.
"""

import io
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    CLUSTER_SUMMARY_FILE,
    EVALUATION_REPORT_FILE,
    INTERIM_DIR,
    MACRO_CLUSTERS_FILE,
    PROCESSED_DIR,
    STREAMLIT_LAYOUT,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_PAGE_TITLE,
    TICKETS_FEATURES_FILE,
)

# Page config
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
)


@st.cache_data
def load_data():
    """Load all processed data files."""
    macro_clusters = pd.read_csv(MACRO_CLUSTERS_FILE)
    cluster_summary = pd.read_csv(CLUSTER_SUMMARY_FILE)
    tickets = pd.read_csv(TICKETS_FEATURES_FILE)

    # Add effectiveness_category if not present
    if "effectiveness_category" not in macro_clusters.columns:
        macro_clusters["effectiveness_category"] = pd.cut(
            macro_clusters["macro_effectiveness_index"],
            bins=[0, 33, 66, 100],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )

    try:
        with open(EVALUATION_REPORT_FILE, "r") as f:
            report_text = f.read()
    except FileNotFoundError:
        report_text = "Report not yet generated. Run the pipeline first."

    # Load macro usage for agent analysis
    try:
        macro_usage = pd.read_csv(INTERIM_DIR / "macro_usage_cleaned.csv")
    except FileNotFoundError:
        macro_usage = pd.DataFrame()

    return macro_clusters, cluster_summary, tickets, report_text, macro_usage


@st.cache_data
def compute_similarity_matrix(macro_clusters: pd.DataFrame):
    """Compute macro similarity for redundancy detection."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = macro_clusters["macro_body"].fillna("").tolist()
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf)
        return similarity
    except Exception:
        return None


@st.cache_data
def find_redundant_pairs(macro_clusters: pd.DataFrame, threshold: float = 0.8):
    """Find pairs of similar macros."""
    similarity = compute_similarity_matrix(macro_clusters)
    if similarity is None:
        return pd.DataFrame()

    pairs = []
    n = len(macro_clusters)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                pairs.append({
                    "macro_a_id": macro_clusters.iloc[i]["macro_id"],
                    "macro_a_name": macro_clusters.iloc[i]["macro_name"],
                    "macro_b_id": macro_clusters.iloc[j]["macro_id"],
                    "macro_b_name": macro_clusters.iloc[j]["macro_name"],
                    "similarity": similarity[i, j],
                    "category_a": macro_clusters.iloc[i]["category"],
                    "category_b": macro_clusters.iloc[j]["category"],
                })

    return pd.DataFrame(pairs).sort_values("similarity", ascending=False)


@st.cache_data
def compute_agent_metrics(macro_usage: pd.DataFrame, tickets: pd.DataFrame):
    """Compute agent-level metrics."""
    if macro_usage.empty:
        return pd.DataFrame()

    # macro_usage already has agent_id, so we just need ticket outcomes
    usage_with_tickets = macro_usage.merge(
        tickets[["ticket_id", "csat_score", "total_handle_time_minutes", "reopens_count"]],
        on="ticket_id",
        how="left",
    )

    # Drop rows with no agent_id
    usage_with_tickets = usage_with_tickets.dropna(subset=["agent_id"])

    if usage_with_tickets.empty:
        return pd.DataFrame()

    agent_metrics = usage_with_tickets.groupby("agent_id").agg(
        total_tickets=("ticket_id", "nunique"),
        total_macro_uses=("macro_id", "count"),
        unique_macros_used=("macro_id", "nunique"),
        avg_csat=("csat_score", "mean"),
        avg_handle_time=("total_handle_time_minutes", "mean"),
        reopen_rate=("reopens_count", lambda x: (x > 0).mean()),
    ).reset_index()

    agent_metrics["macros_per_ticket"] = agent_metrics["total_macro_uses"] / agent_metrics["total_tickets"]
    agent_metrics["macro_diversity"] = agent_metrics["unique_macros_used"] / agent_metrics["total_macro_uses"]

    return agent_metrics


@st.cache_data
def compute_daily_metrics(tickets: pd.DataFrame):
    """Compute daily aggregate metrics for time series."""
    tickets = tickets.copy()
    tickets["created_at"] = pd.to_datetime(tickets["created_at"])
    tickets["date"] = tickets["created_at"].dt.date

    daily = tickets.groupby("date").agg(
        total_tickets=("ticket_id", "count"),
        avg_csat=("csat_score", "mean"),
        avg_handle_time=("total_handle_time_minutes", "mean"),
        tickets_with_macros=("num_macros_used", lambda x: (x > 0).sum()),
        reopen_rate=("reopens_count", lambda x: (x > 0).mean()),
    ).reset_index()

    daily["macro_adoption_rate"] = daily["tickets_with_macros"] / daily["total_tickets"]
    daily["date"] = pd.to_datetime(daily["date"])

    return daily


def create_effectiveness_heatmap(macro_clusters: pd.DataFrame):
    """Create category x effectiveness heatmap."""
    pivot = macro_clusters.groupby(["category", "effectiveness_category"]).size().unstack(fill_value=0)

    # Ensure columns in right order
    col_order = ["Low", "Medium", "High"]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns], fill_value=0)

    fig = px.imshow(
        pivot,
        labels={"x": "Effectiveness Level", "y": "Category", "color": "Macro Count"},
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig.update_layout(title="Macro Distribution: Category × Effectiveness")
    return fig


def create_sunburst(macro_clusters: pd.DataFrame):
    """Create sunburst chart of category > cluster > effectiveness."""
    fig = px.sunburst(
        macro_clusters[macro_clusters["has_sufficient_usage"]],
        path=["category", "cluster_label", "effectiveness_category"],
        values="usage_count",
        color="macro_effectiveness_index",
        color_continuous_scale="RdYlGn",
        title="Macro Hierarchy: Category → Cluster → Effectiveness",
    )
    return fig


def create_similarity_network(redundant_pairs: pd.DataFrame, macro_clusters: pd.DataFrame, top_n: int = 30):
    """Create network visualization of similar macros."""
    if redundant_pairs.empty:
        return None

    pairs = redundant_pairs.head(top_n)

    # Build node positions
    unique_ids = list(set(pairs["macro_a_id"].tolist() + pairs["macro_b_id"].tolist()))
    n_nodes = len(unique_ids)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    x_pos = {id_: np.cos(angles[i]) for i, id_ in enumerate(unique_ids)}
    y_pos = {id_: np.sin(angles[i]) for i, id_ in enumerate(unique_ids)}

    # Edges
    edge_x = []
    edge_y = []
    for _, row in pairs.iterrows():
        edge_x.extend([x_pos[row["macro_a_id"]], x_pos[row["macro_b_id"]], None])
        edge_y.extend([y_pos[row["macro_a_id"]], y_pos[row["macro_b_id"]], None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none",
    )

    # Nodes
    node_x = [x_pos[id_] for id_ in unique_ids]
    node_y = [y_pos[id_] for id_ in unique_ids]
    node_names = []
    for id_ in unique_ids:
        match = macro_clusters[macro_clusters["macro_id"] == id_]
        name = match["macro_name"].values[0][:20] if len(match) > 0 else id_
        node_names.append(name)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=15, color="#1f77b4"),
        text=node_names,
        textposition="top center",
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Macro Similarity Network",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
    )
    return fig


def create_excel_export(macro_clusters, tickets, agent_metrics, redundant_pairs):
    """Create Excel export with multiple sheets."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Summary
        macro_clusters.to_excel(writer, sheet_name="All Macros", index=False)

        # Top performers
        top = macro_clusters.nlargest(20, "macro_effectiveness_index")
        top.to_excel(writer, sheet_name="Top Performers", index=False)

        # Needs attention
        low = macro_clusters.nsmallest(20, "macro_effectiveness_index")
        low.to_excel(writer, sheet_name="Needs Attention", index=False)

        # Agent metrics
        if not agent_metrics.empty:
            agent_metrics.to_excel(writer, sheet_name="Agent Metrics", index=False)

        # Redundant pairs
        if not redundant_pairs.empty:
            redundant_pairs.to_excel(writer, sheet_name="Similar Macros", index=False)

    return output.getvalue()


def main():
    """Main Streamlit app."""
    st.title("📊 Macro Help-Center Intelligence Analyzer")
    st.markdown("*Analyze macro effectiveness, identify redundancies, and optimize your support content*")

    # Load data
    try:
        macro_clusters, cluster_summary, tickets, report_text, macro_usage = load_data()
    except FileNotFoundError as e:
        st.error(f"❌ Data files not found: {e}")
        st.info("Please run the data pipeline first: `python run_pipeline.py`")
        return

    # Compute derived data
    redundant_pairs = find_redundant_pairs(macro_clusters)
    agent_metrics = compute_agent_metrics(macro_usage, tickets)
    daily_metrics = compute_daily_metrics(tickets)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview",
        "🔍 Macro Explorer",
        "🎯 Topic Clusters",
        "👥 Agent Analysis",
        "🔄 Redundancy Detection",
        "📅 Time Trends",
        "📥 Export & Reports"
    ])

    # =========== TAB 1: OVERVIEW ===========
    with tab1:
        st.header("Executive Overview")

        # KPI cards row 1
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Macros", f"{len(macro_clusters):,}")

        with col2:
            unused_pct = (macro_clusters["usage_count"] == 0).mean() * 100
            st.metric("Unused Macros", f"{unused_pct:.1f}%", delta=f"-{unused_pct:.0f}%" if unused_pct > 10 else None, delta_color="inverse")

        with col3:
            avg_effectiveness = macro_clusters["macro_effectiveness_index"].mean()
            st.metric("Avg Effectiveness", f"{avg_effectiveness:.1f}/100")

        with col4:
            top_score = macro_clusters["macro_effectiveness_index"].max()
            st.metric("Top Macro Score", f"{top_score:.1f}/100")

        # KPI cards row 2
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_tickets = len(tickets)
            st.metric("Tickets Analyzed", f"{total_tickets:,}")

        with col2:
            avg_csat = tickets["csat_score"].mean()
            st.metric("Avg CSAT", f"{avg_csat:.2f}/5")

        with col3:
            total_usage = macro_clusters["usage_count"].sum()
            st.metric("Total Macro Uses", f"{total_usage:,}")

        with col4:
            categories = macro_clusters["category"].nunique()
            st.metric("Macro Categories", f"{categories}")

        st.markdown("---")

        # Charts row 1
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Effectiveness Distribution")
            fig_hist = px.histogram(
                macro_clusters[macro_clusters["has_sufficient_usage"]],
                x="macro_effectiveness_index",
                nbins=20,
                labels={"macro_effectiveness_index": "Effectiveness Index"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.subheader("Usage vs Effectiveness")
            fig_scatter = px.scatter(
                macro_clusters[macro_clusters["has_sufficient_usage"]],
                x="usage_count",
                y="macro_effectiveness_index",
                color="macro_category",
                hover_data=["macro_id", "macro_name"],
                color_discrete_map={
                    "Underused Gem": "#27ae60",
                    "Core Performer": "#3498db", 
                    "Under Review": "#f39c12",
                    "Needs Improvement": "#e74c3c",
                },
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Charts row 2 - new enhanced charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Category × Effectiveness Heatmap")
            heatmap = create_effectiveness_heatmap(macro_clusters)
            st.plotly_chart(heatmap, use_container_width=True)

        with col2:
            st.subheader("Macro Hierarchy")
            sunburst = create_sunburst(macro_clusters)
            st.plotly_chart(sunburst, use_container_width=True)

    # =========== TAB 2: MACRO EXPLORER ===========
    with tab2:
        st.header("Macro Explorer")

        # Filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            categories = ["All"] + sorted(macro_clusters["category"].unique().tolist())
            selected_category = st.selectbox("Category", categories)

        with col2:
            clusters = ["All"] + sorted(macro_clusters["cluster_label"].unique().tolist())
            selected_cluster = st.selectbox("Topic Cluster", clusters)

        with col3:
            eff_levels = ["All", "High", "Medium", "Low"]
            selected_eff = st.selectbox("Effectiveness", eff_levels)

        with col4:
            min_usage = st.number_input("Min Usage Count", min_value=0, value=0)

        # Filter data
        filtered = macro_clusters.copy()
        if selected_category != "All":
            filtered = filtered[filtered["category"] == selected_category]
        if selected_cluster != "All":
            filtered = filtered[filtered["cluster_label"] == selected_cluster]
        if selected_eff != "All":
            filtered = filtered[filtered["effectiveness_category"] == selected_eff]
        filtered = filtered[filtered["usage_count"] >= min_usage]

        st.write(f"**{len(filtered)} macros** matching filters")

        # Table
        display_cols = [
            "macro_id", "macro_name", "category", "usage_count",
            "macro_effectiveness_index", "avg_csat", "avg_handle_time",
            "reopen_rate", "effectiveness_category",
        ]
        available_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[available_cols].sort_values("macro_effectiveness_index", ascending=False),
            use_container_width=True,
            height=400,
        )

        # Detail pane with comparison
        if len(filtered) > 0:
            st.markdown("---")
            st.subheader("Macro Comparison")

            col1, col2 = st.columns(2)

            with col1:
                macro_a = st.selectbox("Macro A", filtered["macro_id"].tolist(), key="macro_a")
            with col2:
                remaining = [m for m in filtered["macro_id"].tolist() if m != macro_a]
                macro_b = st.selectbox("Macro B", remaining if remaining else [macro_a], key="macro_b")

            detail_a = filtered[filtered["macro_id"] == macro_a].iloc[0]
            detail_b = filtered[filtered["macro_id"] == macro_b].iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {detail_a['macro_name'][:40]}")
                st.text_area("Macro Body", detail_a["macro_body"], height=150, key="body_a")
                st.metric("Effectiveness", f"{detail_a['macro_effectiveness_index']:.1f}")
                st.metric("Usage", f"{detail_a['usage_count']:,}")
                st.metric("CSAT", f"{detail_a['avg_csat']:.2f}")

            with col2:
                st.markdown(f"### {detail_b['macro_name'][:40]}")
                st.text_area("Macro Body", detail_b["macro_body"], height=150, key="body_b")
                eff_diff = detail_b['macro_effectiveness_index'] - detail_a['macro_effectiveness_index']
                st.metric("Effectiveness", f"{detail_b['macro_effectiveness_index']:.1f}", delta=f"{eff_diff:+.1f}")
                usage_diff = detail_b['usage_count'] - detail_a['usage_count']
                st.metric("Usage", f"{detail_b['usage_count']:,}", delta=f"{usage_diff:+,}")
                csat_diff = detail_b['avg_csat'] - detail_a['avg_csat']
                st.metric("CSAT", f"{detail_b['avg_csat']:.2f}", delta=f"{csat_diff:+.2f}")

    # =========== TAB 3: TOPIC CLUSTERS ===========
    with tab3:
        st.header("Topic / Cluster Analysis")

        # Bar chart
        fig_clusters = px.bar(
            cluster_summary.sort_values("avg_effectiveness", ascending=False),
            x="cluster_label",
            y="avg_effectiveness",
            color="num_macros",
            labels={
                "cluster_label": "Cluster",
                "avg_effectiveness": "Avg Effectiveness",
                "num_macros": "# Macros",
            },
            title="Cluster Effectiveness",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_clusters, use_container_width=True)

        # Treemap of clusters
        st.subheader("Cluster Size Distribution")
        fig_treemap = px.treemap(
            macro_clusters[macro_clusters["has_sufficient_usage"]],
            path=["cluster_label"],
            values="usage_count",
            color="macro_effectiveness_index",
            color_continuous_scale="RdYlGn",
            title="Cluster Usage (color = effectiveness)",
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

        st.markdown("---")

        # Cluster table
        st.subheader("Cluster Summary")
        st.dataframe(
            cluster_summary[[
                "cluster_label", "num_macros", "avg_effectiveness",
                "total_usage", "num_underused_gems", "num_low_effectiveness",
                "consolidation_candidate",
            ]],
            use_container_width=True,
        )

        # Consolidation candidates
        consolidation = cluster_summary[cluster_summary["consolidation_candidate"]]
        if len(consolidation) > 0:
            st.warning(f"⚠️ {len(consolidation)} clusters identified as consolidation candidates")

    # =========== TAB 4: AGENT ANALYSIS ===========
    with tab4:
        st.header("Agent Performance Analysis")

        if agent_metrics.empty:
            st.warning("Agent metrics not available. Make sure macro usage data is loaded.")
        else:
            # Top agents
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Agent CSAT Leaderboard")
                top_agents = agent_metrics.nlargest(15, "avg_csat")
                fig = px.bar(
                    top_agents,
                    x="agent_id",
                    y="avg_csat",
                    color="avg_csat",
                    color_continuous_scale="Greens",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Macro Diversity by Agent")
                fig = px.scatter(
                    agent_metrics,
                    x="unique_macros_used",
                    y="avg_csat",
                    size="total_tickets",
                    color="avg_handle_time",
                    hover_data=["agent_id"],
                    labels={
                        "unique_macros_used": "Unique Macros Used",
                        "avg_csat": "Average CSAT",
                        "avg_handle_time": "Avg Handle Time",
                    },
                    color_continuous_scale="RdYlGn_r",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Agent table
            st.subheader("Agent Metrics Table")
            st.dataframe(
                agent_metrics.sort_values("avg_csat", ascending=False),
                use_container_width=True,
                height=400,
            )

            # Best practices
            st.subheader("Best Practice Insights")
            top_performers = agent_metrics[agent_metrics["avg_csat"] > agent_metrics["avg_csat"].quantile(0.8)]
            avg_macros_top = top_performers["unique_macros_used"].mean()
            avg_macros_all = agent_metrics["unique_macros_used"].mean()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Top Performers Avg Macros", f"{avg_macros_top:.1f}")
            with col2:
                st.metric("All Agents Avg Macros", f"{avg_macros_all:.1f}")
            with col3:
                diff = avg_macros_top - avg_macros_all
                st.metric("Difference", f"{diff:+.1f}")

    # =========== TAB 5: REDUNDANCY DETECTION ===========
    with tab5:
        st.header("Redundancy Detection")

        if redundant_pairs.empty:
            st.success("✅ No highly similar macro pairs found above threshold")
        else:
            st.warning(f"⚠️ Found {len(redundant_pairs)} potentially redundant macro pairs")

            # Threshold slider
            threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.8, 0.05)
            filtered_pairs = redundant_pairs[redundant_pairs["similarity"] >= threshold]

            st.write(f"**{len(filtered_pairs)} pairs** above {threshold:.0%} similarity")

            # Network visualization
            network_fig = create_similarity_network(filtered_pairs, macro_clusters)
            if network_fig:
                st.plotly_chart(network_fig, use_container_width=True)

            # Pairs table
            st.subheader("Similar Macro Pairs")
            st.dataframe(
                filtered_pairs[["macro_a_name", "macro_b_name", "similarity", "category_a", "category_b"]],
                use_container_width=True,
                height=300,
            )

            # Detail comparison
            if len(filtered_pairs) > 0:
                st.markdown("---")
                st.subheader("Compare Similar Pair")

                pair_options = [
                    f"{row['macro_a_name'][:30]} ↔ {row['macro_b_name'][:30]}"
                    for _, row in filtered_pairs.head(20).iterrows()
                ]
                selected_pair = st.selectbox("Select pair to compare", pair_options)
                pair_idx = pair_options.index(selected_pair)
                selected_row = filtered_pairs.iloc[pair_idx]

                col1, col2 = st.columns(2)
                with col1:
                    macro_a_detail = macro_clusters[macro_clusters["macro_id"] == selected_row["macro_a_id"]].iloc[0]
                    st.markdown(f"**{macro_a_detail['macro_name']}**")
                    st.text_area("Content", macro_a_detail["macro_body"], height=200, key="redund_a")
                    st.metric("Effectiveness", f"{macro_a_detail['macro_effectiveness_index']:.1f}")

                with col2:
                    macro_b_detail = macro_clusters[macro_clusters["macro_id"] == selected_row["macro_b_id"]].iloc[0]
                    st.markdown(f"**{macro_b_detail['macro_name']}**")
                    st.text_area("Content", macro_b_detail["macro_body"], height=200, key="redund_b")
                    st.metric("Effectiveness", f"{macro_b_detail['macro_effectiveness_index']:.1f}")

                # Recommendation
                if macro_a_detail['macro_effectiveness_index'] > macro_b_detail['macro_effectiveness_index']:
                    st.info(f"💡 **Recommendation**: Keep '{macro_a_detail['macro_name']}' (higher effectiveness)")
                else:
                    st.info(f"💡 **Recommendation**: Keep '{macro_b_detail['macro_name']}' (higher effectiveness)")

    # =========== TAB 6: TIME TRENDS ===========
    with tab6:
        st.header("Time Trends Analysis")

        if daily_metrics.empty:
            st.warning("Time series data not available.")
        else:
            # Time series charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Daily Ticket Volume")
                fig = px.line(
                    daily_metrics,
                    x="date",
                    y="total_tickets",
                    markers=True,
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Tickets")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("CSAT Trend")
                fig = px.line(
                    daily_metrics,
                    x="date",
                    y="avg_csat",
                    markers=True,
                    color_discrete_sequence=["#27ae60"],
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Avg CSAT")
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Macro Adoption Rate")
                fig = px.area(
                    daily_metrics,
                    x="date",
                    y="macro_adoption_rate",
                    color_discrete_sequence=["#3498db"],
                )
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Handle Time Trend")
                fig = px.line(
                    daily_metrics,
                    x="date",
                    y="avg_handle_time",
                    markers=True,
                    color_discrete_sequence=["#e74c3c"],
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Avg Handle Time (min)")
                st.plotly_chart(fig, use_container_width=True)

            # Combined metrics chart
            st.subheader("Combined Metrics Overview")
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Tickets", "CSAT", "Macro Adoption", "Handle Time"),
            )
            fig.add_trace(go.Scatter(x=daily_metrics["date"], y=daily_metrics["total_tickets"], name="Tickets"), row=1, col=1)
            fig.add_trace(go.Scatter(x=daily_metrics["date"], y=daily_metrics["avg_csat"], name="CSAT"), row=1, col=2)
            fig.add_trace(go.Scatter(x=daily_metrics["date"], y=daily_metrics["macro_adoption_rate"], name="Adoption"), row=2, col=1)
            fig.add_trace(go.Scatter(x=daily_metrics["date"], y=daily_metrics["avg_handle_time"], name="Handle Time"), row=2, col=2)
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # =========== TAB 7: EXPORT & REPORTS ===========
    with tab7:
        st.header("Export & Reports")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Excel Export")
            st.write("Download comprehensive Excel report with multiple sheets")

            if st.button("Generate Excel Report", type="primary"):
                with st.spinner("Generating Excel..."):
                    excel_data = create_excel_export(macro_clusters, tickets, agent_metrics, redundant_pairs)
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"macro_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

        with col2:
            st.subheader("📝 Text Report")
            st.text_area("Evaluation Report", report_text, height=300)

        st.markdown("---")

        # Quick stats
        st.subheader("Report Summary Stats")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            high_eff = (macro_clusters["effectiveness_category"] == "High").sum()
            st.metric("High Effectiveness", high_eff)

        with col2:
            low_eff = (macro_clusters["effectiveness_category"] == "Low").sum()
            st.metric("Low Effectiveness", low_eff)

        with col3:
            unused = (macro_clusters["usage_count"] == 0).sum()
            st.metric("Unused", unused)

        with col4:
            redundant = len(redundant_pairs)
            st.metric("Redundant Pairs", redundant)

        # Action items summary
        st.subheader("Action Items")

        underused_gems = macro_clusters[macro_clusters["macro_category"] == "Underused Gem"]
        if len(underused_gems) > 0:
            with st.expander(f"✅ Promote {len(underused_gems)} Underused High-Impact Macros"):
                st.dataframe(
                    underused_gems[["macro_id", "macro_name", "macro_effectiveness_index", "usage_count"]]
                    .sort_values("macro_effectiveness_index", ascending=False)
                )

        low_effectiveness = macro_clusters[
            (macro_clusters["macro_effectiveness_index"] < 25)
            & (macro_clusters["has_sufficient_usage"])
        ]
        if len(low_effectiveness) > 0:
            with st.expander(f"⚠️ Rewrite {len(low_effectiveness)} Low Effectiveness Macros"):
                st.dataframe(
                    low_effectiveness[["macro_id", "macro_name", "macro_effectiveness_index", "avg_csat"]]
                    .sort_values("macro_effectiveness_index")
                )

        unused = macro_clusters[macro_clusters["usage_count"] == 0]
        if len(unused) > 0:
            with st.expander(f"🗑️ Archive {len(unused)} Unused Macros"):
                st.dataframe(unused[["macro_id", "macro_name", "category"]])


if __name__ == "__main__":
    main()
