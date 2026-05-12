"""
streamlit_app.py
----------------
Multi-page Streamlit dashboard for Star Wars Character Network Analysis.

Pages:
    1. Home                — project overview + key stats
    2. Dataset Insights    — EDA: distributions, top characters
    3. Network Visualizer  — interactive pyvis + static graph
    4. Centrality Rankings — sortable ranked table + radar chart
    5. Community Explorer  — community breakdown + membership
    6. Graph ML Results    — model comparison + confusion matrix

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph_builder  import load_processed, build_graph, graph_summary
from src.centrality     import build_centrality_table
from src.communities    import (
    detect_louvain, community_summary,
    compute_modularity,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Star Wars Network Analysis",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 13px; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 22px; }
    /* Tab styling */
    button[data-baseweb="tab"] {
        font-size: 14px;
        color: #8b949e;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff;
    }
    /* DataFrame */
    [data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; }
    /* Section headers */
    .section-header {
        color: #58a6ff;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    /* Community badge */
    .comm-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #8b949e;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)

# ── Color constants ───────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
TEXT_CLR  = "#e6edf3"
ACCENT    = "#58a6ff"
PALETTE   = [
    "#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff",
    "#ffa657", "#79c0ff", "#56d364", "#ff9bce",
    "#e3b341", "#bc8cff", "#f47067", "#4ac26b",
]

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading graph data...")
def load_data():
    nodes, edges = load_processed()
    return nodes, edges


@st.cache_resource(show_spinner="Building graph...")
def load_graph():
    nodes, edges = load_data()
    G = build_graph(nodes, edges)
    return G


@st.cache_data(show_spinner="Computing centrality metrics...")
def load_centrality():
    G = load_graph()
    return build_centrality_table(G)


@st.cache_data(show_spinner="Detecting communities...")
def load_communities():
    G = load_graph()
    partition   = detect_louvain(G)
    comm_sum    = community_summary(G, partition)
    modularity  = compute_modularity(G, partition)
    return partition, comm_sum, modularity


@st.cache_data
def load_graph_summary():
    G = load_graph()
    return graph_summary(G)


def load_ml_results() -> dict:
    path = ROOT / "outputs" / "reports" / "graph_ml_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_figure(filename: str) -> Path:
    return ROOT / "outputs" / "figures" / filename


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 16px 0 8px'>
            <span style='font-size:36px'>⚔️</span>
            <h2 style='color:#e6edf3; margin:4px 0; font-size:17px'>
                Star Wars Network
            </h2>
            <p style='color:#8b949e; font-size:12px; margin:0'>
                Character Interaction Analysis
            </p>
        </div>
        <hr style='border-color:#21262d; margin:12px 0'>
        """, unsafe_allow_html=True)

        pages = {
            "🏠  Home":                "Home",
            "📊  Dataset Insights":    "Dataset Insights",
            "🌐  Network Visualizer":  "Network Visualizer",
            "📈  Centrality Rankings": "Centrality Rankings",
            "🔵  Community Explorer":  "Community Explorer",
            "🤖  Graph ML Results":    "Graph ML Results",
        }

        selected = st.radio(
            "Navigate",
            list(pages.keys()),
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#21262d; margin:12px 0'>",
                    unsafe_allow_html=True)

        # Quick stats
        summary = load_graph_summary()
        st.markdown("<p class='section-header'>Graph Stats</p>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:12px; color:#8b949e; line-height:2'>
            🔵 <b style='color:#e6edf3'>{summary['num_nodes']}</b> characters<br>
            🔗 <b style='color:#e6edf3'>{summary['num_edges']}</b> interactions<br>
            📐 Density: <b style='color:#e6edf3'>{summary['density']:.4f}</b><br>
            🌀 Clustering: <b style='color:#e6edf3'>
                {summary['avg_clustering_coefficient']:.4f}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <hr style='border-color:#21262d; margin:12px 0'>
        <div style='font-size:11px; color:#8b949e; text-align:center'>
            Built by
            <a href='https://github.com/Stokesy-dev'
               style='color:#58a6ff'>@Stokesy-dev</a><br>
            Graph ML · NetworkX · PyG · Streamlit
        </div>
        """, unsafe_allow_html=True)

    return pages[selected]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.title("⚔️ Star Wars Character Network Analysis")
    st.markdown(
        "<p style='color:#8b949e; font-size:15px'>"
        "Graph ML · Community Detection · Centrality Analysis · "
        "Interactive Network Visualization"
        "</p>", unsafe_allow_html=True,
    )

    summary = load_graph_summary()
    _, comm_sum, modularity = load_communities()
    cent_df = load_centrality()

    # ── Key metrics row ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        ("Characters",     summary["num_nodes"],              ""),
        ("Interactions",   summary["num_edges"],              ""),
        ("Communities",    summary["num_connected_components"]
                           if not summary["is_connected"]
                           else len(comm_sum),                ""),
        ("Graph Density",  f"{summary['density']:.4f}",       ""),
        ("Avg Clustering", f"{summary['avg_clustering_coefficient']:.4f}", ""),
        ("Diameter (LCC)", summary["diameter_lcc"],           ""),
    ]
    for col, (label, val, delta) in zip([c1,c2,c3,c4,c5,c6], metrics):
        col.metric(label, val)

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("🌐 Character Network")
        static_path = load_figure("network_static.png")
        if static_path.exists():
            st.image(str(static_path), use_column_width=True)
        else:
            st.info("Run `scripts/run_visualization.py` to generate figures.")

    with col_r:
        st.subheader("🏆 Most Influential Characters")
        top10 = cent_df.reset_index(drop=True).head(10)[
            ["name", "centrality_score", "pagerank"]
        ].copy()
        top10["centrality_score"] = top10["centrality_score"].round(4)
        top10["pagerank"]         = top10["pagerank"].round(5)
        top10.index = range(1, 11)
        top10.index.name = "Rank"
        st.dataframe(
            top10.rename(columns={
                "name":              "Character",
                "centrality_score":  "Score",
                "pagerank":          "PageRank",
            }),
            use_container_width=True, height=360,
        )

        st.markdown("---")
        st.markdown("<p class='section-header'>Modularity Score</p>",
                    unsafe_allow_html=True)
        st.progress(min(modularity, 1.0),
                    text=f"Louvain Modularity: {modularity:.4f}")

    st.markdown("---")
    st.subheader("📖 About This Project")
    st.markdown("""
    This project builds a **character interaction network** from Star Wars
    episode data (Kaggle), then applies:

    | Module | Techniques |
    |--------|-----------|
    | **Graph Analytics** | Degree, Betweenness, Closeness, Eigenvector Centrality, PageRank |
    | **Community Detection** | Louvain algorithm, Girvan-Newman |
    | **Graph ML** | GCN (Kipf & Welling, 2017), GraphSAGE (Hamilton et al., 2017) |
    | **Visualization** | NetworkX, PyVis, Plotly, Seaborn |
    | **Dashboard** | Streamlit multi-page app |

    > **Task:** Node classification — predict which faction/community a character
    belongs to using graph structure + centrality features.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def page_dataset_insights():
    st.title("📊 Dataset Insights")
    st.markdown(
        "<p style='color:#8b949e'>Exploratory analysis of the "
        "Star Wars character interaction dataset.</p>",
        unsafe_allow_html=True,
    )

    nodes, edges = load_data()
    G            = load_graph()

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Raw Data",
        "📈 Distributions",
        "🔝 Top Characters",
        "🔗 Edge Analysis",
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Character Nodes")
            st.dataframe(nodes, use_container_width=True, height=400)
        with col2:
            st.subheader("Interactions (Edges)")
            st.dataframe(edges, use_container_width=True, height=400)

        st.markdown("**Shape:**")
        col3, col4 = st.columns(2)
        col3.metric("Nodes", nodes.shape[0])
        col4.metric("Edges", edges.shape[0])

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Degree Distribution")
            p = load_figure("degree_distribution.png")
            if p.exists():
                st.image(str(p), use_column_width=True)

        with col2:
            st.subheader("Edge Weight Distribution")
            p = load_figure("edge_weight_distribution.png")
            if p.exists():
                st.image(str(p), use_column_width=True)

        # Interactive degree histogram
        st.subheader("Interactive Degree Distribution")
        degrees = dict(G.degree())
        deg_df  = pd.DataFrame({
            "Character": [G.nodes[n].get("name", str(n)) for n in degrees],
            "Degree":    list(degrees.values()),
        }).sort_values("Degree", ascending=False)

        fig = px.histogram(
            deg_df, x="Degree", nbins=20,
            title="Degree Frequency Distribution",
            color_discrete_sequence=[ACCENT],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT_CLR),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Top Characters by Interaction Degree")
        degrees = dict(G.degree())
        top_df  = pd.DataFrame({
            "Character": [G.nodes[n].get("name", str(n)) for n in degrees],
            "Degree":    list(degrees.values()),
            "Total Weight": [
                G.nodes[n].get("total_weight", 0) for n in degrees
            ],
        }).sort_values("Degree", ascending=False).head(20)

        fig = px.bar(
            top_df, x="Degree", y="Character",
            orientation="h",
            color="Degree",
            color_continuous_scale="Plasma",
            title="Top 20 Characters by Degree",
            template="plotly_dark",
            text="Degree",
        )
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT_CLR),
            yaxis=dict(autorange="reversed"),
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Top 20 Strongest Interactions")
        top_edges = edges.nlargest(20, "weight")[
            ["source_name", "target_name", "weight", "weight_norm"]
        ].reset_index(drop=True)
        top_edges.index += 1
        st.dataframe(
            top_edges.rename(columns={
                "source_name": "Character A",
                "target_name": "Character B",
                "weight":      "Co-occurrences",
                "weight_norm": "Normalized Weight",
            }),
            use_container_width=True,
        )

        fig2 = px.scatter(
            edges, x="weight", y="weight_norm",
            title="Raw vs Normalized Edge Weights",
            labels={"weight": "Co-occurrence Count",
                    "weight_norm": "Normalized Weight"},
            color="weight",
            color_continuous_scale="Viridis",
            template="plotly_dark",
            opacity=0.7,
        )
        fig2.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT_CLR),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — NETWORK VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

def page_network_visualizer():
    st.title("🌐 Network Visualizer")
    st.markdown(
        "<p style='color:#8b949e'>"
        "Explore the full character interaction network. "
        "Hover nodes for details. Drag to rearrange.</p>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎮 Interactive Network",
        "🖼️ Static Network",
        "🔵 Community Network",
        "🔍 Character Search",
    ])

    with tab1:
        pyvis_path = load_figure("network_interactive.html")
        if pyvis_path.exists():
            with open(pyvis_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=750, scrolling=False)
        else:
            st.warning(
                "Interactive network not found. "
                "Run `scripts/run_visualization.py` first."
            )
        st.caption(
            "💡 Drag nodes to rearrange · Scroll to zoom · "
            "Hover for character details"
        )

    with tab2:
        p = load_figure("network_static.png")
        if p.exists():
            st.image(str(p), use_column_width=True,
                     caption="Static network — node size=PageRank, "
                             "color=degree centrality")

    with tab3:
        p = load_figure("network_communities.png")
        if p.exists():
            st.image(str(p), use_column_width=True,
                     caption="Community-colored network (Louvain detection)")

    with tab4:
        st.subheader("🔍 Character Ego Network")
        G = load_graph()
        nodes = list(G.nodes())
        names = [G.nodes[n].get("name", str(n)) for n in nodes]
        name_to_id = dict(zip(names, nodes))
        
        selected_name = st.selectbox("Search Character:", sorted(names))
        if selected_name:
            node_id = name_to_id[selected_name]
            ego_net = nx.ego_graph(G, node_id, radius=1)
            
            st.markdown(f"**{selected_name}** is directly connected to **{ego_net.number_of_nodes() - 1}** characters.")
            
            if ego_net.number_of_nodes() > 1:
                fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1117")
                ax.set_facecolor("#0d1117")
                pos = nx.spring_layout(ego_net, seed=42)
                
                # Draw edges
                nx.draw_networkx_edges(ego_net, pos, alpha=0.5, edge_color="#58a6ff", ax=ax)
                
                # Draw nodes
                node_colors = ["#ff7b72" if n == node_id else "#3fb950" for n in ego_net.nodes()]
                nx.draw_networkx_nodes(ego_net, pos, node_color=node_colors, node_size=600, ax=ax)
                
                # Draw labels
                labels = {n: G.nodes[n].get("name", str(n)) for n in ego_net.nodes()}
                nx.draw_networkx_labels(ego_net, pos, labels=labels, font_size=9, font_color="#e6edf3", font_weight="bold", ax=ax)
                
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CENTRALITY RANKINGS
# ══════════════════════════════════════════════════════════════════════════════

def page_centrality_rankings():
    st.title("📈 Centrality Rankings")
    st.markdown(
        "<p style='color:#8b949e'>"
        "Five centrality metrics reveal who matters most in the galaxy.</p>",
        unsafe_allow_html=True,
    )

    cent_df = load_centrality()

    # ── Controls ──
    col1, col2 = st.columns([2, 1])
    with col1:
        metric = st.selectbox(
            "Sort by metric",
            options=[
                "centrality_score", "degree_centrality",
                "betweenness_centrality", "closeness_centrality",
                "eigenvector_centrality", "pagerank",
            ],
            format_func=lambda x: x.replace("_", " ").title(),
        )
    with col2:
        top_n = st.slider("Show top N", min_value=5, max_value=50, value=20)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📋 Ranked Table",
        "📊 Bar Chart",
        "🕸️ Radar Chart",
    ])

    with tab1:
        display_df = cent_df.reset_index(drop=True).nlargest(top_n, metric).copy()
        display_df.index = range(1, len(display_df) + 1)
        display_df.index.name = "Rank"

        # Format columns
        for col in ["degree_centrality", "betweenness_centrality",
                    "closeness_centrality", "eigenvector_centrality",
                    "pagerank", "centrality_score"]:
            display_df[col] = display_df[col].round(5)

        st.dataframe(
            display_df[[
                "name", "degree_centrality", "betweenness_centrality",
                "closeness_centrality", "eigenvector_centrality",
                "pagerank", "centrality_score",
            ]].rename(columns={
                "name":                  "Character",
                "degree_centrality":     "Degree",
                "betweenness_centrality":"Betweenness",
                "closeness_centrality":  "Closeness",
                "eigenvector_centrality":"Eigenvector",
                "pagerank":              "PageRank",
                "centrality_score":      "Overall Score",
            }),
            use_container_width=True, height=480,
        )

        # Metric explanations
        with st.expander("📖 Metric Definitions"):
            st.markdown("""
            | Metric | Meaning |
            |--------|---------|
            | **Degree** | Fraction of all characters directly connected to this node |
            | **Betweenness** | How often this character sits on shortest paths between others (broker role) |
            | **Closeness** | How quickly this character can reach everyone else |
            | **Eigenvector** | Influence based on quality of connections (connected to influential nodes) |
            | **PageRank** | Random walk prestige — probability a random walk lands here |
            | **Overall Score** | Equal-weighted average of all five metrics |
            """)

    with tab2:
        top_bar = cent_df.reset_index(drop=True).nlargest(top_n, metric)
        fig = px.bar(
            top_bar, x=metric, y="name",
            orientation="h",
            color=metric,
            color_continuous_scale="Plasma",
            title=f"Top {top_n} Characters — {metric.replace('_',' ').title()}",
            template="plotly_dark",
            text=top_bar[metric].round(4).astype(str),
        )
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT_CLR),
            yaxis=dict(autorange="reversed"),
            height=max(400, top_n * 24),
            margin=dict(l=160, r=80, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        radar_path = load_figure("centrality_radar.html")
        if radar_path.exists():
            with open(radar_path, "r") as f:
                st.components.v1.html(f.read(), height=580, scrolling=False)
        else:
            st.info("Run `scripts/run_visualization.py` to generate radar chart.")

    # Heatmap
    st.markdown("---")
    st.subheader("🗺️ Centrality Heatmap")
    heatmap_path = load_figure("centrality_heatmap.png")
    if heatmap_path.exists():
        st.image(str(heatmap_path), use_column_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COMMUNITY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def page_community_explorer():
    st.title("🔵 Community Explorer")
    st.markdown(
        "<p style='color:#8b949e'>"
        "Louvain community detection reveals natural character clusters — "
        "Jedi, Sith, Rebels, and more.</p>",
        unsafe_allow_html=True,
    )

    G                         = load_graph()
    partition, comm_sum, mod  = load_communities()

    # ── Summary metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Communities Found", len(comm_sum))
    c2.metric("Modularity Score",  f"{mod:.4f}")
    c3.metric("Largest Community", int(comm_sum["size"].max()))
    c4.metric("Avg Community Size",f"{comm_sum['size'].mean():.1f}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📋 Community Summary",
        "🔍 Explore a Community",
        "📊 Size Distribution",
    ])

    with tab1:
        st.subheader("All Communities — Overview")
        display_comm = comm_sum.copy()
        display_comm.index = range(1, len(display_comm) + 1)
        st.dataframe(
            display_comm.rename(columns={
                "community":           "ID",
                "size":                "Members",
                "internal_edges":      "Internal Edges",
                "avg_internal_weight": "Avg Edge Weight",
                "top_characters":      "Key Characters",
            }),
            use_container_width=True,
        )

        # Community bar chart
        fig = px.bar(
            comm_sum, x="community", y="size",
            color="size",
            color_continuous_scale="Viridis",
            title="Community Sizes",
            labels={"community": "Community ID", "size": "Members"},
            template="plotly_dark",
            text="size",
        )
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT_CLR), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        comm_ids = sorted(comm_sum["community"].tolist())
        selected_comm = st.selectbox(
            "Select a community to explore",
            options=comm_ids,
            format_func=lambda x: f"Community {x}",
        )

        # Get members
        members = [
            nid for nid, c in partition.items()
            if c + 1 == selected_comm
        ]

        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.markdown(f"**Community {selected_comm} Members**")
            member_data = []
            for nid in members:
                name = G.nodes[nid].get("name", str(nid))
                deg  = G.degree(nid)
                member_data.append({"Character": name, "Degree": deg})
            member_df = pd.DataFrame(member_data).sort_values(
                "Degree", ascending=False
            ).reset_index(drop=True)
            member_df.index += 1
            st.dataframe(member_df, use_container_width=True, height=360)

        with col_r:
            st.markdown(f"**Internal Network — Community {selected_comm}**")
            subG = G.subgraph(members).copy()
            if subG.number_of_nodes() > 1:
                fig2, ax = plt.subplots(figsize=(7, 5),
                                        facecolor="#0d1117")
                ax.set_facecolor("#0d1117")
                pos = nx.spring_layout(subG, k=3, seed=42)
                pr  = nx.pagerank(subG, weight="weight")
                sizes = [pr[n] * 8000 + 200 for n in subG.nodes()]
                nx.draw_networkx_edges(
                    subG, pos, alpha=0.4, edge_color=ACCENT,
                    width=1.5, ax=ax,
                )
                nx.draw_networkx_nodes(
                    subG, pos, node_size=sizes, ax=ax,
                    node_color=PALETTE[selected_comm % len(PALETTE)],
                    alpha=0.9,
                )
                labels = {
                    n: G.nodes[n].get("name", str(n))
                    for n in subG.nodes()
                }
                nx.draw_networkx_labels(
                    subG, pos, labels=labels,
                    font_size=8, font_color="#e6edf3",
                    font_weight="bold", ax=ax,
                )
                ax.axis("off")
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
            else:
                st.info("Community too small to visualize.")

    with tab3:
        fig3 = px.pie(
            comm_sum, values="size", names="community",
            title="Community Size Distribution",
            color_discrete_sequence=PALETTE,
            template="plotly_dark",
            hole=0.35,
        )
        fig3.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )
        fig3.update_layout(
            paper_bgcolor=DARK_BG, font=dict(color=TEXT_CLR),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Internal vs external edges per community
        G_main      = load_graph()
        rows        = []
        for comm_id in sorted(set(partition.values())):
            members = [n for n, c in partition.items() if c == comm_id]
            subG    = G_main.subgraph(members)
            int_e   = subG.number_of_edges()
            ext_e   = sum(
                1 for n in members
                for nb in G_main.neighbors(n)
                if nb not in members
            ) // 2
            rows.append({
                "Community": comm_id + 1,
                "Internal": int_e,
                "External": ext_e,
            })
        edge_df = pd.DataFrame(rows)
        fig4 = px.bar(
            edge_df.melt(
                id_vars="Community",
                value_vars=["Internal", "External"],
            ),
            x="Community", y="value", color="variable",
            barmode="group",
            title="Internal vs External Edges per Community",
            labels={"value": "Edge Count", "variable": "Type"},
            color_discrete_sequence=[ACCENT, "#ff7b72"],
            template="plotly_dark",
        )
        fig4.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT_CLR),
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — GRAPH ML RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def page_graph_ml_results():
    st.title("🤖 Graph ML Results")
    st.markdown(
        "<p style='color:#8b949e'>"
        "Node classification using GCN and GraphSAGE — "
        "predicting character communities from graph structure.</p>",
        unsafe_allow_html=True,
    )

    ml_results = load_ml_results()

    if not ml_results:
        st.warning(
            "No ML results found. "
            "Run `python scripts/run_graph_ml.py` first."
        )
        st.markdown("""
```bash
        source venv/bin/activate
        python scripts/run_graph_ml.py
```
        """)
        return

    # ── Model comparison metrics ──
    st.subheader("📊 Model Comparison")
    
    num_models = len(ml_results)
    cols = st.columns(num_models)

    for col, (model_name, res) in zip(cols, ml_results.items()):
        with col:
            st.markdown(f"#### {model_name}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy",    f"{res['accuracy']:.4f}")
            m2.metric("F1 Weighted", f"{res['f1_weighted']:.4f}")
            m3.metric("F1 Macro",    f"{res['f1_macro']:.4f}")

    # ── Comparison bar chart ──
    comparison_df = pd.DataFrame([
        {
            "Model":        name,
            "Accuracy":     res["accuracy"],
            "F1 Weighted":  res["f1_weighted"],
            "F1 Macro":     res["f1_macro"],
        }
        for name, res in ml_results.items()
    ])

    fig = px.bar(
        comparison_df.melt(id_vars="Model"),
        x="variable", y="value", color="Model",
        barmode="group",
        title="Model Performance Comparison",
        labels={"variable": "Metric", "value": "Score"},
        color_discrete_sequence=[ACCENT, "#3fb950", "#d2a8ff"],
        template="plotly_dark",
        text_auto=".3f",
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR),
        yaxis=dict(range=[0, 1]),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Training curves + confusion matrices ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Training Curves",
        "🔲 Confusion Matrix (GCN)",
        "🔲 Confusion Matrix (GraphSAGE)",
        "🔲 Confusion Matrix (GAT)",
    ])

    with tab1:
        p = load_figure("training_curves.png")
        if p.exists():
            st.image(str(p), use_column_width=True)
        else:
            st.info("Training curves not found.")

    with tab2:
        p = load_figure("confusion_matrix_gcn.png")
        if p.exists():
            st.image(str(p), use_column_width=True)

    with tab3:
        p = load_figure("confusion_matrix_graphsage.png")
        if p.exists():
            st.image(str(p), use_column_width=True)

    with tab4:
        p = load_figure("confusion_matrix_gat.png")
        if p.exists():
            st.image(str(p), use_column_width=True)

    st.markdown("---")
    st.subheader("🏗️ Model Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **GCN (Graph Convolutional Network)**
    Input [N, 7]
      ↓ GCNConv(7 → 64) + ReLU + Dropout(0.4)
      ↓ GCNConv(64 → 64) + ReLU + Dropout(0.4)
      ↓ Linear(64 → n_classes)
      ↓ LogSoftmax
    Output [N, n_classes]
        Reference: Kipf & Welling, ICLR 2017
        """)
    with col2:
        st.markdown("""
        **GraphSAGE (Graph Sample and Aggregate)**
    Input [N, 7]
      ↓ SAGEConv(7 → 64) + ReLU + Dropout(0.4)
      ↓ SAGEConv(64 → 64) + ReLU + Dropout(0.4)
      ↓ Linear(64 → n_classes)
      ↓ LogSoftmax
    Output [N, n_classes]
        Reference: Hamilton et al., NeurIPS 2017
        """)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        **GAT (Graph Attention Network)**
    Input [N, 7]
      ↓ GATConv(7 → 64, heads=4) + ELU + Dropout(0.4)
      ↓ GATConv(64*4 → n_classes, heads=1)
      ↓ LogSoftmax
    Output [N, n_classes]
        Reference: Veličković et al., ICLR 2018
        """)

    with st.expander("📋 Node Features Used (7 features)"):
        st.markdown("""
        | # | Feature | Description |
        |---|---------|-------------|
        | 0 | Degree Centrality | Normalized connection count |
        | 1 | Betweenness Centrality | Broker/bridge score |
        | 2 | Closeness Centrality | Reachability score |
        | 3 | Eigenvector Centrality | Influence score |
        | 4 | PageRank | Random walk prestige |
        | 5 | Raw Degree (normalized) | Direct neighbor count / N |
        | 6 | Total Weight (log) | Log of total interaction weight |
        """)

    # ── Footer ──
    st.markdown("""
    <div class='footer'>
        Star Wars Character Network Analysis ·
        Built with NetworkX · PyTorch Geometric · Streamlit ·
        <a href='https://github.com/Stokesy-dev/character-network-analysis'
           style='color:#58a6ff'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    if page == "Home":
        page_home()
    elif page == "Dataset Insights":
        page_dataset_insights()
    elif page == "Network Visualizer":
        page_network_visualizer()
    elif page == "Centrality Rankings":
        page_centrality_rankings()
    elif page == "Community Explorer":
        page_community_explorer()
    elif page == "Graph ML Results":
        page_graph_ml_results()


if __name__ == "__main__":
    main()
