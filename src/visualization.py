"""
visualization.py
----------------
All visualization functions for the Character Network Analysis project.

Generates:
  1. Static network graph (matplotlib)         → network_static.png
  2. Degree distribution plot (seaborn)        → degree_distribution.png
  3. Edge weight distribution (seaborn)        → edge_weight_distribution.png
  4. Top characters bar chart (plotly)         → top_characters.html
  5. Centrality heatmap (seaborn)              → centrality_heatmap.png
  6. Centrality radar chart (plotly)           → centrality_radar.html
  7. Interactive pyvis network graph           → network_interactive.html
  8. Community-colored network (matplotlib)    → network_communities.png
"""

import logging
import warnings
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyvis.network import Network
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # non-interactive backend for saving

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
TEXT_CLR  = "#e6edf3"
ACCENT    = "#58a6ff"
PALETTE   = [
    "#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff",
    "#ffa657", "#79c0ff", "#56d364", "#ff9bce",
]

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   TEXT_CLR,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
    "text.color":        TEXT_CLR,
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. STATIC NETWORK GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def plot_static_network(
    G: nx.Graph,
    centrality_df: Optional[pd.DataFrame] = None,
    top_n: int = 40,
    filename: str = "network_static.png",
) -> Path:
    """
    Draw a static network graph with:
    - Node size  ∝ PageRank
    - Node color ∝ degree centrality (cool→warm gradient)
    - Edge width ∝ normalized weight
    - Labels only for top-N characters by degree

    Args:
        G: NetworkX Graph.
        centrality_df: Output of build_centrality_table(). If None, computed inline.
        top_n: Number of characters to label.
        filename: Output filename.

    Returns:
        Path to saved figure.
    """
    logger.info("Plotting static network graph...")

    fig, ax = plt.subplots(figsize=(18, 14), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Layout — spring layout with heavy nodes pulled to center
    pos = nx.spring_layout(G, k=2.5, seed=42, weight="weight")

    # Node sizing by PageRank
    pr = nx.pagerank(G, weight="weight")
    node_sizes = [pr[n] * 18000 + 80 for n in G.nodes()]

    # Node coloring by degree centrality
    dc = nx.degree_centrality(G)
    node_colors = [dc[n] for n in G.nodes()]

    # Edge widths by normalized weight
    edge_weights = [G[u][v].get("weight_norm", 0.5) for u, v in G.edges()]
    edge_widths  = [w * 3.5 + 0.3 for w in edge_weights]
    edge_alphas  = [max(0.15, w * 0.8) for w in edge_weights]

    # Draw edges
    for i, (u, v) in enumerate(G.edges()):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=edge_widths[i],
            alpha=edge_alphas[i],
            edge_color=ACCENT,
            ax=ax,
        )

    # Draw nodes
    nc = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        vmin=0, vmax=max(node_colors),
        alpha=0.92,
        ax=ax,
    )

    # Labels for top-N characters only
    top_nodes = sorted(dc, key=dc.get, reverse=True)[:top_n]
    labels = {n: G.nodes[n].get("name", str(n)) for n in top_nodes}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=7.5,
        font_color=TEXT_CLR,
        font_weight="bold",
        ax=ax,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                norm=plt.Normalize(vmin=0, vmax=max(node_colors)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Degree Centrality", color=TEXT_CLR, fontsize=11)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

    ax.set_title(
        "Star Wars Character Interaction Network\n"
        "Node size = PageRank  |  Color = Degree Centrality  |  "
        "Edge width = Interaction strength",
        fontsize=13, color=TEXT_CLR, pad=16,
    )
    ax.axis("off")
    plt.tight_layout()

    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"✅ Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEGREE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_degree_distribution(
    G: nx.Graph,
    filename: str = "degree_distribution.png",
) -> Path:
    """
    Plot degree distribution with KDE overlay and power-law reference.
    """
    logger.info("Plotting degree distribution...")

    degrees = [d for _, d in G.degree()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)

    # Left: histogram + KDE
    ax = axes[0]
    sns.histplot(degrees, bins=20, kde=True, color=ACCENT, ax=ax,
                 line_kws={"linewidth": 2})
    ax.set_title("Degree Distribution", fontsize=13)
    ax.set_xlabel("Degree (number of connections)")
    ax.set_ylabel("Count")
    ax.axvline(np.mean(degrees), color="#ff7b72", linestyle="--",
               linewidth=1.5, label=f"Mean = {np.mean(degrees):.1f}")
    ax.axvline(np.median(degrees), color="#3fb950", linestyle="--",
               linewidth=1.5, label=f"Median = {np.median(degrees):.1f}")
    ax.legend(fontsize=9)

    # Right: log-log scale (check for power-law)
    ax2 = axes[1]
    from collections import Counter
    degree_counts = Counter(degrees)
    x = sorted(degree_counts.keys())
    y = [degree_counts[d] for d in x]
    ax2.scatter(x, y, color=ACCENT, alpha=0.8, s=50, zorder=5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Degree Distribution (Log-Log Scale)", fontsize=13)
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("Network Degree Analysis", fontsize=15, y=1.02, color=TEXT_CLR)
    plt.tight_layout()

    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"✅ Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 3. EDGE WEIGHT DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_edge_weight_distribution(
    edges_df: pd.DataFrame,
    filename: str = "edge_weight_distribution.png",
) -> Path:
    """Plot distribution of raw and normalized edge weights."""
    logger.info("Plotting edge weight distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)

    # Raw weights
    sns.histplot(edges_df["weight"], bins=30, kde=True,
                 color="#d2a8ff", ax=axes[0], line_kws={"linewidth": 2})
    axes[0].set_title("Raw Interaction Weights", fontsize=13)
    axes[0].set_xlabel("Co-occurrence count")
    axes[0].set_ylabel("Frequency")

    # Normalized weights
    sns.histplot(edges_df["weight_norm"], bins=30, kde=True,
                 color="#3fb950", ax=axes[1], line_kws={"linewidth": 2})
    axes[1].set_title("Normalized Edge Weights [0–1]", fontsize=13)
    axes[1].set_xlabel("Normalized weight")
    axes[1].set_ylabel("Frequency")

    fig.suptitle("Edge Weight Distribution", fontsize=15, y=1.02, color=TEXT_CLR)
    plt.tight_layout()

    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"✅ Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 4. TOP CHARACTERS BAR CHART (PLOTLY)
# ══════════════════════════════════════════════════════════════════════════════

def plot_top_characters_plotly(
    centrality_df: pd.DataFrame,
    top_n: int = 15,
    filename: str = "top_characters.html",
) -> Path:
    """
    Interactive Plotly bar chart: top-N characters by each centrality metric.
    Tabs via dropdown selector.
    """
    logger.info("Plotting interactive top-characters bar chart...")

    metrics = {
        "centrality_score":      "Overall Centrality Score",
        "degree_centrality":     "Degree Centrality",
        "betweenness_centrality":"Betweenness Centrality",
        "closeness_centrality":  "Closeness Centrality",
        "eigenvector_centrality":"Eigenvector Centrality",
        "pagerank":              "PageRank",
    }

    df = centrality_df.reset_index(drop=True)
    fig = go.Figure()

    buttons = []
    for i, (col, label) in enumerate(metrics.items()):
        top = df.nlargest(top_n, col)
        fig.add_trace(go.Bar(
            x=top[col],
            y=top["name"],
            orientation="h",
            name=label,
            visible=(i == 0),
            marker=dict(
                color=top[col],
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title=label, thickness=12),
            ),
            text=[f"{v:.4f}" for v in top[col]],
            textposition="outside",
        ))
        visibility = [False] * len(metrics)
        visibility[i] = True
        buttons.append(dict(
            label=label,
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"Top {top_n} Characters — {label}"},
            ],
        ))

    fig.update_layout(
        title=f"Top {top_n} Characters — Overall Centrality Score",
        title_font_size=16,
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0.01, xanchor="left",
            y=1.12, yanchor="top",
            bgcolor="#161b22",
            bordercolor="#30363d",
            font=dict(color=TEXT_CLR),
        )],
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR),
        height=520,
        margin=dict(l=160, r=80, t=80, b=40),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(autorange="reversed"),
    )

    path = FIGURES_DIR / filename
    fig.write_html(str(path))
    logger.info(f"✅ Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 5. CENTRALITY HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_centrality_heatmap(
    centrality_df: pd.DataFrame,
    top_n: int = 20,
    filename: str = "centrality_heatmap.png",
) -> Path:
    """
    Seaborn heatmap of all centrality metrics for top-N characters.
    Each column is min-max normalized for comparability.
    """
    logger.info("Plotting centrality heatmap...")

    metrics = [
        "degree_centrality", "betweenness_centrality",
        "closeness_centrality", "eigenvector_centrality", "pagerank",
    ]

    df = centrality_df.reset_index(drop=True).nlargest(top_n, "centrality_score")
    heatmap_data = df.set_index("name")[metrics].copy()

    # Min-max normalize each column
    heatmap_data = (heatmap_data - heatmap_data.min()) / (
        heatmap_data.max() - heatmap_data.min() + 1e-9
    )
    heatmap_data.columns = [
        "Degree", "Betweenness", "Closeness", "Eigenvector", "PageRank"
    ]

    fig, ax = plt.subplots(figsize=(10, 12), facecolor=DARK_BG)
    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="#0d1117",
        cbar_kws={"shrink": 0.6, "label": "Normalized Score"},
        ax=ax,
    )
    ax.set_title(
        f"Centrality Metrics — Top {top_n} Characters\n(Min-Max Normalized)",
        fontsize=13, pad=14,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"✅ Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 6. CENTRALITY RADAR CHART (PLOTLY)
# ══════════════════════════════════════════════════════════════════════════════

def plot_centrality_radar(
    centrality_df: pd.DataFrame,
    top_n: int = 6,
    filename: str = "centrality_radar.html",
) -> Path:
    """
    Interactive Plotly radar chart comparing top-N characters
    across all centrality dimensions.
    """
    logger.info("Plotting centrality radar chart...")

    metrics = [
        "degree_centrality", "betweenness_centrality",
        "closeness_centrality", "eigenvector_centrality", "pagerank",
    ]
    labels = ["Degree", "Betweenness", "Closeness", "Eigenvector", "PageRank"]

    df = centrality_df.reset_index(drop=True).nlargest(top_n, "centrality_score")

    # Normalize for radar
    for m in metrics:
        col_max = centrality_df[m].max()
        df[m] = df[m] / (col_max + 1e-9)

    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics] + [row[metrics[0]]]  # close polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            name=row["name"],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            fillcolor=PALETTE[i % len(PALETTE)],
            opacity=0.25,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="#30363d", color=TEXT_CLR,
            ),
            angularaxis=dict(gridcolor="#30363d", color=TEXT_CLR),
        ),
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR),
        title=dict(
            text=f"Centrality Profile — Top {top_n} Characters",
            font=dict(size=16),
        ),
        legend=dict(bgcolor=CARD_BG, bordercolor="#30363d"),
        height=550,
    )

    path = FIGURES_DIR / filename
    fig.write_html(str(path))
    logger.info(f"✅ Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 7. INTERACTIVE PYVIS NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def plot_pyvis_network(
    G: nx.Graph,
    centrality_df: Optional[pd.DataFrame] = None,
    community_map: Optional[dict] = None,
    filename: str = "network_interactive.html",
    top_n_labels: int = 50,
) -> Path:
    """
    Generate an interactive pyvis HTML network graph.

    Features:
    - Node size   ∝ PageRank
    - Node color  = community (if provided) or degree centrality gradient
    - Edge width  ∝ interaction weight
    - Hover shows character name + metrics
    - Physics simulation with Barnes-Hut solver

    Args:
        G: NetworkX Graph.
        centrality_df: Centrality table for tooltip enrichment.
        community_map: {node_id: community_int} from communities.py.
        filename: Output HTML filename.
        top_n_labels: Show labels only for top-N characters.

    Returns:
        Path to saved HTML file.
    """
    logger.info("Building interactive pyvis network...")

    net = Network(
        height="750px", width="100%",
        bgcolor=DARK_BG, font_color=TEXT_CLR,
        notebook=False,
    )

    # Physics config
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "stabilization": {"iterations": 150}
      },
      "edges": {
        "smooth": {"type": "continuous"}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # Centrality lookup
    pr = nx.pagerank(G, weight="weight")
    dc = nx.degree_centrality(G)

    # Community colors
    community_colors = [
        "#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff",
        "#ffa657", "#79c0ff", "#f0883e", "#ff9bce",
        "#56d364", "#e3b341", "#bc8cff", "#f47067",
    ]

    top_nodes_set = set(sorted(dc, key=dc.get, reverse=True)[:top_n_labels])

    # Build centrality lookup by name
    cent_lookup = {}
    if centrality_df is not None:
        for _, row in centrality_df.reset_index(drop=True).iterrows():
            cent_lookup[row["node_id"]] = row

    # Add nodes
    for node_id in G.nodes():
        name = G.nodes[node_id].get("name", str(node_id))
        size = pr.get(node_id, 0.01) * 1200 + 8

        if community_map and node_id in community_map:
            color = community_colors[community_map[node_id] % len(community_colors)]
        else:
            # Degree → blue gradient
            d = dc.get(node_id, 0)
            intensity = int(d * 200 + 55)
            color = f"#{intensity:02x}a6ff"

        # Tooltip
        cr = cent_lookup.get(node_id, {})
        title = (
            f"<b>{name}</b><br>"
            f"Degree: {G.degree(node_id)}<br>"
            f"PageRank: {pr.get(node_id, 0):.4f}<br>"
            f"Betweenness: {cr.get('betweenness_centrality', 'N/A')}<br>"
            f"Closeness: {cr.get('closeness_centrality', 'N/A')}"
        )

        net.add_node(
            node_id,
            label=name if node_id in top_nodes_set else "",
            title=title,
            size=size,
            color=color,
            borderWidth=1.5,
            borderWidthSelected=3,
        )

    # Add edges
    for u, v, data in G.edges(data=True):
        w = data.get("weight_norm", 0.3)
        net.add_edge(
            u, v,
            width=w * 4 + 0.5,
            color={"color": "#30363d", "highlight": ACCENT, "hover": ACCENT},
            title=f"Interactions: {int(data.get('weight', 1))}",
        )

    path = FIGURES_DIR / filename
    net.write_html(str(path))
    logger.info(f"✅ Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 8. COMMUNITY-COLORED NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def plot_community_network(
    G: nx.Graph,
    community_map: dict,
    filename: str = "network_communities.png",
    top_n_labels: int = 30,
) -> Path:
    """
    Static matplotlib network with nodes colored by community.

    Args:
        G: NetworkX Graph.
        community_map: {node_id: community_int} from communities.py.
        filename: Output filename.
        top_n_labels: Number of high-degree nodes to label.

    Returns:
        Path to saved figure.
    """
    logger.info("Plotting community network graph...")

    fig, ax = plt.subplots(figsize=(18, 14), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    pos = nx.spring_layout(G, k=2.8, seed=42, weight="weight")

    communities = sorted(set(community_map.values()))
    cmap_colors = plt.cm.Set1.colors + plt.cm.Set2.colors
    color_map = {c: cmap_colors[i % len(cmap_colors)] for i, c in enumerate(communities)}

    node_colors = [color_map[community_map.get(n, 0)] for n in G.nodes()]
    pr = nx.pagerank(G, weight="weight")
    node_sizes = [pr[n] * 16000 + 100 for n in G.nodes()]

    edge_weights = [G[u][v].get("weight_norm", 0.3) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos,
        width=[w * 2.5 + 0.2 for w in edge_weights],
        alpha=0.25,
        edge_color="#58a6ff",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.90,
        ax=ax,
    )

    dc = nx.degree_centrality(G)
    top_nodes = sorted(dc, key=dc.get, reverse=True)[:top_n_labels]
    labels = {n: G.nodes[n].get("name", str(n)) for n in top_nodes}
    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_size=7, font_color=TEXT_CLR,
        font_weight="bold", ax=ax,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=color_map[c], label=f"Community {c + 1}")
        for c in communities[:10]
    ]
    ax.legend(
        handles=legend_patches, loc="lower left",
        fontsize=9, facecolor=CARD_BG,
        edgecolor="#30363d", labelcolor=TEXT_CLR,
    )

    ax.set_title(
        "Star Wars Character Network — Community Structure\n"
        "Node size = PageRank  |  Color = Community (Louvain)",
        fontsize=13, color=TEXT_CLR, pad=16,
    )
    ax.axis("off")
    plt.tight_layout()

    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"✅ Saved: {path}")
    return path
