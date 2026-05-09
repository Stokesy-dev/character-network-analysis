"""
centrality.py
-------------
Computes five centrality metrics for every character node:
  - Degree Centrality
  - Betweenness Centrality
  - Closeness Centrality
  - Eigenvector Centrality
  - PageRank

Generates ranked tables and saves to outputs/reports/.
"""

import logging
import pandas as pd
import networkx as nx
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_degree_centrality(G: nx.Graph) -> dict:
    """
    Degree centrality: fraction of nodes each node is connected to.
    Higher = more direct connections.
    """
    return nx.degree_centrality(G)


def compute_betweenness_centrality(G: nx.Graph) -> dict:
    """
    Betweenness centrality: fraction of shortest paths passing through a node.
    Higher = more of a 'bridge' or 'broker' in the network.
    Uses normalized, weighted betweenness.
    """
    return nx.betweenness_centrality(G, weight="weight", normalized=True)


def compute_closeness_centrality(G: nx.Graph) -> dict:
    """
    Closeness centrality: inverse of average shortest path distance.
    Higher = can reach all others quickly.
    """
    return nx.closeness_centrality(G)


def compute_eigenvector_centrality(G: nx.Graph, max_iter: int = 1000) -> dict:
    """
    Eigenvector centrality: influence of a node based on its neighbours' influence.
    Higher = connected to other highly influential nodes.
    Uses weighted edges.
    """
    try:
        return nx.eigenvector_centrality(G, weight="weight", max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality did not converge — using unweighted.")
        return nx.eigenvector_centrality_numpy(G)


def compute_pagerank(G: nx.Graph, alpha: float = 0.85) -> dict:
    """
    PageRank: probability of arriving at a node via random walk.
    alpha = damping factor (0.85 standard).
    Higher = more 'prestige' in the network.
    """
    return nx.pagerank(G, alpha=alpha, weight="weight")


def build_centrality_table(G: nx.Graph) -> pd.DataFrame:
    """
    Compute all five centrality metrics and return a merged ranked DataFrame.

    Args:
        G: NetworkX Graph with 'name' node attribute.

    Returns:
        DataFrame with columns:
        [node_id, name, degree_centrality, betweenness_centrality,
         closeness_centrality, eigenvector_centrality, pagerank, centrality_score]
    """
    logger.info("Computing centrality metrics...")

    degree      = compute_degree_centrality(G)
    betweenness = compute_betweenness_centrality(G)
    closeness   = compute_closeness_centrality(G)
    eigenvector = compute_eigenvector_centrality(G)
    pagerank    = compute_pagerank(G)

    rows = []
    for node_id in G.nodes():
        rows.append({
            "node_id":               node_id,
            "name":                  G.nodes[node_id].get("name", str(node_id)),
            "degree_centrality":     round(degree.get(node_id, 0), 6),
            "betweenness_centrality":round(betweenness.get(node_id, 0), 6),
            "closeness_centrality":  round(closeness.get(node_id, 0), 6),
            "eigenvector_centrality":round(eigenvector.get(node_id, 0), 6),
            "pagerank":              round(pagerank.get(node_id, 0), 6),
        })

    df = pd.DataFrame(rows)

    # Composite centrality score: equal-weighted average of all 5 (all already in [0,1])
    metrics = [
        "degree_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
        "pagerank",
    ]
    df["centrality_score"] = df[metrics].mean(axis=1).round(6)
    df = df.sort_values("centrality_score", ascending=False).reset_index(drop=True)
    df.index += 1  # rank starts at 1
    df.index.name = "rank"

    logger.info("Centrality computation complete.")
    return df


def print_top_n(df: pd.DataFrame, n: int = 10) -> None:
    """Print top-N characters by centrality score."""
    print("\n" + "═" * 80)
    print(f"  TOP {n} MOST INFLUENTIAL CHARACTERS — Star Wars Network")
    print("═" * 80)
    cols = ["name", "degree_centrality", "betweenness_centrality",
            "closeness_centrality", "eigenvector_centrality",
            "pagerank", "centrality_score"]
    print(df[cols].head(n).to_string())
    print("═" * 80 + "\n")


def save_centrality_report(df: pd.DataFrame) -> Path:
    """Save centrality table to outputs/reports/centrality.csv."""
    path = REPORTS_DIR / "centrality.csv"
    df.reset_index().to_csv(path, index=False)
    logger.info(f"✅ Centrality report saved: {path}")
    return path


def get_top_characters(
    df: pd.DataFrame,
    metric: str = "centrality_score",
    n: int = 10,
) -> pd.DataFrame:
    """
    Return top-N characters sorted by a given metric.

    Args:
        df: Centrality DataFrame from build_centrality_table().
        metric: Column name to sort by.
        n: Number of top characters to return.

    Returns:
        Filtered and sorted DataFrame.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Choose from: {list(df.columns)}")
    return df.reset_index().nlargest(n, metric)[["rank", "name", metric]]
