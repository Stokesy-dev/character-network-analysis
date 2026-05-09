"""
communities.py
--------------
Community detection using:
  1. Louvain algorithm  (fast, resolution-based)
  2. Girvan-Newman      (edge betweenness removal)

Saves community assignments to outputs/reports/communities.csv
"""

import logging
import pandas as pd
import networkx as nx
import community as community_louvain
from networkx.algorithms.community import girvan_newman
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def detect_louvain(
    G: nx.Graph,
    resolution: float = 1.0,
    random_state: int = 42,
) -> dict:
    """
    Detect communities using the Louvain method.

    Args:
        G: NetworkX Graph (weighted).
        resolution: Higher = more, smaller communities.
        random_state: Reproducibility seed.

    Returns:
        {node_id: community_int} mapping.
    """
    logger.info("Running Louvain community detection...")
    partition = community_louvain.best_partition(
        G, weight="weight",
        resolution=resolution,
        random_state=random_state,
    )
    n_communities = len(set(partition.values()))
    logger.info(f"Louvain found {n_communities} communities.")
    return partition


def detect_girvan_newman(
    G: nx.Graph,
    n_communities: int = 6,
) -> dict:
    """
    Detect communities using Girvan-Newman (edge betweenness removal).
    Note: Slow on large graphs — uses top-k communities.

    Args:
        G: NetworkX Graph.
        n_communities: Target number of communities to extract.

    Returns:
        {node_id: community_int} mapping.
    """
    logger.info(f"Running Girvan-Newman for {n_communities} communities...")

    # Use largest connected component to avoid issues
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc).copy()

    comp = girvan_newman(G_lcc)
    communities = None
    for comm in comp:
        if len(comm) >= n_communities:
            communities = comm
            break
        communities = comm

    if communities is None:
        logger.warning("Girvan-Newman returned no communities.")
        return {}

    partition = {}
    for i, community_set in enumerate(communities):
        for node in community_set:
            partition[node] = i

    logger.info(f"Girvan-Newman found {len(communities)} communities.")
    return partition


def community_summary(
    G: nx.Graph,
    partition: dict,
) -> pd.DataFrame:
    """
    Generate a summary table of each community.

    Returns:
        DataFrame with [community, size, top_characters, internal_edges,
                        avg_internal_weight, modularity_contribution]
    """
    from collections import defaultdict

    community_nodes = defaultdict(list)
    for node, comm in partition.items():
        community_nodes[comm].append(node)

    rows = []
    for comm_id, nodes in sorted(community_nodes.items()):
        subgraph = G.subgraph(nodes)
        internal_edges = subgraph.number_of_edges()
        weights = [d.get("weight", 1) for _, _, d in subgraph.edges(data=True)]
        avg_weight = round(sum(weights) / len(weights), 2) if weights else 0

        # Top 3 characters by degree in this community
        dc = nx.degree_centrality(subgraph)
        top_chars = sorted(dc, key=dc.get, reverse=True)[:3]
        top_names = ", ".join(G.nodes[n].get("name", str(n)) for n in top_chars)

        rows.append({
            "community":          comm_id + 1,
            "size":               len(nodes),
            "internal_edges":     internal_edges,
            "avg_internal_weight":avg_weight,
            "top_characters":     top_names,
        })

    df = pd.DataFrame(rows).sort_values("size", ascending=False)
    return df


def save_communities(
    G: nx.Graph,
    partition: dict,
    filename: str = "communities.csv",
) -> Path:
    """Save full node-community mapping to CSV."""
    rows = [
        {
            "node_id":   node_id,
            "name":      G.nodes[node_id].get("name", str(node_id)),
            "community": comm + 1,
        }
        for node_id, comm in partition.items()
    ]
    df = pd.DataFrame(rows).sort_values(["community", "name"])
    path = REPORTS_DIR / filename
    df.to_csv(path, index=False)
    logger.info(f"✅ Community assignments saved: {path}")
    return path


def compute_modularity(G: nx.Graph, partition: dict) -> float:
    """Compute modularity score for a partition."""
    modularity = community_louvain.modularity(partition, G, weight="weight")
    logger.info(f"Modularity score: {modularity:.4f}")
    return modularity
