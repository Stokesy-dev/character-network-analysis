"""
graph_builder.py
----------------
Constructs a weighted, undirected NetworkX graph from processed
node and edge CSVs. Provides graph-level summary statistics.
"""

import logging
import pandas as pd
import networkx as nx
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cleaned nodes and edges from data/processed/.

    Returns:
        Tuple of (nodes_df, edges_df).
    """
    nodes_path = PROCESSED_DIR / "nodes.csv"
    edges_path = PROCESSED_DIR / "edges.csv"

    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run scripts/run_pipeline.py first."
        )

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    logger.info(f"Loaded {len(nodes)} nodes, {len(edges)} edges.")
    return nodes, edges


def build_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    weight_col: str = "weight",
) -> nx.Graph:
    """
    Build a weighted undirected NetworkX graph.

    Node attributes added:
        - name, value, colour, degree, total_weight

    Edge attributes added:
        - weight, weight_norm

    Args:
        nodes_df: Cleaned nodes DataFrame.
        edges_df: Cleaned edges DataFrame.
        weight_col: Column to use as edge weight.

    Returns:
        NetworkX Graph object.
    """
    G = nx.Graph()

    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        G.add_node(
            row["node_id"],
            name=row["name"],
            value=int(row.get("value", 1)),
            colour=row.get("colour", "#cccccc"),
            degree_raw=int(row.get("degree", 0)),
            total_weight=int(row.get("total_weight", 0)),
        )

    # Add edges with attributes
    for _, row in edges_df.iterrows():
        G.add_edge(
            int(row["source"]),
            int(row["target"]),
            weight=float(row[weight_col]),
            weight_norm=float(row.get("weight_norm", 1.0)),
            source_name=row.get("source_name", ""),
            target_name=row.get("target_name", ""),
        )

    logger.info(
        f"Graph built — Nodes: {G.number_of_nodes()}, "
        f"Edges: {G.number_of_edges()}"
    )
    return G


def graph_summary(G: nx.Graph) -> dict:
    """
    Compute and return key graph-level statistics.

    Args:
        G: NetworkX Graph.

    Returns:
        Dictionary of summary statistics.
    """
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    G_lcc = G.subgraph(largest_cc)

    summary = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": round(nx.density(G), 6),
        "num_connected_components": nx.number_connected_components(G),
        "largest_component_size": len(largest_cc),
        "avg_clustering_coefficient": round(nx.average_clustering(G, weight="weight"), 6),
        "avg_shortest_path_lcc": round(
            nx.average_shortest_path_length(G_lcc, weight=None), 4
        ),
        "diameter_lcc": nx.diameter(G_lcc),
        "is_connected": nx.is_connected(G),
        "avg_degree": round(
            sum(d for _, d in G.degree()) / G.number_of_nodes(), 4
        ),
    }

    logger.info("── Graph Summary ─────────────────────────────────")
    for k, v in summary.items():
        logger.info(f"  {k:<35} {v}")
    logger.info("──────────────────────────────────────────────────")

    return summary


def get_node_name(G: nx.Graph, node_id: int) -> str:
    """Helper: return character name for a node ID."""
    return G.nodes[node_id].get("name", str(node_id))
