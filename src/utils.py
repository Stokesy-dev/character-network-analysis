"""
utils.py
--------
Shared utility functions used across the project.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def save_json(data: dict, filename: str, subdir: str = "outputs/reports") -> Path:
    """Save a dictionary as a JSON file."""
    path = ROOT / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON: {path}")
    return path


def load_graph_from_processed() -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """
    Convenience loader: returns (G, nodes_df, edges_df) from processed CSVs.
    Avoids repeating boilerplate in notebooks and scripts.
    """
    from src.graph_builder import load_processed, build_graph
    nodes, edges = load_processed()
    G = build_graph(nodes, edges)
    return G, nodes, edges


def node_id_to_name(G: nx.Graph) -> dict:
    """Return {node_id: name} mapping from graph."""
    return {n: G.nodes[n].get("name", str(n)) for n in G.nodes()}


def name_to_node_id(G: nx.Graph) -> dict:
    """Return {name: node_id} mapping from graph."""
    return {G.nodes[n].get("name", str(n)): n for n in G.nodes()}


def format_number(n: float, decimals: int = 4) -> str:
    """Format a float for display."""
    return f"{n:.{decimals}f}"


def get_largest_component(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component as a subgraph."""
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()
