"""
test_graph_builder.py
---------------------
Unit tests for src/graph_builder.py
"""

import pytest
import pandas as pd
import networkx as nx
from src.graph_builder import build_graph, graph_summary


@pytest.fixture
def sample_data():
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "name": ["Luke", "Vader", "Leia", "Han"],
        "value": [10, 8, 6, 7],
        "colour": ["#ff0000", "#000000", "#ffffff", "#ffff00"],
        "degree": [3, 2, 2, 1],
        "total_weight": [20, 15, 10, 7],
    })
    edges = pd.DataFrame({
        "source": [0, 0, 1, 0],
        "target": [1, 2, 2, 3],
        "weight": [5.0, 3.0, 4.0, 2.0],
        "weight_norm": [1.0, 0.6, 0.8, 0.4],
        "source_name": ["Luke", "Luke", "Vader", "Luke"],
        "target_name": ["Vader", "Leia", "Leia", "Han"],
    })
    return nodes, edges


def test_build_graph_node_count(sample_data):
    nodes, edges = sample_data
    G = build_graph(nodes, edges)
    assert G.number_of_nodes() == 4


def test_build_graph_edge_count(sample_data):
    nodes, edges = sample_data
    G = build_graph(nodes, edges)
    assert G.number_of_edges() == 4


def test_node_has_name_attribute(sample_data):
    nodes, edges = sample_data
    G = build_graph(nodes, edges)
    assert G.nodes[0]["name"] == "Luke"


def test_edge_has_weight_attribute(sample_data):
    nodes, edges = sample_data
    G = build_graph(nodes, edges)
    assert G[0][1]["weight"] == 5.0


def test_graph_summary_keys(sample_data):
    nodes, edges = sample_data
    G = build_graph(nodes, edges)
    summary = graph_summary(G)
    expected_keys = [
        "num_nodes", "num_edges", "density",
        "num_connected_components", "avg_clustering_coefficient",
    ]
    for key in expected_keys:
        assert key in summary, f"Missing key: {key}"


def test_graph_is_undirected(sample_data):
    nodes, edges = sample_data
    G = build_graph(nodes, edges)
    assert isinstance(G, nx.Graph)
    assert not isinstance(G, nx.DiGraph)
