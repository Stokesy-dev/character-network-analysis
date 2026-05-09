"""
test_metrics.py
---------------
Unit tests for src/centrality.py
"""

import pytest
import pandas as pd
import networkx as nx
from src.centrality import (
    compute_degree_centrality,
    compute_betweenness_centrality,
    compute_closeness_centrality,
    compute_pagerank,
    build_centrality_table,
)


@pytest.fixture
def simple_graph():
    G = nx.Graph()
    nodes = [
        (0, {"name": "Luke"}),
        (1, {"name": "Vader"}),
        (2, {"name": "Leia"}),
        (3, {"name": "Han"}),
        (4, {"name": "Yoda"}),
    ]
    G.add_nodes_from(nodes)
    G.add_edges_from([
        (0, 1, {"weight": 5.0}),
        (0, 2, {"weight": 3.0}),
        (0, 3, {"weight": 2.0}),
        (1, 2, {"weight": 4.0}),
        (2, 4, {"weight": 1.0}),
    ])
    return G


def test_degree_centrality_range(simple_graph):
    dc = compute_degree_centrality(simple_graph)
    for v in dc.values():
        assert 0.0 <= v <= 1.0


def test_betweenness_centrality_range(simple_graph):
    bc = compute_betweenness_centrality(simple_graph)
    for v in bc.values():
        assert 0.0 <= v <= 1.0


def test_closeness_centrality_range(simple_graph):
    cc = compute_closeness_centrality(simple_graph)
    for v in cc.values():
        assert 0.0 <= v <= 1.0


def test_pagerank_sums_to_one(simple_graph):
    pr = compute_pagerank(simple_graph)
    assert abs(sum(pr.values()) - 1.0) < 1e-6


def test_centrality_table_columns(simple_graph):
    df = build_centrality_table(simple_graph)
    expected_cols = [
        "node_id", "name",
        "degree_centrality", "betweenness_centrality",
        "closeness_centrality", "eigenvector_centrality",
        "pagerank", "centrality_score",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_centrality_table_sorted(simple_graph):
    df = build_centrality_table(simple_graph)
    scores = df["centrality_score"].tolist()
    assert scores == sorted(scores, reverse=True), "Table must be sorted descending"


def test_top_character_is_hub(simple_graph):
    df = build_centrality_table(simple_graph)
    # Node 0 (Luke) connects to 3 others — should rank highest
    assert df.iloc[0]["name"] == "Luke"
