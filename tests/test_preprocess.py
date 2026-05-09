"""
test_preprocess.py
------------------
Unit tests for src/preprocess.py
"""

import pytest
import pandas as pd
from src.preprocess import clean_nodes, clean_edges, generate_node_features


@pytest.fixture
def sample_nodes():
    return pd.DataFrame({
        "name": ["Luke Skywalker", "Darth Vader", "Leia Organa", "Luke Skywalker"],
        "value": [10, 8, 6, 5],
        "colour": ["#ff0000", "#000000", "#ffffff", "#ff0000"],
    })


@pytest.fixture
def sample_edges(sample_nodes):
    nodes = clean_nodes(sample_nodes)
    return pd.DataFrame({
        "source": [0, 0, 1],
        "target": [1, 2, 2],
        "weight": [5, 3, 4],
    }), nodes


def test_clean_nodes_deduplication(sample_nodes):
    result = clean_nodes(sample_nodes)
    assert result["name"].nunique() == len(result), "Duplicate names must be removed"


def test_clean_nodes_has_node_id(sample_nodes):
    result = clean_nodes(sample_nodes)
    assert "node_id" in result.columns


def test_clean_nodes_value_is_int(sample_nodes):
    result = clean_nodes(sample_nodes)
    assert result["value"].dtype in [int, "int64", "int32"]


def test_clean_edges_no_self_loops(sample_edges):
    raw_edges, nodes = sample_edges
    result = clean_edges(raw_edges, nodes)
    assert (result["source"] != result["target"]).all(), "Self-loops must be removed"


def test_clean_edges_weight_norm_range(sample_edges):
    raw_edges, nodes = sample_edges
    result = clean_edges(raw_edges, nodes)
    assert result["weight_norm"].between(0, 1).all(), "Normalized weights must be in [0,1]"


def test_generate_node_features_degree(sample_edges):
    raw_edges, nodes = sample_edges
    edges = clean_edges(raw_edges, nodes)
    enriched = generate_node_features(nodes, edges)
    assert "degree" in enriched.columns
    assert "total_weight" in enriched.columns
    assert enriched["degree"].min() >= 0
