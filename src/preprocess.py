"""
preprocess.py
-------------
Cleans raw Star Wars node/edge data and produces:
  - data/processed/nodes.csv   — character metadata with numeric ID
  - data/processed/edges.csv   — cleaned, weighted edge list
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise character node metadata.

    Steps:
    - Drop duplicates by character name
    - Fill missing values
    - Normalize character names
    - Add numeric node_id

    Args:
        nodes_df: Raw nodes DataFrame from data_loader.

    Returns:
        Cleaned nodes DataFrame.
    """
    logger.info("Cleaning nodes...")
    df = nodes_df.copy()

    # Standardise column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Rename 'name' variants
    if "name" not in df.columns and "label" in df.columns:
        df.rename(columns={"label": "name"}, inplace=True)

    # Normalize character names
    df["name"] = df["name"].str.strip().str.title()

    # Drop rows with no name
    df.dropna(subset=["name"], inplace=True)

    # Deduplicate: keep highest 'value' (scene count) per character
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0).astype(int)
        df = df.sort_values("value", ascending=False).drop_duplicates(
            subset=["name"], keep="first"
        )
    else:
        df = df.drop_duplicates(subset=["name"])
        df["value"] = 1  # default weight

    # Fill missing colour
    if "colour" not in df.columns and "color" in df.columns:
        df.rename(columns={"color": "colour"}, inplace=True)
    if "colour" not in df.columns:
        df["colour"] = "#cccccc"
    df["colour"] = df["colour"].fillna("#cccccc")

    # Add sequential numeric ID for graph construction
    df = df.reset_index(drop=True)
    df["node_id"] = df.index

    logger.info(f"Clean nodes: {len(df)} unique characters.")
    return df[["node_id", "name", "value", "colour"]]


def clean_edges(
    edges_df: pd.DataFrame, nodes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean and validate edge list.

    Steps:
    - Map source/target from name-based or index-based references
    - Remove self-loops
    - Remove edges referencing unknown nodes
    - Aggregate duplicate edges (sum weights)
    - Normalize edge weights to [0, 1]

    Args:
        edges_df: Raw edges DataFrame.
        nodes_df: Cleaned nodes DataFrame (must have 'node_id' and 'name').

    Returns:
        Cleaned edges DataFrame with columns: [source, target, weight, weight_norm]
    """
    logger.info("Cleaning edges...")
    df = edges_df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Rename weight column
    if "value" in df.columns:
        df.rename(columns={"value": "weight"}, inplace=True)
    if "weight" not in df.columns:
        df["weight"] = 1

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1).astype(int)

    # Build name→id map
    name_to_id = dict(zip(nodes_df["name"], nodes_df["node_id"]))
    id_to_name = dict(zip(nodes_df["node_id"], nodes_df["name"]))

    # If source/target are names, convert to node_id
    if df["source"].dtype == object:
        df["source"] = df["source"].str.strip().str.title().map(name_to_id)
        df["target"] = df["target"].str.strip().str.title().map(name_to_id)
    else:
        # Already numeric indices — validate they exist
        valid_ids = set(nodes_df["node_id"])
        df["source"] = df["source"].apply(lambda x: x if x in valid_ids else np.nan)
        df["target"] = df["target"].apply(lambda x: x if x in valid_ids else np.nan)

    # Drop rows with unknown nodes
    before = len(df)
    df.dropna(subset=["source", "target"], inplace=True)
    df["source"] = df["source"].astype(int)
    df["target"] = df["target"].astype(int)
    logger.info(f"Dropped {before - len(df)} edges with unknown nodes.")

    # Remove self-loops
    df = df[df["source"] != df["target"]]

    # Ensure undirected: sort source/target so (A,B) == (B,A)
    df[["source", "target"]] = np.sort(df[["source", "target"]].values, axis=1)

    # Aggregate duplicate edges
    df = (
        df.groupby(["source", "target"], as_index=False)["weight"]
        .sum()
    )

    # Normalize weights to [0, 1]
    max_w = df["weight"].max()
    df["weight_norm"] = (df["weight"] / max_w).round(4)

    # Add character name columns for readability
    df["source_name"] = df["source"].map(id_to_name)
    df["target_name"] = df["target"].map(id_to_name)

    logger.info(f"Clean edges: {len(df)} unique interactions.")
    return df[["source", "target", "source_name", "target_name", "weight", "weight_norm"]]


def generate_node_features(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich node metadata with degree and total interaction weight.

    Args:
        nodes_df: Cleaned nodes DataFrame.
        edges_df: Cleaned edges DataFrame.

    Returns:
        nodes_df enriched with [degree, total_weight] columns.
    """
    logger.info("Generating node features...")
    df = nodes_df.copy()

    # Degree = number of unique interaction partners
    all_connections = pd.concat([
        edges_df[["source"]].rename(columns={"source": "node_id"}),
        edges_df[["target"]].rename(columns={"target": "node_id"}),
    ])
    degree_map = all_connections["node_id"].value_counts().to_dict()
    df["degree"] = df["node_id"].map(degree_map).fillna(0).astype(int)

    # Total weight = sum of all interaction weights
    src_w = edges_df.groupby("source")["weight"].sum().rename("src_weight")
    tgt_w = edges_df.groupby("target")["weight"].sum().rename("tgt_weight")
    df = df.merge(src_w, left_on="node_id", right_index=True, how="left")
    df = df.merge(tgt_w, left_on="node_id", right_index=True, how="left")
    df["total_weight"] = (
        df["src_weight"].fillna(0) + df["tgt_weight"].fillna(0)
    ).astype(int)
    df.drop(columns=["src_weight", "tgt_weight"], inplace=True)

    logger.info("Node features generated.")
    return df


def save_processed(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """Save processed DataFrames to data/processed/."""
    nodes_path = PROCESSED_DIR / "nodes.csv"
    edges_path = PROCESSED_DIR / "edges.csv"

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    logger.info(f"✅ Saved: {nodes_path}")
    logger.info(f"✅ Saved: {edges_path}")
