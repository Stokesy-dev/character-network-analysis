"""
data_loader.py
--------------
Downloads and loads the Star Wars character interaction dataset from Kaggle.
Dataset: alexataheri/star-wars-interactions (nodes + edges across all episodes)
"""

import os
import json
import logging
import pandas as pd
import kagglehub
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> Path:
    """
    Returns path to the manually downloaded dataset directory.
    """
    dataset_path = RAW_DIR / "star-wars-social-network"
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        logger.info("Please download the dataset from https://www.kaggle.com/datasets/ruchi798/star-wars-social-network")
        logger.info("Extract the ZIP file so that the 'episode-1', 'episode-2' etc. folders are inside: data/raw/star-wars-social-network/")
        raise FileNotFoundError(f"Missing manual download at {dataset_path}")
    
    logger.info(f"Using local dataset at: {dataset_path}")
    return dataset_path


def load_nodes(dataset_path: Path, episode: str = "episode-1") -> pd.DataFrame:
    """
    Load character node metadata from a JSON file.

    Args:
        dataset_path: Root path of downloaded dataset.
        episode: Episode subfolder name (e.g. 'episode-1').

    Returns:
        DataFrame with columns: [id, name, value, colour]
    """
    json_path = dataset_path / episode / "characters.json"

    if not json_path.exists():
        # fallback: search all JSON files
        candidates = list(dataset_path.rglob("characters.json"))
        if not candidates:
            raise FileNotFoundError(f"No characters.json found under {dataset_path}")
        json_path = candidates[0]
        logger.warning(f"Falling back to: {json_path}")

    logger.info(f"Loading nodes from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    nodes = pd.DataFrame(data["nodes"])
    logger.info(f"Loaded {len(nodes)} characters.")
    return nodes


def load_edges(dataset_path: Path, episode: str = "episode-1") -> pd.DataFrame:
    """
    Load character interaction edges from a JSON file.

    Args:
        dataset_path: Root path of downloaded dataset.
        episode: Episode subfolder name.

    Returns:
        DataFrame with columns: [source, target, value]
    """
    json_path = dataset_path / episode / "interactions.json"

    if not json_path.exists():
        candidates = list(dataset_path.rglob("interactions.json"))
        if not candidates:
            raise FileNotFoundError(f"No interactions.json found under {dataset_path}")
        json_path = candidates[0]
        logger.warning(f"Falling back to: {json_path}")

    logger.info(f"Loading edges from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    edges = pd.DataFrame(data["links"])
    logger.info(f"Loaded {len(edges)} interactions.")
    return edges


def load_all_episodes(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and combine nodes and edges across ALL episodes.

    Returns:
        Tuple of (all_nodes_df, all_edges_df) with an 'episode' column added.
    """
    episodes = [
        "episode-1", "episode-2", "episode-3",
        "episode-4", "episode-5", "episode-6",
        "episode-7",
    ]

    all_nodes, all_edges = [], []

    for ep in episodes:
        try:
            nodes = load_nodes(dataset_path, ep)
            edges = load_edges(dataset_path, ep)
            nodes["episode"] = ep
            edges["episode"] = ep
            all_nodes.append(nodes)
            all_edges.append(edges)
            logger.info(f"✅ {ep}: {len(nodes)} chars, {len(edges)} interactions")
        except FileNotFoundError:
            logger.warning(f"⚠️  Skipping {ep} — files not found.")

    combined_nodes = pd.concat(all_nodes, ignore_index=True)
    combined_edges = pd.concat(all_edges, ignore_index=True)
    return combined_nodes, combined_edges


def save_raw(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to data/raw/ as CSV."""
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved raw file: {path}")
