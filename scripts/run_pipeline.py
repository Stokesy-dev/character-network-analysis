"""
run_pipeline.py
---------------
End-to-end runner for Phase 2: Download → Preprocess → Save.

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import download_dataset, load_all_episodes, save_raw
from src.preprocess import clean_nodes, clean_edges, generate_node_features, save_processed
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 55)
    logger.info("  Star Wars Character Network — Data Pipeline")
    logger.info("=" * 55)

    # Step 1: Download
    dataset_path = download_dataset()

    # Step 2: Load all episodes
    raw_nodes, raw_edges = load_all_episodes(dataset_path)
    save_raw(raw_nodes, "raw_nodes.csv")
    save_raw(raw_edges, "raw_edges.csv")

    # Step 3: Clean
    nodes = clean_nodes(raw_nodes)
    edges = clean_edges(raw_edges, nodes)

    # Step 4: Feature engineering
    nodes = generate_node_features(nodes, edges)

    # Step 5: Save processed
    save_processed(nodes, edges)

    # Step 6: Summary
    logger.info("")
    logger.info("── Pipeline Summary ──────────────────────────────")
    logger.info(f"  Characters : {len(nodes)}")
    logger.info(f"  Interactions: {len(edges)}")
    logger.info(f"  Top 5 characters by degree:")
    top5 = nodes.nlargest(5, "degree")[["name", "degree", "total_weight"]]
    for _, row in top5.iterrows():
        logger.info(f"    {row['name']:<20} degree={row['degree']}  weight={row['total_weight']}")
    logger.info("──────────────────────────────────────────────────")
    logger.info("✅ Phase 2 complete. Check data/processed/")


if __name__ == "__main__":
    main()
