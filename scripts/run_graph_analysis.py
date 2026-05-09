"""
run_graph_analysis.py
---------------------
Phase 3 runner: Build graph → Compute centrality → Save reports.

Usage:
    python scripts/run_graph_analysis.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_builder import load_processed, build_graph, graph_summary
from src.centrality import build_centrality_table, print_top_n, save_centrality_report
from src.utils import save_json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 55)
    logger.info("  Star Wars Character Network — Graph Analysis")
    logger.info("=" * 55)

    # Step 1: Load processed data + build graph
    nodes, edges = load_processed()
    G = build_graph(nodes, edges)

    # Step 2: Graph-level summary
    summary = graph_summary(G)
    save_json(summary, "graph_summary.json")

    # Step 3: Centrality analysis
    centrality_df = build_centrality_table(G)
    print_top_n(centrality_df, n=15)
    save_centrality_report(centrality_df)

    # Step 4: Per-metric top-10 tables
    metrics = [
        "degree_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
        "pagerank",
    ]

    print("\n── Per-Metric Top 5 Rankings ─────────────────────────────────────\n")
    for metric in metrics:
        top = centrality_df.reset_index().nlargest(5, metric)[["name", metric]]
        print(f"  {metric.replace('_', ' ').title()}")
        print(top.to_string(index=False))
        print()

    logger.info("✅ Phase 3 complete. Check outputs/reports/")


if __name__ == "__main__":
    main()
