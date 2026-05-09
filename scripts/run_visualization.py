"""
run_visualization.py
--------------------
Phase 4 runner: Build graph → Detect communities → Generate all visualizations.

Usage:
    python scripts/run_visualization.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_builder import load_processed, build_graph, graph_summary
from src.centrality import build_centrality_table, save_centrality_report
from src.communities import (
    detect_louvain, detect_girvan_newman,
    community_summary, save_communities, compute_modularity,
)
from src.visualization import (
    plot_static_network,
    plot_degree_distribution,
    plot_edge_weight_distribution,
    plot_top_characters_plotly,
    plot_centrality_heatmap,
    plot_centrality_radar,
    plot_pyvis_network,
    plot_community_network,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 55)
    logger.info("  Star Wars Character Network — Visualization")
    logger.info("=" * 55)

    # ── Load data ──────────────────────────────────────────
    nodes, edges = load_processed()
    G = build_graph(nodes, edges)

    # ── Centrality ─────────────────────────────────────────
    centrality_df = build_centrality_table(G)
    save_centrality_report(centrality_df)

    # ── Community detection ────────────────────────────────
    louvain_partition = detect_louvain(G)
    gn_partition      = detect_girvan_newman(G, n_communities=6)

    modularity = compute_modularity(G, louvain_partition)
    comm_summary = community_summary(G, louvain_partition)

    save_communities(G, louvain_partition, "communities_louvain.csv")
    save_communities(G, gn_partition,      "communities_girvan_newman.csv")

    logger.info("\n── Community Summary (Louvain) ───────────────────────")
    print(comm_summary.to_string(index=False))
    logger.info(f"\nModularity score: {modularity:.4f}")

    # ── Static plots ───────────────────────────────────────
    plot_static_network(G, centrality_df)
    plot_degree_distribution(G)
    plot_edge_weight_distribution(edges)
    plot_centrality_heatmap(centrality_df)
    plot_community_network(G, louvain_partition)

    # ── Interactive plots ──────────────────────────────────
    plot_top_characters_plotly(centrality_df)
    plot_centrality_radar(centrality_df)
    plot_pyvis_network(G, centrality_df, louvain_partition)

    # ── Summary ────────────────────────────────────────────
    logger.info("\n── Outputs Generated ─────────────────────────────────")
    from src.visualization import FIGURES_DIR
    for f in sorted(FIGURES_DIR.iterdir()):
        logger.info(f"  {f.name}")
    logger.info("\n✅ Phase 4 complete. Check outputs/figures/")


if __name__ == "__main__":
    main()
