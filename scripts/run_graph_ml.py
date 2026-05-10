"""
run_graph_ml.py
---------------
Phase 5 runner: Prepare data → Train GCN + GraphSAGE →
Evaluate → Plot results → Save models.

Usage:
    python scripts/run_graph_ml.py
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_builder   import load_processed, build_graph
from src.centrality      import build_centrality_table
from src.communities     import detect_louvain
from src.graph_ml import (
    prepare_graph_data,
    GCN, GraphSAGE,
    train_model,
    evaluate_model,
    plot_training_curves,
    plot_confusion_matrix,
    save_model,
    DEVICE,
)
from src.utils import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 55)
    logger.info("  Star Wars Character Network — Graph ML")
    logger.info("=" * 55)

    # ── Load graph ─────────────────────────────────────────
    nodes, edges   = load_processed()
    G              = build_graph(nodes, edges)
    centrality_df  = build_centrality_table(G)
    partition      = detect_louvain(G)

    # ── Prepare PyG data ───────────────────────────────────
    data, label_encoder, node_ids = prepare_graph_data(
        G, centrality_df, partition,
        test_size=0.25, min_community_size=3,
    )

    in_channels  = data.x.shape[1]
    n_classes    = int(data.y.max().item()) + 1
    hidden_dim   = 64

    logger.info(f"Input features : {in_channels}")
    logger.info(f"Classes        : {n_classes}")
    logger.info(f"Hidden dim     : {hidden_dim}")
    logger.info(f"Device         : {DEVICE}")

    # ── Train GCN ──────────────────────────────────────────
    gcn_model = GCN(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=n_classes,
        dropout=0.4,
    )
    gcn_model, gcn_train_loss, gcn_test_loss = train_model(
        gcn_model, data,
        model_name="GCN",
        epochs=300, lr=0.005,
        weight_decay=5e-4, patience=35,
    )

    # ── Train GraphSAGE ────────────────────────────────────
    sage_model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=n_classes,
        dropout=0.4,
    )
    sage_model, sage_train_loss, sage_test_loss = train_model(
        sage_model, data,
        model_name="GraphSAGE",
        epochs=300, lr=0.005,
        weight_decay=5e-4, patience=35,
    )

    # ── Evaluate ───────────────────────────────────────────
    gcn_results  = evaluate_model(gcn_model,  data, label_encoder, "GCN")
    sage_results = evaluate_model(sage_model, data, label_encoder, "GraphSAGE")

    # ── Plots ──────────────────────────────────────────────
    plot_training_curves(
        gcn_train_loss,  gcn_test_loss,
        sage_train_loss, sage_test_loss,
    )
    plot_confusion_matrix(gcn_results,  label_encoder, "GCN")
    plot_confusion_matrix(sage_results, label_encoder, "GraphSAGE")

    # ── Save models ────────────────────────────────────────
    save_model(gcn_model,  "gcn")
    save_model(sage_model, "graphsage")

    # ── Save results JSON ──────────────────────────────────
    summary = {
        "GCN": {
            "accuracy":    gcn_results["accuracy"],
            "f1_weighted": gcn_results["f1_weighted"],
            "f1_macro":    gcn_results["f1_macro"],
        },
        "GraphSAGE": {
            "accuracy":    sage_results["accuracy"],
            "f1_weighted": sage_results["f1_weighted"],
            "f1_macro":    sage_results["f1_macro"],
        },
    }
    save_json(summary, "graph_ml_results.json")

    # ── Final comparison ───────────────────────────────────
    print("\n" + "═" * 55)
    print("  Model Comparison Summary")
    print("═" * 55)
    print(f"  {'Model':<14} {'Accuracy':>10} {'F1 (w)':>10} {'F1 (m)':>10}")
    print("  " + "─" * 46)
    for name, res in summary.items():
        print(
            f"  {name:<14} "
            f"{res['accuracy']:>10.4f} "
            f"{res['f1_weighted']:>10.4f} "
            f"{res['f1_macro']:>10.4f}"
        )
    print("═" * 55)

    best = max(summary, key=lambda k: summary[k]["f1_weighted"])
    print(f"\n  🏆 Best model: {best} "
          f"(F1 weighted = {summary[best]['f1_weighted']:.4f})\n")

    logger.info("✅ Phase 5 complete. Check outputs/models/ and outputs/figures/")


if __name__ == "__main__":
    main()
