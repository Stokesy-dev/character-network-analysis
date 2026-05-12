"""
graph_ml.py
-----------
Graph Machine Learning for Character Network Analysis.

Task:
    Node Classification — predict which COMMUNITY a character belongs to,
    using only the graph structure (node features + adjacency).

Model:
    GCN  (Graph Convolutional Network)  — Kipf & Welling 2017
    GraphSAGE                           — Hamilton et al. 2017

Pipeline:
    1. Convert NetworkX graph → PyTorch Geometric Data object
    2. Build node features (centrality metrics + degree)
    3. Train/test split (stratified by community label)
    4. Train GCN + GraphSAGE
    5. Evaluate: Accuracy, F1 (weighted), Confusion Matrix
    6. Save best model to outputs/models/
"""

import logging
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def build_node_features(
    G: nx.Graph,
    centrality_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build a feature matrix X of shape [num_nodes, num_features].

    Features per node (7 total):
        0  degree_centrality
        1  betweenness_centrality
        2  closeness_centrality
        3  eigenvector_centrality
        4  pagerank
        5  degree (raw, normalized)
        6  total_weight (log-normalized)

    Args:
        G: NetworkX Graph.
        centrality_df: Output of build_centrality_table(), indexed by rank.

    Returns:
        Feature matrix as numpy array, shape [N, 7].
    """
    cent = centrality_df.reset_index(drop=True).set_index("node_id")

    features = []
    for node_id in sorted(G.nodes()):
        row = cent.loc[node_id] if node_id in cent.index else {}
        feat = [
            float(row.get("degree_centrality",      0)),
            float(row.get("betweenness_centrality",  0)),
            float(row.get("closeness_centrality",    0)),
            float(row.get("eigenvector_centrality",  0)),
            float(row.get("pagerank",                0)),
            float(G.degree(node_id)) / max(1, G.number_of_nodes()),
            float(np.log1p(G.nodes[node_id].get("total_weight", 0))),
        ]
        features.append(feat)

    X = np.array(features, dtype=np.float32)

    # Standardise features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def networkx_to_pyg(
    G: nx.Graph,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> Data:
    """
    Convert NetworkX graph + numpy arrays → PyTorch Geometric Data object.

    Args:
        G: NetworkX Graph (nodes must be 0-indexed integers).
        features: Node feature matrix [N, F].
        labels: Node class labels [N].
        train_mask: Boolean mask for training nodes [N].
        test_mask: Boolean mask for test nodes [N].

    Returns:
        PyTorch Geometric Data object on DEVICE.
    """
    # Edge index [2, num_edges] — both directions for undirected graph
    edges = list(G.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Edge weights
    edge_weights = []
    for u, v in edges:
        w = G[u][v].get("weight_norm", 1.0)
        edge_weights.extend([w, w])
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    data = Data(
        x           = torch.tensor(features, dtype=torch.float),
        edge_index  = edge_index,
        edge_attr   = edge_attr,
        y           = torch.tensor(labels,     dtype=torch.long),
        train_mask  = torch.tensor(train_mask, dtype=torch.bool),
        test_mask   = torch.tensor(test_mask,  dtype=torch.bool),
    )
    return data.to(DEVICE)


def prepare_graph_data(
    G: nx.Graph,
    centrality_df: pd.DataFrame,
    partition: dict,
    test_size: float = 0.25,
    min_community_size: int = 3,
    random_state: int = 42,
) -> tuple[Data, LabelEncoder, np.ndarray]:
    """
    Full data prep pipeline: features → labels → masks → PyG Data.

    Filters out communities smaller than min_community_size to ensure
    valid train/test split with stratification.

    Returns:
        (data, label_encoder, node_ids_array)
    """
    logger.info("Preparing PyG data object...")

    # Sorted node list (must match feature matrix row order)
    node_ids = sorted(G.nodes())

    # Raw community labels
    raw_labels = np.array([partition.get(n, 0) for n in node_ids])

    # Filter: keep only communities with enough nodes
    from collections import Counter
    comm_counts = Counter(raw_labels)
    valid_comms = {c for c, cnt in comm_counts.items() if cnt >= min_community_size}
    valid_mask  = np.array([raw_labels[i] in valid_comms for i in range(len(node_ids))])

    filtered_ids    = np.array(node_ids)[valid_mask]
    filtered_labels = raw_labels[valid_mask]

    logger.info(
        f"After filtering: {len(filtered_ids)} nodes, "
        f"{len(valid_comms)} communities (removed "
        f"{len(node_ids) - len(filtered_ids)} isolated nodes)"
    )

    # Encode labels to 0-indexed integers
    le = LabelEncoder()
    encoded_labels = le.fit_transform(filtered_labels)
    n_classes = len(le.classes_)
    logger.info(f"Number of classes: {n_classes}")

    # Build full feature matrix for ALL nodes in G
    X_full = build_node_features(G, centrality_df)

    # Subset to filtered nodes
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    filtered_indices = [id_to_idx[nid] for nid in filtered_ids]
    X_filtered = X_full[filtered_indices]

    # Train/test split (stratified)
    train_idx, test_idx = train_test_split(
        np.arange(len(filtered_ids)),
        test_size=test_size,
        stratify=encoded_labels,
        random_state=random_state,
    )

    # Build full-graph arrays (unfiltered nodes get label=0, excluded from masks)
    full_labels     = np.zeros(len(node_ids), dtype=int)
    full_train_mask = np.zeros(len(node_ids), dtype=bool)
    full_test_mask  = np.zeros(len(node_ids), dtype=bool)

    for local_i, global_i in enumerate(filtered_indices):
        full_labels[global_i] = encoded_labels[local_i]

    for local_i in train_idx:
        full_train_mask[filtered_indices[local_i]] = True

    for local_i in test_idx:
        full_test_mask[filtered_indices[local_i]] = True

    logger.info(
        f"Train: {full_train_mask.sum()} nodes | "
        f"Test:  {full_test_mask.sum()} nodes"
    )

    data = networkx_to_pyg(G, X_full, full_labels, full_train_mask, full_test_mask)
    return data, le, np.array(node_ids)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class GCN(nn.Module):
    """
    2-layer Graph Convolutional Network (Kipf & Welling, 2017).

    Architecture:
        Input → GCNConv(hidden) → ReLU → Dropout
              → GCNConv(hidden) → ReLU → Dropout
              → Linear(n_classes) → LogSoftmax
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden_channels)
        self.conv2   = GCNConv(hidden_channels, hidden_channels)
        self.linear  = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    """
    2-layer Graph Attention Network (Veličković et al., 2018).

    Architecture:
        Input → GATConv(hidden, heads=4) → ELU → Dropout
              → GATConv(n_classes, heads=1, concat=False) → LogSoftmax
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.4,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.conv1   = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2   = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGE(nn.Module):
    """
    2-layer GraphSAGE (Hamilton et al., 2017).

    Architecture:
        Input → SAGEConv(hidden) → ReLU → Dropout
              → SAGEConv(hidden) → ReLU → Dropout
              → Linear(n_classes) → LogSoftmax
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.conv1   = SAGEConv(in_channels, hidden_channels)
        self.conv2   = SAGEConv(hidden_channels, hidden_channels)
        self.linear  = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Single training epoch. Returns training loss."""
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a given mask.

    Returns:
        (loss, true_labels, predicted_labels)
    """
    model.eval()
    out    = model(data.x, data.edge_index)
    loss   = F.nll_loss(out[mask], data.y[mask]).item()
    preds  = out[mask].argmax(dim=1).cpu().numpy()
    truths = data.y[mask].cpu().numpy()
    return loss, truths, preds


def train_model(
    model: nn.Module,
    data: Data,
    model_name: str,
    epochs: int = 300,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
    patience: int = 30,
) -> tuple[nn.Module, list, list]:
    """
    Full training loop with early stopping.

    Args:
        model: GCN or GraphSAGE instance.
        data: PyG Data object.
        model_name: Name for logging + checkpoint saving.
        epochs: Maximum training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization.
        patience: Early stopping patience.

    Returns:
        (trained_model, train_losses, test_losses)
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.NLLLoss()

    train_losses, test_losses = [], []
    best_test_loss   = float("inf")
    best_state_dict  = None
    patience_counter = 0

    logger.info(f"\nTraining {model_name} for up to {epochs} epochs...")
    logger.info(f"{'Epoch':>6} {'Train Loss':>12} {'Test Loss':>11} {'Test Acc':>10}")
    logger.info("─" * 46)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, data, optimizer, criterion)
        test_loss, y_true, y_pred = evaluate(model, data, data.test_mask)
        test_acc = accuracy_score(y_true, y_pred)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss  = test_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0 or epoch == 1:
            logger.info(
                f"{epoch:>6}   {train_loss:>10.4f}   "
                f"{test_loss:>9.4f}   {test_acc:>8.4f}"
            )

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    # Restore best weights
    if best_state_dict:
        model.load_state_dict(best_state_dict)

    logger.info(f"Best test loss: {best_test_loss:.4f}")
    return model, train_losses, test_losses


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: nn.Module,
    data: Data,
    label_encoder: LabelEncoder,
    model_name: str,
) -> dict:
    """
    Full evaluation suite: accuracy, F1, confusion matrix, report.

    Returns:
        Dictionary of all metrics.
    """
    _, y_true, y_pred = evaluate(model, data, data.test_mask)

    acc      = accuracy_score(y_true, y_pred)
    f1_w     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    cm       = confusion_matrix(y_true, y_pred)

    class_names = [f"Community {le_class + 1}"
                   for le_class in label_encoder.classes_]

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    print(f"\n{'═' * 55}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'═' * 55}")
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 Score (weighted): {f1_w:.4f}")
    print(f"  F1 Score (macro)   : {f1_macro:.4f}")
    print(f"\n  Per-Class Report:\n")
    print(report)
    print(f"  Confusion Matrix:\n{cm}\n")

    return {
        "model":       model_name,
        "accuracy":    round(acc,      4),
        "f1_weighted": round(f1_w,     4),
        "f1_macro":    round(f1_macro, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — TRAINING CURVES + CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    gcn_train: list,
    gcn_test: list,
    sage_train: list,
    sage_test: list,
    gat_train: list = None,
    gat_test: list = None,
    filename: str = "training_curves.png",
) -> Path:
    """Plot training + test loss curves for the models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DARK_BG  = "#0d1117"
    CARD_BG  = "#161b22"
    TEXT_CLR = "#e6edf3"

    num_plots = 3 if gat_train is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), facecolor=DARK_BG)

    train_lists = [gcn_train, sage_train]
    test_lists = [gcn_test, sage_test]
    names = ["GCN", "GraphSAGE"]
    colors = ["#58a6ff", "#3fb950"]
    
    if gat_train is not None:
        train_lists.append(gat_train)
        test_lists.append(gat_test)
        names.append("GAT")
        colors.append("#d2a8ff")

    for ax, train_l, test_l, name, color in zip(axes, train_lists, test_lists, names, colors):
        ax.set_facecolor(CARD_BG)
        epochs = range(1, len(train_l) + 1)
        ax.plot(epochs, train_l, color=color,    linewidth=2,   label="Train loss")
        ax.plot(epochs, test_l,  color="#ff7b72", linewidth=2,
                linestyle="--", label="Test loss")
        ax.set_title(f"{name} Training Curves", color=TEXT_CLR, fontsize=13)
        ax.set_xlabel("Epoch",      color=TEXT_CLR)
        ax.set_ylabel("NLL Loss",   color=TEXT_CLR)
        ax.tick_params(colors=TEXT_CLR)
        ax.legend(fontsize=9, facecolor=CARD_BG, labelcolor=TEXT_CLR)
        ax.grid(True, alpha=0.3, color="#21262d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig.suptitle("Graph ML — Training Curves", fontsize=15, color=TEXT_CLR, y=1.02)
    plt.tight_layout()

    from src.visualization import FIGURES_DIR
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"✅ Saved: {path}")
    return path


def plot_confusion_matrix(
    results: dict,
    label_encoder: LabelEncoder,
    model_name: str,
    filename: Optional[str] = None,
) -> Path:
    """Plot a styled confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    DARK_BG  = "#0d1117"
    TEXT_CLR = "#e6edf3"

    cm = np.array(results["confusion_matrix"])
    class_names = [f"C{i+1}" for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        linecolor="#0d1117",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(
        f"{model_name} — Confusion Matrix\n"
        f"Accuracy: {results['accuracy']:.4f}  |  "
        f"F1 (weighted): {results['f1_weighted']:.4f}",
        color=TEXT_CLR, fontsize=12, pad=14,
    )
    ax.set_xlabel("Predicted Community", color=TEXT_CLR)
    ax.set_ylabel("True Community",      color=TEXT_CLR)
    ax.tick_params(colors=TEXT_CLR)
    ax.set_facecolor(DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    plt.tight_layout()

    from src.visualization import FIGURES_DIR
    fname = filename or f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    path  = FIGURES_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"✅ Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model: nn.Module, model_name: str) -> Path:
    """Save model state dict to outputs/models/."""
    filename = model_name.lower().replace(" ", "_") + ".pt"
    path = MODELS_DIR / filename
    torch.save(model.state_dict(), path)
    logger.info(f"✅ Model saved: {path}")
    return path


def load_model(
    model_class: type,
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
) -> nn.Module:
    """Load a saved model from outputs/models/."""
    filename = model_name.lower().replace(" ", "_") + ".pt"
    path = MODELS_DIR / filename
    model = model_class(in_channels, hidden_channels, out_channels)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    logger.info(f"Model loaded: {path}")
    return model
