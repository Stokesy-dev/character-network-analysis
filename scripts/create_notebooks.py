"""
create_notebooks.py
-------------------
Generates all 3 Jupyter notebooks programmatically.
Run once after cloning.

Usage:
    python scripts/create_notebooks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True, parents=True)

def cell(source: str, kind: str = "code") -> dict:
    return {
        "cell_type": kind,
        "metadata": {},
        "source": [source],
        **({"outputs": [], "execution_count": None} if kind == "code" else {}),
    }


def md(source: str) -> dict:
    return cell(source, "markdown")


def code(source: str) -> dict:
    return cell(source, "code")


KERNEL = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11.0"},
}

# ── Notebook 1: EDA ───────────────────────────────────────────────────────────
nb1 = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": KERNEL,
    "cells": [
        md("# 01 — Exploratory Data Analysis\n**Star Wars Character Network Analysis**"),
        code(
            "import sys\nfrom pathlib import Path\n"
            "sys.path.insert(0, str(Path('..').resolve()))\n\n"
            "import pandas as pd\nimport numpy as np\nimport networkx as nx\n"
            "import matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "import plotly.express as px\nfrom collections import Counter\n\n"
            "plt.style.use('dark_background')\n%matplotlib inline\nprint('✅ Ready')"
        ),
        md("## 1. Load Data"),
        code(
            "from src.graph_builder import load_processed, build_graph, graph_summary\n\n"
            "nodes, edges = load_processed()\n"
            "G = build_graph(nodes, edges)\n\n"
            "print(f'Characters : {len(nodes)}')\n"
            "print(f'Interactions: {len(edges)}')\n"
            "nodes.head()"
        ),
        md("## 2. Graph Summary"),
        code(
            "summary = graph_summary(G)\n"
            "for k, v in summary.items():\n"
            "    print(f'{k:<40} {v}')"
        ),
        md("## 3. Degree Distribution"),
        code(
            "degrees = [d for _, d in G.degree()]\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "axes[0].hist(degrees, bins=20, color='#58a6ff', edgecolor='#0d1117')\n"
            "axes[0].axvline(np.mean(degrees), color='#ff7b72', linestyle='--',\n"
            "                label=f'Mean={np.mean(degrees):.1f}')\n"
            "axes[0].set_title('Degree Distribution')\naxes[0].legend()\n\n"
            "from collections import Counter\n"
            "dc = Counter(degrees)\n"
            "axes[1].scatter(sorted(dc), [dc[x] for x in sorted(dc)], color='#58a6ff')\n"
            "axes[1].set_xscale('log'); axes[1].set_yscale('log')\n"
            "axes[1].set_title('Log-Log (Power Law Check)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("## 4. Top Characters by Degree"),
        code(
            "top20 = nodes.nlargest(20, 'degree')[['name','degree','total_weight']]\n"
            "fig = px.bar(top20, x='degree', y='name', orientation='h',\n"
            "             color='degree', color_continuous_scale='Plasma',\n"
            "             title='Top 20 Characters by Degree', template='plotly_dark')\n"
            "fig.update_layout(yaxis=dict(autorange='reversed'), height=550)\n"
            "fig.show()"
        ),
        md("## 5. Edge Weight Analysis"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(14,5))\n"
            "axes[0].hist(edges['weight'], bins=30, color='#d2a8ff',\n"
            "             edgecolor='#0d1117')\n"
            "axes[0].set_title('Raw Edge Weights')\n"
            "axes[1].hist(edges['weight_norm'], bins=30, color='#3fb950',\n"
            "             edgecolor='#0d1117')\n"
            "axes[1].set_title('Normalized Edge Weights')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(edges['weight'].describe())"
        ),
        md("## 6. Clustering Coefficient Distribution"),
        code(
            "cc = nx.clustering(G, weight='weight')\n"
            "plt.figure(figsize=(10, 4))\n"
            "plt.hist(list(cc.values()), bins=20, color='#ffa657',\n"
            "         edgecolor='#0d1117')\n"
            "plt.title('Clustering Coefficient Distribution')\n"
            "plt.xlabel('Clustering Coefficient'); plt.ylabel('Count')\n"
            "plt.show()\n\n"
            "cc_df = pd.DataFrame([{'name': G.nodes[n]['name'], 'cc': v}\n"
            "                       for n, v in cc.items()])\n"
            "print(cc_df.nlargest(10,'cc').to_string(index=False))"
        ),
    ],
}

# ── Notebook 2: Graph Analysis ────────────────────────────────────────────────
nb2 = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": KERNEL,
    "cells": [
        md("# 02 — Graph Analysis\n**Centrality + Community Detection**"),
        code(
            "import sys\nfrom pathlib import Path\n"
            "sys.path.insert(0, str(Path('..').resolve()))\n\n"
            "import pandas as pd\nimport networkx as nx\n"
            "import matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "import plotly.express as px\n\n"
            "plt.style.use('dark_background')\n%matplotlib inline"
        ),
        md("## 1. Build Graph"),
        code(
            "from src.graph_builder import load_processed, build_graph\n"
            "from src.centrality import build_centrality_table, print_top_n\n\n"
            "nodes, edges = load_processed()\n"
            "G = build_graph(nodes, edges)\n"
            "print(f'G: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')"
        ),
        md("## 2. Centrality Analysis"),
        code(
            "cent_df = build_centrality_table(G)\n"
            "print_top_n(cent_df, n=15)"
        ),
        code(
            "# Heatmap of top 20\n"
            "metrics = ['degree_centrality','betweenness_centrality',\n"
            "           'closeness_centrality','eigenvector_centrality','pagerank']\n"
            "top20 = cent_df.reset_index(drop=True).head(20).set_index('name')[metrics]\n"
            "top20_norm = (top20 - top20.min()) / (top20.max() - top20.min() + 1e-9)\n\n"
            "fig, ax = plt.subplots(figsize=(10, 10))\n"
            "sns.heatmap(top20_norm, annot=True, fmt='.2f', cmap='YlOrRd',\n"
            "            linewidths=0.4, ax=ax)\n"
            "ax.set_title('Centrality Heatmap — Top 20 Characters')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("## 3. Community Detection — Louvain"),
        code(
            "from src.communities import (\n"
            "    detect_louvain, community_summary, compute_modularity\n"
            ")\n\n"
            "partition = detect_louvain(G)\n"
            "comm_sum  = community_summary(G, partition)\n"
            "mod       = compute_modularity(G, partition)\n\n"
            "print(f'Modularity: {mod:.4f}')\n"
            "print(comm_sum.to_string(index=False))"
        ),
        md("## 4. Community Visualization"),
        code(
            "from src.visualization import plot_community_network\n"
            "plot_community_network(G, partition)\n"
            "from IPython.display import Image\n"
            "Image('outputs/figures/network_communities.png')"
        ),
        md("## 5. Interactive Network"),
        code(
            "from src.visualization import plot_pyvis_network\n"
            "plot_pyvis_network(G, cent_df, partition)\n\n"
            "from IPython.display import IFrame\n"
            "IFrame('outputs/figures/network_interactive.html', width=900, height=600)"
        ),
        md("## 6. PageRank vs Degree Scatter"),
        code(
            "import plotly.express as px\n"
            "df = cent_df.reset_index(drop=True).head(30)\n"
            "fig = px.scatter(\n"
            "    df, x='degree_centrality', y='pagerank',\n"
            "    text='name', size='centrality_score',\n"
            "    color='betweenness_centrality',\n"
            "    color_continuous_scale='Plasma',\n"
            "    title='PageRank vs Degree Centrality',\n"
            "    template='plotly_dark'\n"
            ")\n"
            "fig.update_traces(textposition='top center')\n"
            "fig.show()"
        ),
    ],
}

# ── Notebook 3: Graph ML ──────────────────────────────────────────────────────
nb3 = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": KERNEL,
    "cells": [
        md("# 03 — Graph Machine Learning\n**GCN + GraphSAGE Node Classification**"),
        code(
            "import sys\nfrom pathlib import Path\n"
            "sys.path.insert(0, str(Path('..').resolve()))\n\n"
            "import torch\nimport pandas as pd\nimport numpy as np\n"
            "import matplotlib.pyplot as plt\nimport seaborn as sns\n\n"
            "plt.style.use('dark_background')\n%matplotlib inline\n\n"
            "print(f'PyTorch: {torch.__version__}')\n"
            "print(f'Device : {\"cuda\" if torch.cuda.is_available() else \"cpu\"}')"
        ),
        md("## 1. Load Data + Build Graph"),
        code(
            "from src.graph_builder import load_processed, build_graph\n"
            "from src.centrality import build_centrality_table\n"
            "from src.communities import detect_louvain\n\n"
            "nodes, edges = load_processed()\n"
            "G            = build_graph(nodes, edges)\n"
            "cent_df      = build_centrality_table(G)\n"
            "partition    = detect_louvain(G)\n\n"
            "print(f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')\n"
            "print(f'Communities: {len(set(partition.values()))}')"
        ),
        md("## 2. Prepare PyG Data"),
        code(
            "from src.graph_ml import prepare_graph_data\n\n"
            "data, label_encoder, node_ids = prepare_graph_data(\n"
            "    G, cent_df, partition,\n"
            "    test_size=0.25, min_community_size=3,\n"
            ")\n\n"
            "print(f'Feature matrix : {data.x.shape}')\n"
            "print(f'Labels         : {data.y.shape}')\n"
            "print(f'Train nodes    : {data.train_mask.sum().item()}')\n"
            "print(f'Test nodes     : {data.test_mask.sum().item()}')\n"
            "print(f'Classes        : {len(label_encoder.classes_)}')"
        ),
        md("## 3. Train GCN"),
        code(
            "from src.graph_ml import GCN, train_model, evaluate_model\n\n"
            "in_ch  = data.x.shape[1]\n"
            "n_cls  = int(data.y.max().item()) + 1\n\n"
            "gcn = GCN(in_channels=in_ch, hidden_channels=64,\n"
            "          out_channels=n_cls, dropout=0.4)\n\n"
            "gcn, gcn_train_loss, gcn_test_loss = train_model(\n"
            "    gcn, data, model_name='GCN',\n"
            "    epochs=300, lr=0.005, patience=35\n"
            ")"
        ),
        md("## 4. Train GraphSAGE"),
        code(
            "from src.graph_ml import GraphSAGE\n\n"
            "sage = GraphSAGE(in_channels=in_ch, hidden_channels=64,\n"
            "                 out_channels=n_cls, dropout=0.4)\n\n"
            "sage, sage_train_loss, sage_test_loss = train_model(\n"
            "    sage, data, model_name='GraphSAGE',\n"
            "    epochs=300, lr=0.005, patience=35\n"
            ")"
        ),
        md("## 5. Evaluate Both Models"),
        code(
            "gcn_res  = evaluate_model(gcn,  data, label_encoder, 'GCN')\n"
            "sage_res = evaluate_model(sage, data, label_encoder, 'GraphSAGE')\n\n"
            "print(f\"\\nGCN  Accuracy: {gcn_res['accuracy']:.4f}  \"\n"
            "      f\"F1: {gcn_res['f1_weighted']:.4f}\")\n"
            "print(f\"SAGE Accuracy: {sage_res['accuracy']:.4f}  \"\n"
            "      f\"F1: {sage_res['f1_weighted']:.4f}\")"
        ),
        md("## 6. Training Curves"),
        code(
            "from src.graph_ml import plot_training_curves\n"
            "plot_training_curves(\n"
            "    gcn_train_loss, gcn_test_loss,\n"
            "    sage_train_loss, sage_test_loss\n"
            ")\n"
            "from IPython.display import Image\n"
            "Image('outputs/figures/training_curves.png')"
        ),
        md("## 7. Confusion Matrices"),
        code(
            "from src.graph_ml import plot_confusion_matrix\n"
            "plot_confusion_matrix(gcn_res,  label_encoder, 'GCN')\n"
            "plot_confusion_matrix(sage_res, label_encoder, 'GraphSAGE')\n\n"
            "from IPython.display import Image, display\n"
            "display(Image('outputs/figures/confusion_matrix_gcn.png'))\n"
            "display(Image('outputs/figures/confusion_matrix_graphsage.png'))"
        ),
        md("## 8. Save Models"),
        code(
            "from src.graph_ml import save_model\n"
            "save_model(gcn,  'gcn')\n"
            "save_model(sage, 'graphsage')\n"
            "print('Models saved to outputs/models/ ✅')"
        ),
    ],
}

# ── Write files ───────────────────────────────────────────────────────────────
for filename, nb in [
    ("01_EDA.ipynb",           nb1),
    ("02_Graph_Analysis.ipynb",nb2),
    ("03_Graph_ML.ipynb",      nb3),
]:
    path = NB_DIR / filename
    path.write_text(json.dumps(nb, indent=1))
    print(f"✅ Created: {path}")

print("\nAll notebooks created. Run: jupyter notebook notebooks/")
