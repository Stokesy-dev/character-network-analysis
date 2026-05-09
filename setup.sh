#!/bin/bash
# ============================================================
# setup.sh — Bootstrap script for Character Network Analysis
# Run: bash setup.sh
# ============================================================

set -e

PROJECT="character-network-analysis"
PYTHON="python3"

echo "🚀 Setting up $PROJECT..."

# Create folder structure
mkdir -p data/raw data/processed
mkdir -p notebooks
mkdir -p src
mkdir -p app
mkdir -p tests
mkdir -p outputs/figures outputs/reports outputs/models
mkdir -p .github/workflows

# Create placeholder Python files in src/
touch src/__init__.py
touch src/data_loader.py
touch src/preprocess.py
touch src/graph_builder.py
touch src/centrality.py
touch src/communities.py
touch src/visualization.py
touch src/graph_ml.py
touch src/utils.py

# Create placeholder test files
touch tests/__init__.py
touch tests/test_preprocess.py
touch tests/test_graph_builder.py
touch tests/test_metrics.py

# Create placeholder app
touch app/__init__.py
touch app/streamlit_app.py

# Create placeholder notebooks
touch notebooks/01_EDA.ipynb
touch notebooks/02_Graph_Analysis.ipynb
touch notebooks/03_Graph_ML.ipynb

# Create placeholder root files
touch PROJECT_REPORT.md
touch CONTRIBUTING.md

echo "✅ Folder structure created."

# Create virtual environment
echo "🐍 Creating virtual environment..."
$PYTHON -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Phase 1 complete! Run: source venv/bin/activate"
