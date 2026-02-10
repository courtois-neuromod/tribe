#!/bin/bash
# TRIBE - Cluster Environment Setup
# Alliance Canada (rorqual / narval / beluga / cedar / graham)
#
# This script sets up everything needed to run TRIBE on an Alliance Canada
# SLURM cluster. Run it once from a login node.
#
# Usage:
#   bash setup_cluster.sh
#
# Prerequisites:
#   - Access to an Alliance Canada cluster
#   - HuggingFace account with access to meta-llama/Llama-3.2-3B

set -euo pipefail

SCRATCH="${SCRATCH:?SCRATCH env var not set}"
INSTALL_DIR="${INSTALL_DIR:-$SCRATCH/tribe}"
VENV_DIR="${INSTALL_DIR}/envs/tribe"
DATA_DIR="${DATAPATH:-$SCRATCH/algonauts_2025.competitors}"

echo "=== TRIBE Cluster Setup ==="
echo "Install dir: $INSTALL_DIR"
echo "Venv dir:    $VENV_DIR"
echo "Data dir:    $DATA_DIR"
echo ""

# --- Step 1: Load required modules ---
# Arrow must be loaded BEFORE creating the venv (Alliance Canada requirement)
echo "[1/7] Loading modules..."
module load python/3.12 gcc arrow git-annex
echo "  Loaded: python/3.12, gcc, arrow, git-annex"

# --- Step 2: Clone the repository ---
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[2/7] Repository already exists at $INSTALL_DIR, pulling latest..."
    cd "$INSTALL_DIR" && git pull
else
    echo "[2/7] Cloning repository..."
    git clone git@github.com:courtois-neuromod/tribe.git "$INSTALL_DIR"
fi

# --- Step 3: Create virtual environment ---
if [ -d "$VENV_DIR" ]; then
    echo "[3/7] Venv already exists at $VENV_DIR, skipping creation..."
else
    echo "[3/7] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- Step 4: Install dependencies ---
echo "[4/7] Installing dependencies..."

# PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Local packages (editable mode)
cd "$INSTALL_DIR"
pip install -e data_utils/ -e modeling_utils/

# Additional dependencies
pip install transformers moviepy spacy nilearn Levenshtein "huggingface_hub[cli]" julius

# Datalad (for dataset download)
pip install datalad

echo "  All dependencies installed."

# --- Step 5: Download the Algonauts 2025 dataset ---
if [ -d "$DATA_DIR/.git" ]; then
    echo "[5/7] Dataset already cloned at $DATA_DIR, skipping..."
else
    echo "[5/7] Downloading Algonauts 2025 dataset (this will take a while)..."
    cd "$(dirname "$DATA_DIR")"
    datalad install -r -s https://github.com/courtois-neuromod/algonauts_2025.competitors.git
    cd "$DATA_DIR"
    datalad get -r -J8 .
    echo "  Dataset download complete."
fi

# --- Step 6: HuggingFace login ---
echo "[6/7] HuggingFace login (needed for LLAMA 3.2-3B access)..."
echo "  You need a 'read' token from https://huggingface.co/settings/tokens"
echo "  And access to https://huggingface.co/meta-llama/Llama-3.2-3B"
huggingface-cli login

# --- Step 7: Set environment variables ---
echo "[7/7] Setting environment variables..."

# Only append if not already present
if ! grep -q "TRIBE environment" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << BASHEOF

# TRIBE environment
export DATAPATH="$DATA_DIR"
export SAVEPATH="$SCRATCH"
BASHEOF
    echo "  Added DATAPATH and SAVEPATH to ~/.bashrc"
else
    echo "  Environment variables already in ~/.bashrc, skipping..."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment in future sessions / SLURM jobs:"
echo "  module load python/3.12 gcc arrow"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Submit feature extraction:  sbatch run_feature_extraction.sh"
echo "  2. Run a test training:        python -m algonauts2025.grids.test_run"
echo "  3. Run full grid search:       python -m algonauts2025.grids.run_grid"
