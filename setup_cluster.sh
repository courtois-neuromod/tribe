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
#   - The Algonauts 2025 dataset (see README for download instructions)

set -euo pipefail

SCRATCH="${SCRATCH:?SCRATCH env var not set}"
INSTALL_DIR="${INSTALL_DIR:-$SCRATCH/tribe}"
VENV_DIR="${INSTALL_DIR}/envs/tribe"

echo "=== TRIBE Cluster Setup ==="
echo "Install dir: $INSTALL_DIR"
echo "Venv dir:    $VENV_DIR"
echo ""

# --- Step 1: Load required modules ---
# Arrow must be loaded BEFORE creating the venv (Alliance Canada requirement)
echo "[1/5] Loading modules..."
module load python/3.12 gcc arrow
echo "  Loaded: python/3.12, gcc, arrow"

# --- Step 2: Clone the repository ---
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[2/5] Repository already exists at $INSTALL_DIR, pulling latest..."
    cd "$INSTALL_DIR" && git pull
else
    echo "[2/5] Cloning repository..."
    git clone git@github.com:courtois-neuromod/tribe.git "$INSTALL_DIR"
fi

# --- Step 3: Create virtual environment ---
if [ -d "$VENV_DIR" ]; then
    echo "[3/5] Venv already exists at $VENV_DIR, skipping creation..."
else
    echo "[3/5] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- Step 4: Install dependencies ---
echo "[4/5] Installing dependencies..."

# PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Local packages (editable mode)
cd "$INSTALL_DIR"
pip install -e data_utils/ -e modeling_utils/

# Additional dependencies from paper
pip install transformers moviepy spacy nilearn Levenshtein "huggingface_hub[cli]" julius

echo "  All dependencies installed."

# --- Step 5: HuggingFace login ---
echo "[5/5] HuggingFace login (needed for LLAMA 3.2-3B access)..."
echo "  You need a 'read' token from https://huggingface.co/settings/tokens"
echo "  And access to https://huggingface.co/meta-llama/Llama-3.2-3B"
huggingface-cli login

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment in future sessions, add to your jobs:"
echo "  module load python/3.12 gcc arrow"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Set these env vars (or add to ~/.bashrc):"
echo "  export DATAPATH=\"/path/to/algonauts/dataset\""
echo "  export SAVEPATH=\"$SCRATCH\""
echo "  export SLURM_PARTITION=\"gpubase_bynode_b3\""
