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
DATA_DIR="${DATAPATH:-$SCRATCH/algonauts_2025.competitors}"

echo "=== TRIBE Cluster Setup ==="
echo "Install dir: $INSTALL_DIR"
echo "Data dir:    $DATA_DIR"
echo ""

# --- Step 1: Load required modules ---
# Arrow must be loaded BEFORE creating the venv (Alliance Canada requirement)
echo "[1/8] Loading modules..."
module load python/3.12 gcc arrow git-annex
echo "  Loaded: python/3.12, gcc, arrow, git-annex"

# --- Step 2: Clone the repository ---
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[2/8] Repository already exists at $INSTALL_DIR, pulling latest..."
    cd "$INSTALL_DIR" && git pull
else
    echo "[2/8] Cloning repository..."
    git clone git@github.com:courtois-neuromod/tribe.git "$INSTALL_DIR"
fi

# --- Step 3: Create a temporary venv for setup tasks ---
# This venv is only used on the login node for downloading models and data.
# Compute nodes build their own venv on local SSD at job start.
echo "[3/8] Creating temporary setup venv..."
SETUP_VENV="$SCRATCH/envs/tribe-setup"
if [ ! -d "$SETUP_VENV" ]; then
    python3 -m venv "$SETUP_VENV"
fi
source "$SETUP_VENV/bin/activate"

pip install --quiet transformers spacy "huggingface_hub[cli]" datalad

# Pre-download wheels for packages not in Alliance wheelhouse.
# Compute nodes have no internet, so these must be cached on $SCRATCH.
echo "  Pre-downloading wheels for packages not in Alliance wheelhouse..."
mkdir -p "$SCRATCH/wheels"
pip download --dest "$SCRATCH/wheels" exca x_transformers
echo "  Wheels saved to $SCRATCH/wheels"

# --- Step 4: Download the Algonauts 2025 dataset ---
if [ -d "$DATA_DIR/.git" ]; then
    echo "[4/8] Dataset already cloned at $DATA_DIR, skipping..."
else
    echo "[4/8] Downloading Algonauts 2025 dataset (this will take a while)..."
    cd "$(dirname "$DATA_DIR")"
    datalad install -r -s https://github.com/courtois-neuromod/algonauts_2025.competitors.git
    cd "$DATA_DIR"
    datalad get -r -J8 .
    echo "  Dataset download complete."
fi

# --- Step 5: Create expected directory structure ---
# The code expects data at DATAPATH/algonauts2025/download/algonauts_2025.competitors/
echo "[5/8] Setting up dataset directory structure..."
mkdir -p "$SCRATCH/data/algonauts2025/download"
ln -sf "$DATA_DIR" "$SCRATCH/data/algonauts2025/download/algonauts_2025.competitors"
echo "  Symlinked dataset to expected location"

# --- Step 6: HuggingFace login ---
echo "[6/8] HuggingFace login (needed for LLAMA 3.2-3B access)..."
echo "  You need a 'read' token from https://huggingface.co/settings/tokens"
echo "  And access to https://huggingface.co/meta-llama/Llama-3.2-3B"
huggingface-cli login

# --- Step 7: Pre-download models ---
# Compute nodes have NO internet access on Alliance Canada clusters.
# All models must be cached on the login node first.
echo "[7/8] Pre-downloading models (compute nodes have no internet)..."

echo "  Downloading spacy English model (en_core_web_lg)..."
python -m spacy download en_core_web_lg

echo "  Downloading LLAMA 3.2-3B..."
python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
AutoModel.from_pretrained('meta-llama/Llama-3.2-3B')
print('  LLAMA 3.2-3B: OK')
"

echo "  Downloading Wav2Vec2-BERT..."
python -c "
from transformers import AutoModel, AutoFeatureExtractor
AutoModel.from_pretrained('facebook/w2v-bert-2.0')
AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')
print('  Wav2Vec2-BERT: OK')
"

echo "  Downloading VJEPA2 (this is large, ~3.6GB)..."
python -c "
from transformers import AutoModel, AutoVideoProcessor
AutoVideoProcessor.from_pretrained('facebook/vjepa2-vitg-fpc64-256', do_rescale=True)
AutoModel.from_pretrained('facebook/vjepa2-vitg-fpc64-256', output_hidden_states=True)
print('  VJEPA2: OK')
"

echo "  All models downloaded."

# --- Step 8: Set environment variables ---
echo "[8/8] Setting environment variables..."

# Only append if not already present
if ! grep -q "TRIBE environment" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << BASHEOF

# TRIBE environment
export DATAPATH="$SCRATCH/data"
export SAVEPATH="$SCRATCH"
BASHEOF
    echo "  Added DATAPATH and SAVEPATH to ~/.bashrc"
else
    echo "  Environment variables already in ~/.bashrc, skipping..."
fi

echo ""
echo "=== Running diagnostics ==="
python "$INSTALL_DIR/diagnose.py"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Run feature extraction:  bash run_feature_extraction.sh"
echo "  2. Run a test training:     python -m algonauts2025.grids.test_run"
echo "  3. Run full grid search:    python -m algonauts2025.grids.run_grid"
