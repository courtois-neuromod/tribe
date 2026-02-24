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
echo "[1/9] Loading modules..."
module load python/3.12 gcc arrow git-annex
echo "  Loaded: python/3.12, gcc, arrow, git-annex"

# --- Step 2: Clone the repository ---
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[2/9] Repository already exists at $INSTALL_DIR, pulling latest..."
    cd "$INSTALL_DIR" && git pull
else
    echo "[2/9] Cloning repository..."
    git clone git@github.com:courtois-neuromod/tribe.git "$INSTALL_DIR"
fi

# --- Step 3: Create virtual environment ---
if [ -d "$VENV_DIR" ]; then
    echo "[3/9] Venv already exists at $VENV_DIR, skipping creation..."
else
    echo "[3/9] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- Step 4: Install dependencies ---
echo "[4/9] Installing dependencies..."

# PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Local packages (editable mode)
cd "$INSTALL_DIR"
pip install -e data_utils/ -e modeling_utils/

# Core scientific packages (not provided by Alliance Canada modules when
# the venv is created with include-system-site-packages=false)
pip install numpy scipy pandas pyarrow packaging

# Additional dependencies
pip install transformers moviepy spacy nilearn Levenshtein "huggingface_hub[cli]" julius h5py \
    decorator matplotlib platformdirs pygments pillow

# Datalad (for dataset download)
pip install datalad

echo "  All dependencies installed."

# --- Step 5: Download the Algonauts 2025 dataset ---
if [ -d "$DATA_DIR/.git" ]; then
    echo "[5/9] Dataset already cloned at $DATA_DIR, skipping..."
else
    echo "[5/9] Downloading Algonauts 2025 dataset (this will take a while)..."
    cd "$(dirname "$DATA_DIR")"
    datalad install -r -s https://github.com/courtois-neuromod/algonauts_2025.competitors.git
    cd "$DATA_DIR"
    datalad get -r -J8 .
    echo "  Dataset download complete."
fi

# --- Step 6: Create expected directory structure ---
# The code expects data at DATAPATH/algonauts2025/download/algonauts_2025.competitors/
echo "[6/9] Setting up dataset directory structure..."
mkdir -p "$SCRATCH/data/algonauts2025/download"
ln -sf "$DATA_DIR" "$SCRATCH/data/algonauts2025/download/algonauts_2025.competitors"
echo "  Symlinked dataset to expected location"

# --- Step 7: HuggingFace login ---
echo "[7/9] HuggingFace login (needed for LLAMA 3.2-3B access)..."
echo "  You need a 'read' token from https://huggingface.co/settings/tokens"
echo "  And access to https://huggingface.co/meta-llama/Llama-3.2-3B"
huggingface-cli login

# --- Step 8: Pre-download models ---
# Compute nodes have NO internet access on Alliance Canada clusters.
# All models must be cached on the login node first.
echo "[8/9] Pre-downloading models (compute nodes have no internet)..."

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

# --- Step 9: Set environment variables ---
echo "[9/9] Setting environment variables..."

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

# --- Step 10 (optional): Build Apptainer container ---
echo ""
echo "[10] Building Apptainer container (optional, speeds up imports)..."
echo "  This bundles all Python deps into a container image, avoiding"
echo "  slow CVMFS imports on cold compute nodes."
bash "$INSTALL_DIR/containers/build.sh" || echo "  WARNING: Container build failed. You can still use venv mode."

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Run feature extraction (venv):       bash run_feature_extraction.sh 02:30:00"
echo "  2. Run feature extraction (container):  bash run_feature_extraction.sh --container 02:30:00"
echo "  3. Run a test training:                 python -m algonauts2025.grids.test_run"
echo "  4. Run full grid search:                python -m algonauts2025.grids.run_grid"
