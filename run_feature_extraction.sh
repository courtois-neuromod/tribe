#!/bin/bash
# Run feature extraction interactively via salloc + srun.
# Usage: bash run_feature_extraction.sh [--slow] [time]
#   --slow: use venv on $SCRATCH directly (slow imports on cold nodes)
#   time:   walltime in HH:MM:SS (default: 04:00:00)
#
# By default, builds a fresh venv on the node-local SSD ($SLURM_TMPDIR)
# using Alliance Canada's wheelhouse. This avoids the 5-10 min CVMFS
# import hangs on cold compute nodes.
#
# Run this from a login node on the cluster.

set -euo pipefail

MODE=local
TIME="04:00:00"

for arg in "$@"; do
    case "$arg" in
        --slow) MODE=slow ;;
        *) TIME="$arg" ;;
    esac
done

echo "Requesting GPU allocation (time=$TIME, mode=$MODE)..."
salloc --account=rrg-pbellec_gpu \
  --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 \
  --cpus-per-task=4 --mem=30G \
  --time="$TIME" \
  srun bash -c '
    MODE='"$MODE"'

    if [ "$MODE" = "local" ]; then
        # ── Local venv mode (default) ─────────────────────────────────
        # Build a venv on node-local SSD from Alliance wheelhouse.
        # No network needed — pip install --no-index uses local wheels.
        echo "Building local venv on $SLURM_TMPDIR..."
        module load python/3.12 gcc arrow ffmpeg
        python3 -m venv "$SLURM_TMPDIR/venv"
        source "$SLURM_TMPDIR/venv/bin/activate"

        pip install --no-index --quiet \
            torch torchvision torchaudio \
            numpy scipy pandas packaging \
            transformers spacy nilearn h5py \
            matplotlib lightning einops torchmetrics wandb \
            Levenshtein julius moviepy decorator platformdirs pygments pillow \
            huggingface_hub exca 2>&1 | tail -3

        echo "  Venv ready ($(du -sh "$SLURM_TMPDIR/venv" | cut -f1))"
    else
        # ── Slow mode (venv on $SCRATCH) ──────────────────────────────
        echo "Using venv on \$SCRATCH (imports may be slow on cold nodes)..."
        module load python/3.12 gcc arrow ffmpeg
        source "$SCRATCH/envs/tribe/bin/activate"
    fi

    export DATAPATH="$SCRATCH/data"
    export SAVEPATH="$SCRATCH"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    echo ""
    echo "=== Job Info ==="
    echo "Job ID:    $SLURM_JOB_ID"
    echo "Node:      $SLURM_NODELIST"
    echo "GPUs:      $CUDA_VISIBLE_DEVICES"
    echo "Started:   $(date)"
    echo ""

    cd "$SCRATCH/tribe"
    python -u extract_features_only.py

    echo ""
    echo "Finished: $(date)"
  '
