#!/bin/bash
# Run feature extraction interactively via salloc + srun.
# Usage: bash run_feature_extraction.sh [time]
#   time: walltime in HH:MM:SS (default: 00:30:00)
#
# Run this from a login node on the cluster.

set -euo pipefail

TIME="${1:-00:30:00}"

echo "Requesting GPU allocation (time=$TIME)..."
salloc --account=rrg-pbellec_gpu \
  --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 \
  --cpus-per-task=4 --mem=30G \
  --time="$TIME" \
  srun bash -c '
    module load python/3.12 gcc arrow ffmpeg
    source "$SCRATCH/envs/tribe/bin/activate"

    export DATAPATH="$SCRATCH/data"
    export SAVEPATH="$SCRATCH"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

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
