#!/bin/bash
# Run feature extraction interactively via salloc + srun inside Apptainer.
# Usage: bash run_feature_extraction.sh [time]
#   time: walltime in HH:MM:SS (default: 00:30:00)
#
# Run this from a login node on the cluster.

set -euo pipefail

TIME="${1:-00:30:00}"
SIF="$SCRATCH/containers/tribe.sif"

if [ ! -f "$SIF" ]; then
    echo "Error: Container not found at $SIF"
    echo "Build it first:  bash containers/build.sh"
    exit 1
fi

echo "Requesting GPU allocation (time=$TIME)..."
salloc --account=rrg-pbellec_gpu \
  --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 \
  --cpus-per-task=4 --mem=30G \
  --time="$TIME" \
  srun apptainer exec --nv \
    --bind "$SCRATCH/tribe:/tribe" \
    --bind "$HOME/.cache/huggingface:/hf_cache" \
    --bind "$SCRATCH/data:/data" \
    --bind "$SCRATCH:/scratch_host" \
    "$SIF" \
    bash -c '
      export DATAPATH=/data
      export SAVEPATH=/scratch_host

      echo "=== Job Info ==="
      echo "Job ID:    $SLURM_JOB_ID"
      echo "Node:      $SLURM_NODELIST"
      echo "GPUs:      $CUDA_VISIBLE_DEVICES"
      echo "Started:   $(date)"
      echo ""

      python3 -u /tribe/extract_features_only.py

      echo ""
      echo "Finished: $(date)"
    '
