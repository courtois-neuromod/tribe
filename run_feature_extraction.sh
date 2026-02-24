#!/bin/bash
# Run feature extraction interactively via salloc + srun.
# Usage: bash run_feature_extraction.sh [--container] [time]
#   --container: use Apptainer container (faster imports, requires build first)
#   time:        walltime in HH:MM:SS (default: 00:30:00)
#
# By default uses module/venv. Pass --container for the containerized version
# which avoids slow CVMFS imports on cold compute nodes.
#
# Run this from a login node on the cluster.

set -euo pipefail

USE_CONTAINER=false
TIME="04:00:00"

for arg in "$@"; do
    case "$arg" in
        --container) USE_CONTAINER=true ;;
        *) TIME="$arg" ;;
    esac
done

if [ "$USE_CONTAINER" = true ]; then
    # ── Container mode ──────────────────────────────────────────────
    module load apptainer 2>/dev/null || true
    SIF="$SCRATCH/containers/tribe.sif"

    if [ ! -f "$SIF" ]; then
        echo "Error: Container not found at $SIF"
        echo "Build it first:  bash containers/build.sh"
        exit 1
    fi

    echo "Requesting GPU allocation (time=$TIME, mode=container)..."
    export TRIBE_SIF="$SIF"
    salloc --account=rrg-pbellec_gpu \
      --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 \
      --cpus-per-task=4 --mem=30G \
      --time="$TIME" \
      srun bash -c '
        module load apptainer 2>/dev/null || true

        # Copy SIF to node-local SSD — avoids slow squashfuse reads over Lustre.
        # This takes ~4 min but makes imports instant (seconds vs 5-10 min).
        LOCAL_SIF="$SLURM_TMPDIR/tribe.sif"
        echo "Copying container to local SSD ($SLURM_TMPDIR)..."
        cp "$TRIBE_SIF" "$LOCAL_SIF"
        echo "  Done ($(du -h "$LOCAL_SIF" | cut -f1))"
        echo ""

        echo "=== Job Info ==="
        echo "Job ID:    $SLURM_JOB_ID"
        echo "Node:      $SLURM_NODELIST"
        echo "GPUs:      $CUDA_VISIBLE_DEVICES"
        echo "Started:   $(date)"
        echo ""

        apptainer exec --nv \
          --bind "$SCRATCH/tribe:/tribe" \
          --bind "$HOME/.cache/huggingface:/hf_cache" \
          --bind "$SCRATCH/data:/data" \
          --bind "$SCRATCH:/scratch_host" \
          "$LOCAL_SIF" \
          bash -c "
            export DATAPATH=/data
            export SAVEPATH=/scratch_host

            python3 -u /tribe/extract_features_only.py

            echo \"\"
            echo \"Finished: \$(date)\"
          "
      '
else
    # ── Venv mode (default) ─────────────────────────────────────────
    echo "Requesting GPU allocation (time=$TIME, mode=venv)..."
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
fi
