#!/bin/bash
#SBATCH --job-name=tribe_features
#SBATCH --account=rrg-pbellec_gpu
#SBATCH --partition=gpubase_bynode_b3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# --- Environment setup ---
module load python/3.12 gcc arrow
source "$SCRATCH/tribe/envs/tribe/bin/activate"

# --- Job info ---
echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo ""

# --- Run feature extraction ---
cd "$SCRATCH/tribe"
python extract_features_only.py

echo ""
echo "Finished: $(date)"
