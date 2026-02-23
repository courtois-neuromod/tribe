#!/bin/bash
# Build the TRIBE Apptainer container on a login node.
#
# Usage:
#   bash containers/build.sh
#
# Run from the repository root ($SCRATCH/tribe).

set -euo pipefail

SCRATCH="${SCRATCH:?SCRATCH env var not set}"
DEF_FILE="$(cd "$(dirname "$0")" && pwd)/tribe.def"
SIF_DIR="$SCRATCH/containers"
SIF_FILE="$SIF_DIR/tribe.sif"

mkdir -p "$SIF_DIR"

echo "Building container..."
echo "  Definition: $DEF_FILE"
echo "  Output:     $SIF_FILE"
echo ""

apptainer build --fakeroot "$SIF_FILE" "$DEF_FILE"

echo ""
echo "Build complete: $SIF_FILE"
echo "Size: $(du -h "$SIF_FILE" | cut -f1)"
echo ""
echo "Smoke test:"
echo "  apptainer exec --nv $SIF_FILE python3 -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
