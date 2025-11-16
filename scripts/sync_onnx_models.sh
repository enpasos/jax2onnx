#!/bin/bash

set -e  # Exit on error

# Resolve absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
JAX2ONNX_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$JAX2ONNX_DIR/docs/onnx"
DST_DIR="$JAX2ONNX_DIR/../jax2onnx-models"

echo "🔄 Syncing ONNX files from:"
echo "   $SRC_DIR"
echo "   → $DST_DIR"

# 1. Ensure destination exists and pull latest
mkdir -p "$DST_DIR"
cd "$DST_DIR"
git checkout main
git pull

# 2. Ensure LFS handles ONNX assets (idempotent). Sync only .onnx (no .onnx.data).
git lfs install --local
git lfs track "*.onnx" "examples/**/*.onnx" "primitives/**/*.onnx" "docs/onnx/**/*.onnx"
git add .gitattributes

# 3. Remove existing ONNX payloads (drop any lingering .onnx.data; we don't sync them)
find . -type f -name "*.onnx" -delete
find . -type f -name "*.onnx.data" -delete

# 4. Copy only .onnx files from source into repo root (maintain original layout)
rsync -av --include='*/' --include='*.onnx' --exclude='*' "$SRC_DIR"/ "$DST_DIR"/

# 5. Commit and force-push (scope to synced assets + attributes)
git add .gitattributes *.onnx docs/onnx examples primitives
git status --short
git commit -m "🔁 Replace all ONNX test models with latest output"
git push --force

echo "✅ Done. Hugging Face repo updated:"
echo "   https://huggingface.co/enpasos/jax2onnx-models"
