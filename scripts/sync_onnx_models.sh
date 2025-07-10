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
cd "$DST_DIR"
git checkout main
git pull

# 2. Remove all existing .onnx files
find . -type f -name "*.onnx" -delete

# 3. Copy new ONNX files from source
cp -R "$SRC_DIR"/* "$DST_DIR"/

# 4. Commit and force-push
git add .
git commit -m "🔁 Replace all ONNX test models with latest output"
git push --force

echo "✅ Done. Hugging Face repo updated:"
echo "   https://huggingface.co/enpasos/jax2onnx-models"
