#!/bin/bash

set -e  # Exit on error

# Resolve absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
JAX2ONNX_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$JAX2ONNX_DIR/onnx"
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
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs not found; install Git LFS before syncing models." >&2
  exit 1
fi

HOOK="$DST_DIR/.git/hooks/pre-push"
# Refresh the LFS hook if an older pre-push hook blocks installation.
if ! git lfs install --local; then
  if [ -f "$HOOK" ] && grep -q "git lfs pre-push" "$HOOK"; then
    echo "Refreshing existing Git LFS hook with 'git lfs update --force'..."
    git lfs update --force
    git lfs install --local --force
  else
    echo "Pre-push hook at $HOOK is not an LFS hook. Merge manually via 'git lfs update --manual'." >&2
    exit 1
  fi
fi
git lfs track "*.onnx" "examples/**/*.onnx" "primitives/**/*.onnx" "onnx/**/*.onnx"
git add .gitattributes

# 3. Remove existing ONNX payloads (drop any lingering .onnx.data; we don't sync them)
find . -type f -name "*.onnx" -delete
find . -type f -name "*.onnx.data" -delete

# 4. Copy only .onnx files from source into repo root (maintain original layout)
rsync -av --include='*/' --include='*.onnx' --exclude='*' "$SRC_DIR"/ "$DST_DIR"/

# 5. Commit and force-push (scope to synced assets + attributes)
TRACK_TARGETS=(".gitattributes")
if compgen -G "*.onnx" >/dev/null; then
  TRACK_TARGETS+=(*.onnx)
fi
for dir in onnx examples primitives; do
  if [ -d "$dir" ]; then
    TRACK_TARGETS+=("$dir")
  fi
done
git add "${TRACK_TARGETS[@]}"
git status --short
git commit -m "🔁 Replace all ONNX test models with latest output"
git push --force

echo "✅ Done. Hugging Face repo updated:"
echo "   https://huggingface.co/enpasos/jax2onnx-models"
