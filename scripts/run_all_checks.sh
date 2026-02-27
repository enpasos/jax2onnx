#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "[1/3] Running pre-commit on all files..."
poetry run pre-commit run --all-files

echo "[2/3] Generating tests..."
poetry run python scripts/generate_tests.py

echo "[3/3] Running pytest..."
if poetry run python -c "import xdist" >/dev/null 2>&1; then
  echo "Using pytest-xdist: -n auto --dist=loadscope"
  poetry run pytest -n auto --dist=loadscope
else
  echo "pytest-xdist is not installed in the Poetry environment."
  echo "Falling back to serial pytest."
  echo "Install it with: poetry add --group dev pytest-xdist"
  poetry run pytest
fi
