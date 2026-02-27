#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_MAXTEXT="${JAX2ONNX_RUN_MAXTEXT:-0}"
RUN_MAXTEXT_ACTIVE=0

cd "${REPO_ROOT}"

step=1

echo "[${step}] Running pre-commit on all files..."
poetry run pre-commit run --all-files
step=$((step + 1))

if [[ "${RUN_MAXTEXT}" == "1" ]]; then
  PYTHON_MM="$(poetry run python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  PYTHON_MINOR="$(poetry run python -c 'import sys; print(sys.version_info.minor)')"
  if (( PYTHON_MINOR < 12 )); then
    echo "[${step}] Skipping MaxText SotA checks on Python ${PYTHON_MM} (requires >= 3.12)."
    unset JAX2ONNX_MAXTEXT_SRC || true
    unset JAX2ONNX_MAXTEXT_MODELS || true
  else
    RUN_MAXTEXT_ACTIVE=1
  fi
fi

if [[ "${RUN_MAXTEXT_ACTIVE}" == "1" ]]; then
  MAXTEXT_SRC="${JAX2ONNX_MAXTEXT_SRC:-${REPO_ROOT}/tmp/maxtext}"
  MAXTEXT_MODELS="${JAX2ONNX_MAXTEXT_MODELS:-all}"
  export JAX2ONNX_MAXTEXT_SRC="${MAXTEXT_SRC}"
  export JAX2ONNX_MAXTEXT_MODELS="${MAXTEXT_MODELS}"

  echo "[${step}] Preparing MaxText SotA checks..."
  if [[ ! -d "${MAXTEXT_SRC}" ]]; then
    mkdir -p "$(dirname "${MAXTEXT_SRC}")"
    git clone https://github.com/AI-Hypercomputer/maxtext.git "${MAXTEXT_SRC}"
  fi

  poetry install --with maxtext
  step=$((step + 1))
fi

echo "[${step}] Generating tests..."
poetry run python scripts/generate_tests.py
step=$((step + 1))

if [[ "${RUN_MAXTEXT_ACTIVE}" == "1" ]]; then
  echo "[${step}] Running MaxText SotA tests..."
  poetry run pytest -q tests/examples/test_maxtext.py
  step=$((step + 1))
fi

echo "[${step}] Running pytest..."
if poetry run python -c "import xdist" >/dev/null 2>&1; then
  echo "Using pytest-xdist: -n auto --dist=loadscope"
  poetry run pytest -n auto --dist=loadscope
else
  echo "pytest-xdist is not installed in the Poetry environment."
  echo "Falling back to serial pytest."
  echo "Install it with: poetry add --group dev pytest-xdist"
  poetry run pytest
fi
