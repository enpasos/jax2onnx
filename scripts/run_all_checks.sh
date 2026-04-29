#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_MAXTEXT="${JAX2ONNX_RUN_MAXTEXT:-0}"
RUN_MAXTEXT_ACTIVE=0
RUN_MAXDIFFUSION="${JAX2ONNX_RUN_MAXDIFFUSION:-0}"
RUN_MAXDIFFUSION_ACTIVE=0
PYTEST_DURATION_ARGS=(--durations=50 --durations-min=1)

cd "${REPO_ROOT}"

step=1

echo "[${step}] Running pre-commit on all files..."
poetry run python scripts/ensure_path_markers.py --fix
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

if [[ "${RUN_MAXDIFFUSION}" == "1" ]]; then
  PYTHON_MM="$(poetry run python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  PYTHON_MINOR="$(poetry run python -c 'import sys; print(sys.version_info.minor)')"
  if (( PYTHON_MINOR < 12 )); then
    echo "[${step}] Skipping MaxDiffusion SotA checks on Python ${PYTHON_MM} (requires >= 3.12)."
    unset JAX2ONNX_MAXDIFFUSION_SRC || true
    unset JAX2ONNX_MAXDIFFUSION_MODELS || true
  else
    RUN_MAXDIFFUSION_ACTIVE=1
  fi
fi

if [[ "${RUN_MAXTEXT_ACTIVE}" == "1" ]]; then
  MAXTEXT_SRC="${JAX2ONNX_MAXTEXT_SRC:-${REPO_ROOT}/tmp/maxtext}"
  MAXTEXT_MODELS="${JAX2ONNX_MAXTEXT_MODELS:-all}"
  MAXTEXT_REF="${JAX2ONNX_MAXTEXT_REF:-17d805e3488104b5de96bd19be09491ff73c57c1}"
  export JAX2ONNX_MAXTEXT_SRC="${MAXTEXT_SRC}"
  export JAX2ONNX_MAXTEXT_MODELS="${MAXTEXT_MODELS}"
  export JAX2ONNX_MAXTEXT_REF="${MAXTEXT_REF}"

  echo "[${step}] Preparing MaxText SotA checks..."
  if [[ ! -d "${MAXTEXT_SRC}" ]]; then
    mkdir -p "$(dirname "${MAXTEXT_SRC}")"
    git clone --depth 1 https://github.com/AI-Hypercomputer/maxtext.git "${MAXTEXT_SRC}"
  else
    echo "Using existing MaxText checkout at ${MAXTEXT_SRC}. Target ref: ${MAXTEXT_REF}."
  fi
  if [[ ! -d "${MAXTEXT_SRC}/.git" ]]; then
    echo "Error: ${MAXTEXT_SRC} exists but is not a git checkout." >&2
    exit 1
  fi
  if git -C "${MAXTEXT_SRC}" fetch --depth 1 origin "${MAXTEXT_REF}"; then
    git -C "${MAXTEXT_SRC}" checkout --detach FETCH_HEAD
  else
    echo "Direct fetch for '${MAXTEXT_REF}' failed; retrying with full history..."
    git -C "${MAXTEXT_SRC}" fetch --tags origin
    git -C "${MAXTEXT_SRC}" fetch --unshallow origin || true
    git -C "${MAXTEXT_SRC}" checkout --detach "${MAXTEXT_REF}"
  fi
  echo "MaxText checkout: $(git -C "${MAXTEXT_SRC}" rev-parse --short HEAD)"

  poetry install --with maxtext
  step=$((step + 1))
fi

if [[ "${RUN_MAXDIFFUSION_ACTIVE}" == "1" ]]; then
  MAXDIFFUSION_SRC="${JAX2ONNX_MAXDIFFUSION_SRC:-${REPO_ROOT}/tmp/maxdiffusion}"
  MAXDIFFUSION_MODELS="${JAX2ONNX_MAXDIFFUSION_MODELS:-all}"
  MAXDIFFUSION_REF="${JAX2ONNX_MAXDIFFUSION_REF:-b4f95730bf4f00c4fd9e3dd2fdda1b50484afda2}"
  export JAX2ONNX_MAXDIFFUSION_SRC="${MAXDIFFUSION_SRC}"
  export JAX2ONNX_MAXDIFFUSION_MODELS="${MAXDIFFUSION_MODELS}"
  export JAX2ONNX_MAXDIFFUSION_REF="${MAXDIFFUSION_REF}"

  echo "[${step}] Preparing MaxDiffusion SotA checks..."
  if [[ ! -d "${MAXDIFFUSION_SRC}" ]]; then
    mkdir -p "$(dirname "${MAXDIFFUSION_SRC}")"
    git clone --depth 1 https://github.com/google/maxdiffusion.git "${MAXDIFFUSION_SRC}"
  else
    echo "Using existing MaxDiffusion checkout at ${MAXDIFFUSION_SRC}. Target ref: ${MAXDIFFUSION_REF}."
  fi
  if [[ ! -d "${MAXDIFFUSION_SRC}/.git" ]]; then
    echo "Error: ${MAXDIFFUSION_SRC} exists but is not a git checkout." >&2
    exit 1
  fi
  if git -C "${MAXDIFFUSION_SRC}" fetch --depth 1 origin "${MAXDIFFUSION_REF}"; then
    git -C "${MAXDIFFUSION_SRC}" checkout --detach FETCH_HEAD
  else
    echo "Direct fetch for '${MAXDIFFUSION_REF}' failed; retrying with full history..."
    git -C "${MAXDIFFUSION_SRC}" fetch --tags origin
    git -C "${MAXDIFFUSION_SRC}" fetch --unshallow origin || true
    git -C "${MAXDIFFUSION_SRC}" checkout --detach "${MAXDIFFUSION_REF}"
  fi
  echo "MaxDiffusion checkout: $(git -C "${MAXDIFFUSION_SRC}" rev-parse --short HEAD)"

  poetry install --with maxdiffusion
  step=$((step + 1))
fi

echo "[${step}] Generating tests..."
poetry run python scripts/generate_tests.py
step=$((step + 1))

if [[ "${RUN_MAXTEXT_ACTIVE}" == "1" ]]; then
  echo "[${step}] Running MaxText SotA tests..."
  poetry run pytest -q tests/examples/test_maxtext.py "${PYTEST_DURATION_ARGS[@]}"
  step=$((step + 1))
fi

if [[ "${RUN_MAXDIFFUSION_ACTIVE}" == "1" ]]; then
  echo "[${step}] Running MaxDiffusion SotA tests..."
  poetry run pytest -q tests/examples/test_maxdiffusion.py "${PYTEST_DURATION_ARGS[@]}"
  step=$((step + 1))
fi

echo "[${step}] Running pytest..."
if poetry run python -c "import xdist" >/dev/null 2>&1; then
  echo "Using pytest-xdist: -n auto --dist=loadscope"
  poetry run pytest -n auto --dist=loadscope "${PYTEST_DURATION_ARGS[@]}"
else
  echo "pytest-xdist is not installed in the Poetry environment."
  echo "Falling back to serial pytest."
  echo "Install it with: poetry add --group dev pytest-xdist"
  poetry run pytest "${PYTEST_DURATION_ARGS[@]}"
fi
