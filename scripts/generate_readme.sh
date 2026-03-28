#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PRIMARY_MAXTEXT_SRC="${JAX2ONNX_MAXTEXT_SRC:-${REPO_ROOT}/tmp/maxtext}"
MAXTEXT_MODELS="${JAX2ONNX_MAXTEXT_MODELS:-all}"
MAXTEXT_REF="${JAX2ONNX_MAXTEXT_REF:-17d805e3488104b5de96bd19be09491ff73c57c1}"
FALLBACK_MAXTEXT_SRC="${REPO_ROOT}/tmp/maxtext-readme-${MAXTEXT_REF:0:7}"

cd "${REPO_ROOT}"

ensure_git_checkout() {
  local repo_path="$1"

  echo "Preparing MaxText checkout at ${repo_path}..."
  if [[ ! -d "${repo_path}" ]]; then
    mkdir -p "$(dirname "${repo_path}")"
    git clone --depth 1 https://github.com/AI-Hypercomputer/maxtext.git "${repo_path}"
  else
    echo "Using existing MaxText checkout at ${repo_path}."
  fi

  if [[ ! -d "${repo_path}/.git" ]]; then
    echo "Error: ${repo_path} exists but is not a git checkout." >&2
    exit 1
  fi

  if git -C "${repo_path}" fetch --depth 1 origin "${MAXTEXT_REF}"; then
    git -C "${repo_path}" checkout --detach FETCH_HEAD
  else
    echo "Direct fetch for '${MAXTEXT_REF}' failed; retrying with full history..."
    git -C "${repo_path}" fetch --tags origin
    git -C "${repo_path}" fetch --unshallow origin || true
    git -C "${repo_path}" checkout --detach "${MAXTEXT_REF}"
  fi
}

resolve_active_maxtext_src() {
  local candidate="$1"

  if [[ -d "${candidate}/.git" ]] && [[ -n "$(git -C "${candidate}" status --porcelain)" ]]; then
    local fallback="${FALLBACK_MAXTEXT_SRC}"
    if [[ -e "${fallback}" ]] && { [[ ! -d "${fallback}/.git" ]] || [[ -n "$(git -C "${fallback}" status --porcelain 2>/dev/null)" ]]; }; then
      fallback="$(mktemp -d "${REPO_ROOT}/tmp/maxtext-readme-${MAXTEXT_REF:0:7}-XXXXXX")"
    fi
    echo "Primary MaxText checkout is dirty; preserving it and using clean checkout ${fallback}." >&2
    printf '%s\n' "${fallback}"
    return
  fi

  printf '%s\n' "${candidate}"
}

echo "Installing optional MaxText dependencies..."
poetry install --with maxtext

MAXTEXT_SRC="$(resolve_active_maxtext_src "${PRIMARY_MAXTEXT_SRC}")"
ensure_git_checkout "${MAXTEXT_SRC}"

export JAX2ONNX_MAXTEXT_SRC="${MAXTEXT_SRC}"
export JAX2ONNX_MAXTEXT_MODELS="${MAXTEXT_MODELS}"
export JAX2ONNX_MAXTEXT_REF="${MAXTEXT_REF}"

echo "MaxText checkout: $(git -C "${MAXTEXT_SRC}" rev-parse --short HEAD)"
echo "JAX2ONNX_MAXTEXT_MODELS=${JAX2ONNX_MAXTEXT_MODELS}"
echo "Running README generation with MaxText enabled..."

poetry run python scripts/generate_readme.py
