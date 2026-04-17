#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PRIMARY_MAXTEXT_SRC="${JAX2ONNX_MAXTEXT_SRC:-${REPO_ROOT}/tmp/maxtext}"
MAXTEXT_MODELS="${JAX2ONNX_MAXTEXT_MODELS:-all}"
MAXTEXT_REF="${JAX2ONNX_MAXTEXT_REF:-17d805e3488104b5de96bd19be09491ff73c57c1}"
FALLBACK_MAXTEXT_SRC="${REPO_ROOT}/tmp/maxtext-readme-${MAXTEXT_REF:0:7}"
PRIMARY_MAXDIFFUSION_SRC="${JAX2ONNX_MAXDIFFUSION_SRC:-${REPO_ROOT}/tmp/maxdiffusion}"
MAXDIFFUSION_MODELS="${JAX2ONNX_MAXDIFFUSION_MODELS:-all}"
MAXDIFFUSION_REF="${JAX2ONNX_MAXDIFFUSION_REF:-b4f95730bf4f00c4fd9e3dd2fdda1b50484afda2}"
FALLBACK_MAXDIFFUSION_SRC="${REPO_ROOT}/tmp/maxdiffusion-readme-${MAXDIFFUSION_REF:0:7}"

cd "${REPO_ROOT}"

ensure_git_checkout() {
  local repo_name="$1"
  local repo_url="$2"
  local repo_path="$3"
  local repo_ref="$4"

  echo "Preparing ${repo_name} checkout at ${repo_path}..."
  if [[ ! -d "${repo_path}" ]]; then
    mkdir -p "$(dirname "${repo_path}")"
    git clone --depth 1 "${repo_url}" "${repo_path}"
  else
    echo "Using existing ${repo_name} checkout at ${repo_path}."
  fi

  if [[ ! -d "${repo_path}/.git" ]]; then
    echo "Error: ${repo_path} exists but is not a git checkout." >&2
    exit 1
  fi

  if git -C "${repo_path}" fetch --depth 1 origin "${repo_ref}"; then
    git -C "${repo_path}" checkout --detach FETCH_HEAD
  else
    echo "Direct fetch for '${repo_ref}' failed; retrying with full history..."
    git -C "${repo_path}" fetch --tags origin
    git -C "${repo_path}" fetch --unshallow origin || true
    git -C "${repo_path}" checkout --detach "${repo_ref}"
  fi
}

resolve_active_checkout() {
  local candidate="$1"
  local fallback_base="$2"
  local repo_name="$3"

  if [[ -d "${candidate}/.git" ]] && [[ -n "$(git -C "${candidate}" status --porcelain)" ]]; then
    local fallback="${fallback_base}"
    if [[ -e "${fallback}" ]] && { [[ ! -d "${fallback}/.git" ]] || [[ -n "$(git -C "${fallback}" status --porcelain 2>/dev/null)" ]]; }; then
      fallback="$(mktemp -d "${fallback_base}-XXXXXX")"
    fi
    echo "Primary ${repo_name} checkout is dirty; preserving it and using clean checkout ${fallback}." >&2
    printf '%s\n' "${fallback}"
    return
  fi

  printf '%s\n' "${candidate}"
}

echo "Installing optional MaxText and MaxDiffusion dependencies..."
poetry install --with maxtext,maxdiffusion

MAXTEXT_SRC="$(resolve_active_checkout "${PRIMARY_MAXTEXT_SRC}" "${FALLBACK_MAXTEXT_SRC}" "MaxText")"
ensure_git_checkout \
  "MaxText" \
  "https://github.com/AI-Hypercomputer/maxtext.git" \
  "${MAXTEXT_SRC}" \
  "${MAXTEXT_REF}"

export JAX2ONNX_MAXTEXT_SRC="${MAXTEXT_SRC}"
export JAX2ONNX_MAXTEXT_MODELS="${MAXTEXT_MODELS}"
export JAX2ONNX_MAXTEXT_REF="${MAXTEXT_REF}"

echo "MaxText checkout: $(git -C "${MAXTEXT_SRC}" rev-parse --short HEAD)"
echo "JAX2ONNX_MAXTEXT_MODELS=${JAX2ONNX_MAXTEXT_MODELS}"

MAXDIFFUSION_SRC="$(resolve_active_checkout "${PRIMARY_MAXDIFFUSION_SRC}" "${FALLBACK_MAXDIFFUSION_SRC}" "MaxDiffusion")"
ensure_git_checkout \
  "MaxDiffusion" \
  "https://github.com/google/maxdiffusion.git" \
  "${MAXDIFFUSION_SRC}" \
  "${MAXDIFFUSION_REF}"

export JAX2ONNX_MAXDIFFUSION_SRC="${MAXDIFFUSION_SRC}"
export JAX2ONNX_MAXDIFFUSION_MODELS="${MAXDIFFUSION_MODELS}"
export JAX2ONNX_MAXDIFFUSION_REF="${MAXDIFFUSION_REF}"

echo "MaxDiffusion checkout: $(git -C "${MAXDIFFUSION_SRC}" rev-parse --short HEAD)"
echo "JAX2ONNX_MAXDIFFUSION_MODELS=${JAX2ONNX_MAXDIFFUSION_MODELS}"

echo "Running README generation with MaxText and MaxDiffusion enabled..."

poetry run python scripts/generate_readme.py
