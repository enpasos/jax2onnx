#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

RUNNER="${JAX2ONNX_ONNXRUNTIME_WEB_RUNNER:-node}"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required for onnxruntime-web smoke checks." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/node_modules/onnxruntime-web/package.json" ]]; then
  echo "onnxruntime-web is not installed. Run: npm install" >&2
  exit 1
fi

if [[ "${RUNNER}" == "chrome" || "${RUNNER}" == "browser" ]]; then
  if [[ ! -f "${REPO_ROOT}/node_modules/playwright/package.json" ]]; then
    echo "Playwright is required for onnxruntime-web Chrome smoke checks. Run: npm install" >&2
    exit 1
  fi
fi

export JAX2ONNX_VALIDATE_ONNXRUNTIME_WEB=1
export JAX2ONNX_ONNXRUNTIME_WEB_RUNNER="${RUNNER}"

echo "Running onnxruntime-web smoke checks with runner: ${RUNNER}"

poetry run pytest -q \
  tests/extra_tests/test_quickstart.py \
  tests/extra_tests/test_return_modes.py \
  tests/examples/test_lax.py \
  tests/examples/test_jnp.py \
  tests/primitives/test_nn.py::Test_gelu::test_jaxnn_gelu
