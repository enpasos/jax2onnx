#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

RUNNER="${JAX2ONNX_ONNXRUNTIME_WEB_RUNNER:-node}"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required for full onnxruntime-web checks." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/node_modules/onnxruntime-web/package.json" ]]; then
  echo "onnxruntime-web is not installed. Run: npm install" >&2
  exit 1
fi

if [[ "${RUNNER}" == "chrome" || "${RUNNER}" == "browser" ]]; then
  if [[ ! -f "${REPO_ROOT}/node_modules/playwright/package.json" ]]; then
    echo "Playwright is required for full onnxruntime-web Chrome checks. Run: npm install" >&2
    exit 1
  fi
fi

export JAX2ONNX_EXPORT_MODE=web
export JAX2ONNX_VALIDATE_ONNXRUNTIME_WEB=1
export JAX2ONNX_ONNXRUNTIME_WEB_RUNNER="${RUNNER}"

echo "Running full pytest with onnxruntime-web runner: ${RUNNER}"
echo "Web validation export mode: ${JAX2ONNX_EXPORT_MODE}"

if [[ "$#" -gt 0 ]]; then
  exec poetry run pytest "$@"
fi

exec poetry run pytest
