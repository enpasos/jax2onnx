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

# These exact CPU-ORT cases remain required in every full Python lane. Keep
# complete item IDs so new parameters and both comparator controls remain here.
CPU_ONLY_DESELECT=()
for caller in raw-x64-off raw-x64-on; do
  for index in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21; do
    CPU_ONLY_DESELECT+=("--deselect=tests/extra_tests/converter/test_source_original22_state_preparation.py::test_original22_source_state_preparation_cpu_feedback_and_reset[${caller}-${index}]")
  done
  for index in 0 1 3 4 20; do
    CPU_ONLY_DESELECT+=("--deselect=tests/extra_tests/converter/test_source_model_native_bindings_exports.py::test_original_full_model_native_bindings_cpu_gate[${caller}-${index}]")
  done
done

if [[ "$#" -gt 0 ]]; then
  exec poetry run pytest "$@" "${CPU_ONLY_DESELECT[@]}"
fi

exec poetry run pytest "${CPU_ONLY_DESELECT[@]}"
