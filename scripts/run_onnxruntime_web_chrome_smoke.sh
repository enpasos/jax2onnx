#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export JAX2ONNX_ONNXRUNTIME_WEB_RUNNER="${JAX2ONNX_ONNXRUNTIME_WEB_RUNNER:-chrome}"

exec "${SCRIPT_DIR}/run_onnxruntime_web_smoke.sh"
