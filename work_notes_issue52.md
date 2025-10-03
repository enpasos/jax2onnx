# Work Notes – Issue 52 Scatter Window Broadcast 

## Current Status (2025-10-02)

- Updated `jax2onnx/plugins2/jax/lax/scatter_utils.py`
  - `_compute_window_sizes` now returns the unsqueezed window-size tensors and stores dynamic window extents per operand axis on the IR context via `_scatter_window_hints`.
- Updated `jax2onnx/plugins2/jax/lax/broadcast_in_dim.py`
  - Broadcast lowering consumes those hints (only when the primitive actually broadcasts) to build `Concat` inputs for both the expand target and reshape shape from live IR values instead of hard-coded `1`s.
- Sandbox repro (`poetry run python jax2onnx/sandbox/issue52_scatter_payload_repro.py`) now gets past the original `Expand_13` ShapeInferenceError. ONNX export succeeds but onnxruntime fails inside `scan_loop_0/scan_loop_0/Mul_15` because the loop body still receives mismatched window extents.
- Added temporary debug prints during the last run; they are already removed.

## What Remains

1. For loop bodies, propagate scatter window hints in the same way as at top-level. The inner scatter lowering is producing hints, but nested `broadcast_in_dim` calls still emit partial shapes (leading dimension stays 1). Need to ensure the loop IR context reuses the stored hints when building reshape/expand shapes inside the scan body.
2. After fixing the loop broadcast, re-run the sandbox script and confirm the exported model loads in ORT and produces numerically correct outputs.
3. Update `tests/extra_tests2/loop/test_loop_scatter_payload_regression.py`:
   - Remove the `xfail` guard and assert that ORT results match the JAX feed-forward outputs.
4. Remove any remaining debug logging once the fix is confirmed.

## Useful Commands

```bash
poetry run python jax2onnx/sandbox/issue52_scatter_payload_repro.py
poetry run pytest -q tests/extra_tests2/scatter_utils/test_padding_and_expected_shape.py
poetry run pytest -q tests/extra_tests2/loop/test_loop_scatter_payload_regression.py
```

## Quick Code References

- Scatter/window hints: `jax2onnx/plugins2/jax/lax/scatter_utils.py`
- Broadcast lowering consuming hints: `jax2onnx/plugins2/jax/lax/broadcast_in_dim.py`
- Regression test to flip back on: `tests/extra_tests2/loop/test_loop_scatter_payload_regression.py`
