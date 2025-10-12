# Work Notes – Issue 52 Scatter Window Broadcast 

## Current Status (2025-10-12)

- Added an exploratory ONNX Script playground (`jax2onnx/sandbox/onnxscript/play01.py` / `play02.py`) to poke at shape rewrites post-export. `play02.py` can splice a dynamic dim into the top-level broadcast reshape, which confirmed the downstream `Slice` constants are the remaining culprits.
- Tweaked `broadcast_in_dim` so it always considers `_scatter_window_hints`; debug logs show the reshape path now tries to consume `scatter_window_size_vec_*`, but those hints still evaluate to `1`, so the exported graph remains inconsistent.
- No pipeline changes yet—the converter still emits constant-backed `Concat` / `Slice` tensors (e.g. `shape_const_4_0`, `slice_limits_14`), so the sandbox `poetry run python jax2onnx/sandbox/issue52_scatter_payload_repro.py` continues to fail in ORT inside `scan_loop_0/.../node_Mul_102`.
- The ONNX-script probe highlighted exactly which constants need to be replaced by `_scatter_window_hints` outputs inside the plugin lowerings rather than patched after the fact.

## What Remains

1. Fix the scatter lowering itself (`jax2onnx/plugins/jax/lax/scatter_utils.py`) so `_compute_window_sizes` records the correct loop-body extent in `_scatter_window_hints` instead of the current `1` fallback. Once those hints are accurate, `broadcast_in_dim` and the slice helpers will consume live dims automatically.
2. After hints carry the right values, rerun the sandbox repro and verify onnxruntime completes without the `node_Mul_102` mismatch; compare outputs against the JAX reference.
3. Update `tests/extra_tests/loop/test_loop_scatter_payload_regression.py`:
   - Remove the `xfail` guard and assert that ORT results match the JAX feed-forward outputs.
4. Remove any remaining debug logging once the fix is confirmed.

## Useful Commands

```bash
poetry run python jax2onnx/sandbox/issue52_scatter_payload_repro.py
poetry run pytest -q tests/extra_tests/scatter_utils/test_padding_and_expected_shape.py
poetry run pytest -q tests/extra_tests/loop/test_loop_scatter_payload_regression.py
```

## Quick Code References

- Scatter/window hints: `jax2onnx/plugins/jax/lax/scatter_utils.py`
- Broadcast lowering consuming hints: `jax2onnx/plugins/jax/lax/broadcast_in_dim.py`
- Regression test to flip back on: `tests/extra_tests/loop/test_loop_scatter_payload_regression.py`
