# Work Notes – Issue 52 Scatter Window Broadcast 

## Current Status (2025-10-02)

- Added `_jaxpr_contains_scatter` in `scan.py` so we only emit loop extent hints when a scan (or nested subgraph) actually contains a scatter; the hints ride along via `make_subgraph_context`.
- `broadcast_in_dim` now falls back to loop extent hints (axis `0` only) when scatter hints are absent, and continues to use the scatter-derived vectors for the other axes. Reshape/expand shapes inside loop bodies now pick up dynamic values instead of always hard-coding `1`.
- Regression coverage:
  - `tests/extra_tests/loop/test_loop_ff_like_broadcast_mul_regression.py`
  - `tests/primitives/test_lax.py::Test_scan::test_scan_identity_slice_helper{,_f64}`
  - `tests/primitives/test_lax.py::Test_slice::test_slice_scan_axis_drop{,_f64}`
  all pass with the new loop-hint gating.
- Sandbox repro still fails inside `node_Mul_*` with a reshape mismatch. The extent vectors we emit for loop broadcasts stop at the inner `scan` boundary because the trip-count is still materialised via a constant `1`; we’re not yet deriving the interior update length from scatter metadata.

## What Remains

1. Source the loop extent from the scatter metadata when a scan has no explicit scanned inputs (e.g. derive it from the scatter updates/indices shape) so the inner loop no longer holds a constant `1` trip-count.
2. Once the reshape mismatch is resolved, rerun the sandbox script and validate the ONNX graph executes end-to-end in ORT with numerically correct outputs.
3. Un-XFAIL `tests/extra_tests/loop/test_loop_scatter_payload_regression.py` and tighten assertions to compare ORT vs. JAX outputs.
4. Drop any temporary debug prints (`J2O_DEBUG_LOOP_HINTS`, etc.) before landing the fix.

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
