# Work Notes – Issue 52 Scatter Window Broadcast 

## Current Status (2025-10-15)

- Added `_jaxpr_contains_scatter` in `scan.py` so we only emit loop extent hints when a scan (or nested subgraph) actually contains a scatter; the hints ride along via `make_subgraph_context`.
- Scans that contain scatters now record a static loop-extent override (axis `0`) on their subgraph contexts and reuse that value for the trip-count as well as hint propagation.
- `broadcast_in_dim` prefers the loop override on axis `0` (falling back to scatter/window hints when present) so Expand nodes build `[extent, …]` shapes instead of hard-coded ones.
- Gathered per-step tensors in the nested scans now expand their leading dimension to the override length, and both `slice` / `squeeze` plug-ins re-expand axis `0` when the override is active.
- Regression coverage:
  - `tests/extra_tests/loop/test_loop_ff_like_broadcast_mul_regression.py`
  - `tests/primitives/test_lax.py::Test_scan::test_scan_identity_slice_helper{,_f64}`
  - `tests/primitives/test_lax.py::Test_slice::test_slice_scan_axis_drop{,_f64}`
  all pass with the new loop-hint gating.
- Sandbox repro still fails inside `node_Mul_*` with a reshape mismatch. Even though the loop hints now advertise the 5-element scatter window (and per-step gathers/slices expand to that length), later slice/squeeze chains still collapse axis `0` back to `1` before the final Expand, so ORT still observes a `(1, …)` vs `(5, …)` multiply.

## What Remains

1. Ensure inner slice/squeeze paths (e.g. `slice_out_24` → `squeeze_out_20`) rebuild axis `0` to the scatter extent so no `(…, 1, …)` tensors reach the broadcast/mul nodes.
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
