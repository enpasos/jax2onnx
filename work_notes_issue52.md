# Work Notes – Issue 52 Scatter Window Broadcast

## Status Update – 2025-10-18

- Added a focused xfail regression (`tests/extra_tests/loop/test_loop_scatter_payload_regression.py::test_issue52_const_slice_broadcast`) capturing the constant slice vs. loop extent mismatch as a small JAX example.
- Initial attempts to clamp loop overrides inside `broadcast_in_dim` and `slice` revealed that axis-0 hints are leaking even when the broadcast dimensions don’t include the leading axis. The minimal repro still fails under ORT (`node_Expand_179`), so the next iteration should concentrate on padding the constant slice path before the add/mul combination.
- With instrumentation, the failing adds all showed mismatched overrides (`bcast_out_*` @ axis0=5, `slice_out_*` @ axis0=3). We’ll start a new conversation to explore targeted padding or alternative expansion strategies for that constant branch.

## Current Status (2025-10-16)

- Added `_jaxpr_contains_scatter` in `scan.py` so we only emit loop extent hints when a scan (or nested subgraph) actually contains a scatter; the hints ride along via `make_subgraph_context`.
- Scans that contain scatters now record a static loop-extent override (axis `0`) on their subgraph contexts and reuse that value for the trip-count as well as hint propagation.
- `broadcast_in_dim` now prefers loop overrides (checking operand, out-spec, and loop context) and uses them when constructing reshape/expand shapes, so axis `0` no longer defaults to hard-coded `1`s.
- Added an `_axis0_utils` helper that can re-expand scalars using a reference tensor; `slice`, `squeeze`, and the elementwise primitives (add/div/mul/sub/square) all leverage it so constants inherit the scan override.
- Gathered per-step tensors in the nested scans expand their leading dimension to the override length, and `slice`/`squeeze` now propagate the override even when their inputs were generated from constant paths.
- Regression coverage:
  - `tests/extra_tests/loop/test_loop_ff_like_broadcast_mul_regression.py`
  - `tests/primitives/test_lax.py::Test_scan::test_scan_identity_slice_helper{,_f64}`
  - `tests/primitives/test_lax.py::Test_slice::test_slice_scan_axis_drop{,_f64}`
  all pass with the new loop-hint gating.
- Sandbox repro still fails at `node_Mul_166` with ORT reporting shape merges like `{1,1,1}` vs `{5,1,1}`. Axis overrides now survive most paths, but several broadcast nodes (`bcast_out_{14..18}` etc.) still inherit aval metadata with leading `1`s, so reshape constants remain `(1, …)` even after expansion.

## What Remains

1. Track the scan-body producers that feed the problematic broadcasts (`bcast_out_{14..18}`, `bcast_out_{27..32}`, etc.) and ensure their metadata (`set_axis0_override` / aval shape) reflects the length-5 window before hitting `broadcast_in_dim`.
2. Update those producers (likely slice/gather/scatter utilities or scan state wiring) to copy the override onto the values or adjust aval shapes when the override is known.
3. Re-run the sandbox (`issue52_scatter_payload_repro.py --dump-nodes`) and confirm ORT stops emitting `{1,1,1}` merge warnings.
4. Once ORT is happy, rerun relevant pytest targets (scatter regression + loop tests) and un-XFAIL `tests/extra_tests/loop/test_loop_scatter_payload_regression.py` with stricter ORT vs JAX comparisons.
5. Remove temporary debug environment hooks before landing the fix.

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
