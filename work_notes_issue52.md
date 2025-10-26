## Issue 52 – Loop Concat Metadata Regression

- Goal: make the nested-scan sandbox (`jax2onnx/sandbox/issue52_loop_concat_bug.py`) round-trip without metadata hacks by keeping the 5-wide stack extent visible to downstream ops.

- Plan (done):
  1. Lift the sandbox script into a pytest regression so the failure reproduces in CI.
  2. Teach `scan` lowering to carry per-step axis-0 extents into the loop outputs (via `loop_axis0_override`) and keep that metadata through clones.
  3. Ensure the metadata survives ONNX serialization and update tests accordingly.

- Implementation state:
  * Added `tests/extra_tests/loop/test_loop_concat_extent_regression.py` which imports the sandbox repro and checks both the ONNX squeeze output and the IR override.
  * Updated `jax2onnx/plugins/jax/lax/scan.py` to derive static loop extents from the scan body, stamp them on the loop results, and copy the metadata to the builder outputs before binding.
  * Patched `clone_graph` so `ir.Value.meta` entries (e.g. `loop_axis0_override`) survive IR cloning/optimisation.
  * Converted the sandbox repro into an in-tree module so tests can import its helpers.
  * Loop lowering now restamps axis-0 shapes (and metadata) before serialization so broadcast/Squeeze value infos record the real stack width and ORT loads the unpatched export.
  * Broadcast-in-dim now prefers loop/scatter overrides when building Expand target shapes, so axis-0 remains concrete instead of falling back to 1.
  * Added a broadcast-specific sandbox/test pair (`jax2onnx/sandbox/issue52_broadcast_bug.py`, `tests/extra_tests/loop/test_loop_broadcast_extent_regression.py`).
  * `poetry run pytest -q` currently yields `1614 passed, 3 xfailed, 122 warnings`.

---

## Outstanding Work – Issue 52 Scatter/Broadcast Payload

- `jax2onnx/sandbox/issue52_scatter_payload_repro.py` still restamps axis-0 to 1 for the full feed-forward export. Nodes such as `bcast_reshape_out_*` and `dyn_slice_out_*` remain `[1, 5, …]`, so ORT fails unless we patch the metadata manually.
- The remaining gap is the nested scatter window → broadcast chain used in the payload. The simple repros are fixed, but the real Sod export still collapses axis 0.

### Next steps

1. Instrument the payload repro (and/or Sod export) to log axis-0 metadata after each stage—loop output, scatter result, slice/reshape, expand—so we can pinpoint where the override drops.
2. Inspect `scatter_utils` and related hint plumbing (`_loop_extent_hints`, window size helpers) to ensure windowed scatters set `loop_axis0_override` correctly. Patch them as needed.
3. Restamp axis-0 for the reshape/expand stack that feeds the broadcast inside the payload (using `_restamp_axis0`/`ensure_axis0_extent`).
4. Add a regression mirroring the payload path (likely reusing helpers from the sandbox) and re-run `issue52_scatter_payload_repro.py` until it prints the "metadata fixed" line without manual edits.

### Gather deep dive work-in-progress

- Pulled the `jax.lax.gather` semantics from the JAX docs and the XLA Gather specification to understand how `slice_sizes`, `offset_dims`, and `collapsed_slice_dims` determine the result shape.
- Confirmed via instrumentation that our GIR still stamps `target_dimensions_shape=[201, 6]`, so every downstream restamp rebuilds the 201-wide dimension even after we try to override it.
- Experiments adding post-hoc `Slice` nodes (in `ensure_axis0_extent` and the gather plugin) shrink the tensor but introduce invalid Slices on scalar constants; reverting those changes kept the repo clean.
- Next concrete step: teach `gather_compile.py` to rewrite the GIR before lowering so the first gather dimension reflects the loop override (5) while the metadata still records the override for later ops. Once that lands we can re-run the sandbox and then wire in the regression.
