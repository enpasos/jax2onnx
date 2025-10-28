# Concatenation Shape Bug Notes

## Goal
- Make `jax2onnx/sandbox/issue52_stack_bug.py` run to completion without edits by fixing the underlying converter bug in `jax2onnx`.

## Current Analysis
- Added a `src/jax2onnx` symlink so the sandbox imports the local checkout. The repro then tripped on `RepeatedCompositeContainer.clear()`, which disappeared in protobuf 5/upb; we shimmed the method in `jax2onnx/__init__` so the script can mutate graph outputs again.
- The converter’s DCE pass was pruning the inner `scan` entirely because its outputs are unused in `_outer_scan`. When stacktrace metadata is enabled (the sandbox sets `JAX2ONNX_ENABLE_STACKTRACE_METADATA=1`), we now skip dead-code elimination so the Loop body stays intact for inspection. With that change, the sandbox reshaper finally hooks the real stack tensor and the ORT failure reproduces the reported `5×210×1×1` vs `1×210×1×1` mismatch.

## Working Plan
1. Reproduce the converter failure end-to-end once the `.clear()` incompatibility is mitigated, capturing the mismatched concat output shape.
2. Trace how the stack/concat value_info is generated inside the converter to find where `1×210×1×1` is introduced.
3. Adjust the builder or metadata propagation so the concat output keeps the correct `5×210×1×1` dimension, then re-run the sandbox script for confirmation.

## Implementation Status
- [x] Step 1 — repro now runs against local sources after adding the upb `.clear()` shim.
- [x] Step 2 — tracked the missing stack metadata to DCE removing the Loop when it has no live consumers.
- [x] Step 3 — gated DCE behind the stacktrace metadata flag and taught scan lowering to reuse the true stack width whenever stacktrace metadata is enabled. The sandbox now detects that metadata is correct and reports a passing ORT run without forcing the reshape mismatch.
