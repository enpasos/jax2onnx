## Issue trigger

- Source: https://github.com/enpasos/jax2onnx/issues/109
- Excerpt: “I am curious if there is a way to preserve the source code call stack into the onnx graph to help debugging? If the information is available, we can consider using the `pkg.jax2onnx.stacktrace` metadata field to store this in every node.”

## Current interpretation

- Goal is to enrich exported `onnx_ir` graphs with provenance so downstream debugging (typically after conversion succeeds but runtime fails elsewhere) can trace nodes back to their originating JAX call sites.
- Proposed metadata key follows the existing `metadata_props` namespace pattern (`pkg.jax2onnx.*`), but stack traces should be opt-in to avoid constant overhead/noise.
- The feature targets post-export analysis; it does not alter failure handling during conversion.

## Implementation sketch

1. Add a debug toggle (context manager, CLI flag, or config) that enables call-stack capture during lowering.
2. When the flag is active, collect the Python stack (`traceback.extract_stack()` stripped of framework internals) at each node creation point and serialize it (e.g., JSON list or newline-joined string).
3. Store the user-facing metadata (`pkg.jax2onnx.callsite` = `function:line`, `pkg.jax2onnx.plugin` = `Plugin.lower:line` pointing at the builder call site), with optional full traces behind a debug flag.
4. Update `expect_graph` fixtures/tests to mirror the new metadata when the flag is engaged; document the workflow in the relevant dev guide.

## Status

- Awaiting clarification from the issue author on desired workflow, granularity, and prior experience with stacktrace metadata.
- Prototype implemented behind `JAX2ONNX_ENABLE_STACKTRACE_METADATA`; captures succinct provenance via `pkg.jax2onnx.callsite` and `pkg.jax2onnx.plugin`, with optional full traces gated by `JAX2ONNX_STACKTRACE_DETAIL=full`.
- End-to-end export verified via `to_onnx(..., return_mode="file")`; `sandbox_stacktrace_sample.onnx` now serves as a reference artifact with both metadata fields on each node.

## Sample artifact

```bash
PYTHONWARNINGS=ignore JAX2ONNX_ENABLE_STACKTRACE_METADATA=1 python - <<'PY'
import jax
import jax.numpy as jnp
from jax2onnx import to_onnx

jax.config.update("jax_traceback_filtering", "off")

def wide_fn(x):
    a = jnp.sin(x)   # wide_fn.py:8
    b = jnp.cos(x)   # wide_fn.py:9
    c = jnp.tanh(x)  # wide_fn.py:10
    d = jnp.exp(x)   # wide_fn.py:11
    return a + b * c + d  # wide_fn.py:12

def fn(x):
    return jnp.sin(x) * 2.0

to_onnx(
    wide_fn,
    [jax.ShapeDtypeStruct((2,), jnp.float32)],
    return_mode="file",
    output_path="sandbox_stacktrace_sample.onnx",
)
PY
```

- Inspect with `onnx.load("sandbox_stacktrace_sample.onnx")` (or `wide_stacktrace_sample.onnx`) to confirm each node exposes the reduced metadata `pkg.jax2onnx.callsite`/`pkg.jax2onnx.plugin`; enable `JAX2ONNX_STACKTRACE_DETAIL=full` if you also need the legacy trace keys.
