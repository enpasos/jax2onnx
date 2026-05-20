# API Reference

`jax2onnx` exposes three top-level entry points: `to_onnx(...)` for export, `@onnx_function` for reusable subgraphs, and `allclose(...)` for JAX-vs-ONNX validation. The experimental `jax2onnx.diagnostics` module provides structured report helpers for exported ONNX models.

## Common Export Flow

```python
from jax2onnx import to_onnx


model_path = to_onnx(
    fn,
    inputs=[("B", 128)],
    return_mode="file",
    output_path="model.onnx",
)
```

Use string dimensions such as `"B"` when you want symbolic dynamic axes. For direct file export, set `return_mode="file"` and provide `output_path`.

## Parameters To Reach For First

- `inputs`: Positional input specs, either concrete arrays, `ShapeDtypeStruct` values, or shape tuples like `("B", 128)`.
- `input_params`: Runtime flags or keyword-like values that should stay model inputs instead of being baked into the export.
- `return_mode`: `"proto"` for an `onnx.ModelProto`, `"ir"` for the intermediate `onnx_ir.Model`, or `"file"` to serialize directly to disk.
- `enable_double_precision`: Temporarily enables x64 export and emits `tensor(double)` where appropriate.
- `inputs_as_nchw` / `outputs_as_nchw`: Adapt the external ONNX interface to NCHW while keeping the traced JAX computation in its original layout.
- `input_names` / `output_names`: Apply stable user-facing names after conversion.

Use [`@onnx_function`](onnx_functions.md) when repeated callables should become
reusable ONNX functions, and `allclose(...)` when you want a quick numerical
check against ONNX Runtime after export.

## Experimental Diagnostics Reports

Use `jax2onnx.diagnostics` when you need a reusable model inventory, runtime
smoke test, or CI gate around an exported ONNX model. Reports include ONNX
checker status, strict shape inference, opset imports, operator and dtype
counts, public input/output metadata, optional ONNX Runtime CPU execution, and
target-specific findings for `ort-cpu`, `ort-web`, and `ort-mobile`.

This API is experimental. Static target findings are not compatibility
guarantees; only checks backed by a concrete runtime execution should be treated
as validated behavior.

```python
import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.diagnostics import (
    analyze_jax_export,
    evaluate_model_report_gate,
    format_model_report_markdown,
)


def add(lhs, rhs):
    return lhs + rhs


analysis = analyze_jax_export(
    add,
    [
        jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        jax.ShapeDtypeStruct(("B", 3), jnp.float32),
    ],
    sample_inputs=(
        np.ones((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    ),
    input_names=["lhs", "rhs"],
    output_names=["sum"],
    targets=("ort-cpu",),
)

print(format_model_report_markdown(analysis.report))
assert evaluate_model_report_gate(analysis.report).passed
```

For existing ONNX models, call `analyze_model(...)` with an `onnx.ModelProto`,
serialized bytes, or a path. Multi-profile exports can use
`analyze_jax_export_profiles(...)` with named `RuntimeProfile` values.

::: jax2onnx.user_interface

::: jax2onnx.diagnostics
