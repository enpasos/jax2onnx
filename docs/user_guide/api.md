# API Reference

`jax2onnx` exposes three public entry points: `to_onnx(...)` for export, `@onnx_function` for reusable subgraphs, and `allclose(...)` for JAX-vs-ONNX validation.

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

Use `@onnx_function` when repeated callables should become reusable ONNX functions, and `allclose(...)` when you want a quick numerical check against ONNX Runtime after export.

::: jax2onnx.user_interface
