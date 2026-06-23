# API Reference

`jax2onnx` exposes four public entry points:

- `to_onnx(...)` for export
- `@onnx_function` for reusable subgraphs
- `allclose(...)` for JAX-vs-ONNX validation
- `allclose_onnxruntime_web(...)` for ONNX Runtime Web/WASM parity checks

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

Use string dimensions such as `"B"` when you want symbolic dynamic axes.

For direct file export, set `return_mode="file"` and provide `output_path`.

For a full export validation checklist, see
[Validation & Deployment Readiness](validation.md).

For browser deployment with [`onnxruntime-web`](https://onnxruntime.ai/docs/tutorials/web/), use the Web export profile:

```python
model_path = to_onnx(
    fn,
    inputs=[("B", 128)],
    return_mode="file",
    output_path="model.web.onnx",
    export_mode="web",
)
```

`export_mode="web"` keeps the ONNX graph semantics unchanged, but serializes a
single self-contained `.onnx` file instead of spilling large initializers into a
`.onnx.data` sidecar. That is the easiest artifact shape to serve to
`onnxruntime-web/wasm`.

## Parameters To Reach For First

- `inputs`: Positional input specs, either concrete arrays, `ShapeDtypeStruct` values, or shape tuples like `("B", 128)`.
- `input_params`: Runtime flags or keyword-like values that should stay model inputs instead of being baked into the export.
- `return_mode`: `"proto"` for an `onnx.ModelProto`, `"ir"` for the intermediate `onnx_ir.Model`, or `"file"` to serialize directly to disk.
- `export_mode`: `"standard"` for normal serialization, or `"web"` for single-file browser/WASM artifacts.
- `enable_double_precision`: Temporarily enables x64 export and emits `tensor(double)` where appropriate.
- `inputs_as_nchw` / `outputs_as_nchw`: Adapt the external ONNX interface to NCHW while keeping the traced JAX computation in its original layout.
- `input_names` / `output_names`: Apply stable user-facing names after conversion.

## Browser/WASM Validation

The generated test harness can optionally validate exported models with
`onnxruntime-web/wasm` in Node.js or Chrome/Chromium:

```bash
npm install
JAX2ONNX_VALIDATE_ONNXRUNTIME_WEB=1 poetry run pytest -q tests/primitives/test_nn.py
```

For the browser runner, add:

```bash
npx playwright install chromium
JAX2ONNX_VALIDATE_ONNXRUNTIME_WEB=1 \
JAX2ONNX_ONNXRUNTIME_WEB_RUNNER=chrome \
poetry run pytest -q tests/primitives/test_nn.py
```

When this flag is enabled, generated tests export with `export_mode="web"`, keep
the existing JAX-vs-Python-ONNX-Runtime CPU check, then compare the same ONNX
model and inputs against `onnxruntime-web/wasm`.

For a smaller local smoke run that covers the Quickstart Web model plus
representative generated LAX/JAX NumPy examples, run the explicit smoke scripts:

```bash
scripts/run_onnxruntime_web_smoke.sh
scripts/run_onnxruntime_web_chrome_smoke.sh
```

When Web runtime validation is requested through the central repository check
runner, it runs the full pytest suite with `export_mode="web"` and the selected
runtime runner:

```bash
JAX2ONNX_RUN_ONNXRUNTIME_WEB=1 ./scripts/run_all_checks.sh
JAX2ONNX_RUN_ONNXRUNTIME_WEB_CHROME=1 ./scripts/run_all_checks.sh
```

For browser loading code, validation helpers, CI usage, and troubleshooting, see
[Browser/WASM Deployment](browser_wasm.md).

Use [`@onnx_function`](onnx_functions.md) when repeated callables should become
reusable ONNX functions, and `allclose(...)` when you want a quick numerical
check against ONNX Runtime after export.

::: jax2onnx.user_interface
