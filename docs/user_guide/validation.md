# Validation & Deployment Readiness

Use this page as a short checklist before sharing an exported ONNX model or
opening an issue about runtime behavior.

`jax2onnx` exports JAX-derived callables to ONNX. For deployment confidence,
validate four things:

1. The exported artifact is structurally valid ONNX.
2. ONNX shape inference can process the model.
3. ONNX Runtime can load and execute the model.
4. ONNX Runtime produces numerically close outputs compared with the original JAX callable.

## Minimal Validation Workflow

```python
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort

from jax2onnx import allclose, to_onnx


def model(x):
    return jnp.sin(x) + 0.5 * x


model_path = Path("model.onnx")

to_onnx(
    model,
    inputs=[("B", 16)],
    return_mode="file",
    output_path=str(model_path),
)

# 1. ONNX structural validation
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

# 2. Shape inference smoke check
onnx.shape_inference.infer_shapes(onnx_model)

# 3. ONNX Runtime load and execution smoke check
session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
session.run(None, {input_name: np.zeros((2, 16), dtype=np.float32)})

# 4. Numerical parity against JAX
is_match, message = allclose(
    model,
    str(model_path),
    inputs=[np.zeros((2, 16), dtype=np.float32)],
    rtol=1e-5,
    atol=1e-5,
)

assert is_match, message
```

The zero-valued arrays keep the example minimal. For deployment decisions,
repeat the runtime and numerical checks with representative inputs and
tolerances appropriate for the model and dtype.

## Dynamic Dimensions

Use symbolic dimensions such as `"B"` for export when the model should accept
dynamic batch sizes:

```python
to_onnx(
    model,
    inputs=[("B", 16)],
    return_mode="file",
    output_path="model.onnx",
)
```

For validation, pass concrete arrays:

```python
allclose(
    model,
    "model.onnx",
    inputs=[np.zeros((4, 16), dtype=np.float32)],
)
```

## Browser/WASM Validation

For browser deployment, export with `export_mode="web"`:

```python
to_onnx(
    model,
    inputs=[("B", 16)],
    return_mode="file",
    output_path="model.web.onnx",
    export_mode="web",
)
```

Validate the exported Web artifact directly with `allclose_onnxruntime_web(...)`:

```bash
npm install
```

```python
import numpy as np

from jax2onnx import allclose_onnxruntime_web


is_match, message = allclose_onnxruntime_web(
    "model.web.onnx",
    inputs=[np.zeros((2, 16), dtype=np.float32)],
    rtol=1e-5,
    atol=1e-5,
)

assert is_match, message
```

The repository smoke scripts validate representative built-in exports. Use them
to check the local development environment, not as a substitute for validating a
specific deployment model:

```bash
scripts/run_onnxruntime_web_smoke.sh
npx playwright install chromium
scripts/run_onnxruntime_web_chrome_smoke.sh
```

For full-suite validation, use:

```bash
JAX2ONNX_RUN_ONNXRUNTIME_WEB=1 ./scripts/run_all_checks.sh
JAX2ONNX_RUN_ONNXRUNTIME_WEB_CHROME=1 ./scripts/run_all_checks.sh
```

## What To Include In Bug Reports

When reporting an export or runtime issue, include:

- the `jax2onnx` version,
- the JAX, ONNX, and ONNX Runtime versions,
- the target opset if explicitly configured,
- the minimal JAX callable or module,
- the `to_onnx(...)` call,
- the validation failure message,
- whether the failure is from ONNX checker, shape inference, ONNX Runtime load,
  ONNX Runtime execution, or numerical parity.

See [Dependencies](../about/dependencies.md) for the currently documented
dependency stack.
