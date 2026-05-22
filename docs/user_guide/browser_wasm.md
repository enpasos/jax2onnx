# Browser/WASM Deployment

`jax2onnx` can export models in a browser-friendly profile and validate them
against `onnxruntime-web/wasm`. Use this flow when the deployment target is a
web app, a Node.js WASM runner, or any environment where a single `.onnx` file is
easier to ship than an ONNX model plus external tensor data.

## Export A Web Model

Use `return_mode="file"` together with `export_mode="web"`:

```python
from jax2onnx import to_onnx
from jax2onnx.quickstart import build_quickstart_web_model

model = build_quickstart_web_model()

model_path = to_onnx(
    model,
    [("B", 8)],
    return_mode="file",
    output_path="web_mlp.onnx",
    export_mode="web",
)
```

The Web export profile keeps the ONNX graph semantics unchanged, but serializes
a self-contained `.onnx` artifact and removes stale `.onnx.data` sidecars. That
artifact shape is the safest default for `onnxruntime-web/wasm`, static hosting,
and browser caches.

Symbolic dimensions such as `"B"` remain symbolic in the exported model. At
runtime, feeds still use concrete tensor shapes such as `[2, 8]`.

## Run In The Browser

Install `onnxruntime-web` in the application that will load the model:

```bash
npm install onnxruntime-web
```

Then load the exported `.onnx` file with the WASM backend:

```javascript
import * as ort from "onnxruntime-web/wasm";

ort.env.wasm.numThreads = 1;

const session = await ort.InferenceSession.create("/models/web_mlp.onnx", {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "disabled",
});

const input = new ort.Tensor(
  "float32",
  Float32Array.from([
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875,
    1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
  ]),
  [2, 8],
);

const feeds = { [session.inputNames[0]]: input };
const results = await session.run(feeds);
const output = results[session.outputNames[0]];
```

If the app serves WASM assets from a custom location, configure
`ort.env.wasm.wasmPaths` according to the bundler or CDN layout used by the app.
The `.onnx` file itself should be served as a static binary asset.

## Validate With Node.js Or Chrome

The repository includes a Node.js validation path that runs the same exported
model through Python ONNX Runtime CPU and `onnxruntime-web/wasm`, then compares
the outputs. A second runner launches the same WASM validation inside
Chrome/Chromium via Playwright.

Install the repository's JavaScript dependency first:

```bash
npm install
```

For direct validation in Python:

```python
import numpy as np

from jax2onnx import allclose_onnxruntime_web

passed, message = allclose_onnxruntime_web(
    "web_mlp.onnx",
    [np.arange(16, dtype=np.float32).reshape(2, 8) / np.float32(8.0)],
    rtol=1e-5,
    atol=1e-5,
)

assert passed, message
```

For generated plugin/example tests, enable Web validation with:

```bash
JAX2ONNX_VALIDATE_ONNXRUNTIME_WEB=1 poetry run pytest -q tests/primitives/test_nn.py
```

With that flag, generated tests export with `export_mode="web"`, keep the normal
JAX-vs-Python-ONNX-Runtime CPU check, and add a Web/WASM comparison for the same
model and inputs.

To run the same generated checks in a browser instead of Node.js:

```bash
npx playwright install chromium
JAX2ONNX_VALIDATE_ONNXRUNTIME_WEB=1 \
JAX2ONNX_ONNXRUNTIME_WEB_RUNNER=chrome \
poetry run pytest -q tests/primitives/test_nn.py
```

The Chrome runner uses Playwright Chromium by default. Set
`JAX2ONNX_ONNXRUNTIME_WEB_BROWSER=chrome` when a local Google Chrome install
should be used instead.

## Repository Checks

Run the focused browser/WASM smoke suite explicitly with:

```bash
scripts/run_onnxruntime_web_smoke.sh
```

Run the equivalent Chrome/Chromium smoke suite explicitly with:

```bash
npx playwright install chromium
scripts/run_onnxruntime_web_chrome_smoke.sh
```

The central check runner uses the full pytest suite when Web runtime validation
is requested. This is intentionally much heavier than the smoke scripts: every
generated test exports with `export_mode="web"` and performs the Web/WASM
comparison in addition to the normal CPU check.

```bash
JAX2ONNX_RUN_ONNXRUNTIME_WEB=1 ./scripts/run_all_checks.sh
JAX2ONNX_RUN_ONNXRUNTIME_WEB_CHROME=1 ./scripts/run_all_checks.sh
```

Pull request CI runs only the small Node.js `onnxruntime-web` smoke job. The
full Node.js and Chrome/Chromium Web validation jobs live in the scheduled/manual
nightly workflow and skip themselves when no recent commits exist, so the heavy
browser gate does not consume CI minutes on unchanged code or slow the standard
PR matrix.

## Troubleshooting

- **`Node.js command not found`:** Install Node.js before running
  `allclose_onnxruntime_web(...)` or the smoke script.
- **`onnxruntime-web is not installed`:** Run `npm install` at the repository
  root, or add `onnxruntime-web` to the consuming web application.
- **Chrome runner cannot launch:** Run `npx playwright install chromium`, or set
  `JAX2ONNX_ONNXRUNTIME_WEB_BROWSER=chrome` when Google Chrome is installed
  separately.
- **Missing WASM files in the browser:** Configure `ort.env.wasm.wasmPaths` so
  the runtime can find the files emitted by the bundler or served from the CDN.
- **Shape mismatch at runtime:** The ONNX model may use symbolic axes, but the
  browser feed must still provide a concrete tensor with the expected rank and
  dtype.
- **Unsupported dtype:** The validator supports common ONNX Runtime Web tensor
  types, including float16, float32, float64, integer, boolean, and string
  tensors. Convert unsupported application-side arrays before creating
  `ort.Tensor` feeds.
