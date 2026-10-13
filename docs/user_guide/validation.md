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

For reinforcement-learning actors, also validate the policy-specific public
interface and action contract in [RL Policy Deployment](rl_policy_deployment.md).

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

## Decoder Normalization Benchmark

To compare `normalization_mode="auto"` and `"prefer_native"` on the built-in tiny
NNX decoder, run from the repository root on an otherwise idle machine:

```bash
JAX_PLATFORMS=cpu poetry run python scripts/benchmark_decoder_normalization.py \
  --warmup 100 --iterations 5000 --repeats 7 \
  --output /tmp/decoder-normalization.json
```

The benchmark uses identical deterministic weights and inputs for all four
combinations of static/symbolic export and normalization mode. The default is
float32, width 16, one CPU thread, and all runtime graph optimizations enabled.
The reported warmed latency is the median of the per-repeat mean inference
times, including Python call overhead. Session initialization and the first run
are reported separately as single samples; export, JAX reference execution, and
warmup are excluded from warmed latency. Optimized graphs are collected in a
separate session so saving them does not inflate initialization time.

To measure native FP16 normalization on a larger decoder, use:

```bash
JAX_PLATFORMS=cpu poetry run python scripts/benchmark_decoder_normalization.py \
  --dtype float16 --embed-dim 128 --num-heads 4 --ff-dim 256 \
  --intra-op-threads 2 --warmup 100 --iterations 1000 --repeats 7 \
  --output /tmp/decoder-normalization-fp16.json
```

For an isolated RMSNorm comparison, select `--workload rmsnorm --dtype float16
--embed-dim 1024`. Decoder widths must be divisible by `--num-heads`.
`--intra-op-threads 0` selects ONNX Runtime's default thread count. Compare
initialization, first-run, and warmed latency independently when changing
thread counts.

ONNX Runtime 1.31 accelerates native FP16 LayerNorm/RMSNorm on eligible x86
CPUs and normalized widths of at least 16. `prefer_native` exposes those
operators; `auto` retains the explicit normalization policy. Actual model
speedups depend on the full graph and hardware. See the
[upstream kernel change](https://github.com/microsoft/onnxruntime/pull/32715).

The JSON report records dependency versions, CPU, source revision, input and
output dtypes, errors against JAX, and exported/optimized operator counts.
Parameters and inputs use the selected dtype. JAX parity uses `rtol=atol=1e-4`
for float32 and `2e-3` for float16; these benchmark tolerances are separate from
the fixed float32 LayerNorm acceptance gates. Error calculations use float64
without implying a float64 reference. Generated ONNX graphs are temporary.
Use `--help` to adjust input sizes and measurement repetitions.

A sample run on an AMD Ryzen 9 7950X3D under Linux/WSL2, Python 3.12.9,
ONNX Runtime 1.30.0, JAX 0.11.2, Flax 0.12.10, and NumPy 2.5.3 produced:

| Export shape | `auto` latency | `prefer_native` latency | Latency reduction |
| --- | ---: | ---: | ---: |
| Static | 0.0443 ms | 0.0343 ms | 22.6% |
| Symbolic | 0.0514 ms | 0.0410 ms | 20.2% |

This run used the command above with the default inputs `(2, 8, 16)` and
`(2, 4, 16)`, model seed 0, input seed 1, and opset 23. All four cases had a
maximum absolute error of `7.15e-7` against the same JAX float32 output.
ONNX Runtime fused the native exports into three `SkipLayerNormalization`
nodes. These measurements describe one small model and input sample; the
benchmark checks JAX parity with `rtol=atol=1e-4` and does not establish an
independent float64 accuracy guarantee. Runtime and error measurements on the
actual deployment model should guide use of `prefer_native`. The stricter
[LayerNorm accuracy gates](known_limitations.md#scope-of-the-fixed-layernorm-limits)
continue to determine the default `auto` policy.

## ConvTranspose and BatchNorm Fusion

ONNX Runtime 1.31 can fold inference BatchNorm into an immediately preceding
ConvTranspose when weights and normalization parameters are constant and the
convolution output has no other consumers. The exporter attaches eligible
constant float32/float64 channel biases to ConvTranspose so a separate bias Add
does not obstruct this fusion. It preserves additional outputs, existing bias
inputs, and required padding. See the
[upstream fusion](https://github.com/microsoft/onnxruntime/pull/29542).

For example, export a frozen NNX upsampling block and inspect its optimized graph:

```python
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from flax import nnx
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
from jax2onnx import to_onnx


class UpsamplingBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv = nnx.ConvTranspose(
            4, 8, (3, 3), strides=(2, 2), padding="VALID", rngs=rngs
        )
        self.conv.bias = nnx.Param(jnp.linspace(-0.2, 0.2, 8, dtype=jnp.float32))
        self.norm = nnx.BatchNorm(8, use_running_average=True, rngs=rngs)
        self.norm.scale = nnx.Param(jnp.linspace(0.75, 1.25, 8, dtype=jnp.float32))

    def __call__(self, x):
        return self.norm(self.conv(x))


block = UpsamplingBlock(nnx.Rngs(0))
x = np.random.default_rng(1).normal(size=(1, 3, 3, 4)).astype(np.float32)
model = to_onnx(block, [x])

with TemporaryDirectory() as directory:
    optimized_path = Path(directory) / "optimized.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(optimized_path)
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {session.get_inputs()[0].name: x})[0]
    np.testing.assert_allclose(actual, np.asarray(block(x)), rtol=1e-5, atol=1e-5)
    print("Exported:", Counter(node.op_type for node in model.graph.node))
    optimized = onnx.load(optimized_path)
    print("Optimized:", Counter(node.op_type for node in optimized.graph.node))
```

For this eligible block on ORT 1.31, BatchNormalization disappears from the
optimized graph. These temporary optimized graphs describe the measured runtime
and hardware; distribute the standard exported model and let the deployment
runtime optimize it for its target.

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

Both runners above use the WASM execution provider, including Chromium. They
do not validate WebGPU execution.

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
