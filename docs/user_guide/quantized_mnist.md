# A quantized MNIST classifier

This optional example trains one small FP32 Linen classifier, prepares its
weights and activations with Qwix, and exports both computations through ordinary
`to_onnx(...)`. A separate NumPy/ONNX Runtime script evaluates the frozen models
without installing JAX, Flax, Qwix or jax2onnx. Training, downloads and complete
held-out evaluation are explicit commands outside the STANDARD test suite.

## Model and quantization

The model is `784 → Dense(128) → ReLU → Dense(10)`, without biases. Preprocessing
preserves pixel order: original `uint8` images become FP32 pixels divided by 255,
then flatten to `(1, 784)` for inference. Logits have FP32 shape `(1, 10)`.

Training uses seed 0 and one NumPy `PCG64(0)` generator. Its initial permutation
splits the 60,000 training images into 55,000 training and 5,000 development
examples; subsequent permutations shuffle each epoch. The final checkpoint
comes from exactly 20 epochs, batch size 128 including the remainder, mean
softmax cross-entropy, and Adam at `1e-3`. Adam uses `b1=0.9`, `b2=0.999`,
`eps=1e-8`, `eps_root=0`, default moment dtype and no Nesterov acceleration.
There is no checkpoint selection or test-guided tuning.

Qwix prepares INT8 weights once using `absmax`. Both contractions use INT8
activations, INT32 accumulation and FP32 scales/results, without zero points.
The example uses public layer-specific `QuantizationRule` settings:
`act_static_scale=False`, `tile_size=None`, and
`PtqProvider(..., disable_jit=True)`. Each fixed symmetric activation range is
`127.5 * 2**k`. The first scale is `1/64`, covering the known input range
`[0, 1]`; the second is the smallest power of two whose 127.5-times bound covers
the largest FP32 post-ReLU activation across all 5,000 development images.
Calibration runs once before test access and freezes with the checkpoint and
recipe. Unseen activations can still clip, so held-out accuracy remains a gate.

Fixed ranges address a numerical portability issue: with dynamic activation
`absmax`, FP32 division immediately before nearest-even rounding can land on
opposite sides of a half-integer in JAX and ORT. Power-of-two scales make this
model's activation scaling portable without changing generic exporter rounding.
Preparation belongs to Qwix; the exporter follows the prepared computation:

```python
model = to_onnx(
    prepared_serving, [jax.ShapeDtypeStruct((1, 784), jnp.float32)], opset=23
)
```

Here `prepared_serving` captures the actual Qwix-prepared parameters. The script
exports fresh serving callables before obtaining their native references.

## Prepare the optional environment and data

Run preparation from the repository root with Python 3.12 and official CPU JAX
wheels. These pins are example-local; they do not change the project's normal
environment. Qwix 0.1.8 is installed from immutable commit
`9ee1079245682ed82e5da00ce855e4daf77a086b`.

```bash
python3.12 -m venv /tmp/mnist-prepare
export VIRTUAL_ENV=/tmp/mnist-prepare
export POETRY_CACHE_DIR=/tmp/mnist-poetry-cache
poetry env info --path
# Verify that the printed path is /tmp/mnist-prepare before installing.
poetry install --no-interaction
/tmp/mnist-prepare/bin/python -m pip install --no-deps \
  -r scripts/requirements-quantized-mnist.txt
/tmp/mnist-prepare/bin/python -m pip check
```

The isolated Poetry cache selects the explicit Python 3.12 environment; the
repository's existing `poetry.lock` fixes its dependencies. Run this setup in a
dedicated shell, then use the explicit interpreter paths below.

Use the canonical archive described by
[Keras MNIST](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data).
The [Keras loader source](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py)
records its distribution URL, checksum and attribution: MNIST by Yann LeCun and
Corinna Cortes, derived from NIST data, under
[Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
Keep that attribution and comply with the licence when redistributing data.
No TensorFlow installation is needed:

```bash
/tmp/mnist-prepare/bin/python - <<'PY'
import hashlib
from pathlib import Path
import urllib.request

path = Path("/tmp/mnist.npz")
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
urllib.request.urlretrieve(url, path)
with path.open("rb") as archive:
    actual = hashlib.file_digest(archive, "sha256").hexdigest()
expected = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
if actual != expected:
    raise SystemExit("MNIST checksum mismatch")
PY
```

Keep datasets, checkpoints and generated ONNX files outside git. Do not inspect
the official test arrays before freezing weights, calibration and export hashes.
Run these stages serially, preserving the work directory if any gate fails:

```bash
/tmp/mnist-prepare/bin/python scripts/quantized_mnist.py train \
  --data /tmp/mnist.npz --workdir /tmp/mnist-run
/tmp/mnist-prepare/bin/python scripts/quantized_mnist.py export \
  --data /tmp/mnist.npz --workdir /tmp/mnist-run
/tmp/mnist-prepare/bin/python scripts/quantized_mnist.py validate \
  --data /tmp/mnist.npz --workdir /tmp/mnist-run
```

The scripts record the split, recipe, checkpoint and artifact hashes. Export
checks ONNX validity, inferred shapes, both stored INT8 kernels and integer
products. Validation produces `native_outputs.npz` containing eager, fresh-JIT
and warmed-JIT references for every official test image. Both ONNX files are
self-contained and need no external tensor files.

## Run independent CPU inference

Create a separate environment containing only NumPy, ORT and their pinned
dependencies. Copy the standalone runner outside the checkout. It checks that
JAX, jaxlib, Flax, Qwix and jax2onnx are neither installed nor importable.

```bash
python3.12 -m venv /tmp/mnist-runtime
/tmp/mnist-runtime/bin/python -m pip install \
  -r scripts/requirements-quantized-mnist-runtime.txt
/tmp/mnist-runtime/bin/python -m pip check
mkdir -p /tmp/mnist-deploy
cp scripts/run_quantized_mnist.py /tmp/mnist-deploy/
runtime=/tmp/mnist-runtime/bin/python
runner=/tmp/mnist-deploy/run_quantized_mnist.py
run=/tmp/mnist-run
data=/tmp/mnist.npz
env -u PYTHONPATH "$runtime" -I "$runner" --model "$run/fp32.onnx" \
  --variant fp32 --data "$data" --native-outputs "$run/native_outputs.npz" \
  --optimization off --output "$run/fp32-off.json" --logits "$run/fp32-off.npz"
env -u PYTHONPATH "$runtime" -I "$runner" --model "$run/fp32.onnx" \
  --variant fp32 --data "$data" --native-outputs "$run/native_outputs.npz" \
  --optimization on --output "$run/fp32-on.json" --logits "$run/fp32-on.npz"
env -u PYTHONPATH "$runtime" -I "$runner" --model "$run/w8a8.onnx" \
  --variant w8a8 --data "$data" --native-outputs "$run/native_outputs.npz" \
  --optimization off --output "$run/w8a8-off.json" --logits "$run/w8a8-off.npz"
env -u PYTHONPATH "$runtime" -I "$runner" --model "$run/w8a8.onnx" \
  --variant w8a8 --data "$data" --native-outputs "$run/native_outputs.npz" \
  --optimization on --output "$run/w8a8-on.json" --logits "$run/w8a8-on.npz"
```

Each command is a fresh process evaluating all 10,000 images at batch size 1,
with `CPUExecutionProvider`, sequential execution and one intra/inter-op thread.
Optimization `off` means `ORT_DISABLE_ALL`; `on` means `ORT_ENABLE_ALL`.
JSON records artifact bytes/SHA256, errors, hardware, versions, import/session/
inference RSS and peak RSS in bytes. Latencies separate imports, session creation,
first inference, and 1,000 timed calls after 100 warmups on the first test image.
Session timing excludes file reads; inference timing excludes preprocessing.
Native reference loading occurs after runtime memory measurements.

## Acceptance and results

The fixed example reached the following results on all 10,000 official test
images, using the final epoch-20 checkpoint and the prescribed development-only
calibration:

| Measurement | FP32 | W8A8 |
| --- | ---: | ---: |
| Correct predictions | 9,796 | 9,793 |
| Accuracy | 97.96% | 97.93% |
| Self-contained ONNX bytes | 407,044 | 106,082 |
| Warmed median inference, ORT optimizations enabled | 0.00656 ms | 0.05261 ms |
| Process peak RSS through evaluation | 72.6 MiB | 72.3 MiB |

The artifact is **73.9% smaller**, with three additional classification errors
(0.03 percentage points). W8A8 logits match JAX eager, fresh-JIT and warmed-JIT
and both ORT optimization settings exactly on this test set. FP32 predictions
also agree, with a maximum native/ORT logit difference of `1.90735e-5`.

Measurements used Python 3.12.9, NumPy 2.4.6 and ORT 1.31.1 on an AMD Ryzen 9
7950X3D under Linux/WSL2, CPU provider, one thread and batch size 1. Each model
and optimization setting ran in a separate process. Latency uses 100 warmups
and 1,000 repeated calls on test image 0, excluding preprocessing. The small
INT8 graph is about eight times slower here; memory use is similar. Peak RSS
uses Linux `ru_maxrss`; the JSON also retains separate `/proc` RSS snapshots.
Small differences between those counters preclude a memory-improvement claim.
File-size
savings do not imply lower latency or runtime memory on other models or hardware.

The frozen checkpoint SHA-256 is
`8d0802292268f6f051714720519666dffdba6c5be84788d98dc46be1977d559d`.
Its hidden activation scale is `1/8`, with fixed bound `15.9375`, selected from
the development maximum `15.180764198303223`. Generated datasets, weights,
reference arrays and ONNX files remain local and are reproduced by the commands
above; they are not distributed with the repository.

The acceptance criteria remain: FP32 native and ORT each reach 97% accuracy. W8A8 may add at
most 50 errors (0.5 percentage points) against its respective FP32 reference.
All W8A8 eager/JIT/ORT references, including both ORT optimization settings, must
agree over the full test set at `rtol=1e-6`, `atol=1e-6`, with identical class
predictions, finite FP32 logits and exact shapes. FP32 logit differences are
reported separately. Zero-valued inputs and outputs are valid.

Compare the four run records against `export.json`, `native_outputs.npz` and the
actual model files, using the same isolated runner:

```bash
env -u PYTHONPATH "$runtime" -I "$runner" \
  --compare-results "$run" --output "$run/runtime-summary.json"
```

This checks both ORT settings, FP32/W8A8 error differences and artifact bytes:
W8A8 must occupy at most 40% of FP32 bytes. The summary links the frozen export
record; training provenance and ONNX graph/storage qualification remain
preparation-stage checks. Memory and speed have no improvement threshold;
report measured results even when quantization makes them worse.
