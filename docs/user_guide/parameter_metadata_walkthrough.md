# Parameter Metadata: A First Export

This walkthrough adds labels to a model's weights, exports the model, and reads
the labels back. You will also inspect the result in Netron. No ONNX internals
are needed to follow the example.

The example is `shared-linear-general-fp32`: one Flax NNX linear layer with
512 inputs and 512 outputs, used twice. Both calls use the **same weights**.
FP32 means that each number uses 32 bits, or four bytes.

## What Are We Adding?

An ONNX file contains the computation and its stored arrays. ONNX calls a stored
array an **initializer**. Without extra information, a weight might be named
`kernel2d_0`, which says little about where it came from.

We will attach two labels to the weight matrix:

| Label | Value | Meaning chosen by this example |
| --- | --- | --- |
| `source_path` | `shared.kernel` | This array came from our shared layer's kernel. |
| `trainable` | `true` | A consumer may select this parameter for training. |

These labels are called **metadata**. They travel inside the ONNX file and do
not change its calculation. Your application chooses the keys and their
meaning; jax2onnx preserves them as strings. Writing `"true"` does not turn on
training, and writing `"false"` does not make ONNX Runtime freeze a weight.
A tool that reads these labels must implement that policy itself.

## 1. Prepare the Model and Its Labels

Use a checkout containing the `initializer_metadata` feature, planned for
jax2onnx 0.19.0. The released 0.18.0 API does not include this keyword. In that
checkout, install the dependencies with:

```bash
poetry install
```

See [Getting Started](getting_started.md) for the general setup. This example
uses Flax NNX, JAX, ONNX, and ONNX Runtime from the project environment. Netron
is only needed for the optional visual inspection later.

Before exporting a real model, you need to decide:

1. Which arrays you want to label and how to obtain them from your framework.
2. Which source names and training policy your application will use.
3. The input shape and dtype expected by the model.

Here, the array is `layer.kernel.get_value()`. Keep that original JAX array and
the layer alive during export. Use the array itself, rather than its `nnx.Param`
wrapper or a NumPy copy. Two arrays can contain identical numbers and still be
different parameters.

## 2. Export and Read the Result

Save the following complete example as `export_shared_linear.py` in the checkout
and run `poetry run python export_shared_linear.py`:

```python
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort

from jax2onnx import to_onnx


layer = nnx.LinearGeneral(
    512,
    512,
    dtype=jnp.float32,
    param_dtype=jnp.float32,
    rngs=nnx.Rngs(0),
)
kernel = layer.kernel.get_value()


def apply(features):
    # Call the same layer twice, so both calls share its weights.
    return layer(layer(features))


def metadata(source):
    # `is` checks whether this is the original array object.
    if source is kernel:
        return {"source_path": "shared.kernel", "trainable": "true"}
    return None


path = Path("artifacts/initializer-metadata/shared-linear-general-fp32.onnx")
path.parent.mkdir(parents=True, exist_ok=True)

to_onnx(
    apply,
    inputs=[jax.ShapeDtypeStruct((1, 512), jnp.float32)],
    input_names=["features"],
    initializer_metadata=metadata,
    opset=18,
    return_mode="file",
    output_path=path,
)

# Read the saved file, so we also check that the labels survived saving.
model = onnx.load_model(path)
onnx.checker.check_model(model, full_check=True)
tagged = {}
for tensor in model.graph.initializer:
    labels = {item.key: item.value for item in tensor.metadata_props}
    if labels:
        tagged[tensor.name] = labels
        print(tensor.name, list(tensor.dims), labels)

# These expectations belong to this particular example.
assert len(tagged) == 1
kernel_name = next(iter(tagged))
assert tagged[kernel_name] == {
    "source_path": "shared.kernel",
    "trainable": "true",
}
gemms = [node for node in model.graph.node if node.op_type == "Gemm"]
assert len(gemms) == 2
assert all(node.input[1] == kernel_name for node in gemms)

# Run the exported model and compare its numbers with JAX.
features = np.linspace(-0.5, 0.5, 512, dtype=np.float32).reshape(1, 512)
session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
actual = session.run(None, {"features": features})[0]
expected = np.asarray(apply(jnp.asarray(features)))
np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
print(f"Saved and checked: {path}")
```

`metadata` is a **callback**: a function jax2onnx calls while capturing model
constants. For a source you recognize, return its labels. Return `None` for
anything else. Here we label only the kernel; the layer's bias remains
unannotated, which does not mean frozen. Keys and values must be strings.

The callback may be called more than once for one array. Return the same labels
each time and do not change the model during export. The identity check happens
only in Python; no Python object ID is written to the file.

`ShapeDtypeStruct((1, 512), jnp.float32)` describes one row of 512 input numbers.
It tells the exporter what to trace without providing actual input data. The
later `features` array supplies real values for the runtime check.

## 3. What Do You Get?

The script writes `artifacts/initializer-metadata/shared-linear-general-fp32.onnx`.
The printed metadata will look like this; the generated tensor name can vary:

```text
kernel2d_0 [512, 512] {'source_path': 'shared.kernel', 'trainable': 'true'}
```

For this example, the graph has two `Gemm` operations, ONNX's matrix multiply
with bias. Both refer to the same stored kernel. That kernel occupies
`512 × 512 × 4 = 1,048,576` bytes: **1 MiB of weight data**, instead of two
separate copies. The file also contains the bias and graph information.
Other models can need several stored representations of one source, for
example in different scopes or shapes; one copy per model is not a general
API guarantee.

A consumer can now find `shared.kernel` from its label instead of guessing
from `kernel2d_0`, its position in a list, or its numerical contents. The
consumer can also read your declared training policy. Reconstructing a JAX
module, applying optimizer masks, and keeping multiple representations in sync
remain that consumer's work. jax2onnx does not provide an ONNX-to-JAX importer.

No extra metadata file is required to use or share this example: the labels
are already inside the `.onnx` file. The Python reader above is the direct way
to check them. `read_parameter_metadata` belongs to a
[different annotation API](parameter_metadata.md#annotate-an-existing-artifact)
and does not read these tensor labels.

## 4. Inspect the Model in Netron

Open the `.onnx` file in [Netron](https://netron.app/). Click either `Gemm` node
and expand its inputs. `B` is the kernel and `C` is the bias. Check the other
`Gemm`: its `B` input has the same name, showing that both operations use the
same stored kernel. The tensor button beside that input opens its contents.

Tensor metadata display depends on your Netron version. The
[ONNX reader](https://github.com/lutzroeder/netron/blob/main/source/onnx.js)
checked for this walkthrough does not expose initializer
`TensorProto.metadata_props` in the sidebar. Seeing no labels there does not
mean they are missing from the file. The Python output above reads the actual
ONNX properties.

For Netron versions supporting
[metadata attachments](https://github.com/lutzroeder/netron/blob/main/source/view.js),
you can create an optional display file. Append this to the example and rerun
it:

```python
import json


attachment = {
    "signature": "netron:attachment",
    "metadata": [
        {
            "kind": "tensor",
            "target": tensor.name,
            "name": item.key,
            "value": item.value,
            "type": "string",
        }
        for tensor in model.graph.initializer
        for item in tensor.metadata_props
    ],
}
attachment_path = path.with_suffix(".netron.json")
attachment_path.write_text(json.dumps(attachment, indent=2), encoding="utf-8")
print(f"Optional Netron attachment: {attachment_path}")
```

1. Open `shared-linear-general-fp32.onnx` in Netron first.
2. With that model open, drag `shared-linear-general-fp32.netron.json` into the
   same Netron window. Compatible versions recognize it as an attachment.
3. Select a `Gemm`, then open the tensor view for input `B`. Its **Metadata**
   section shows `source_path` and `trainable`.

The JSON is only a display aid copied from the saved ONNX properties. It is
neither a second model nor an input to jax2onnx. Regenerate it after changing
the export, because it refers to that file's generated tensor names. If your
Netron version does not support attachments, use the Python reader.

## 5. What Changes for Your Own Model?

Replace `layer` and `apply` with your model, adjust the input specifications,
and build the callback from your framework's arrays and labels. For larger
models, a framework adapter can maintain a lookup by array identity. If one
array has several source paths, decide how your labels describe that sharing.
Do not match parameters by their numbers: two independent zero biases look
identical.

The promise is: **once jax2onnx accepts labels for a captured source, a
successful export preserves them on the correct stored representations. If it
cannot verify that, export fails.** A preservation failure raises
`InitializerMetadataError`; keep the error and reduce the model to a small
reproduction rather than treating a failed export as an annotated result.
The [contract](initializer_metadata_contract.md) defines the exact boundary,
supported storage, and error cases.

That boundary starts after JAX tracing. JAX can remove or transform arrays
before jax2onnx sees them. A callback returning `None` for everything can still
produce a valid, unannotated model. If you require particular parameters,
compare the expected labels with the saved artifact, as this example does.
Models with control flow or ONNX Functions also need readers that inspect
nested scopes and tensors in `Constant` nodes, beyond the root initializer
loop shown here.

Accepted annotations currently use a more conservative optimization path and
require temporary validation memory. Some exports may be larger, and an
unsupported transformation may fail. Subsequent rewrites by other tools need
their own preservation checks. See the
[technical guide](parameter_metadata.md#captures-scopes-and-transformations)
for these details and the distinction between metadata and training semantics.
