# State Inputs Walkthrough

This page follows one small model through everything that
[Model State as Inputs](state_inputs.md) makes possible: export it with named
state, save and reload the file, run it with the saved defaults and with
replacement arrays, read the labels back, rebuild the Python model from the
file, take an optimizer step and save the new defaults.

The model is a small Equinox attention block with rotary position embeddings
(RoPE). It illustrates the sharing and freeze-policy questions in
[discussion #290](https://github.com/enpasos/jax2onnx/discussions/290); it is not
an adapter for arbitrary Equimo, ViT or DINO models. Its code is in
`scripts/state_inputs_rope.py` in the repository. Only `StateInput` and
`to_onnx` belong to the installed `jax2onnx` package; everything else on this
page is example code to adapt.

## Try the complete example first

Use a checkout of the repository. From its root, prepare the normal
development environment and run:

```bash
poetry install
poetry run python -m scripts.state_inputs_rope \
  --output-dir /tmp/state-inputs-rope
```

The output directory contains:

| File | What it contains |
| --- | --- |
| `attention.onnx` | The inference graph, four named state inputs, their default arrays and labels |
| `report.json` | Versions, the opset, numerical comparisons and the result of the freezing check |

The command saves and reloads the model, compares ONNX Runtime with JAX,
replaces all four arrays and compares the result with a Python model rebuilt
from the same arrays, and takes one AdamW step. Both arrays labelled frozen
must remain unchanged.

To store the tensor payloads in a separate file, use a fresh output directory:

```bash
poetry run python -m scripts.state_inputs_rope \
  --output-dir /tmp/state-inputs-rope-external --external-data
```

This also writes `attention.onnx.data`. Keep that file beside `attention.onnx`
when copying or opening the model. Generated files stay outside git.

## The model and its state

The example declares four arrays:

| Input name | Fields in the Python model | Shape | Label |
| --- | --- | --- | --- |
| `shared_weight` | `shared_weight` | `(4, 4)` | trainable |
| `independent_weight` | `independent_weight` | `(4, 4)` | trainable |
| `rope_rotations` | `rope.rotations` | `(3, 2, 2)` | frozen |
| `shared_frozen` | `q_projection.matrix` and `k_projection.matrix` | `(4, 4)` | frozen |

State follows array objects, not values. Both frozen projections hold the same
array object, so one entry covers both fields and they become one input.
`independent_weight` starts as a copy of `shared_weight`: the values are equal,
but it is a separate object and stays a separate input.

`shared_weight` is used in four places: a direct product, a reshaped view and
two calls of a free ONNX function that receives the weight as an argument. All
four read the one input.

## Walkthrough

The following Python blocks form one sequential session. Run them from the
repository root in `poetry run python`.

### 1. Prepare the model and data

```python
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx import StateInput, to_onnx
from scripts.state_inputs_rope import (
    SmallAttention,
    adamw_step,
    read_state,
    reconstruct,
    runtime,
    save_reload,
    state_inputs,
)

model = SmallAttention(key=jax.random.key(0))
x = jnp.linspace(-0.8, 0.9, 12, dtype=jnp.float32).reshape(3, 4)
inputs = [jax.ShapeDtypeStruct(x.shape, x.dtype)]

# The ordinary export has just the data input.
ordinary = to_onnx(model, inputs=inputs, input_names=["x"])
print([value.name for value in ordinary.graph.input])  # ['x']
```

`x` contains three tokens with four features each. The model's weights and
buffers are FP32 arrays.

### 2. Declare the state

```python
state = state_inputs(model)
for entry in state:
    print(entry.name, entry.array.shape, dict(entry.metadata))
```

`state_inputs` is this example's own function. It returns one `StateInput` per
array object, in the order of the table above. This is the entry for the RoPE
buffer:

```python
rope_entry = StateInput(
    model.rope.rotations,
    "rope_rotations",
    metadata={
        "pkg.example.path": "rope.rotations",
        "pkg.example.trainable": "false",
    },
)
```

`state` already contains this entry. An array object can be declared only
once, so do not append it again.

The labels are this example's own convention; jax2onnx stores them and does
not interpret them. The combined path `q_projection.matrix;k_projection.matrix`
names both fields of the tied array.

### 3. Export

```python
proto = to_onnx(model, inputs=inputs, input_names=["x"], state=state)
print([value.name for value in proto.graph.input])
# ['x', 'shared_weight', 'independent_weight', 'rope_rotations', 'shared_frozen']
```

Pass the model itself. jax2onnx finds the declared arrays among its pytree
leaves and traces them as inputs, so no wrapper function is needed. A
`lambda x: model(x)` would hide the arrays in a closure and raise
`StateCaptureError`.

The computation is the same as in the ordinary export. For this model both
files contain the same operators and the same four weight tensors, and ONNX
Runtime returns the same outputs; a regression test compares them. Only the
way the weights enter the graph differs:

| Ordinary export | With `state=` |
| --- | --- |
| Initializer with a generated name, for example `const_0: float[4, 4]` | Graph input `shared_weight: float[4, 4]` with a same-named initializer as its default |
| No labels | `pkg.example.path=shared_weight` and `pkg.example.trainable=true` on that initializer |
| Uses read the constant, for example `project(x, const_0)` | Uses read the input, for example `project(x, shared_weight)` |

### 4. Save and reload the file

```python
output_dir = Path("/tmp/state-inputs-walkthrough")
output_dir.mkdir(parents=True, exist_ok=True)
path = output_dir / "attention.onnx"

proto = save_reload(proto, path, external_data=False)
```

`save_reload` writes the file with `onnx.save_model`, reloads it, runs the ONNX
checker and compares every state array and its labels with the model in
memory. To write a file without these checks,
`to_onnx(..., return_mode="file", output_path=path)` is enough.

The check runs after writing. If it raises, do not use the files it left
behind; retry in a fresh directory.

### 5. Run inference with the defaults, then replace one weight

```python
session = runtime(proto)
feed = {"x": np.asarray(x)}

default_output = session.run(None, feed)[0]
np.testing.assert_allclose(default_output, model(x), rtol=2e-5, atol=2e-6)

print([item.name for item in session.get_inputs()])  # ['x']
print([item.name for item in session.get_overridable_initializers()])
# ['shared_weight', 'independent_weight', 'rope_rotations', 'shared_frozen']

replacement = np.asarray(model.shared_weight) + np.float32(0.2)
changed_output = session.run(None, {**feed, "shared_weight": replacement})[0]
assert not np.allclose(changed_output, default_output)
```

A replacement array needs the declared shape and dtype. The other three inputs
keep their defaults. Replacing `shared_weight` changes all four of its uses; it
does not change `independent_weight`.

An override affects this one call. It does not edit the file, change the
Python model, or become the default for later calls. Running again with only
`feed` uses the saved defaults.

`runtime` lowers the ONNX Runtime log level. With default logging, ONNX Runtime
warns for each state input that the initializer "appears in graph inputs and
will not be treated as constant value/weight". That is expected for
overridable defaults. Do not remove the inputs to silence the warning; that
removes the override interface.

### 6. Read the labels and rebuild the Python model

```python
stored = read_state(proto)
for entry in stored:
    print(entry.name, entry.metadata)

rebuilt = reconstruct(model, stored)
np.testing.assert_allclose(rebuilt(x), model(x), rtol=2e-5, atol=2e-6)
```

`read_state` reads the graph inputs that have an initializer. The labels are
stored on each initializer as `TensorProto.metadata_props`. The RoPE
initializer, for example, carries `pkg.example.path=rope.rotations` and
`pkg.example.trainable=false`.

`reconstruct` uses the four path labels to place the arrays into the given
Python model. Every path must appear exactly once; missing, duplicate or
unknown labels raise an error. The combined frozen path puts one array object
into both projections, so they stay tied.

The Python architecture is still required. The file provides state and an
inference graph; this reader is not a general ONNX-to-JAX importer.

### 7. Take a training step and save new defaults

```python
updated, loss = adamw_step(model, stored, x, jnp.zeros_like(x))
trained = reconstruct(model, updated)
print(loss)

trained_proto = to_onnx(
    trained, inputs=inputs, input_names=["x"], state=state_inputs(trained)
)
trained_proto = save_reload(
    trained_proto, output_dir / "attention-trained.onnx", external_data=False
)
np.testing.assert_allclose(
    runtime(trained_proto).run(None, feed)[0], trained(x), rtol=2e-5, atol=2e-6
)
```

Training runs on the Python model code with the arrays from the file. The
example reads its own labels and gives AdamW only the two trainable arrays.
Both frozen arrays stay outside the optimizer, so weight decay cannot change
them. Exporting the trained model writes its values as the defaults of a
second file.

## Open the result in Netron

Open `/tmp/state-inputs-walkthrough/attention.onnx` in Netron. Find
`shared_weight` and follow its connections to the operations and function
calls that read it. `independent_weight` is a different value although its
data is initially equal.

Netron shows tensor properties in the details of an initializer if your
version supports them. Otherwise read them directly:

```python
import onnx

saved = onnx.load(path)
state_names = {value.name for value in saved.graph.input}
for tensor in saved.graph.initializer:
    if tensor.name in state_names:
        print(tensor.name, {p.key: p.value for p in tensor.metadata_props})
```

Netron displays the file on disk, not an override from an earlier run. Open
`attention-trained.onnx` to see the newly saved defaults.

## Why not label an ordinary export afterwards?

An ordinary export does not record which model field a weight came from; its
initializer names are generated. Values cannot identify the field either.
Here, `shared_weight` and `independent_weight` start with identical values,
and freshly initialized models typically contain identical LayerNorm scales.
Some weights are also stored in a derived form, for example expanded by
`jnp.broadcast_to` to a fixed shape. Declaring the arrays before the export
records the mapping by construction.

## What this example leaves to you

- `state_inputs`, `read_state`, `reconstruct`, `save_reload` and `adamw_step`
  know this model's four inputs and its path labels. Adapt them for your own
  model.
- Labels do not make an array trainable or frozen. The example's Python code
  interprets `pkg.example.trainable`; jax2onnx and ONNX Runtime do not.
- The file does not carry gradient rules. `stop_gradient`, custom JVP and VJP
  rules, optimizer state and the training modes of Dropout or BatchNorm are
  not recovered from it.
- ONNX Runtime does not fold overridable initializers, which costs inference
  speed. For a fixed inference file, declare the entries with
  `overridable=False`; see [Fixed inference files](state_inputs.md#fixed-inference-files).

The guarantees of `state=` itself are listed under
[Guarantees](state_inputs.md#guarantees).
