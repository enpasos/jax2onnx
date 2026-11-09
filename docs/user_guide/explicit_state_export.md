# Export and use explicit model state

Suppose you want an ONNX model that contains its weights but also lets a caller
replace selected weights or buffers. This example makes those values named
inputs with saved defaults. A caller can run inference with the defaults or
supply replacement arrays by name.

For your own models, use `to_onnx(..., state=...)` as described in
[Model State as Inputs](state_inputs.md). This page builds the same file by
hand with helpers from the repository's `scripts/` directory, to show what
that option does, and then continues with saving, reloading and an optimizer
step. The helpers are not public imports from the installed `jax2onnx`
package. A test checks that `state=` reproduces this walkthrough's file byte
for byte.

The walkthrough uses a small Equinox attention model with rotary position
embeddings (RoPE). It illustrates the sharing and freeze-policy questions in
[discussion #290](https://github.com/enpasos/jax2onnx/discussions/290); it is not
an automatic adapter for arbitrary Equimo, ViT or DINO models.

## Try the complete example first

Use a checkout containing this example. From its repository root, prepare the
normal development environment and run:

```bash
poetry install
poetry run python -m scripts.explicit_state_rope \
  --output-dir /tmp/explicit-state-rope
```

The output directory contains:

| File | What it contains |
| --- | --- |
| `attention.onnx` | The inference graph, four named state inputs, their default arrays and tensor metadata |
| `report.json` | Versions, the actual opset, numerical comparisons and the result of the freezing check |

The command saves and reloads the model, compares ONNX Runtime with JAX, checks
all four replacement arrays against a separately reconstructed Python model,
and performs one AdamW step. Both buffers declared frozen must remain unchanged.

To store tensor payloads in a separate file, use a fresh output directory:

```bash
poetry run python -m scripts.explicit_state_rope \
  --output-dir /tmp/explicit-state-rope-external --external-data
```

This also writes `attention.onnx.data`. Keep that file beside `attention.onnx`
when copying or opening the model. Generated artifacts stay outside git.

## Exactly which interface changes?

The Python model still runs as `source(x)`. For export, an adapter supplies a
second callable whose state is explicit:

```text
Original model:  source(x)
Export wrapper:  forward(x, shared_weight, independent_weight,
                        rope_rotations, shared_frozen)
```

The helper passes typed specifications for **data first, then state in slot
order** to the ordinary `to_onnx` function. After conversion, it attaches each
state array as the default of its corresponding ONNX input.

| Interface | Ordinary closed-model export | This explicit-state recipe |
| --- | --- | --- |
| Python model call | `source(x)` | `source(x)` |
| Callable passed to conversion | The model captures its state | A wrapper receives each selected state slot as an argument |
| Public `to_onnx` signature | Existing API | Existing API; no new keyword |
| Recipe entry point | Not needed | `export_with_state(forward, data_specs, slots, data_input_names=...)` returns an IR model |
| ONNX data input | `x`, required | `x`, required |
| Selected state in ONNX | Stored constants | Named inputs with same-named initializer defaults |
| ONNX Runtime feed for default inference | `{"x": x_array}` | `{"x": x_array}` |
| Replace one selected weight for a run | No declared override interface | Add `"shared_weight": replacement` to the feed |
| Path and training labels | Not supplied by this recipe | String properties on the state initializer tensors |

The wrapper's Python state arguments are required during tracing. In the saved
ONNX model, these state inputs have defaults, so inference callers may omit
them. An omitted slot uses its saved default. A supplied slot uses the caller's
array for that run.

Metadata does not make an input trainable or frozen. The example's Python
consumer interprets its own `pkg.example.trainable` property. The exporter and
ONNX Runtime do not interpret that property.

### What changes in the ONNX file

The recipe does not rewrite the computation. For this example, both exports
contain the same operators and the same four weight tensors, and ONNX Runtime
returns the same outputs. Only the way the weights enter the graph differs:

| Ordinary closed-model export | This explicit-state recipe |
| --- | --- |
| Initializer with a generated name, for example `const_0: float[4, 4]` | Graph input `shared_weight: float[4, 4]` with a same-named initializer default |
| No tensor properties | `pkg.example.path=shared_weight`, `pkg.example.trainable=true` |
| Consumers read the constant, for example `project(x, const_0)` | Consumers read the input, for example `project(x, shared_weight)` |

Other models can need a few extra shape operations where the ordinary export
uses a constant view and the explicit export computes that view from the shared
input. A regression test compares both exports of this example.

### Why not annotate the ordinary export afterwards?

An ordinary export does not record which model field each weight initializer
came from; its names are generated. Values cannot reliably identify the field
either. Here, `shared_weight` and `independent_weight` start with identical
values, and freshly initialized models typically contain identical LayerNorm
scales. Some patterns are also stored in a derived form, for example a weight
expanded by `jnp.broadcast_to` to a fixed shape. Naming the state at the export
interface records the mapping by construction, without provenance tracking in
the converter.

## Walkthrough: prepare, export and run the model

The following Python blocks form one sequential session. Run them from the
repository root in `poetry run python` or a script using that environment.

### 1. Prepare the original model and data

```python
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx import to_onnx
from scripts.explicit_state_export import StateSlot, export_with_state
from scripts.explicit_state_rope import (
    SmallAttention,
    adamw_step,
    explicit_forward,
    read_slots,
    reconstruct,
    runtime,
    save_reload,
    state_slots,
)

source = SmallAttention(key=jax.random.key(0))
x = jnp.linspace(-0.8, 0.9, 12, dtype=jnp.float32).reshape(3, 4)
data_specs = [jax.ShapeDtypeStruct(x.shape, x.dtype)]

# The ordinary export has just the data input.
closed = to_onnx(
    source,
    inputs=data_specs,
    input_names=["x"],
    return_mode="ir",
)
print([value.name for value in closed.graph.inputs])  # ['x']
```

`x` contains three tokens with four features each. The example constructs its
weights and buffers as FP32 arrays before export.

### 2. Declare which values form the state interface

```python
slots = state_slots(source)
for slot in slots:
    print(slot.name, slot.value.shape, dict(slot.metadata))
```

A **slot** is one array you deliberately expose as one changeable value. This
model-specific function declares four slots:

| Slot name | Fields in the Python model | Shape | Training policy |
| --- | --- | --- | --- |
| `shared_weight` | `shared_weight` | `(4, 4)` | Trainable |
| `independent_weight` | `independent_weight` | `(4, 4)` | Trainable |
| `rope_rotations` | `rope.rotations` | `(3, 2, 2)` | Frozen |
| `shared_frozen` | Both `q_projection.matrix` and `k_projection.matrix` | `(4, 4)` | Frozen |

For your own adapter, construct these records with `StateSlot`. This is how
the example declares the RoPE buffer:

```python
rope_slot = StateSlot(
    name="rope_rotations",
    value=source.rope.rotations,
    metadata={
        "pkg.example.path": "rope.rotations",
        "pkg.example.trainable": "false",
    },
)
```

`state_slots(source)` already includes this record in its four-element result;
do not append it again. Slot order must match the wrapper's positional state
arguments.

The two trainable weights initially have equal values, but remain independent
slots. Both frozen projection fields deliberately share one slot. Neither
array identity nor equal contents cause the export helper to infer sharing.
The combined path label for that shared slot is this example's own convention:
`q_projection.matrix;k_projection.matrix`.

### 3. Give the export wrapper all selected state explicitly

```python
forward = explicit_forward(source)

exported = export_with_state(
    forward,
    data_specs,
    slots,
    data_input_names=["x"],
    opset=23,
)
print([value.name for value in exported.graph.inputs])
# ['x', 'shared_weight', 'independent_weight', 'rope_rotations', 'shared_frozen']
```

`explicit_forward` is an adapter for **this architecture**. It partitions the
Equinox model into array values and static structure, replaces all five selected
array fields from the four arguments, and evaluates the rebuilt model. The
`shared_frozen` argument fills both frozen projection fields.

Every use of a selected value must use the corresponding argument. In this
example, `shared_weight` reaches direct operations, reshaped views and two
calls to a free ONNX Function. That Function also receives the weight explicitly.

For your own model, you must write the slot declaration and the wrapper that
replaces its selected fields. The helper does not discover model leaves or
rewrite a closure automatically. Test replacements against an independently
reconstructed source model: a wrapper can otherwise accidentally keep using an
old captured array in part of the computation.

The complete `partition` / `tree_at` / `combine` implementation is in
`explicit_forward` in `scripts/explicit_state_rope.py`. `StateSlot` and
`export_with_state` provide the reusable export mechanism within the supported
scope. The slot declaration, `read_slots`, `reconstruct`, `save_reload` and
`adamw_step` know this model's four slots; adapt those parts for your own model.

### 4. Save and reload the artifact

```python
output_dir = Path("/tmp/explicit-state-walkthrough")
output_dir.mkdir(parents=True, exist_ok=True)
path = output_dir / "attention.onnx"

proto = save_reload(exported, slots, path, external_data=False)
```

`save_reload` is also specific to this four-slot example. It serializes a fresh
`onnx_ir.to_proto(exported)` with `onnx.save_model`, reloads it, runs the ONNX
checker and verifies exact state values and properties. Use this serializer
path when adapting the recipe; the reviewed ONNX IR 1.1.0 `ir.save` path with
External Data loses tensor metadata.

The save check writes to the requested directory before validation. If it
raises, those files are not validated output; retry in a fresh directory.
There is no atomic publication guarantee across the model and data files.

### 5. Run inference with the defaults, then replace one weight

```python
session = runtime(proto)
feed = {"x": np.asarray(x)}

default_output = session.run(None, feed)[0]
np.testing.assert_allclose(default_output, source(x), rtol=2e-5, atol=2e-6)

print([item.name for item in session.get_inputs()])  # ['x']
print([item.name for item in session.get_overridable_initializers()])
# ['shared_weight', 'independent_weight', 'rope_rotations', 'shared_frozen']

replacement = np.asarray(source.shared_weight) + np.float32(0.2)
changed_output = session.run(None, {**feed, "shared_weight": replacement})[0]
assert not np.allclose(changed_output, default_output)
```

Supply replacement arrays with the slot's declared shape and dtype. The other
three slots retain their defaults. Replacing `shared_weight` changes all its
uses, including the two Function calls; it does not change `independent_weight`.

An override affects this inference call. It does not edit the ONNX file, mutate
the Python source model, or become the default for later calls. Running again
with only `feed` uses the saved defaults again.

`runtime` lowers the ONNX Runtime log level. With default logging, ONNX Runtime
warns for each state input that the initializer "appears in graph inputs and
will not be treated as constant value/weight". This is expected for overridable
defaults and reflects the constant-folding trade-off described at the end of
this page. Do not remove the inputs to silence the warning; that removes the
override interface.

### 6. Inspect metadata and reconstruct the known Python model

```python
loaded_slots = read_slots(proto)
for slot in loaded_slots:
    print(slot.name, dict(slot.metadata))

rebuilt = reconstruct(source, loaded_slots)
np.testing.assert_allclose(rebuilt(x), source(x), rtol=2e-5, atol=2e-6)
```

`read_slots` reads the intersection of ONNX graph inputs and initializers.
The properties are stored on each initializer as `TensorProto.metadata_props`.
For example, the RoPE initializer carries `pkg.example.path=rope.rotations`
and `pkg.example.trainable=false`.

`reconstruct` uses the four known path labels to replace fields in the supplied
Python model structure. Every example path must appear exactly once; missing,
duplicate or unknown labels raise an error rather than retaining source values.
For this example, the combined frozen path assigns the same slot to both branches.

The original Python architecture is still required. The file provides state
and an inference graph; this reader is not a general ONNX-to-JAX importer.

### 7. Take a training step and save new defaults

```python
updated_slots, loss = adamw_step(source, loaded_slots, x, jnp.zeros_like(x))
trained = reconstruct(source, updated_slots)
print(loss)

trained_slots = state_slots(trained)
trained_export = export_with_state(
    explicit_forward(trained),
    data_specs,
    trained_slots,
    data_input_names=["x"],
    opset=23,
)
trained_proto = save_reload(
    trained_export, trained_slots, output_dir / "attention-trained.onnx",
    external_data=False,
)
trained_session = runtime(trained_proto)
np.testing.assert_allclose(
    trained_session.run(None, feed)[0], trained(x), rtol=2e-5, atol=2e-6,
)
```

Training runs on the **original Python model code with the loaded arrays**.
The example reads its trainability labels and gives AdamW only the two trainable
arrays. Both frozen buffers stay outside the optimizer state, so weight decay
cannot update them. Re-exporting the trained model writes new default values
to a separate artifact. Create a fresh runtime session to use that artifact.

Being an ONNX graph input does not imply trainability. Forward parity also does
not establish preservation of arbitrary `stop_gradient`, custom JVP/VJP,
optimizer state or training modes through a general importer. The example does
not recover gradient rules from ONNX or model Dropout/BatchNorm training modes.

## Open the result in Netron

Open `/tmp/explicit-state-walkthrough/attention.onnx` in Netron. Find
`shared_weight` and follow its connections to the operations and Function calls
that consume it. The same saved default supplies the shared uses. The separate
`independent_weight` slot remains a different value despite initially equal data.

For tensor properties, select the initializer/input details if your Netron
version displays them. Viewer support varies; inspect the saved properties
directly when they are not shown:

```python
import onnx

saved = onnx.load(path)
state_names = {value.name for value in saved.graph.input}
for tensor in saved.graph.initializer:
    if tensor.name in state_names:
        print(tensor.name, {p.key: p.value for p in tensor.metadata_props})
```

Netron displays the artifact on disk. It does not display a previous runtime
feed override; open `attention-trained.onnx` to inspect the newly saved defaults.

## Helper contract and model requirements

The example-local `StateSlot` and `export_with_state` enforce these rules:

- State is nonempty dense native-endian FP32 with static shapes. Scalars and
  noncontiguous arrays are supported; other state dtypes and empty arrays are
  rejected. Data types remain subject to ordinary exporter support.
- State names are unique C90-style identifiers and do not collide with data
  names. Framework paths belong in properties. Inputs retain data-then-state
  order and every slot must have the expected shape and dtype.
- Each slot gets one same-named root initializer on its input value. Distinct
  slots remain distinct even when their source arrays are identical. Keep these
  entries in `model.graph.inputs` to retain the override convention.
- The helper derives typed specifications including `weak_type` from the source.
  Snapshots preserve exact FP32 bits, including signed zero and NaN payloads,
  without casting. Later changes to source arrays or metadata dictionaries do
  not change the copied defaults. The returned IR model is mutable.
- Properties are copied strings: nonempty UTF-8 keys and UTF-8 values. The
  helper does not interpret their meaning.
- Unused slots are errors unless explicitly declared with `allow_unused=True`.
  A direct graph output counts as a use. This checks local connections, not
  gradient behavior or liveness.
- Records and all exported targets are checked before defaults are attached.
  The helper owns the fresh IR until it completes and writes no files itself.
  Ordinary conversion failures propagate.

These checks do not prove that a wrapper forwards its arguments everywhere.
The example separately tests individual and simultaneous overrides against a
reconstructed JAX reference and compares non-slot tensor inventories across
exports with changed state. An intentional partial closure leak demonstrates
the boundary. Those are regression tests, not extra export requirements for
every user model.

Selected state must remain in traced JAX computations. Host evaluation such as
converting a state tracer to NumPy during the forward pass is unsupported.
Constructing an initial buffer before tracing is different and supported here.

The Equinox module classes in this example are undecorated. Only a free function
uses `@onnx_function`, with state passed explicitly. Stateful decorated classes
and captured state tracers can fail during Function lowering; a concrete-array
closure can instead silently become a constant. This helper does not discover
or repair those patterns.

The recipe introduces no closure conversion, automatic alias detection or
provenance registry. Its suitability for the motivating models in discussion
#290 and for a general ONNX-to-JAX training importer still needs separate
validation. Other runtimes
and external optimizer outputs need their own compatibility checks.

Keeping state dynamic can reduce constant folding and precomputation. Use the
ordinary closed-model export for a fixed inference artifact. This example
promises no runtime-speed or peak-memory improvement.
