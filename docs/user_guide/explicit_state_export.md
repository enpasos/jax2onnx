# Export explicit model state

This experimental example exports selected weights and buffers as named ONNX
inputs, then adds their default values and tensor metadata. It uses the ordinary
`to_onnx` pipeline. The adapter lives in `scripts/`; it is not a new jax2onnx
API or an automatic adapter for arbitrary models.

The example addresses the parameter-sharing and freeze-policy questions in
[discussion #290](https://github.com/enpasos/jax2onnx/discussions/290). It uses a
small Equinox attention model with rotary position embeddings (RoPE). It is a
demonstration of the export convention, not an adapter for Clément's complete
Equimo, ViT or DINO models.

## What the model exports

A parameter can feed several operations while retaining one explicit input:

```text
shared_weight + its initializer default
    -> direct projection
    -> reshape/broadcast -> another use
    -> ONNX Function argument -> projection inside the Function
```

The model has four declared state slots:

| Slot | Purpose | Training policy in this example |
| --- | --- | --- |
| `shared_weight` | A weight used directly, through views and through explicit Function arguments | Trainable |
| `independent_weight` | A separate weight with the same initial values | Trainable, independently |
| `rope_rotations` | The RoPE buffer used through traced JAX operations | Frozen |
| `shared_frozen` | One matrix deliberately shared by two projection branches | Frozen |

The framework adapter defines this sharing. It reconstructs the two known
frozen branch fields from the same slot. Equal values or equal array identities
do not cause the helper to combine slots. The two initially equal trainable
weights remain separate inputs and separate defaults.

## Run the example

From the repository root, use the normal Poetry environment:

```bash
poetry run python -m scripts.explicit_state_rope \
  --output-dir /tmp/explicit-state-rope
```

The command exports the model, saves and reloads it, compares ONNX Runtime with
JAX, exercises changed state, and checks the example's buffer-freezing policy.
It prints the artifact location and validation results. Generated models stay
outside git.

To exercise the supported External Data save path:

```bash
poetry run python -m scripts.explicit_state_rope \
  --output-dir /tmp/explicit-state-rope-external --external-data
```

Both paths serialize a fresh `ir.to_proto(model)` with `onnx.save_model`, then
reload and check the result. The example avoids `ir.save` with External Data:
the reviewed ONNX IR 1.1.0 path loses tensor metadata. There is no promise of
atomic publication across model and data files after filesystem failures.

## What the caller prepares

The model is exported as a function with explicit state arguments. In this
example, Equinox partitioning separates model structure from array values, and
the adapter reconstructs the known model using those arguments.

The example-local helper accepts records like this:

```python
from scripts.explicit_state_export import StateSlot, export_with_state

slot = StateSlot(
    name="shared_weight",
    value=weight,
    metadata={
        "pkg.example.path": "shared_weight",
        "pkg.example.trainable": "true",
    },
)

model = export_with_state(
    forward,  # forward(features, shared_weight)
    [features_spec],
    [slot],
    data_input_names=["features"],
    opset=23,
)
```

`StateSlot` and `export_with_state` belong to the example script. They are not
exports from `jax2onnx`. The helper builds each state specification from its
source shape, dtype and `weak_type`; weak scalar promotion can otherwise change
the computation's dtype.

For each declared slot, the helper checks its target input and installs an
owned tensor snapshot on the same IR value as an initializer. It keeps that
value in `model.graph.inputs`. An input and same-named initializer represent an
input with a default in ONNX; the example checks default and override behavior
with ONNX Runtime. Other runtimes need their own compatibility checks.

Keep model configuration and source values stable while exporting. Subsequent
changes to the original arrays or metadata dictionaries do not alter the
accepted snapshots. The returned IR model is itself mutable.

## The bounded helper contract

The first example supports nonempty dense native-endian FP32 state with static shapes,
including a tested rank-zero weak scalar. Noncontiguous arrays are copied in
C order. Other state dtypes and empty tensors are rejected; they are not
silently converted. Data-input types remain subject to ordinary exporter
support, including the tested FP16-activation case.

The helper enforces these rules:

- State names are unique C90-style identifiers and do not collide with data
  names. Framework paths belong in properties, not automatically joined names.
- Every declared state input must exist with exactly the expected shape and
  dtype. Existing initializer conflicts and missing targets are errors.
- Each slot gets one same-named root initializer on its input value. Different
  slots remain different values, even if their source arrays are identical.
- Payload snapshots retain exact FP32 bits, including signed zero and NaN
  payloads. Properties are copied strings with nonempty UTF-8 keys and UTF-8
  values; the helper does not interpret their meaning.
- Unused slots are errors unless their record explicitly sets
  `allow_unused=True`. A direct graph output counts as a use. This is a local
  connection check, not a liveness or gradient analysis.
- Records and all target inputs are checked before attachment. The helper owns
  the fresh IR until it completes and writes no files itself. Ordinary export
  failures propagate; there is no fallback to captured constant state.

These are checks on declared slots. They do not prove that an arbitrary wrapper
uses its arguments everywhere. A wrapper can still accidentally capture an
old concrete array. The example tests forwarding separately by overriding all
slots and comparing with a reconstructed JAX reference, and by comparing the
non-slot tensor inventory across exports with changed state. Negative controls
demonstrate the difference between these tests and the helper's local checks.
Those comparisons are regression tests, not additional export requirements
executed for every user model.

## Read and use the state

Properties live on initializer tensors as `TensorProto.metadata_props`. Inspect
the saved model directly:

```python
import onnx

model = onnx.load("/tmp/explicit-state-rope/attention.onnx")
input_names = {value.name for value in model.graph.input}
for tensor in model.graph.initializer:
    if tensor.name in input_names:
        print(tensor.name, {p.key: p.value for p in tensor.metadata_props})
```

ONNX Runtime lists these defaults under `get_overridable_initializers()`,
separately from `get_inputs()`. Keep their graph-input entries: removing an
entry loses the override convention and can let runtime optimization create
separate stored views again. The saved original model is the artifact checked
by this example; external optimizer output needs separate validation.

In Netron, follow the shared input/default to its consumers and view
operations. Tensor-metadata display depends on the viewer version; the Python
inspection above provides a direct check of the actual saved properties.

The example's reader reconstructs the **known example architecture** from the
saved state. It is not a general ONNX-to-JAX importer. Its AdamW demonstration
puts only the metadata-selected trainable arrays into the optimizer state.
Both frozen buffers are excluded, so weight decay cannot update them.

Being a graph input does not mean being trainable. The example interprets
`pkg.example.trainable` in its own reader; jax2onnx and the export helper do not.
Forward parity also does not establish arbitrary `stop_gradient`, custom
JVP/VJP, optimizer-state or training-mode preservation. The small example has
no training/inference-dependent Dropout or BatchNorm behavior.

## Model requirements and remaining questions

Selected state must remain in traced JAX computations. Converting a state
tracer with NumPy during the forward pass is unsupported. This is different
from creating an initial buffer before export.

The example's Equinox classes are not decorated with `@onnx_function`. Only a
free function is decorated, and it receives selected state explicitly. Current
stateful decorated classes and captured state tracers can fail during Function
lowering. A concrete-array closure can instead silently become a local
constant. The helper does not promise to discover or repair either pattern.

Supporting those model patterns would be a separate converter decision. This
example introduces no closure conversion, automatic alias detection, provenance
registry or special optimizer path. Its suitability for Clément's actual
export path remains to be checked with a concrete model.

Keeping weights dynamic can reduce constant-folding and precomputation
opportunities. Use the usual closed-model export when a fixed inference
artifact is the desired result. No runtime speed or peak-memory improvement is
promised by this example.
