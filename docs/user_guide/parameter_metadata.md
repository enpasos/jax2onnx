# Parameter Metadata

Use `to_onnx(..., initializer_metadata=callback)` to attach framework-owned
metadata to captured parameter tensors during conversion. The callback can
associate captured array objects with source paths and an explicit training
policy, without depending on generated ONNX names. ONNX runtimes do not
interpret these properties automatically.

For manual annotation of an already exported artifact, the optional
`annotate_parameters` helper provides a separate, typed model-level convention
described [below](#annotate-and-read-an-export).

## Capture Source Metadata During Export

The callback receives each captured source object before conversion to NumPy.
Return a mapping of string keys to string values, or `None` to leave that tensor
unannotated. Keys must be nonempty. The mapping is copied, and callback errors
fail the export. The exporter does not interpret keys or infer parameter paths,
trainability, or which framework leaves are buffers.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import onnx

from jax2onnx import to_onnx


weight = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
bias = jnp.zeros((2,), dtype=jnp.float32)
properties = {
    id(weight): {"pkg.myapp.path": "weight", "pkg.myapp.trainable": "true"},
    id(bias): {"pkg.myapp.path": "bias", "pkg.myapp.trainable": "false"},
}


def model_fn(x):
    return x @ weight + bias


def initializer_metadata(source):
    return properties.get(id(source))


model = to_onnx(
    model_fn,
    [jax.ShapeDtypeStruct((1, 3), jnp.float32)],
    initializer_metadata=initializer_metadata,
)
with TemporaryDirectory() as directory:
    path = Path(directory) / "annotated.onnx"
    onnx.save_model(model, path)
    restored = onnx.load_model(path)

annotations = {
    tensor.name: {entry.key: entry.value for entry in tensor.metadata_props}
    for tensor in restored.graph.initializer
    if tensor.metadata_props
}
assert {info["pkg.myapp.path"] for info in annotations.values()} == {"weight", "bias"}
onnx.checker.check_model(restored, full_check=True)
```

Array identity distinguishes independent parameters whose current values happen
to be equal. Keep the source model and its leaves alive during export. If two
framework paths refer to the same array, the callback sees that one object;
the framework must define its alias policy rather than choose a path silently.

The callback is also available with `return_mode="ir"` or `"file"`. For IR
initializers, properties live on `value.const_value.metadata_props`, which
serializes to `TensorProto.metadata_props`. This is different from
`value.metadata_props`, which belongs to ONNX value information.

### Captures, Scopes, and Transformations

Captured constants in `jit`, ONNX Functions, and supported control-flow bodies
use the same callback. Child graphs inherit the callback but keep their own IR
values. Functions cannot contain initializers, so their annotations live on
the tensor attributes of `Constant` nodes. Consumers handling these models must
inspect the relevant scopes, not only `model.graph.initializer`.

The callback may run more than once for the same source object. It should be
deterministic and should not mutate arrays. Generated shape constants and
literal operands do not invoke it. Arrays copied or transformed before capture
may no longer be the original model leaves. JAX may also wrap NumPy arrays in
new literal objects, so Python identity is not guaranteed for NumPy source
arrays. Unused leaves may not be exported. Validate the expected source paths
in the completed artifact when completeness matters.

Constant reshape lowering caches views by source IR value and target shape
within one graph, preserving metadata on the reshaped tensor. Independent
equal-valued tensors remain independent. Eqx Conv biases and ConvTranspose bias
folding preserve the source annotation when reshaping a single bias. An
annotated broadcast remains a graph operation so updating one source element
continues to affect all its uses. Tensor constants carrying metadata are
protected from content-based common-subexpression elimination.

Annotations identify a source association, not necessarily an unchanged storage
shape or dtype. A consumer that reconstructs a source module must validate the
representation and any reshape it applies. One source can still have multiple
exported views or appear in multiple scopes; the callback does not supply a
general alias or view-transformation schema. This hook does not guarantee
metadata preservation through arbitrary downstream optimizers.

### Framework-Owned Training Policy

For an Equimo-style workflow, the framework can construct the callback from its
parameter paths and explicit fine-tuning mask, write `pkg.equimo.path` and
`pkg.equimo.trainable`, then interpret those keys in its importer. Excluding a
buffer from the optimizer also prevents AdamW weight decay from updating it;
merely obtaining a zero gradient does not establish that policy.

The regression suite checks a small RoPE attention example with serialization,
ONNX Runtime forward parity, source-module restoration, and selective AdamW
updates. The broader ONNX-to-JAX importer remains outside this package. These
checks do not establish arbitrary JAX gradient preservation; see
[Training Semantics](#training-semantics).

## Annotate an Existing Artifact

Annotations apply to the final artifact after `to_onnx` has completed lowering,
optimization, and input/output renaming. They leave the computation unchanged.
The helpers accept both `onnx.ModelProto` and `onnx_ir.Model`.

These helpers use the model-level `pkg.jax2onnx.parameter_metadata` manifest.
They neither read nor write the framework-owned tensor properties produced by
`initializer_metadata`; the two conventions can coexist independently.

## Annotate and Read an Export

Making parameters explicit function arguments lets you give them stable exported
input names. This example marks `weight` as trainable and `bias` as frozen, then
saves and reloads the annotations:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import onnx

from jax2onnx import (
    ParameterMetadata,
    annotate_parameters,
    read_parameter_metadata,
    to_onnx,
)


def linear(x, weight, bias):
    return x @ weight + bias


model = to_onnx(
    linear,
    inputs=[
        jax.ShapeDtypeStruct((1, 2), jnp.float32),
        jax.ShapeDtypeStruct((2, 1), jnp.float32),
        jax.ShapeDtypeStruct((1,), jnp.float32),
    ],
    input_names=["x", "weight", "bias"],
)
annotate_parameters(
    model,
    {
        "weight": ParameterMetadata(trainable=True, source_path="params.weight"),
        "bias": ParameterMetadata(trainable=False, source_path="params.bias"),
    },
)

with TemporaryDirectory() as directory:
    path = Path(directory) / "linear.onnx"
    onnx.save_model(model, path)
    restored = onnx.load_model(path)
    annotations = read_parameter_metadata(restored)

assert annotations["weight"].trainable is True
assert annotations["bias"].trainable is False
assert "x" not in annotations
onnx.checker.check_model(restored, full_check=True)
```

The same workflow works with `return_mode="ir"`; serialize with
`onnx_ir.to_proto(model)` or `onnx_ir.save(model, path)` after annotating. For an
existing file, load it with ONNX, annotate the model, then save it using the
external-data settings appropriate for that artifact.

For weights captured by a closure or framework module, inspect the exported
initializers and supply their actual names. Names such as `const_0` are generated
for that particular export and are not stable source parameter identifiers.
The helper does not infer a mapping from JAX arrays or framework state to ONNX
values. If that mapping is unavailable, leave the values unspecified rather
than guessing from dtype, shape, array contents, or initializer order.

## Annotation Contract

`ParameterMetadata` is an immutable record with these fields:

| Field | Meaning |
| --- | --- |
| `trainable=True` | The caller selects this exported value for updates by a consumer. |
| `trainable=False` | The caller explicitly freezes this exported value. |
| `trainable=None` (default) | The caller has not specified an update policy. |
| `source_path` | Optional, nonempty, caller-supplied source label. |

An absent annotation also means unspecified, not frozen. A source path is
informational: equal paths do not tie two weights, and different paths do not
split one shared exported value. The helper does not check whether a consumer
can differentiate or update a value's dtype or representation.

Keys must be exact names of inputs or initializers in the root graph. A legal
input/initializer overlap identifies the same value. Intermediate node outputs,
Function-local values, and subgraph-local values are outside this first schema.

`annotate_parameters(model, parameters)` modifies the model in place and returns
`None`. It merges entries by exported name; an entry replaces the entire previous
record for that name, including `source_path`. It validates the existing manifest
and every new entry before writing anything. An empty mapping leaves the model
unchanged. Unrelated metadata, graph structure, tensor contents, training-info
fields, and external-data references are left untouched by these helpers.

`read_parameter_metadata(model)` returns a new dictionary of immutable records.
It returns `{}` when the manifest is absent. Both helpers reject malformed
manifests, unsupported versions, ambiguous names, and references to values no
longer present in the root graph. These checks are not a replacement for the
ONNX model checker.

## Serialized Convention and Consumers

The convention is stored as one JSON string under the model-level
`metadata_props` key `pkg.jax2onnx.parameter_metadata`:

```json
{
  "version": 1,
  "parameters": {
    "weight": {"trainable": true, "source_path": "params.weight"},
    "bias": {"trainable": false, "source_path": "params.bias"},
    "other_weight": {"trainable": null}
  }
}
```

Version 1 requires `version` and `parameters`, and each record requires
`trainable`; `source_path` is optional. Additional fields and duplicate JSON
keys are rejected. The example illustrates the schema; every parameter name
must exist in the artifact being annotated. An unannotated export has no such
metadata key. Tensor-level and node-level metadata are not used by this API.

A downstream consumer can select parameters explicitly:

```python
annotations = read_parameter_metadata(model)
selected = [name for name, info in annotations.items() if info.trainable is True]
frozen = [name for name, info in annotations.items() if info.trainable is False]
```

The consumer must validate its own requirements. For example, an adapter for
ONNX Runtime's training artifact generator must select supported initializers
for its `requires_grad` and `frozen_params` arguments. The explicit graph inputs
in the first example are not initializers; an annotation does not supply tensor
data, move a value into the initializer list, create a loss, or generate a
gradient graph.

Apply annotations after graph transformations. Reading catches removed or
renamed targets, but cannot detect a changed meaning under an unchanged name.
After further rewrites, reassess the selection and source mapping. Distinct
parameters with equal current contents must not be inferred to be shared.
If an existing manifest contains stale references, discard its reserved
metadata entry and rebuild the annotations after checking the new graph; the
helpers reject stale entries instead of silently carrying them into an update.

## Training Semantics

This API records a caller's selection; it does not reconstruct original JAX
training behavior. In particular, the exported forward graph may no longer
contain `stop_gradient` barriers or custom JVP/VJP rules. For
`w + jax.lax.stop_gradient(w)`, JAX's derivative is `1`, whereas differentiating
the ordinary exported `Identity` and `Add` gives `2`. Marking `w` trainable
cannot restore that distinction.

Framework parameter discovery, source-to-export provenance, optimizer state,
and an ONNX-to-JAX importer remain separate work. See
[Known Limitations](known_limitations.md#training-is-out-of-scope).

## API Reference

::: jax2onnx.parameter_metadata
