# Parameter Metadata

Use `annotate_parameters` to record which exported values a downstream consumer
should treat as trainable or frozen. This is an optional convention between the
exporter and a consumer; ONNX runtimes do not interpret it automatically.

Annotations apply to the final artifact after `to_onnx` has completed lowering,
optimization, and input/output renaming. They leave the computation unchanged.
The helpers accept both `onnx.ModelProto` and `onnx_ir.Model`.

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
