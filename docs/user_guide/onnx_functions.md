# ONNX Functions

Use `@onnx_function` when a callable should appear as a named, reusable function
boundary in the exported ONNX model. This keeps repeated subgraphs readable in
tools such as Netron and lets you give important model blocks stable names.

## Goals and Defaults

- `@onnx_function` marks a callable as an ONNX function boundary. Each invocation
  emits a function call in the exported model, subject to the transformation
  behavior described below.
- By default, function definitions are cached per callable or module instance,
  input shapes/dtypes, and capture signature. Matching repeated calls reuse the
  definition. The node's `op_type` mirrors the callable name, and the domain uses
  the `"custom"` namespace.
- `unique=True` also permits compatible module instances with equivalent state
  to share a definition. `namespace=...` controls the domain prefix.

## Minimal Example

```python
import jax
import jax.numpy as jnp

from jax2onnx import onnx_function, to_onnx


@onnx_function
def block(x):
    return jnp.tanh(x) + 1.0


def model(x):
    return block(block(x))


to_onnx(
    model,
    [jax.ShapeDtypeStruct((2, 4), jnp.float32)],
    return_mode="file",
    output_path="model_with_function.onnx",
)
```

Open the exported model in Netron to inspect the function boundary.

## JAX Transformations

The current `vmap` batching rule inlines the mapped callable's body. Its
computations and dynamic keyword values are exported, but the mapped call does
not retain a separate ONNX Function boundary. A model containing only such
mapped calls can therefore have an empty `model.functions` collection.

Function-boundary inspection and numerical validation answer different
questions: check `model.functions` when a named boundary is required, and
compare runtime outputs when validating transformed computations.

## Flags

```python
@onnx_function(unique=False, namespace="custom", type="MyFn")
def my_fn(...):
    ...
```

- `unique=True` uses callable/module state rather than instance identity when
  deciding whether matching calls can share a function definition.
- `namespace` overrides the domain prefix. When omitted, it defaults to `"custom"`.
- `type` (preferred) or `name` (alias) overrides the human-readable base
  name/op_type used for the function. When omitted, the callable’s Python name
  is used. If both are supplied, `type` takes precedence.

### Domain Naming Rules

- Non-unique functions use `{namespace}.{base}.{counter}`.
- Unique functions use `{namespace}.{base}.unique` for the first instance and
  `{namespace}.{base}.unique.{N}` for additional variants when the capture differs.
- `op_type` equals the sanitized base name (`type`/`name` when provided,
  otherwise the callable name) so node “types” stay human-friendly.

Examples (default namespace):

| Setting          | First FunctionProto Domain        | Second Variant            |
|------------------|-----------------------------------|---------------------------|
| unique=False     | `custom.MyBlock.1`                | `custom.MyBlock.2`        |
| unique=True      | `custom.MyBlock.unique`           | `custom.MyBlock.unique.2` |

Custom namespace:

```python
@onnx_function(unique=True, namespace="my.model")
def square(...):
    ...
```

Produces `domain="my.model.square.unique"` for all reused call-sites.

## Reuse Mechanics

- Function identity considers:
  - Qualified target name.
  - Input shapes/dtypes from the parent equation.
  - Captured parameters. For classes, we fingerprint the module state via
    `jax.tree_util.tree_flatten`.
- With `unique=False`, reuse requires the same callable or module instance.
- With `unique=True`, compatible instances with equivalent state can also reuse
  the same `FunctionProto`. Different keys receive separate domain suffixes.

Static Python keywords specialize the function body. Built-in containers and
NumPy arrays are snapshotted when each call is traced, so changing a dictionary
between calls does not alter an earlier call. Configuration such as string
modes, Python booleans, and transpose-axis tuples remains static during abstract
evaluation as well. Opaque Python objects are passed through, must remain stable
during conversion, and conservatively receive separate definitions.

## Examples

The [Examples](examples.md) reference table contains two kinds of ONNX
Function exports:

- Dedicated decorator examples in the `onnx_functions_*` rows. These cover
  function boundaries, nested functions, call parameters, and `unique=True`
  reuse.
- Larger model-family examples that use `@onnx_function` internally. These rows
  are named after their exported components, not after the decorator feature.
  Look for GPT, ViT, DINOv3, GPT-OSS, `Flax*`, and `NnxDino*` examples.

Those links open representative ONNX models in Netron.

## Best Practices

- Use `type="..."` when you want a stable display name that is independent of
  the Python callable name.
- Use `namespace="..."` when several model families or libraries may define
  functions with similar names.
- Use `unique=True` for repeated call-sites that should share one function body.
- When migrating existing decorators, ensure no conflicting namespace choices are
  applied to the same target; the decorator raises if mixed namespacing is detected.
- Keep display-name overrides stable on repeated decoration. The decorator accepts
  identical `type`/`name` overrides but raises when the same target is registered
  with conflicting display names.
