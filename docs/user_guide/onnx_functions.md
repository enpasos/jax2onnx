# ONNX Functions

Use `@onnx_function` when a callable should appear as a named, reusable function
boundary in the exported ONNX model. This keeps repeated subgraphs readable in
tools such as Netron and lets you give important model blocks stable names.

## Goals and Defaults

- `@onnx_function` marks a callable as an ONNX function boundary. Each invocation
  emits a `FunctionProto` in the exported model.
- By default, every call-site receives its own function instance; the node’s
  `op_type` mirrors the callable name, and the domain uses the `"custom"` namespace.
- Optional flags allow you to reuse function bodies (`unique=True`) and control the
  domain prefix (`namespace=...`), without sacrificing readability in tools like
  Netron.

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

## Flags

```python
@onnx_function(unique=False, namespace="custom", type="MyFn")
def my_fn(...):
    ...
```

- `unique=True` de-duplicates call-sites that share the same capture signature.
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
- When `unique=True`, call-sites with identical captures share the same
  `FunctionProto`; otherwise each cooperative invocation gets its own domain suffix.

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
