# ONNX Function Decorator Guidelines

This guide summarizes the current behaviour and guardrails for the `@onnx_function`
decorator. It complements the plugin quickstart by detailing how function reuse,
namespacing, and testing conventions work in practice.

## Goals and Defaults

- `@onnx_function` marks a callable as an ONNX function boundary. Each invocation
  emits a `FunctionProto` in the exported model.
- By default, every call-site receives its own function instance; the node’s
  `op_type` mirrors the callable name, and the domain uses the `"custom"` namespace.
- Optional flags allow you to reuse function bodies (`unique=True`) and control the
  domain prefix (`namespace=...`), without sacrificing readability in tools like
  Netron.

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
  is used.

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

## Testing & Examples

- The regression suite lives at
  `tests/extra_tests/converter/test_onnx_function_unique.py`.
  It covers duplicate reuse, distinct captures, and custom namespaces.
- Example `onnx_functions_017` demonstrates two call-sites sharing a unique function.
- Regenerate fixtures after behaviour changes:
  ```bash
  poetry run python scripts/generate_tests.py
  ```

## Best Practices

- Use `construct_and_call(...).with_requested_dtype(...).with_rng_seed(...)` in
  metadata so tests can rebuild deterministic f32/f64 variants.
- Keep `post_check_onnx_graph` expectations focused on meaningful structural
  checks. Function selectors accept either the exact `domain:name` or a domain
  prefix (e.g. `"custom.MyFn.1"` matches `"custom.MyFn.1:MyFn"`).
- When migrating existing decorators, ensure no conflicting namespace choices are
  applied to the same target; the decorator raises if mixed namespacing is detected.
