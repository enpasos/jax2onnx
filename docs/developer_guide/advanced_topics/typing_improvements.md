# Typing Guardrails

The converter and plugin stack is typed as first-party code. Treat typing as a
guardrail for IR ownership, shape/dtype metadata, PRNG discipline, and plugin
metadata contracts rather than as a separate cleanup project.

## Current Check

Run:

```bash
./scripts/check_typing.sh
```

The wrapper runs:

```bash
poetry run mypy --config-file pyproject.toml
poetry run python scripts/report_rng_traces.py
```

`pyproject.toml` currently checks the `jax2onnx` package, skips optional SOTA
example integrations and sandbox repros, and follows `onnx_ir` imports normally
so ONNX IR annotations are enforced.

## Shared Types

Use the shared protocols and dataclasses in
`jax2onnx.converter.typing_support` instead of local `Any`/`dict` shapes:

| Type | Use |
| --- | --- |
| `LoweringContextProtocol` | Plugin `lower(...)` methods and helpers that need the converter context. |
| `IRBuilderProtocol` | Helpers that only need builder operations, constants, graph containers, or names. |
| `SymbolicDimOrigin` | Provenance for symbolic dimensions materialized by `dim_as_value`. |
| `AxisOverrideInfo` / `AxisOverrideMap` | Loop/scan axis-0 extent metadata that must survive nested graph rewrites. |
| `RngTrace` | Metadata used by RNG trace reporting. |
| `PrimitiveLowering` / `FunctionLowering` | Dispatch protocols for primitive and ONNX Function plugin instances. |

## Authoring Rules

- Annotate new plugin `lower(...)` methods with `LoweringContextProtocol` and the
  concrete JAX equation type when it is available.
- Keep helper signatures narrow. If a helper only needs `ctx.builder`, accept
  `IRBuilderProtocol` or pass the builder directly.
- Prefer `ir.Value`, `ir.Node`, `ir.Graph`, `ir.Attr`, and `ir.DataType`
  annotations over protobuf types in converter/plugin code.
- Use shared shape/dtype helpers from `plugins/_ir_shapes.py`,
  `jax2onnx.ir_utils`, and `converter/typing_support.py`.
- Add a shared protocol/helper when the same `cast(...)` or duck-typed fallback
  appears in more than one module.

## Boundary Exceptions

Some code is intentionally less strict:

- `jax2onnx/plugins/examples/maxtext` and `jax2onnx/plugins/examples/maxdiffusion`
  depend on optional external checkouts.
- `jax2onnx/sandbox` contains exploratory repros.
- Structural test helpers may accept both ONNX IR and ONNX `ModelProto`-like
  objects.

Keep exceptions in those boundaries; do not let them leak into normal lowering
modules.
