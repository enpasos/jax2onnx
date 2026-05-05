# Using `onnx_ir` Types

`jax2onnx` depends on `onnx-ir>=0.2.1`. The package ships inline type
information and a `py.typed` marker, so mypy can check converter and plugin code
against the public ONNX IR API.

## Project Configuration

The repository keeps `onnx_ir` imports typed in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.11"
files = ["jax2onnx"]
follow_imports = "skip"
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = [
    "onnx_ir",
    "onnx_ir.*",
]
follow_imports = "normal"
```

Run the project typing check with:

```bash
./scripts/check_typing.sh
```

That wrapper runs mypy with the repo config and then reports RNG metadata traces.

## Constructing Typed IR Objects

Prefer `ir.val(...)`, `ir.tensor(...)`, `ir.Node`, `ir.Graph`, and `ir.Model`
directly when writing low-level tests or utility code:

```python
import numpy as np
import onnx_ir as ir


def build_constant_add() -> ir.Model:
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3])
    weight = ir.val(
        "weight",
        dtype=ir.DataType.FLOAT,
        shape=[3],
        const_value=ir.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
    )
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[3])

    node = ir.Node("", "Add", [x, weight], outputs=[y])
    graph = ir.Graph(
        inputs=[x],
        outputs=[y],
        nodes=[node],
        initializers=[weight],
        opset_imports={"": 23},
        name="constant_add",
    )
    return ir.Model(graph=graph, ir_version=10, producer_name="jax2onnx-test")
```

For plugin lowering, prefer `ctx.builder` instead of constructing `ir.Node`
manually. The builder keeps graph containers, initializers, names, and metadata
aligned with project policy.

## Typing Patterns

- Annotate lowerings with `LoweringContextProtocol` from
  `jax2onnx.converter.typing_support`.
- Use `IRBuilderProtocol` for helpers that only need the builder surface.
- Use `SymbolicDimOrigin` when passing symbolic-dimension provenance between the
  converter, `dim_as_value`, and shape helpers.
- Use `AxisOverrideInfo` / `AxisOverrideMap` only for loop/scan extent metadata
  where axis-0 overrides must survive nested graph rewrites.
- Use `PrimitiveLowering` and `FunctionLowering` protocols when dispatch code
  needs to accept plugin instances without depending on concrete base classes.

## What to Avoid

- Do not add local stubs for `onnx_ir`; rely on the package marker.
- Do not use `Any` to bypass IR types in normal lowering code. If a compatibility
  boundary needs duck typing, isolate it and document why.
- Do not copy ONNX protobuf access patterns into converter/plugin code. Keep
  protobuf compatibility inside test helpers and serialization boundaries.
- Do not mutate private graph mirrors when public containers or helpers exist.

## Troubleshooting

- **mypy cannot find `onnx_ir` types:** confirm dependencies are installed in the
  Poetry environment and that `onnx-ir>=0.2.1` is active.
- **A helper needs both ONNX IR and ONNX `ModelProto`:** keep the typed IR path
  first, then add a narrow duck-typed fallback at the boundary.
- **A plugin suddenly needs `cast(...)`:** prefer a small typed helper in
  `converter/typing_support.py`, `plugins/_ir_shapes.py`, or `jax2onnx.ir_utils`
  if the pattern will recur.
