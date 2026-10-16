# Using `onnx_ir` Types

`jax2onnx` depends on `onnx-ir>=1.0.0`. The package ships inline type
information and a `py.typed` marker, so mypy can check converter and plugin code
against the public ONNX IR API.

## Project Configuration

The repository keeps `onnx_ir` imports typed in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
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

- Annotate new native lowerings with `NativeLoweringContext` from
  `jax2onnx.converter.typing_support`. Its `builder` is already the closed
  `IRBuilderAPI`, so direct calls are checked without a local type annotation.
- Use `ConverterContext` for internal equation dispatch and Function/scope
  management. It includes the converter's bookkeeping and compatible builder
  surface. `LoweringContextProtocol` remains a legacy alias for that contract;
  existing plugin annotations continue to work.
- Use the closed `IRBuilderAPI` for helpers that only need the builder surface.
  Its methods and input/output types are checked statically. The compatibility
  `IRBuilderProtocol` adds unrestricted dynamic operator shorthand for existing
  leaf plugins; it cannot reject unknown operator attributes.
- Prefer its explicitly typed `op` / `op_multi_out` methods. `op` returns
  `ir.Value`; `op_multi_out` returns `Sequence[ir.Value]`. Optional input slots
  are typed `ir.Value | None`, and `output=` / `outputs=` accept existing IR
  values without casts or result-shape guessing.
- Use `SymbolicDimOrigin` when passing symbolic-dimension provenance between the
  converter, `dim_as_value`, and shape helpers.
- Use `AxisOverrideInfo` / `AxisOverrideMap` only for loop/scan extent metadata
  where axis-0 overrides must survive nested graph rewrites.
- Use `PrimitiveLowering` and `FunctionLowering` protocols when dispatch code
  needs to accept plugin instances without depending on concrete base classes.

### A Typed Builder Call

```python
import onnx_ir as ir

from jax2onnx.converter.typing_support import IRBuilderAPI


def transpose(builder: IRBuilderAPI, value: ir.Value) -> ir.Value:
    output = ir.val("transposed", dtype=value.dtype, shape=[3, 2])
    return builder.op(
        "Transpose",
        [value],
        {"perm": ir.AttrInt64s("perm", [1, 0])},
        output=output,
    )
```

Use public `ir.Attr` constructors for sequence/tensor/graph attributes and
scalar Python values for simple attributes. The public ONNX IR module does
not expose Tape's complete attribute union; do not import that alias from a
private module or create local stubs. The dynamic operator shorthand retains
its existing compatibility contract, while the typed methods make inputs,
outputs, and node metadata parameters checkable at each call site.

See the [typed emission pattern](onnx_ir_builder.md#typed-emission-pattern)
for the complete lowering contract, with `NegPlugin` and `TopKPlugin` as
executable single-/multi-output examples.
The shared unary helper in `plugins/jax/numpy/_unary_utils.py`, Function call
emission, and layout output binding apply the same pattern in production paths.
For graph transformations, the
[graph rewrite reference](ir_optimizer.md#graph-rewrite-reference) uses the same
public IR value types and checks ownership after replacement.

## What to Avoid

- Do not add local stubs for `onnx_ir`; rely on the package marker.
- Do not use `Any` to bypass IR types in normal lowering code. If a compatibility
  boundary needs duck typing, isolate it and document why.
- Do not copy ONNX protobuf access patterns into converter/plugin code. Keep
  protobuf compatibility inside test helpers and serialization boundaries.
- Do not mutate private graph mirrors when public containers or helpers exist.

## Troubleshooting

- **mypy cannot find `onnx_ir` types:** confirm dependencies are installed in the
  Poetry environment and that `onnx-ir>=1.0.0` is active.
- **A helper needs both ONNX IR and ONNX `ModelProto`:** keep the typed IR path
  first, then add a narrow duck-typed fallback at the boundary.
- **A plugin suddenly needs `cast(...)`:** prefer a small typed helper in
  `converter/typing_support.py`, `plugins/_ir_shapes.py`, or `jax2onnx.ir_utils`
  if the pattern will recur.
