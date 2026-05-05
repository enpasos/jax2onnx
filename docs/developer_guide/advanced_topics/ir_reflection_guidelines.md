# IR Reflection & Typed API Guidelines

Use typed `onnx_ir` APIs by default. Reflection and duck typing are still useful
at compatibility boundaries, but they should be deliberate and isolated so the
converter remains maintainable across `onnx_ir` releases.

## Default Rule

- Treat `ir.Model`, `ir.Graph`, `ir.Node`, `ir.Value`, `ir.Attr`, and `ir.Shape`
  as the canonical runtime model inside `converter/` and `plugins/`.
- Prefer `value.dtype`, `value.shape`, `value.producer()`,
  `value.consumers()`, `value.is_graph_input()`, and
  `value.is_graph_output()` over probing private fields.
- Prefer graph/container helpers such as `graph.all_nodes()`, `graph.remove(...)`,
  `graph.subgraphs(...)`, and `ir.convenience.replace_all_uses_with(...)`.
- Keep lowering code ONNX-IR only. Do not import ONNX protobuf types in
  `converter/` or `plugins/`.

## Allowed Reflection Boundaries

Reflection is acceptable when the code is intentionally bridging multiple graph
representations:

- Structural test helpers such as `plugins/_post_check_onnx_graph.py`, which
  accept both ONNX IR objects and ONNX `ModelProto`-like objects.
- Patching and plugin-discovery code that must inspect optional JAX/Flax/Equinox
  attributes across versions.
- Narrow compatibility shims where an `onnx_ir` release exposes equivalent state
  through different public containers.

Keep those shims close to the boundary. Do not copy their duck-typed access
patterns into normal lowering code.

## Rewrite Rules

- When eliminating or replacing nodes, preserve graph outputs first, then call
  `ir.convenience.replace_all_uses_with(...)`, then remove nodes through
  `graph.remove(...)`.
- When traversing nested graphs, prefer `graph.subgraphs(...)` or
  `graph.all_nodes()` where that gives the required scope. If a helper only
  supports top-level graphs or imported ONNX Functions, say so in its docs.
- When reading shape/dtype metadata, use shared helpers from
  `plugins/_ir_shapes.py`, `jax2onnx.ir_utils`, and `converter/typing_support.py`
  instead of cloning conversion logic locally.
- When a fallback `getattr(...)` is necessary, make the expected shapes explicit
  with type annotations and keep the fallback branch small.

## Audit Workflow

1. Run `poetry run python scripts/audit_ir_dynamic_access.py` when changing
   converter/plugin graph manipulation.
2. Classify new dynamic access as either a boundary shim or a typing gap.
3. Replace typing gaps with typed helpers before broadening mypy coverage.
4. Verify with the focused pytest target plus `./scripts/check_typing.sh`.
