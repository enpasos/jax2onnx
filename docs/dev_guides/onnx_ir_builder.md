# ONNX IR Builder Guide

This guide distills the guardrails we enforce around `onnx_ir._tape.Builder`: how to wire values, record initializers, and keep tests green now that the IR pipeline is builder-first.

## Policy Checklist
- Always pass `name=` when calling `builder.initializer(...)` or `ctx.builder.add_initializer_from_*`. `tests/extra_tests/framework/test_no_onnx_in_converter2_plugins2.py` verifies this.
- `_outputs` must be a list/tuple (or alias that resolves to one); string literals are rejected by `tests/extra_tests/framework/test_ir_builder_contracts.py` and `scripts/check_ir_builder_usage.py`.
- Keep converter/plugins IR-only—no `onnx` protobuf helpers—per the same policy suite.
- Run `scripts/check_ir_builder_usage.py` before sending patches (it is also wired into the pre-commit stack).

## Quick Checklist
- Emit ops through `ctx.builder` (or `_tape.Builder`) rather than constructing `ir.Node` manually. Fall back only in function-mode/legacy paths where the builder cannot express the behaviour.
- After every builder call, stamp dtype and shape with `_stamp_type_and_shape(...)` and register the value via `_ensure_value_info(...)` so later eqns see consistent metadata.
- Register constants via `builder.initializer(...)` / `ctx.bind_const_for_var(...)`; never smuggle tensors through ad-hoc `ir.Value(const_value=...)` without keeping the initializer list in sync.
- When defining plugin metadata, use `construct_and_call(...).with_requested_dtype(...).with_rng_seed(...)` to honour the single-use RNG policy and keep tests deterministic across f32/f64 runs.
- Run the validation hooks listed below (Ruff, builder usage checker, pytest) before landing a change; the pre-commit stack invokes them automatically.

## Plugin Metadata Requirements
- Construct callable metadata with `construct_and_call(...)` so the test harness can rebuild modules for each dtype. Pair it with `with_requested_dtype()` and `with_rng_seed(...)`/`with_prng_key(...)` helpers instead of inlining lambdas or seeding at import time.
- Avoid `callable_factory`. The test generator now raises if metadata still relies on factories—`callable` entries must be concrete `construct_and_call(...)` results.
- When you need constants inside plugin lowers, prefer shared helpers (for example, `_const_i64`) that delegate to `ctx.builder` so they participate in initializer bookkeeping.
- Respect the single-use RNG rule: split keys per consumer and never cache module instances inside traced calls—`construct_and_call(...).with_dtype(...)` already handles per-dtype reuse.

## Validation Hooks
- `tests/extra_tests/framework/test_no_onnx_in_converter2_plugins2.py` enforces both the "no protobuf" policy and initializer naming for every builder call.
- `tests/extra_tests/framework/test_ir_builder_contracts.py` walks the AST to guarantee `_outputs=` uses sequence types.
- `scripts/check_ir_builder_usage.py` wraps the same heuristics for local iteration and runs as a pre-commit hook. Invoke it manually with `poetry run python scripts/check_ir_builder_usage.py` when editing converter/plugins code.

Everything below expands on the why and how behind those rules.

## Prerequisites and Imports
- The ONNX IR package ships with ONNX Script and is available as `onnx_ir`; install `onnx-script` or `onnx-ir` and ensure runtime dependencies (notably `numpy`) are available.
- When working from a source checkout, set `PYTHONPATH=src` before importing.
- The builder lives in an internal module. Import it explicitly:

```python
import onnx_ir as ir
from onnx_ir._tape import Builder
```

> **Stability note**: `_tape.Builder` is currently internal API (the leading underscore is intentional) and can change. Keep the wrapper that instantiates it confined to XY so updates are easy.

## Core Concept
`Builder` subclasses `onnx_ir.tape.Tape`. It records nodes, initializers, and the opsets they require while exposing every ONNX operator as a dynamic method (for example, `builder.Add`, `builder.Conv`).

Use it when you want to script graph construction but still hand the collected nodes to `ir.Graph` or `ir.Function` later. If you need finer-grained control (custom outputs, metadata, overload selection, or pre-existing `ir.Value` objects), drop down to `Tape.op` / `Tape.op_multi_out` or construct `ir.Node` directly.

## End-to-End Workflow
```python
import numpy as np
import onnx_ir as ir
from onnx_ir._tape import Builder

# 1. Provide typed graph values up front.
X = ir.val("X", dtype=ir.DataType.FLOAT, shape=[1])
Y = ir.val("Y", dtype=ir.DataType.FLOAT, shape=[1])

# 2. Create a builder (optionally tie it to an existing graph/function).
builder = Builder()

# 3. Register any constant tensors through the builder so outputs stay in sync.
weight_init = builder.initializer(
    ir.tensor(np.array([0.25], dtype=np.float32)),
    name="weight",
)

# 4. Emit operators. Positional args become inputs; keyword args become ONNX attributes.
scaled = builder.Mul(X, weight_init, _outputs=["scaled"])  # returns ir.Value
summed = builder.Add(scaled, Y, _domain="", _version=18)

# 5. Package the recording into a graph/model when ready.
def to_opset_imports(used_opsets: set[tuple[str, int | None]]):
    result: dict[str, int] = {}
    for domain, version in used_opsets:
        if version is None:
            continue  # fall back to the containing graph's default
        previous = result.get(domain)
        if previous is not None and previous != version:
            raise ValueError(
                f"Mixed opset versions requested for domain '{domain}': {previous} vs {version}"
            )
        result[domain] = version
    return result or {"": 18}  # choose an explicit default for the model

graph = ir.Graph(
    inputs=[X, Y],
    outputs=[summed],
    nodes=builder.nodes,
    initializers=builder.initializers,
    opset_imports=to_opset_imports(builder.used_opsets),
    name="scale_and_sum",
)
model = ir.Model(graph=graph, ir_version=10)
```

## Bringing Existing Models Into the Builder
The official docs highlight converting `onnx.ModelProto` to the IR via `ir.from_proto` or `onnx_ir.load`. That makes it easy to combine scripted nodes with imported graphs:

```python
import onnx
import onnx_ir as ir
from onnx_ir._tape import Builder

model_proto = onnx.parser.parse_model(MODEL_TEXT)
model = ir.from_proto(model_proto)

builder = Builder(model.graph)
extra = builder.Identity(model.graph.outputs[0])
model.graph.outputs.append(extra)
```

You can reverse the process with `ir.to_proto(model)` when you need to serialize back to protobuf.

## What the Builder Does for You
- Tracks every created `ir.Node` in insertion order via `builder.nodes` so you can extend a graph or build a new one.
- Keeps initializers created through `builder.initializer` aligned with the eventual graph. When the builder is constructed with `graph_like=ir.Graph(...)`, the initializer is immediately registered on that graph.
- Records `builder.used_opsets` as `(domain, version)` pairs so you can populate `Graph.opset_imports` consistently.

## Reserved Keyword Arguments
`Builder` intercepts a few keyword arguments before treating the remainder as ONNX attributes:
- `_domain`: operator domain (default `""`).
- `_version`: opset version for the operator. Use one consistent value per domain.
- `_outputs`: either an `int` (number of outputs) or a *sequence* of output names.
  - When you pass a sequence, make it a list/tuple of strings; plain strings count as sequences of characters and will be split unintentionally.

Everything else in `**kwargs` is fed to `_convenience.convert_attributes`, which automatically turns Python scalars, sequences, tensors, and graphs into the right `ir.Attr` instances.

## Tape API Highlights
The public documentation for `onnx_ir.tape` at <https://onnx.ai/ir-py/api/ir_tape.html> spells out the signatures for `Tape.op`, `Tape.op_multi_out`, and `Tape.initializer`:
- `Tape.op` returns the first output `ir.Value` and accepts keyword-only arguments such as `overload`, `graph`, `name`, `doc_string`, `metadata_props`, and `output`.
- `Tape.op_multi_out` requires either `num_outputs` or `outputs` (but not both) and returns an immutable tuple of `ir.Value` objects.
- `Tape.initializer` insists on a name and on the provided tensor having `const_value` set; ONNX functions intentionally reject initializers.

Keep these signatures in mind when deciding between builder convenience and direct tape usage.

## Handling Multi-Output Operators
```python
values = builder.If(condition, _outputs=["then_out", "else_out"], _version=18)
then_out, else_out = values
```
- The return type is a tuple of `ir.Value`. Pull out the node again with `then_out.producer()` if you need to mutate metadata.
- For heterogeneous arity where ONNX requires empty slots, pass `None` in the positional inputs (for example, `builder.MaxPool(X, None, strides=[1, 1], _outputs=2)`).

## Managing Attributes Explicitly
- Python types are auto-inferred. For ambiguous cases (empty lists or `None`) create the attribute yourself: `builder.Cast(X, to=ir.Attr("to", ir.AttributeType.INT, 1))`.
- Tensor attributes should be created with `ir.tensor(...)` to guarantee dtype/shape correctness.
- Graph-typed attributes must be wrapped with `ir.AttrGraph` or `ir.AttrGraphs`.

## Integrating with Existing Graphs or Functions
```python
graph = ir.Graph(inputs=[X], outputs=[Z], nodes=[])
builder = Builder(graph)
intermediate = builder.Relu(X)
# The node is already appended to `graph`, and names are assigned by the graph's name authority.
```
- When bound to a graph, builder calls reuse the graph's naming authority and automatically respect graph invariants.
- Initializers are registered only for graphs. ONNX functions do not permit initializers, so the builder simply stores them locally when `graph_like` is an `ir.Function`.

## Limitations Compared to `Tape.op`
Because `_make_node` forwards the remaining keyword arguments into the attribute map, the builder cannot set certain `Tape` parameters at construction time:
- `overload`, `graph`, `name`, `doc_string`, `metadata_props`, and `output` are interpreted as attributes. Set them on the resulting node (`value.producer()`) after creation or call `Tape.op` directly when you need those parameters.
- To attach a node to a different graph than `builder.graph_like`, instantiate another builder or fall back to `Tape.op(graph=...)`.
- To reuse pre-created `ir.Value` outputs, call `Tape.op(output=existing_value)` or `Tape.op_multi_out(outputs=[...])` rather than relying on `_outputs`.

## Common Pitfalls and How to Avoid Them
- **Node metadata via kwargs**: `builder.Add(..., name="foo")` creates an attribute named `name`; it does *not* rename the node. Use `summed.producer().name = "foo"` after creation instead.
- **Doc strings & metadata props**: assign them on the node object (`node = summed.producer(); node.doc_string = "..."`).
- **Output naming**: pass a list (`_outputs=["y"]`), not a bare string.
- **Initializer naming**: provide a name whenever the tensor lacks one; `Tape.initializer` raises otherwise.
- **Multiple opset versions**: if two builder calls request different versions for the same domain, detect and reconcile before finishing the graph.
- **Optional inputs**: include explicit `None` placeholders to maintain positional semantics.
- **Attribute values of `None`**: build an `ir.Attr` with an explicit `AttributeType`; automatic conversion rejects bare `None`.
- **Graph ownership**: do not reuse a builder-generated node inside another graph without detaching it first (`graph.remove(node)`), because each node tracks its owning graph.

## Checklist Before Serializing
- All graph inputs/outputs are `ir.Value` instances with types and shapes populated (consider using `ir.val` for convenience).
- Initializers created through the builder are either registered on the target graph or injected via `graph.initializers.add(...)`.
- `graph.opset_imports` reflects the versions implied by `builder.used_opsets`.
- Any node-level metadata (names, doc strings, annotations, overloads) is set on the node objects after creation.
- Perform optional validation such as `ir.to_proto(model)`, ONNX checker runs, or `onnx_ir.load` round-trips if XY integrates them.

Keeping these conventions in one place ensures the "builder" layer stays predictable for Codex agents and humans alike, reducing churn when the upstream library evolves.

## Validation Routine
1. `poetry run python scripts/check_ir_builder_usage.py --diff` (lints only staged files; drop `--diff` to scan the whole tree).
2. `poetry run ruff check .` followed by `poetry run ruff format --check .` (or let the pre-commit hooks fix issues automatically).
3. `poetry run pytest -q` plus any focused suites you touched (for example `tests/primitives/test_jnp.py::Test_linspace`).
4. For builder-heavy refactors, run the structural policy tests directly: `poetry run pytest -q tests/extra_tests/framework/test_ir_builder_contracts.py`.
