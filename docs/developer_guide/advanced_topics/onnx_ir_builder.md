# ONNX IR Builder Guide

This guide distills the guardrails we enforce around the public
`onnx_ir.tape.Tape` API and the project wrapper
`jax2onnx.converter.ir_builder.IRBuilder`: how to wire values, record constants,
and keep tests green now that the converter is IR-first.

## Policy Checklist
- Always pass `name=` when calling `builder.initializer(...)` or `ctx.builder.add_initializer_from_*`. `tests/extra_tests/framework/test_no_onnx_in_converter_plugins.py` verifies this.
- `_outputs` must be a list/tuple (or alias that resolves to one); string literals are rejected by `tests/extra_tests/framework/test_ir_builder_contracts.py` and `scripts/check_ir_builder_usage.py`.
- Keep converter/plugins IR-only—no `onnx` protobuf helpers—per the same policy suite.
- Run `scripts/check_ir_builder_usage.py` before sending patches (it is also wired into the pre-commit stack).

## Quick Checklist
- Emit ops through `ctx.builder` rather than constructing `ir.Node` manually.
  For lower-level construction, use the public `Tape.op` / `Tape.op_multi_out`
  methods that `IRBuilder` forwards.
- After every builder call, stamp dtype and shape with `_stamp_type_and_shape(...)` and run `_ensure_value_metadata(...)` so the `ir.Value` carries normalized shape/type metadata (no separate `value_info` bucket).
- Register constants via `builder.initializer(...)` / `ctx.bind_const_for_var(...)`; never smuggle tensors through ad-hoc `ir.Value(const_value=...)` without keeping the initializer list in sync.
- When defining plugin metadata, use `construct_and_call(...)` with placeholder
  arguments such as `with_requested_dtype()` and `with_rng_seed(...)` to honour
  the single-use RNG policy and keep tests deterministic across f32/f64 runs.
- Run the validation hooks listed below (Ruff, builder usage checker, pytest) before landing a change; the pre-commit stack invokes them automatically.

## Plugin Metadata Requirements
- Construct callable metadata with `construct_and_call(...)` so the test harness can rebuild modules for each dtype. Pair it with `with_requested_dtype()` and `with_rng_seed(...)`/`with_prng_key(...)` helpers instead of inlining lambdas or seeding at import time.
- Avoid `callable_factory`. The test generator now raises if metadata still relies on factories—`callable` entries must be concrete `construct_and_call(...)` results.
- When you need constants inside plugin lowers, prefer shared helpers (for example, `_const_i64`) that delegate to `ctx.builder` so they participate in initializer bookkeeping.
- Respect the single-use RNG rule: split keys per consumer and never cache
  module instances inside traced calls. `construct_and_call(...)` plus
  `with_requested_dtype()` placeholders handles per-dtype rebuilds.

## Validation Hooks
- `tests/extra_tests/framework/test_no_onnx_in_converter_plugins.py` enforces
  the "no protobuf" and "no private onnx-ir API" policies plus initializer
  naming for every builder call.
- `tests/extra_tests/framework/test_ir_builder_contracts.py` walks the AST to guarantee `_outputs=` uses sequence types.
- `scripts/check_ir_builder_usage.py` wraps the same heuristics for local iteration and runs as a pre-commit hook. Invoke it manually with `poetry run python scripts/check_ir_builder_usage.py` when editing converter/plugins code.

Everything below expands on the why and how behind those rules.

## Prerequisites and Imports
- The ONNX IR package ships with ONNX Script and is available as `onnx_ir`; install `onnx-script` or `onnx-ir` and ensure runtime dependencies (notably `numpy`) are available.
- When working from a source checkout, set `PYTHONPATH=src` before importing.
- Import the public tape API for low-level graph construction, or the project
  wrapper for converter/plugin work:

```python
import onnx_ir as ir
from onnx_ir.tape import Tape

from jax2onnx.converter.ir_builder import IRBuilder
```

> **Stability note**: Do not import `onnx_ir._tape`. `onnx-ir` 1.x documents
> `onnx_ir.tape.Tape` as the public API. `IRBuilder` implements its dynamic
> `builder.Add(...)` shorthand locally on top of that public surface.

> **Legacy note**: The converter no longer maintains a `builder.value_info` list. Plugins should rely exclusively on `_ensure_value_metadata(...)` and the fields on each `ir.Value` when they need shape/type information. Avoid appending to or expecting a global `value_info` registry.

## Core Concept
`Tape` records nodes, initializers, and the opsets they require. `IRBuilder`
adds the dynamic operator methods used throughout the converter (for example,
`builder.Add` and `builder.Conv`) without depending on an upstream private
builder class.

Use `IRBuilder` for normal converter work. If you need finer-grained control
(custom outputs, metadata, overload selection, or pre-existing `ir.Value`
objects), call its forwarded `Tape.op` / `Tape.op_multi_out` methods. Use a
standalone `Tape` when extending an existing `ir.Graph` or `ir.Function`.

## End-to-End Workflow
```python
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import IRBuilder

# 1. Provide typed graph values up front.
X = ir.val("X", dtype=ir.DataType.FLOAT, shape=[1])
Y = ir.val("Y", dtype=ir.DataType.FLOAT, shape=[1])

# 2. Create the project builder and register graph inputs.
builder = IRBuilder(opset=18, enable_double_precision=False)
builder.inputs.extend([X, Y])

# 3. Register any constant tensors through the builder so outputs stay in sync.
weight_init = builder.add_initializer_from_array(
    name="weight",
    array=np.array([0.25], dtype=np.float32),
)

# 4. Emit operators. Positional args become inputs; keyword args become ONNX attributes.
scaled = builder.Mul(X, weight_init, _outputs=["scaled"])  # returns ir.Value
summed = builder.Add(scaled, Y, _domain="", _version=18)

# 5. Mark graph outputs and package the graph into a model.
builder.outputs.append(summed)
model = builder.to_ir_model(name="scale_and_sum", ir_version=10)
```

## Bringing Existing Models Into a Tape
The official docs highlight converting `onnx.ModelProto` to the IR via `ir.from_proto` or `onnx_ir.load`. That makes it easy to combine scripted nodes with imported graphs:

```python
import onnx
import onnx_ir as ir
from onnx_ir.tape import Tape

model_proto = onnx.parser.parse_model(MODEL_TEXT)
model = ir.from_proto(model_proto)

tape = Tape(model.graph)
extra = tape.op("Identity", [model.graph.outputs[0]])
model.graph.outputs.clear()
model.graph.outputs.append(extra)
```

You can reverse the process with `ir.to_proto(model)` when you need to serialize back to protobuf.

## What Tape and IRBuilder Do for You
- `Tape` tracks every node it creates in insertion order and records
  `used_opsets` as `(domain, version)` pairs.
- `Tape.initializer(...)` immediately registers the value when the tape is bound
  to an `ir.Graph`.
- `IRBuilder` owns the converter graph, exposes its live node/input/output
  containers, adds initializer deduplication, and preserves the established
  dynamic operator shorthand.

## Reserved Keyword Arguments
The dynamic `IRBuilder` shorthand intercepts a few keyword arguments before
treating the remainder as ONNX attributes:
- `_domain`: operator domain (default `""`).
- `_version`: opset version for the operator. Use one consistent value per domain.
- `_outputs`: either an `int` (number of outputs) or a *sequence* of output names.
  - When you pass a sequence, make it a list/tuple containing only strings.
    Plain strings, bytes, and sequences containing non-string values raise
    `TypeError` before a node is added to the graph.

Everything else in `**kwargs` is fed to `_convenience.convert_attributes`, which automatically turns Python scalars, sequences, tensors, and graphs into the right `ir.Attr` instances.

## Tape API Highlights
The public documentation for `onnx_ir.tape` at <https://onnx.ai/ir-py/api/ir_tape.html> spells out the signatures for `Tape.op`, `Tape.op_multi_out`, and `Tape.initializer`:
- `Tape.op` returns the first output `ir.Value` and accepts keyword-only arguments such as `overload`, `graph`, `name`, `doc_string`, `metadata_props`, and `output`.
- `Tape.op_multi_out` requires either `num_outputs` or `outputs` (but not both)
  and returns a sequence of `ir.Value` objects.
- `Tape.initializer` requires `name=` unless the tensor itself is named. A tape
  bound to an ONNX function keeps the initializer only on the tape because
  functions cannot register graph initializers.

Keep these signatures in mind when deciding between builder convenience and direct tape usage.

## Handling Multi-Output Operators
```python
values, indices = builder.TopK(
    X,
    K,
    axis=-1,
    _outputs=["top_values", "top_indices"],
    _version=18,
)
```
- The return type is a sequence of `ir.Value`. Pull out the node again with
  `values.producer()` if you need to mutate metadata.
- For heterogeneous arity where ONNX requires empty slots, pass `None` in the positional inputs (for example, `builder.MaxPool(X, None, strides=[1, 1], _outputs=2)`).

## Managing Attributes Explicitly
- Python types are auto-inferred. For ambiguous cases (empty lists or `None`) create the attribute yourself: `builder.Cast(X, to=ir.Attr("to", ir.AttributeType.INT, 1))`.
- Tensor attributes should be created with `ir.tensor(...)` to guarantee dtype/shape correctness.
- Graph-typed attributes must be wrapped with `ir.AttrGraph` or `ir.AttrGraphs`.

## Graph Ownership & Cloning
- `IRBuilder` now keeps its `inputs`, `outputs`, and nodes in sync with the underlying `onnx_ir.Graph` via proxy setters. Reassigning `builder.inputs = [...]` (or `.outputs`/`.nodes`) clears and repopulates the graph-side containers, while `builder.initializers` remains a list-like shim that delegates to `graph.initializers`. Prefer mutating these sequences in place, but reassignment is safe when you need to reset them.
- When exporting a staged graph—either to an `ir.Model` or into ONNX graph-typed attributes—clone it first using `jax2onnx.converter.ir_clone.clone_graph`. The helper copies values, initializers, metadata, and nested graphs so the detached graph can be owned by another model/function without triggering “Value … is already owned by a different graph” errors. Function scopes and control-flow plugins (`cond`, `fori_loop`, `scan`, `while_loop`) already adopt this pattern; follow suit for any new subgraph emission.

## Integrating with Existing Graphs or Functions
```python
graph = ir.Graph(inputs=[X], outputs=[Z], nodes=[])
tape = Tape(graph)
intermediate = tape.op("Relu", [X])
# The node is already appended to `graph`, and names are assigned by the graph's name authority.
```
- When bound to a graph, tape calls reuse the graph's naming authority and
  automatically respect graph invariants.
- Initializers are registered only for graphs. ONNX functions do not permit
  initializers, so the tape simply stores them locally when `graph_like` is an
  `ir.Function`.

## Limitations of the Dynamic Shorthand
Because the dynamic shorthand forwards the remaining keyword arguments into
the attribute map, it cannot set certain `Tape` parameters at construction
time:
- `overload`, `graph`, `name`, `doc_string`, `metadata_props`, and `output` are interpreted as attributes. Set them on the resulting node (`value.producer()`) after creation or call `Tape.op` directly when you need those parameters.
- To attach a node to a different graph, use `builder.op(..., graph=target_graph)`
  or create a standalone tape for that graph.
- To reuse pre-created `ir.Value` outputs, call
  `builder.op(..., output=existing_value)` or
  `builder.op_multi_out(..., outputs=[...])` rather than relying on `_outputs`.

## Common Pitfalls and How to Avoid Them
- **Node metadata via kwargs**: `builder.Add(..., name="foo")` creates an attribute named `name`; it does *not* rename the node. Use `summed.producer().name = "foo"` after creation instead.
- **Doc strings & metadata props**: assign them on the node object (`node = summed.producer(); node.doc_string = "..."`).
- **Debug provenance metadata**: setting `JAX2ONNX_ENABLE_STACKTRACE_METADATA=1` (or `stacktrace_metadata=True`) records a concise call-site (`pkg.jax2onnx.callsite`, formatted as `function:line`) plus the plugin invocation site (`pkg.jax2onnx.plugin`, formatted as `Plugin.lower:line` pointing at the builder call) on each node. This is the default reduced payload surfaced in tools like Netron. Set `JAX2ONNX_STACKTRACE_DETAIL=full` when you also need the full Python (`pkg.jax2onnx.stacktrace`) and JAX (`pkg.jax2onnx.jax_traceback`) traces.

  Example (line numbers annotated to mirror the metadata):
  ```python
  def wide_fn(x):
      a = jnp.sin(x)   # wide_fn.py:8
      b = jnp.cos(x)   # wide_fn.py:9
      c = jnp.tanh(x)  # wide_fn.py:10
      d = jnp.exp(x)   # wide_fn.py:11
      return a + b * c + d  # wide_fn.py:12
  ```
- **Output naming**: pass a list (`_outputs=["y"]`), not a bare string; bare
  strings are rejected with `TypeError`.
- **Initializer naming**: provide a name whenever the tensor lacks one; `Tape.initializer` raises otherwise.
- **Multiple opset versions**: if two builder calls request different versions for the same domain, detect and reconcile before finishing the graph.
- **Optional inputs**: include explicit `None` placeholders to maintain positional semantics.
- **Attribute values of `None`**: build an `ir.Attr` with an explicit `AttributeType`; automatic conversion rejects bare `None`.
- **Graph ownership**: do not reuse a builder-generated node or value inside
  another graph directly. Clone staged graphs with
  `jax2onnx.converter.ir_clone.clone_graph` before moving them into a model,
  control-flow body, or graph-typed attribute.

## Initializer Deduplication
- Prefer `ctx.builder.add_initializer_from_scalar/array(...)` or `ctx.builder.const_i64(...)` to create constants. Avoid writing directly to `graph.initializers[...]`.
- The upstream `GraphInitializers.add(value)` overwrites by name. Our builder layer enforces a stricter policy to preserve IR value connections:
  - Identical duplicate (same name + same data/shape/dtype) → reuse existing initializer; do not replace the object.
  - Conflicting duplicate (same name + different payload) → raise a `ValueError`.
  - In function-mode, constants are emitted as `Constant` nodes; graph initializers are not allowed in ONNX Functions.

Example
```python
import numpy as np

w1 = builder.add_initializer_from_array("weight", np.array([1.0], dtype=np.float32))
# Re-adding with identical payload reuses the same Value (no-op):
w2 = builder.add_initializer_from_array("weight", np.array([1.0], dtype=np.float32))
assert w1 is w2

# Re-adding with different payload raises:
builder.add_initializer_from_array("weight", np.array([2.0], dtype=np.float32))  # ValueError
```

Rationale
- Preserving object identity prevents subtle mismatches where nodes still reference the old `ir.Value` even though the `graph.initializers` dict now points to a new one. This keeps optimizer passes, cloning, and structural tests stable and predictable.

## Checklist Before Serializing
- All graph inputs/outputs are `ir.Value` instances with types and shapes populated (consider using `ir.val` for convenience).
- Initializers created through the builder are either registered on the target graph or injected via `graph.initializers.add(...)`.
  - Duplicate policy: attempting to re-add an initializer with the same name and different data raises. Re-adding an identical initializer reuses the existing value without replacing it, preserving value connections.
- `graph.opset_imports` reflects the versions implied by `builder.used_opsets`.
- Any node-level metadata (names, doc strings, annotations, overloads) is set on the node objects after creation.
- Perform optional validation such as `ir.to_proto(model)`, ONNX checker runs,
  or `onnx_ir.load` round-trips when the code path serializes models.

Keeping these conventions in one place ensures the "builder" layer stays predictable for Codex agents and humans alike, reducing churn when the upstream library evolves.

## Validation Routine
1. `poetry run python scripts/check_ir_builder_usage.py --diff` (lints only staged files; drop `--diff` to scan the whole tree).
2. `poetry run ruff check .` followed by `poetry run black --check .` (or let the pre-commit hooks fix issues automatically).
3. `poetry run pytest -q` plus any focused suites you touched (for example `tests/primitives/test_jnp.py::Test_linspace`).
4. For builder-heavy refactors, run the structural policy tests directly: `poetry run pytest -q tests/extra_tests/framework/test_ir_builder_contracts.py`.
