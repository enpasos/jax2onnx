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
  Prefer typed `op` / `op_multi_out` calls for new or touched lowerings. Their
  explicit signatures forward the public `Tape` API and retain project metadata.
- Preserve dtype and shape on explicit output values. Attach missing metadata to
  intermediates with `_stamp_type_and_shape(...)` and normalize it with
  `_ensure_value_metadata(...)` where needed; there is no `value_info` bucket.
- Register constants via named `builder.add_initializer_from_*` helpers or
  `ctx.bind_const_for_var(...)` so Function mode emits `Constant` nodes.
  Do not bypass their scope policy with raw initializer insertion.
- When defining plugin metadata, use `construct_and_call(...)` with placeholder
  arguments such as `with_requested_dtype()` and `with_rng_seed(...)` to honour
  the single-use RNG policy and keep tests deterministic across f32/f64 runs.
- Run Ruff, typing, builder-policy checks, and appropriate pytest coverage before
  landing a change. Pre-commit runs the static hooks; pytest runs separately.

## Plugin Metadata Requirements
- Construct module metadata with `construct_and_call(...)` so the test harness
  can rebuild state for each dtype. Pair it with `with_requested_dtype()` and
  `with_rng_seed(...)`/`with_prng_key(...)`; do not create seeded module state in
  import-time lambdas.
- Avoid `callable_factory`. Module/state/RNG construction uses concrete
  `construct_and_call(...)` results; stateless pure callables can be supplied
  directly in testcase metadata.
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

Use `IRBuilder` for normal converter work. Its typed `op` / `op_multi_out`
methods expose custom outputs, metadata, overload selection, and pre-existing
`ir.Value` objects directly. Use a standalone `Tape` when extending an existing
`ir.Graph` or `ir.Function`.

Type new emission helpers with the closed `IRBuilderAPI` from
`converter.typing_support`. Existing `IRBuilderProtocol` adds dynamic shorthand
for compatibility; native helpers do not need that unrestricted attribute API.

## Typed Emission Pattern

The canonical lowering reads required inputs, allocates explicit outputs, and
binds the emitted values. `NegPlugin` is the single-output reference:

```python
from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.typing_support import IRBuilderAPI, LoweringContextProtocol


def lower(ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
    x = ctx.require_value_for_var(eqn.invars[0])
    outvar = eqn.outvars[0]
    output = ctx.allocate_value_for_var(outvar, name_hint=ctx.fresh_name("neg_out"))
    builder: IRBuilderAPI = ctx.builder
    result = builder.op("Neg", [x], output=output)
    ctx.bind_value_for_var(outvar, result)
```

The shared NumPy unary helper, Function call emission, and layout output binding
use the same required-input/explicit-output contract. This avoids duplicating
dynamic result inspection across individual operators.

`op` returns `ir.Value`. Passing `output=` keeps the placeholder's name, dtype,
shape, and identity; the node becomes its producer. It does not infer missing
metadata. Set dtype/shape on new intermediate values, using
`_stamp_type_and_shape` and `_ensure_value_metadata` when shape normalization is
needed. An explicit output must not already have a producer.

`op_multi_out` returns `Sequence[ir.Value]`. Specify either `outputs=` with
existing values or `num_outputs=`; specifying both fails before graph mutation.
For example, split a tensor evenly with the explicit output-count attribute:

```python
first = ir.val("first", dtype=ir.DataType.FLOAT, shape=[2])
second = ir.val("second", dtype=ir.DataType.FLOAT, shape=[2])
values = builder.op_multi_out(
    "Split",
    [x],
    {"axis": 0, "num_outputs": 2},
    outputs=[first, second],
    version=18,
)
assert values[0] is first
```

Optional ONNX input slots use `None`. Clip with no lower bound keeps its missing
middle input explicit:

```python
maximum = builder.add_initializer_from_scalar(name="clip_max", value=1.0)
clipped = builder.op(
    "Clip",
    [x, None, maximum],
    output=ir.val("clipped", dtype=x.dtype, shape=x.shape),
)
```

`TopKPlugin` is the multi-output lowering reference. It emits a typed `TopK`
output sequence, preserves the ONNX INT64 index output, and casts indices to
the JAX output dtype before binding them.

The typed attribute map accepts `ir.Attr` and Python scalar `str`/`int`/`float`
values. For sequences, tensors, and graphs, use public constructors such as
`ir.AttrInt64s`, `ir.AttrTensor`, and `ir.AttrGraph`. This avoids relying on the
upstream private attribute union:

```python
transposed = builder.op("Transpose", [x], {"perm": ir.AttrInt64s("perm", [1, 0])})
```

Both typed methods retain stacktrace/plugin metadata and Tape opset tracking.
The dynamic `builder.Neg(...)` shorthand remains supported for existing code.
Keep constants in the named initializer helpers so graph and function scopes
retain their established constant representation.
The [graph rewrite reference](ir_optimizer.md#graph-rewrite-reference) shows
the corresponding value/metadata contract when transforming an existing graph.

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
- When attaching a staged graph to another model/function or graph attribute,
  use the public `graph.clone(allow_outer_scope_values=True)` API. It copies
  graph-owned values, nodes, initializers, metadata, and nested graphs while
  retaining lexical references to outer values. Use it only when those outer
  references are valid for the target scope. Function scopes and control-flow
  plugins already use this pattern; no project-specific clone helper is needed.

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
- **Debug provenance metadata**: setting `JAX2ONNX_ENABLE_STACKTRACE_METADATA=1` (or `IRBuilder(..., enable_stacktrace_metadata=True)`) records a concise call-site (`pkg.jax2onnx.callsite`, formatted as `function:line`) plus the plugin invocation site (`pkg.jax2onnx.plugin`, formatted as `Plugin.lower:line` pointing at the builder call) on each node. This is the default reduced payload surfaced in tools like Netron. Set `JAX2ONNX_STACKTRACE_DETAIL=full` when you also need the full Python (`pkg.jax2onnx.stacktrace`) and JAX (`pkg.jax2onnx.jax_traceback`) traces.

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
  the public `graph.clone(allow_outer_scope_values=True)` API before attaching
  them elsewhere, retaining only valid lexical outer references.

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
