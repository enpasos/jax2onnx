# `expect_graph` checklist for plugins

`expect_graph` (from `jax2onnx.plugins._post_check_onnx_graph`) is the lightweight
structural assertion helper used by plugin tests and examples. It lets a test
express the operators, ordering, and shapes that should appear in a converted
IR/ONNX graph without dumping the full model. This document captures the
conventions we rely on when writing or reviewing `post_check_onnx_graph`
expectations.

> **Test metadata reminder:** when wiring new examples/tests, construct callables
> with `construct_and_call(...)` and placeholder arguments such as
> `with_requested_dtype()` / `with_rng_seed(...)` so the harness can rebuild
> deterministic f32/f64 variants. See the builder guide for the full randomness
> and dtype rules.

## Import

```python
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
```

Alias it to `EG` inside tests to keep callsites short.

> **Builder reminder:** structural tests assume plugins emitted nodes via
> `ctx.builder`. Review the [ONNX IR Builder Guide](onnx_ir_builder.md) if
> `_outputs` naming or initializer wiring looks suspicious; policy tests now
> enforce those contracts.

## Basic usage

Pass a list of patterns to `expect_graph`. Each pattern is either a string or a
`(string, options)` tuple. Nodes are written in evaluation order with `->`
separating them.

```python
EG([
    "Transpose -> Conv -> Relu -> AveragePool",
])
```

The pattern above requires the graph to contain that exact operator chain.
Failing to find it raises an assertion with a summarized diff of the graph.

## Encoding shapes

Append `:shape` to a node name to assert the *output* shape of that node. Use
`x` separators (e.g. `Bx32x28x28`). Leave dimensions symbolic by reusing the
string symbol that the test harness passed as an input shape (for example
`"B"`).

```python
EG([
    "Gemm:Bx256 -> Relu:Bx256 -> Gemm:Bx10",
])
```

Write concrete integers for known static sizes (`3x1x28x28`). Symbols and
integers can be mixed (`B?x256` is not supported; prefer `symbols={"B": None}`
if you need to unify multiple strings).

## Spec Forms and Match Options

Use plain strings for simple paths. Use a mapping or `(path, options)` tuple when
you need predicates:

```python
EG([
    (
        "Transpose:3x1x28x28 -> Conv:3x32x28x28 -> Relu:3x32x28x28 -> Gemm:3x256",
        {
            "counts": {"Transpose": 1, "Conv": 1, "Relu": 1, "Gemm": 1},
            "inputs": {1: {"initializer_name": "kernel"}},
        },
    ),
],
no_unused_inputs=True,
mode="all",
must_absent=["Not"],
)
```

### Entry options (per pattern)

Use these in the options dictionary of a `(pattern, options)` tuple, or directly
inside a mapping spec with a `path` key:

- `attrs`: Map of operator name to required attribute values (e.g. `{"Softmax": {"axis": -1}}`).
- `counts`: map of op type to the exact number of occurrences expected.
- `graph`: graph/function selector. Use `"top"` for the top graph, a function
  name/domain selector, or the `fn:<selector>` prefix when `search_functions=True`.
- `inputs`: map of 0-based input index to constraints. Supported predicates are
  `{"const": value}`, `{"const_bool": bool}`, `{"initializer_name": "name"}`,
  and `{"absent": True}`.
- `must_absent`: list of operator names that must not appear anywhere in the searched graph(s).
- `symbols`: dictionary mapping symbolic dim labels to `None` (any value) or an integer.

### Global options (function arguments)

Pass these as keyword arguments to `expect_graph`:

- `mode`: one of `"all"` (default; all patterns must match) or `"any"` (at least one matches).
- `must_absent`: global list of operator names that must not appear anywhere.
- `no_unused_inputs`: when `True`, fail if the graph retains dangling inputs.
- `no_unused_function_inputs`: extend the check to imported ONNX function bodies (requires `search_functions=True`).
- `search_functions`: include imported ONNX Function bodies in the search. It
  does not walk arbitrary `If`/`Loop` graph attributes.
- `symbols`: dictionary mapping symbolic dim labels to `None` or an integer (unifies across all patterns).
- `explain_on_fail`: print the compact graph diagnostic when a match fails
  (enabled by default).

The matcher automatically walks through helper nodes that frequently sit on the
main data edge (by default we skip `Reshape`, `Identity`, `Cast`, `CastLike`,
`Squeeze`, `Unsqueeze`, `Flatten`, `Shape`, `Gather`, `Concat`, `Add`, and
`Where`). This lets a single pattern cover sequential graphs where tensors fan
out into shape-building side chains, such as the CNN dynamic example where the
`Transpose` output feeds both `Reshape` and the shape-construction subgraph.

### Function naming compatibility

Function exports now keep the original callable name as the node `op_type`
(`TransformerBlock`, `MLPBlock`, …) and move the numeric suffix into
`node.name`/`domain` (`TransformerBlock_2`, `custom.TransformerBlock_2`, …). To
keep older expectations valid, `expect_graph` automatically strips trailing
`_123` suffixes when comparing `op_type` and normalises graph filters such as
`fn:custom.TransformerBlock_2`. Prefer matching on the base `op_type` unless a
specific call-site needs to be distinguished by name.

## Practical tips

- Prefer a single path that covers the interesting operators rather than every
  node in the graph. Keep counts strict if extra occurrences would signal a
  regression.
- Include shapes for layers where layout or dimension handling is important
  (transposes, pooling, reshapes). Shape assertions catch missing
  `_stamp_type_and_shape` calls and layout errors quickly.
- Keep expectations small for dynamic tests; the static counterpart usually
  asserts shapes, while a dynamic test covers symbolic behaviour or flags.
- Use `mode="all"` with multiple patterns to check disjoint subgraphs. Use
  `mode="any"` when an operator can lower through more than one valid shape or
  opset path.
- If the graph contains fused or optimizer-inserted elementwise ops, anchor the
  pattern on the surrounding operators and rely on `counts` to constrain the
  totals.


## Maintaining Expectations

- Every plugin/example should keep its `expect_graph(...)` snippet next to the
  testcase metadata that owns the behavior.
- Regenerate a candidate snippet with
  `poetry run python scripts/emit_expect_graph.py <testcase>` when lowering
  behavior changes, then simplify the output before committing it.
- Keep expectations focused on durable structure. Do not paste a full generated
  graph into metadata unless the complete graph shape is the contract.
- Run the focused generated pytest target for the component you touched before
  widening to the broader suite.

## Guardrails

- Converter/plugins must remain ONNX-IR only; do not import ONNX protobuf types
  in lowering code.
- Use `construct_and_call(...)` with `with_requested_dtype()` and
  `with_rng_seed(...)` placeholders; split PRNG keys before reuse.
- Attention plugins must retain masked-weight normalisation; structural checks
  should reflect the normalised path when that path is part of the contract.
- Run core tooling (`poetry run pytest -q`, `poetry run ruff check .`,
  `./scripts/check_typing.sh`) for larger sweeps.

## Where to Use It

`post_check_onnx_graph` entries appear inside example/plugin test metadata (see
`jax2onnx/plugins/examples/nnx/cnn.py` for a compact reference). The helper works
with ONNX IR models and with ONNX `ModelProto`-like objects used by policy tests
under `tests/extra_tests`.

When adding new metadata entries, seed them with a minimal structural check, run
the example once to capture the intended op sequence, and then layer on shape
assertions, counts, attributes, or input predicates only where they prevent a
real regression.

## Known Boundaries

- `search_functions=True` searches imported ONNX Function bodies. It is not a
  general recursive control-flow graph walker.
- Shape matching depends on available metadata. If a value has no stamped shape,
  anchor the check on operator order or counts instead.
- The helper is a structural assertion tool, not a semantic verifier. Keep
  numeric correctness in the generated test comparison unless the structural
  shape itself is the behavior under test.
