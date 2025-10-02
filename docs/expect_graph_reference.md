# `expect_graph` checklist for plugins

`expect_graph` (from `jax2onnx.plugins2._post_check_onnx_graph`) is the lightweight
structural assertion helper used by plugin tests and examples. It lets a test
express the operators, ordering, and shapes that should appear in a converted
IR/ONNX graph without dumping the full model. This document captures the
conventions we rely on when writing or reviewing `post_check_onnx_graph`
expectations.

## Import

```python
from jax2onnx.plugins2._post_check_onnx_graph import expect_graph as EG
```

Alias it to `EG` inside tests to keep callsites short.

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

## Additional match options

Attach an options dictionary to require counts, forbid nodes, or tweak the
search.

```python
EG([
    (
        "Transpose:3x1x28x28 -> Conv:3x32x28x28 -> Relu:3x32x28x28 -> Gemm:3x256",
        {
            "counts": {"Transpose": 1, "Conv": 1, "Relu": 1, "Gemm": 1},
        },
    ),
],
no_unused_inputs=True,
mode="all",
must_absent=["Not"],
)
```

Common fields:

- `counts`: map of op type to the exact number of occurrences expected.
- `must_absent`: list of operator names that must not appear anywhere.
- `symbols`: dictionary mapping symbolic dim labels to `None` (any value) or an
  integer (specific size). Use it when multiple patterns should share the same
  symbolic dimension.
- `mode`: one of `"all"` (default; all patterns must match), `"any"` (at least
  one matches), or `"exact"` (the entire graph must equal the pattern).
- `no_unused_inputs`: when `True`, fail if the graph retains dangling inputs
  after conversion.
- `search_functions`: include function bodies (control-flow subgraphs) in the
  search.

The matcher automatically walks through helper nodes that frequently sit on the
main data edge (by default we skip `Reshape`, `Identity`, `Cast`, `CastLike`,
`Squeeze`, `Unsqueeze`, `Flatten`, `Shape`, `Gather`, `Concat`, `Add`, and
`Where`). This lets a single pattern cover sequential graphs where tensors fan
out into shape-building side chains, such as the CNN dynamic example where the
`Transpose` output feeds both `Reshape` and the shape-construction subgraph.

## Practical tips

- Prefer a single path that covers the interesting operators rather than every
  node in the graph. Keep counts strict if extra occurrences would signal a
  regression.
- Include shapes for layers where layout or dimension handling is important
  (transposes, pooling, reshapes). Shape assertions catch missing
  `_stamp_type_and_shape` calls and layout errors quickly.
- Keep expectations small for dynamic tests; the static counterpart usually
  asserts shapes, while a dynamic test covers symbolic behaviour or flags.
- Use `mode="all"` with multiple patterns to check disjoint subgraphs, or
  `mode="exact"` when the entire graph must be anchored (rare; harder to
  maintain).
- If the graph contains fused or optimizer-inserted elementwise ops, anchor the
  pattern on the surrounding operators and rely on `counts` to constrain the
  totals.

## Where to use it

`post_check_onnx_graph` entries appear inside example/plugin test metadata (see
`jax2onnx/plugins2/examples2/nnx/cnn.py` for a reference). The helper works with
any object that produces an ONNX IR graph compatible with
`onnx_ir.GraphProto`. The same API is shared by policy tests under
`tests/extra_tests2`.

When adding new metadata entries, seed them with a minimal structural check,
run the example once to capture the intended op sequence, and then layer on
shape assertions and counts to guard against regressions.
