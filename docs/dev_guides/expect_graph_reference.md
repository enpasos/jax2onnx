# `expect_graph` checklist for plugins

`expect_graph` (from `jax2onnx.plugins._post_check_onnx_graph`) is the lightweight
structural assertion helper used by plugin tests and examples. It lets a test
express the operators, ordering, and shapes that should appear in a converted
IR/ONNX graph without dumping the full model. This document captures the
conventions we rely on when writing or reviewing `post_check_onnx_graph`
expectations.

> **Test metadata reminder:** when wiring new examples/tests, construct callables
> via `construct_and_call(...).with_requested_dtype(...).with_rng_seed(...)` so the
> harness can rebuild deterministic f32/f64 variants. See the builder guide for the
> full randomness and dtype rules.

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
  after conversion. Combine with `no_unused_function_inputs=True` to extend the
  check to every imported ONNX function body (requires `search_functions=True`).
- `search_functions`: include function bodies (control-flow subgraphs) in the
  search.

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
- Use `mode="all"` with multiple patterns to check disjoint subgraphs, or
  `mode="exact"` when the entire graph must be anchored (rare; harder to
  maintain).
- If the graph contains fused or optimizer-inserted elementwise ops, anchor the
  pattern on the surrounding operators and rely on `counts` to constrain the
  totals.

## Where to use it

`post_check_onnx_graph` entries appear inside example/plugin test metadata (see
`jax2onnx/plugins/examples/nnx/cnn.py` for a reference). The helper works with
any object that produces an ONNX IR graph compatible with
`onnx_ir.GraphProto`. The same API is shared by policy tests under
`tests/extra_tests`.

When adding new metadata entries, seed them with a minimal structural check,
run the example once to capture the intended op sequence, and then layer on
shape assertions and counts to guard against regressions.

## Reference snippets (Oct 2025 refresh)

> **NNX reminder:** follow the Oct 2025 AGENTS note—seed nnx fixtures via
> `with_rng_seed(...)` (no inline lambdas) so callables stay hashable under JAX
> 0.7. Attention plugins must keep masked-weight normalisation enabled; retain
> the helper path when updating metadata or docs.

### Scatter add sweep (`primitives.lax.scatter_add`)

The converter now anchors the full regression matrix on `ScatterND`. These
snippets were regenerated with `JAX_ENABLE_X64=1` to keep f64 parity.

```python
EG(['ScatterND:4'], no_unused_inputs=True)  # scatter_add_vector
EG([{'path': 'ScatterND:6', 'inputs': {2: {'const': 5.0}}}], no_unused_inputs=True)  # scatter_add_scalar
EG(['ScatterND:5'], no_unused_inputs=True)  # scatter_add_simple_1d / scatter_add_batch_updates_1d_operand
EG(['ScatterND:2x3'], no_unused_inputs=True)  # scatter_add_window_2d_operand_1d_indices
EG(['ScatterND:5x208x1x1'], no_unused_inputs=True)  # scatter_add_mismatched_window_dims_from_user_report
```

Additional user-report variants (`report2`, `report3`, fluids pattern, depth
helpers) share the same `ScatterND:<shape>` signature and reuse the wrapper
helpers documented in `jax2onnx/plugins/jax/lax/scatter_add.py`.

### Issue 18 loop fixtures (`examples.jnp.issue18`)

Regenerated loop traces now expose the control-flow helpers and the loop-carried
symbol. Remember to pass `search_functions=True` when validating subgraph bodies.

```python
EG([{'path': 'Loop', 'inputs': {0: {'const': 5.0}, 1: {'const_bool': True}}}],
   search_functions=True, no_unused_inputs=True)  # fori_loop_fn
EG([{'path': 'Less -> Loop', 'inputs': {0: {'const': 9.223372036854776e18}, 3: {'const': 0.0}}}],
   no_unused_inputs=True)  # while_loop_fn
EG(['Loop:B'], symbols={'B': None}, search_functions=True, no_unused_inputs=True)  # scan_fn
EG(['Greater:3 -> Where:3'], no_unused_inputs=True)  # where_fn
```

### Flax/NNX GRU cell (`examples.nnx.gru_cell_basic`)

The ONNX lowering now fuses the `Tanh` stage, resulting in twin `Add` paths off
the Sigmoid gate outputs. Regenerate the snippet after adjusting module wiring,
and keep the RNG helpers in place so the sample stays deterministic.

```python
EG(
    [
        "Gemm:2x12 -> Split:2x4 -> Add:2x4 -> Sigmoid:2x4 -> Sub:2x4 -> Mul:2x4 -> Add:2x4",
        "Gemm:2x12 -> Split:2x4 -> Add:2x4 -> Sigmoid:2x4 -> Sub:2x4 -> Mul:2x4 -> Add:2x4 -> Add:2x4",
    ],
    no_unused_inputs=True,
)
```

### Equinox DINOv3 vision transformer (`examples.eqx_dino`)

Use the bundled helper to emit snippets for each variant (`eqx_dinov3_vit_Ti14`,
`_S14`, `_B14`, `_S16`). All of them collapse into a single `VisionTransformer`
node with the expected patch/token layout. Ensure `no_unused_inputs=True` stays
set so cached weights or mask inputs do not linger.

```python
EG(['VisionTransformer:Bx257x192'], symbols={'B': None}, no_unused_inputs=True)
# S16/S14/B14 variants only differ in the trailing dimension (384/768) and the token count.
```
