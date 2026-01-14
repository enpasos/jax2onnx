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


## Maintaining Coverage

- Every plugin/example should ship an `expect_graph(...)` snippet alongside tests; rerun `python scripts/emit_expect_graph.py <testcase>` whenever behaviour changes.
- Regenerate and update this guide after each sweep so metadata and documentation stay in sync.
- When new fixtures land, add them to the coverage snapshot and verify the relevant pytest target before updating this doc.

## Workflow Checklist

1. Identify the next uncovered component (scan for missing `register_example` / `register_primitive` entries if needed).
2. Capture the snippet via `poetry run python scripts/emit_expect_graph.py <testcase>`.
3. Update metadata/tests with the snippet, run the focused pytest target, then expand to the broader suite if applicable.
4. Refresh this guide (and coverage tables) with the new snippet.
5. Before wrapping, ensure everything is documented and guardrails (RNG helpers, ONNX-IR boundaries, attention normalisation) are respected.

## Guardrails

- Converter/plugins must remain ONNX-IR only (no protobuf imports).
- Use `construct_and_call(...).with_requested_dtype()` and `with_rng_seed(...)`; split PRNG keys before reuse.
- Attention plugins must retain masked-weight normalisation; expect_graph snippets should reflect the normalised path.
- Run core tooling (`poetry run pytest -q`, `poetry run ruff check .`, `poetry run mypy src`) for larger sweeps.

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

### NNX Multi-head attention (`examples.nnx.multi_head_attention`)

Both the pure JAX (`multihead_attention_nn`) and nnx-driven (`multihead_attention_nnx`,
`multihead_attention_2_nnx`) variants emit the same core chain after reshaping
queries/keys/values. Symbols capture the batch size; `search_functions=True`
keeps subgraph rewrites under the function export.

```python
EG(
    [
        "Reshape:?x256 -> Gemm:?x256 -> Reshape:Bx4x8x32 -> "
        "Transpose:Bx8x4x32 -> MatMul:Bx8x4x4 -> Mul:Bx8x4x4 -> "
        "Softmax:Bx8x4x4 -> MatMul:Bx8x4x32 -> Transpose:Bx4x8x32 -> "
        "Reshape:?x256 -> Gemm:?x256 -> Reshape:Bx4x256"
    ],
    symbols={"B": None},
    search_functions=True,
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

### Equinox DINO building blocks

While covering the DINO stack, keep these helpers in sync:

```python
EG(['PatchEmbed_1:1x256x384'], no_unused_inputs=True)  # PatchEmbed
EG([{'path': 'Gemm', 'counts': {'Gemm': 4}},
    {'path': 'MatMul', 'counts': {'MatMul': 2}},
    {'path': 'Softmax', 'counts': {'Softmax': 1}}],
   symbols={'B': None}, search_functions=True, no_unused_inputs=True)  # AttentionCore
EG([{'path': 'MatMul', 'counts': {'MatMul': 2}}, 'Softmax'],
   symbols={'B': None}, search_functions=True)  # Attention
EG(['Block_1:Bx257x384'], symbols={'B': None}, must_absent=['Identity'])  # Block
```
### GPT components (`examples.gpt`)

`GPT_Attention` and `GPT_CausalSelfAttention` rely on the shared `_no_cast_where`
helper, which fails the test if any `Cast -> Where` chain appears in the exported
graph (and reruns with diagnostics to surface the offending path). The rest of
the GPT stack leans on compact structural checks:

```python
EG(['MLP_1:Bx1024x768'], symbols={'B': None}, no_unused_inputs=True)  # GPT_MLP
EG(['Block_1:Bx1024x768'], symbols={'B': None}, no_unused_inputs=True)  # GPT_TransformerBlock
EG(['TokenEmbedding_1:Bx1024x768'], symbols={'B': None}, no_unused_inputs=True)  # GPT_TokenEmbedding
EG(['PositionEmbedding_1:1x1024x768'], no_unused_inputs=True)  # GPT_PositionEmbedding
EG(['GPTTransformerStack_1:Bx1024x768'], symbols={'B': None}, no_unused_inputs=True)  # GPT_TransformerStack
EG(['GPTEmbeddings_1:Bx1024x768'], symbols={'B': None}, no_unused_inputs=True)  # GPT_Embeddings
EG(['GPTHead_1:Bx1024x3144'], symbols={'B': None}, no_unused_inputs=True)  # GPT_Head
EG(['Add:Bx4x5'], symbols={'B': None}, no_unused_inputs=True)  # GPT_broadcast_add
EG(
    [{'graph': 'custom.PositionEmbedding.1:PositionEmbedding',
      'path': 'Range -> Unsqueeze -> Expand -> Gather',
      'must_absent': ['Cast']}],
    no_unused_inputs=True,
    no_unused_function_inputs=True,
    search_functions=True,
)  # GPT
```

### Vision Transformer components (`examples.vit`, `examples.vit_flat`)

Patch/conv embeddings and the final classifier keep the ViT snippets compact.
When working on the flattened variants, keep the reshape/transposes anchored so
the sequence length stays guarded.

```python
EG(['PatchEmbedding_1:Bx49x256'], symbols={'B': None}, no_unused_inputs=True)
EG(['ConvEmbedding_1:Bx49x128'], symbols={'B': None}, no_unused_inputs=True)
EG(['FeedForward_1:Bx10x256'], symbols={'B': None}, no_unused_inputs=True)
EG(['TransformerBlock_1:Bx10x256'], symbols={'B': None}, no_unused_inputs=True)
EG(['TransformerStack_1:Bx10x256'], symbols={'B': None}, no_unused_inputs=True)
EG(['ConcatClsToken_1:Bx50x256'], symbols={'B': None}, no_unused_inputs=True)
EG(['PositionalEmbedding_1:Bx50x256'], symbols={'B': None}, no_unused_inputs=True)
EG(['VisionTransformer_1:Bx10'], symbols={'B': None}, no_unused_inputs=True)  # conv embedding model
EG(['VisionTransformer_1:2x10'], no_unused_inputs=True)  # patch embedding model
EG(['LayerNormalization -> Gemm -> LogSoftmax'], symbols={'B': None}, no_unused_inputs=True)  # flattened ViT heads
EG(
    [
        "Reshape:Bx7x4x7x4x1 -> Transpose:Bx7x7x4x4x1 -> "
        "Reshape:Bx49x16 -> Reshape:?x16 -> Gemm:?x256 -> Reshape:Bx49x256"
    ],
    symbols={'B': None},
    no_unused_inputs=True,
)  # PatchEmbeddingFlatten
EG(
    ['Slice -> Squeeze', {'path': 'Transpose:50xBx256 -> Gather:Bx256', 'inputs': {1: {'const': 0.0}}}],
    mode='any',
    symbols={'B': None},
    no_unused_inputs=True,
)  # GetToken
EG(['ClassificationHead_1:Bx10'], symbols={'B': None}, no_unused_inputs=True)
```

### Transformer decoder variants (`examples.nnx.transformer_decoder_*`)

Both decoder flavours share the same residual-add/LayerNorm cadence; the
sequential version also has a dynamic-shape testcase.

```python
EG(
    ["Add:2x8x16 -> LayerNormalization:2x8x16 -> Add:2x8x16 -> LayerNormalization:2x8x16 -> Add:2x8x16 -> LayerNormalization:2x8x16"],
    search_functions=True,
    no_unused_inputs=True,
)  # TransformerDecoderWithSequential (static shapes)
EG(
    ["Add:BxHx16 -> LayerNormalization:BxHx16 -> Add:BxHx16 -> LayerNormalization:BxHx16 -> Add:BxHx16 -> LayerNormalization:BxHx16"],
    search_functions=True,
    symbols={'B': None, 'H': None},
    no_unused_inputs=True,
)  # TransformerDecoderWithSequential (dynamic shapes)
EG(
    ["Add:Bx8x16 -> LayerNormalization:Bx8x16 -> Add:Bx8x16 -> LayerNormalization:Bx8x16 -> Add:Bx8x16 -> LayerNormalization:Bx8x16"],
    symbols={'B': 2},
    no_unused_inputs=True,
)  # TransformerDecoderWithoutSequential
```
