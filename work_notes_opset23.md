# Opset 23 Work Notes

## Goal
- Deliver ONNX opset ≥23 support focused on emitting fused `Attention` and `RotaryEmbedding` ops instead of the current decomposed subgraphs, while retaining the existing pre-23 lowering paths for backward compatibility.

## Analysis
- Issue #110 highlights that opset 23 introduces fused `Attention`/`RotaryEmbedding` operators that better match the Equinox lowering patterns already present in the converter.
- Repository guardrails from `AGENTS.md` apply:
  - All lowering lives in `converter/` and `plugins/` using the `onnx_ir` builder; no direct ONNX protobuf usage in those directories.
  - Deterministic module construction and single-use PRNG keys are mandatory for new paths and accompanying tests.
  - Structural expectations must be refreshed via `scripts/emit_expect_graph.py` whenever the graph shape changes.
- ONNX Runtime 1.23.1 (CPU) is confirmed (per issue comments) to ship kernels for the new ops, so runtime coverage is available once the converter emits them.
- Need to review `docs/dev_guides/onnx_ir_builder.md` and attention-related plugin guides to ensure the fused lowering follows established patterns (mask handling, metadata parity, etc.).
- Current Equinox lowering (`jax2onnx/plugins/equinox/eqx/nn/multihead_attention.py`) reshapes queries/keys/values via Gemm → Reshape → Transpose chains then performs MatMul + scaling + Softmax + MatMul before a final Gemm; rotary processing hooks into this path via `_apply_rotary_process_heads_lowering`.
- The Equinox rotary plugin (`jax2onnx/plugins/equinox/eqx/nn/rotary_positional_embedding.py`) materializes sin/cos caches and emits Split/Concat/Mul/Add sequences; TODO already notes switching to the fused op when available.
- `jax.nn.dot_product_attention` lowering follows a similar decomposed flow with extensive mask logic; if we adopt fused ops there as well we must preserve mask normalization semantics highlighted in `AGENTS.md`.
- Existing expect-graph assertions (e.g. plugin-level `post_check_onnx_graph` specs and example coverage in `tests/examples/test_eqx_dino.py`) currently pin the decomposed shapes, so they will need regenerated variants once the fused path lands.
- Equinox DINOv3 example exports still fall back to the legacy Split/Neg/Concat rotary lowering because the current `process_heads` hook applies RoPE inside JAX before the fused path runs. When we try to force the fused lowering the ONNX graph loses type information (`Input 0 expected to have type but instead is null`). We need a dedicated integration that leverages the standalone `RotaryPositionalEmbedding` plugin so the exported graph matches Meta's behaviour while keeping the fused op.

## Design Notes
- **Opset gating:** Detect `builder.opset >= 23` and availability of `builder.Attention` / `builder.RotaryEmbedding`. Preserve legacy lowering for lower opsets or when guardrails fail (e.g. odd head size).
- **Equinox MHA fused path:**
  - Reuse existing projection reshapes to obtain tensors shaped `(batch, seq, heads, width)`; transpose to `(batch, heads, seq, width)` right before invoking `Attention`.
  - Configure `Attention` with `scale=1/sqrt(qk_size)` and request only the primary output; optional caches stay unset.
  - After the fused op, transpose back to `(batch, seq, heads, width)` and continue with the existing flatten + output projection.
  - Maintain fallback logic (MatMul + Softmax) behind an `else` branch so policy tests for opset < 23 remain valid.
- **Rotary integration:**
  - When fused ops are enabled, slice sin/cos caches to the first half-width (the second half duplicates pairs) and call `RotaryEmbedding` instead of the manual Split/Concat flow.
  - Support both the standalone primitive (2D inputs reshaped to 3D via temporary unsqueeze) and the MHA `process_heads` hook (operate on tensors once they are in `(batch, heads, seq, width)` form, then transpose back).
  - Retain the legacy path when caches cannot be safely normalized (e.g. odd embedding size, dynamic sequence without cached tables).
- **Testing strategy:** add opset-23 variants for existing Equinox rotary/attention testcases so CI exercises both fused and fallback paths; ensure `expect_graph` specs explicitly look for `Attention`/`RotaryEmbedding`.

## Plan
1. Audit existing attention-related converters/plugins and catalog current IR graph patterns plus tests covering them.
2. Introduce opset gating: detect target opset ≥23 and branch to new fused lowering while retaining existing decomposition for older opsets.
3. Implement fused `Attention` and `RotaryEmbedding` builders using the IR-only API, ensuring masked weight normalization persists.
4. Update / add structural + regression tests (`expect_graph`, runtime execution) for both fused and legacy paths; regenerate expectations.
5. Integrate the standalone `RotaryPositionalEmbedding` plugin into the Equinox `process_heads` path (DINOv3) so opset ≥23 emits the fused op without breaking Meta weight compatibility.
6. Document behavior changes (release notes, relevant dev guide references) and verify lint/mypy/test suites stay green.

## Implementation Status
- [x] Context gathering and code reconnaissance
- [x] Design doc / detailed lowering sketch
- [x] Fused attention lowering implemented with gating
- [x] Rotary embedding lowering implemented with gating for the base attention path
- [ ] Rotary embedding lowering integrated with Equinox process_heads / DINOv3 exports
- [x] Tests and expect_graph artifacts updated
- [ ] Docs and release notes updated
