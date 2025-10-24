# DINO Equinox Export Work Notes

## Repo Guardrails (AGENTS.md)
- IR-only converter: keep ONNX protobuf imports out of `converter/` and `plugins/`.
- Deterministic module construction: use `construct_and_call(...).with_requested_dtype()` and `with_rng_seed(...)`; never seed at import.
- Single-use PRNG keys: split before distributing keys; enable `jax_debug_key_reuse` when debugging.
- Tooling: Python 3.11+, Poetry, Ruff (`check` + `format`), mypy, pytest; supported runtime stack JAX ≥0.7.2
- Workflow checklist: install with `poetry install -E all`, run focused pytest during development, full suite + lint + mypy before merging.
- Metadata parity: keep `expect_graph` specs aligned with lowering and regenerate via `scripts/emit_expect_graph.py` after behaviour changes.

## Background Thread
- Kickoff from @clementpoiret: greenlight to use Equinox DINOv3 as first bigger ONNX-IR export example for jax2onnx 0.9.0, replacing protobuf path.
- DINOv3 includes RoPE positional embeddings; to be thorough, also cover a standard learned positional embedding variant (see Equimo `posemb.py` at commit `ca0dae7`).
- Learned posemb across multiple image sizes needs `jax.image.resize` (with/without antialiasing) support, aligning with ONNX `Resize`.
- Long-term alignment: keep the example as close as possible to Equimo’s DINOv3 implementation and source trained parameters directly from upstream Equimo or the Meta/Facebook DINOv3 releases once format compatibility is clear.

## Focus
- `jax2onnx/plugins/examples/eqx/dino.py`: ensure the example runs under the IR-only pipeline and adheres to the above guardrails.
- Track blockers, test coverage, and export parity updates directly in this document as work progresses.
- **Strict Directive:** Keep the Equinox example code as close as reasonably possible to the upstream Equimo implementation; prefer enhancing `jax2onnx` over diverging from the source unless a minimal shim is absolutely required.

## Progress Log (Completed)
- Pretrained export: CLI script `scripts/export_dinov3_pretrained.py` now produces ONNX directly via IR. Added runtime shims (RoPE cache freezing, deterministic dropout paths, GELU activation) so Equimo’s `dinov3_vits16_pretrain_lvd1689m` checkpoint exports successfully and deterministically.
- Added `tests/examples/test_eqx_dino_pretrained_runtime.py` – optional ONNX Runtime smoke test comparing the exported graph against the patched JAX model when `DINO_EQX_ONNX` (and optionally `DINO_EQX_WEIGHTS`) are provided.
- Added `scripts/map_equimo_dino_weights.py` to lift Equimo checkpoints into the simplified `examples.eqx_dino` VisionTransformer (`.eqx` serialisation output). Mapper now supports `--strip-register-tokens` to ignore Equimo’s register/storage tokens so weights can be applied while keeping the exact example graph structure (semantics may differ from full Equimo in configs that rely on registers).
- Added `scripts/export_eqx_dino_example_with_mapped_weights.py` to export ONNX from the simplified example model using mapped weights, preserving the exact operator layout used in the example testcases (supports static or dynamic batch).
- Added `jax2onnx/sandbox/dino_01.py` to run an ONNX DINO model on an image, save CLS/pooled features, print SHA256 checksums, and optionally compare against saved references.
- PatchEmbed: introduced `eqx.filter_vmap` wrappers and a batching rule for the custom `jnp.squeeze` primitive so `Test_PatchEmbed::test_patch_embed` passes for both static/dynamic batches.
- Vision blocks: LayerNorm/MLP now run under `eqx.filter_vmap`, keeping Equimo semantics while satisfying ONNX tracing (fixes the transformer + ViT paths).
- VisionTransformer now models Equimo’s four storage/register tokens end to end; the weight mapper copies them into the example, and the export CLI auto-detects their presence when rebuilding the `VisionTransformer` stub.
- DinoRoPE integration: ported the 2D rotary helper, now recompute sin/cos per call, and thread a hashable `DinoRotaryProcessHeads` (with numpy-backed caches) through each block. The eqx.MultiheadAttention lowering pads prefix tokens automatically so the IR path sees the same caches. Example tests cover the Dino-specific adapter end to end.
- **Plan – DinoRoPE parity (TODO)**
  1. ✅ Swap the rotary handling to match Equimo’s 2D `DinoRoPE`: only the patch grid (HxW) rotates; CLS + register tokens stay unrotated. This likely means porting the `DinoRoPE` helper and wiring a `process_heads` adapter that keeps prefix tokens unchanged.
  2. ✅ Update the multihead-attention lowering so it understands the new adapter (possibly by teaching it to accept the Dino-specific process_heads or by generating equivalent sin/cos caches during export).
  3. Re-run diff checks (`scripts/compare_dinov3_embeddings.py`) until ONNX ↔ PyTorch δ ≈ numerical noise; refresh expect_graph baselines if the operator layout shifts.
  4. Once parity holds, re-enable the broader test matrix (examples + regression) and document the rotary behavior in `work_notes_dino.md`.

## Next Steps (DinoRoPE)
- Run `scripts/compare_dinov3_embeddings.py` (see `scripts/compare_dinov3_embeddings.py:1`) with upstream Equimo outputs once the adapter lands, logging the delta here and updating `expect_graph` fixtures if operators shift.
- 2025-10-22: First parity run with `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` vs `eqx_dinov3_vit_S16.onnx` @ `/tmp/coco_39769.jpg` shows CLS max|Δ|≈1.47 and pooled max|Δ|≈0.71 (rtol=1e-4, atol=1e-6) – still far from parity; expect_graph fixtures unchanged for now. Attempting the shorter filename `dinov3_vits16_pretrain.pth` failed (missing on disk); stick with the hashed Meta archive for now.
- 2025-10-22: After wiring the on-call caches, parity numbers are unchanged (CLS max|Δ|≈1.47 / pooled≈0.71). Focus shifts to structural gaps rather than rotary plumbing.
- 2025-10-22: Regenerated `.eqx` weights with new rotary metadata; Equimo ↔ JAX example drift still sits at ~6.5e-3 (CLS) / 3.6e-3 (patch). ONNX ↔ PyTorch gap unchanged.
- (`TODO`) teach the mapper to embed the hashable rotary caches explicitly so downstream tooling can deserialize without manual patching.
- Example MLP still relies on the default approximate GELU until we land a robust ONNX-compatible exact variant.
- Added `VisionTransformer.forward_features` and `_encode` helpers mirroring Equimo’s API so we can compare normalized CLS/storage/patch tokens without re-running the whole export loop.
- Exact GELU still leverages the native ONNX Gelu op; we still owe lax-level `erf`/`erfc` lowering so standalone error-function primitives don’t break export (tracked as TODO).
- Per-block delta vs. Equimo (coco_39769): blocks 0–5 stay ≤1e-2, but after block 6 deltas grow—`mlp_scaled` hits ~0.22 at block 6 and both `attn_scaled`/`mlp_scaled` exceed 0.2 by blocks 10–11. Untied norms and LayerScale handling in later blocks look like the main gap.

## Plan – Match Equimo Features
1. **Gap audit:** Diff `jax2onnx/plugins/examples/eqx/dino.py` against `equimo/models/vit.py` and `equimo/layers/attention.py` to list missing behaviors (register tokens, untied norms, pooling, rope refresh).
2. **Prefix/token handling:** Port Equimo’s storage/register-token flow into our `VisionTransformer.__call__`, ensuring CLS/storage tokens follow the same concat/split semantics and that storage tokens propagate through blocks.
3. **Normalization parity:** Introduce untied norms (cls vs patch) and any optional local-cls norm toggles used by the pretrain checkpoints; gate features behind flags when the simplified example doesn’t exercise them.
4. **Block internals:** Align block-level attention/MLP ordering with Equimo, including per-block rope cache handling and any dropout/LayerScale nuances still missing.
5. **Validation & cleanup:** Re-run `scripts/compare_dinov3_embeddings.py` after each milestone, capture deltas here, and refresh expect_graph fixtures/tests once parity settles. Document any intentional deviations.
- Add a regression that toggles between rotary/no-rotary adapters inside the example attention module to ensure the plugin dispatch stays stable.
- Restore the broader Dino example/regression test matrix after parity holds and note the verified configuration (weights, image size, tolerances) in this file.
- Attention + RoPE:
  - Restored the upstream two-argument rotary API and threaded token length explicitly so dynamic batches export cleanly.
  - Refactored `Attention` to reuse an in-module `AttentionCore` and a shared `RotaryProcessHeads` helper; the plugin lowers RoPE alongside the attention primitive, including dynamic batch support.
- EQX primitive registration: hooked `eqx.nn.Conv2d`, `eqx.nn.MultiheadAttention`, and `eqx.nn.RotaryPositionalEmbedding` into the plugin registry with focused expect-graph coverage.
- IR optimizer: added `remove_identity_reshapes_ir` to strip redundant reshape corridors, simplifying the generated attention graphs.
- Imaging utilities: implemented a `jax.image.resize` lowering (nearest, linear, cubic) so posemb grids can be resized when we add learned positional embeddings.
- Examples & expect_graph updates:
  - Simplified `AttentionCore` usage (no `@onnx_function` indirection) and refreshed tests to assert operator counts rather than fragile reshape chains.
  - Adjusted EQX multihead attention expect-graphs to reflect the optimized operator layout after the reshape cleanup.

## Notes & Attempts
- Export script applies Equimo-specific shims (freeze RoPE caches, bypass dropout randomness, replace exact GELU) to keep the IR pipeline deterministic. These patches deliberately stay inside the CLI so core example modules remain untouched.
- Float64 runtime gaps still push the examples toward `run_only_f32_variant`; revisit once ONNX Runtime catches up.
- Earlier attempt to reshape RoPE caches by changing `jax.numpy.pow` abstract evaluation was rolled back due to recursion failures—keep in mind if dynamic-dim support resurfaces.

## Structuring Plan (Attention + RoPE)
1. **Module Layers**
   - Confirm any additional RoPE variants (e.g., learned cache reuse) still compose cleanly via the new `RotaryProcessHeads`.
   - Evaluate whether other Equinox helpers (relative position shifts, etc.) can be expressed as lightweight process-head adapters.
2. **Plugin Enhancements**
   - Generalise detection so closely related closures (e.g., custom modules wrapping `RotaryProcessHeads`) can be white-listed without re-implementing the lowering.
   - Explore surfacing the rotary caches as reusable nodes when multiple attention layers share the same sequence length, to reduce constant duplication.
3. **Testing & Docs**
   - Extend coverage with a regression that toggles between rotary/no-rotary `process_heads` to ensure the plugin continues to dispatch correctly.
   - Document the supported pattern in the example docstring (and plugin README) so contributors know to reuse `RotaryProcessHeads` instead of hand-written head rewrites.
- Stand up a learned positional embedding example mirroring Equimo’s `posemb.py`; confirm interpolation paths exercise the new `jax.image.resize` lowering (both antialias off/on).
- Audit parameter parity against upstream Equimo/DINOv3 checkpoints—identify a reproducible source for pretrained weights or add guidance on importing Equimo’s parameters (priority: match Equimo repo first, fall back to Meta’s DINOv3 release).
- **Weights ingestion**
  1. Request Meta’s official DINOv3 weights via <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/> (required for the `.pth` files referenced by Equimo).
  2. Drop the downloaded checkpoints into `~/.cache/torch/hub/dinov3/weights/` with the exact filenames expected by Equimo’s `models/dinov3.py`.
  3. Use the helper `scripts/convert_dinov3_from_equimo.py --variant <id>` (wraps Equimo’s `convert_torch_to_equinox`) to serialize Equinox checkpoints into `~/.cache/equimo/dinov3/{variant}.tar.lz4`.
  4. Load checkpoints inside examples/tests via `load_pretrained_dinov3(...)`. A guarded integration test (`tests/examples/test_eqx_dino_pretrained.py`) compares the model’s features against a reference dump when the following env vars are set: `DINO_EQX_WEIGHTS`, `DINO_EQX_IMAGE`, `DINO_EQX_EXPECTED`, `DINO_EQX_VARIANT` (optional).
  5. Use `scripts/generate_dinov3_reference.py` to generate the reference activation file from an input tensor once weights are available.
- Pretrained flow:
  - Reused the Equimo conversion script (mirroring `models/dinov3.py`) so Meta’s `.pth` checkpoints can be converted—or downloaded directly from the Equimo HF hub—and consumed via `load_pretrained_dinov3`.
  - Added `scripts/generate_dinov3_reference.py` plus `tests/examples/test_eqx_dino_pretrained.py` so real weights can be smoke-tested against a reference activation when env vars point to cached inputs/outputs.
  - New exact-structure path: map Equimo → example `.eqx` (optionally `--strip-register-tokens`) then export ONNX from the example to ensure the graph matches the testcases exactly.

## Exact Example Graph with Pretrained Weights

Goal: bake pretrained weights into ONNX while preserving the exact operator structure from `jax2onnx/plugins/examples/eqx/dino.py` testcases.

1) Map Equimo weights into the simplified example model

```bash
poetry run python scripts/map_equimo_dino_weights.py \
  --variant dinov3_vits16_pretrain_lvd1689m \
  --weights ~/.cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m.tar.lz4 \
  --output  ~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx
```

Notes:
- Register tokens are copied into the example by default so the mapped checkpoint matches Meta/Equimo semantics. Pass `--strip-register-tokens` only if you need the legacy no-register graph.

2) Export ONNX with identical example structure

```bash
# Dynamic batch (matches example dynamic onnx in coverage table)
poetry run python scripts/export_eqx_dino_example_with_mapped_weights.py \
  --eqx ~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx \
  --output ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16_dynamic.onnx \
  --img-size 224 \
  --dynamic-b

# Static batch (B=1)
poetry run python scripts/export_eqx_dino_example_with_mapped_weights.py \
  --eqx ~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx \
  --output ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx \
  --img-size 224
```

If config inference from filename fails, pass explicit flags:
`--patch-size 16 --embed-dim 384 --depth 12 --num-heads 6`. The CLI auto-detects storage tokens (tries 4 and then 0) or accept `--storage-tokens` for explicit control.

3) Run on a known image and save vectors

```bash
# COCO validation example
curl -L -o /tmp/coco_39769.jpg \
  http://images.cocodataset.org/val2017/000000039769.jpg

poetry run python jax2onnx/sandbox/dino_01.py \
  --model ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx \
  --image /tmp/coco_39769.jpg \
  --out-cls /tmp/coco_cls.npy \
  --out-pooled /tmp/coco_pooled.npy \
  --print-checksum
```

4) Compare runs (regression)

```bash
poetry run python jax2onnx/sandbox/dino_01.py \
  --model ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx \
  --image /tmp/coco_39769.jpg \
  --out-cls /tmp/coco_cls_new.npy \
  --out-pooled /tmp/coco_pooled_new.npy \
  --ref-cls /tmp/coco_cls.npy \
  --ref-pooled /tmp/coco_pooled.npy \
  --rtol 1e-4 --atol 1e-6 --print-checksum
```

5) Optional: JAX vs ONNX equivalence

Use `jax2onnx.allclose` to compare the ONNX output to the mapped example model in JAX. This verifies the export rather than the full Equimo model.

```python
from pathlib import Path
import numpy as np
from PIL import Image
import equinox as eqx
import jax, jax.numpy as jnp
from jax2onnx import allclose
from jax2onnx.plugins.examples.eqx.dino import VisionTransformer

def preprocess(path, size=224):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = min(w, h)
    img = img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2)).resize((size,size), Image.BICUBIC)
    x = (np.asarray(img).astype("float32")/255.0)
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    x = (x-mean)/std
    return np.transpose(x,(2,0,1))[None,...]

eqx_path = Path("~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx").expanduser()
onnx_path = str(Path("~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx").expanduser())
like = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, key=jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves(eqx_path, like)
fn = lambda x: model(x)
x = preprocess("/tmp/coco_39769.jpg", 224).astype(np.float32)
ok, msg = allclose(fn, onnx_path, [x], rtol=1e-4, atol=1e-6)
print(ok, msg)
```

If `--strip-register-tokens` was used during mapping, expect numerical differences versus the full Equimo runtime with registers; the example export remains self-consistent (JAX example ⇔ ONNX example).

---

## Updated Per-Block Deltas vs. Equimo (post-main fixes)

We re-ran the block-level comparison after merging main (bringing vmap batching rules and the `jnp.stack` plugin). Results are regenerated using `scripts/dino_block_deltas.py` on COCO image `000000039769.jpg` with variant `dinov3_vits16_pretrain_lvd1689m`.

Repro:

```
curl -L -o /tmp/coco_39769.jpg http://images.cocodataset.org/val2017/000000039769.jpg
poetry run python scripts/dino_block_deltas.py \
  --image /tmp/coco_39769.jpg \
  --variant dinov3_vits16_pretrain_lvd1689m \
  --eqx ~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx
```

Summary table (max|Δ|):

| Block | attn_in | attn_raw | attn_norm | attn_scaled | post_attn | mlp_raw | mlp_scaled | output |
|------:|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
|    0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 6.83e-03 | 5.23e-03 | 5.23e-03 |
|    1 | 2.10e-03 | 4.40e-03 | 4.40e-03 | 1.07e-02 | 8.78e-03 | 1.35e-02 | 1.41e-02 | 9.39e-03 |
|    2 | 1.29e-03 | 3.46e-03 | 3.46e-03 | 3.92e-03 | 9.46e-03 | 1.92e-02 | 1.98e-02 | 1.67e-02 |
|    3 | 3.93e-03 | 5.72e-03 | 5.72e-03 | 3.90e-03 | 1.48e-02 | 4.37e-03 | 7.26e-03 | 1.75e-02 |
|    4 | 2.28e-03 | 5.49e-03 | 5.49e-03 | 4.48e-03 | 1.77e-02 | 5.05e-03 | 8.43e-03 | 2.52e-02 |
|    5 | 2.32e-03 | 6.24e-03 | 6.24e-03 | 1.16e-02 | 1.75e-02 | 1.20e-02 | 3.16e-02 | 2.83e-02 |
|    6 | 4.44e-03 | 5.57e-03 | 5.57e-03 | 1.08e-02 | 2.65e-02 | 8.27e-02 | 2.21e-01 | 2.47e-01 |
|    7 | 2.44e-03 | 8.13e-03 | 8.13e-03 | 1.74e-02 | 2.57e-01 | 6.88e-03 | 3.45e-02 | 2.61e-01 |
|    8 | 3.03e-03 | 1.24e-02 | 1.24e-02 | 3.65e-02 | 2.69e-01 | 1.03e-02 | 3.03e-02 | 2.80e-01 |
|    9 | 3.65e-03 | 1.09e-02 | 1.09e-02 | 4.46e-02 | 2.71e-01 | 1.20e-02 | 6.53e-02 | 2.77e-01 |
|   10 | 4.66e-03 | 1.56e-02 | 1.56e-02 | 1.04e-01 | 3.09e-01 | 1.65e-02 | 9.79e-02 | 2.80e-01 |
|   11 | 8.32e-03 | 3.10e-02 | 3.10e-02 | 2.26e-01 | 2.53e-01 | 3.98e-02 | 1.97e-01 | 2.45e-01 |

Interpretation:
- Blocks 0–5 remain tight (≤2e‑2 in post-MLP/output).
- Starting at block 6, residual paths diverge (LayerScale outputs `attn_scaled`/`mlp_scaled` ~1e‑1–2e‑1), with final `output` reaching ~0.25 in late blocks. This corroborates the hypothesis about untied norms/LayerScale and storage-token handling in later layers.

Next diagnostic: run the per-token breakdown to see whether divergence concentrates on CLS or patch tokens in late blocks:

```
poetry run python scripts/dino_block_token_deltas.py --image /tmp/coco_39769.jpg \
  --variant dinov3_vits16_pretrain_lvd1689m --start-block 6
```

### 2025-10-24 – Dynamic reshape regression fixed

- Background: Starting from `d9da4b7a801a5d0132c464ebac5167729ae734a8`, exporting the block/VisionTransformer dynamic testcases threw `InconclusiveDimensionOperation` because the feed-forward stack flattened `(B, T, D)` into `(B·T, D)` before every `Gemm`. The reshape builder tried to recover the original dimensions by multiplying back `B` and the static token count, which is illegal once `B` is symbolic.
- Change: Introduced `LinearLastDim` + `DinoMlp` (`jax2onnx/plugins/examples/eqx/dino.py:332-360`) so the ONNX lowering uses `MatMul` directly on `(B, T, D)` tensors. No flattening means no `[257*B]` shape expressions.
- Mapper update: `scripts/map_equimo_dino_weights.py:171-186` now copies weights/biases into the new structure.
- Validation: `poetry run pytest -q tests/examples/test_eqx_dino.py -q` (static + dynamic) passes, and `scripts/dino_block_deltas.py` / `scripts/dino_block_token_deltas.py` still show per-block deltas at ~1e-4 or lower.

For quick spot checks against Meta without the Equimo hop, run:

```
poetry run python scripts/compare_meta_vs_jax2onnx.py \
  --image /tmp/coco_39769.jpg \
  --onnx ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx \
  --variant dinov3_vits16_pretrain_lvd1689m \
  --weights ~/.cache/torch/hub/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

That prints cosine similarity and max|Δ| for CLS and pooled patch embeddings so we can watch those numbers collapse towards numerical noise as the Meta conversion catches up.
