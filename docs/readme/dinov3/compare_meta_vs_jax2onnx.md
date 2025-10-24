# Comparing Meta PyTorch vs jax2onnx ONNX

After exporting the IR-only DINOv3 model to ONNX, you can quantify how closely
it matches Meta’s original PyTorch checkpoint. The repository ships a helper
script that performs the comparison end to end. Our ONNX is generated from
Equimo’s clean-room Equinox/JAX reimplementation of the architecture described
in Meta AI’s [DINOv3 paper](https://arxiv.org/abs/2508.10104); using Meta’s
pretrained weights remains optional and is governed by the
[DINOv3 license](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).

## 1. Requirements

Ensure the optional test dependencies are installed so PyTorch and torchvision
are available:

```bash
poetry install --with test
curl -L -o /tmp/coco_39769.jpg \
  http://images.cocodataset.org/val2017/000000039769.jpg
```

You’ll also need:

- Meta’s `.pth` checkpoint (see `getting_weights.md`)
- The mapped Equinox checkpoint (`~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx`)
- The ONNX export produced from that checkpoint

## 2. Basic comparison (Meta ⇔ ONNX)

The CLI reports cosine similarity and max absolute error for the CLS token and
the mean pooled patch embeddings:

```bash
poetry run python scripts/compare_meta_vs_jax2onnx.py \
  --image /tmp/coco_39769.jpg \
  --variant dinov3_vits16_pretrain_lvd1689m \
  --weights ~/.cache/torch/hub/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --onnx ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx
```

If the export is faithful, cosine ≈ 1 and max |Δ| should sit near machine
precision (≈ 5e‑6).

## 3. Triangulate with the Equinox example (Meta ⇔ Eqx ⇔ ONNX)

To isolate where mismatches arise, include the Equinox checkpoint and enable
per-block debugging:

```bash
poetry run python scripts/compare_meta_vs_jax2onnx.py \
  --image /tmp/coco_39769.jpg \
  --variant dinov3_vits16_pretrain_lvd1689m \
  --weights ~/.cache/torch/hub/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --onnx ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx \
  --eqx ~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx \
  --block-debug
```

This prints:

- Meta ⇔ ONNX cosine / max |Δ|
- Meta ⇔ Equinox cosine / max |Δ|
- EQX ⇔ ONNX cosine / max |Δ|
- For each transformer block, the max absolute difference per stage (input,
  attention norms, LayerScale outputs, MLP, etc.) between the Equinox example
  and Meta’s original model.

The block table highlights exactly where any remaining drift originates.

## 4. Interpreting results

- Cosine values close to 1 and max |Δ| in the 1e‑6 band indicate parity.
- If Meta ⇔ Eqx is clean but Meta ⇔ ONNX drifts, regenerate the ONNX export.
- If Meta ⇔ Eqx diverges, inspect the highest per-block stage in the debug
  table—it pinpoints the first tensor that needs correction.
