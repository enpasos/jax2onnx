# DINOv3 Weights Workflow

This guide walks through the “from scratch” path for bringing Meta’s official
DINOv3 checkpoints into the IR-only `jax2onnx` example, and how to obtain the
final ONNX export. The architecture we ship is the Equimo project’s clean-room
Equinox/JAX reimplementation based on Meta AI’s [DINOv3 paper](https://arxiv.org/abs/2508.10104).
Using Meta’s pretrained weights is optional and subject to the
[DINOv3 license](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).

All commands assume you are at the project root with the Poetry environment
available.

## 1. Fetch Meta’s PyTorch checkpoint

Meta publishes DINOv3 weights on Hugging Face. The commands below download the
`dinov3_vits16_pretrain_lvd1689m` checkpoint directly into the cache path that
Equimo-based scripts expect.

1. Visit https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m,
   accept the model’s license *and* request access to the gated repository. Wait
   for the approval email—Meta sends presigned download links that remain valid
   for a limited time.
2. Use one of the approved links (they start with `https://dinov3.llamameta.net/...`)
   to fetch the checkpoint directly, for example:

   ```bash
   mkdir -p ~/.cache/equimo/dinov3

   curl -L "https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?...Key-Pair-Id=..." \
     -o ~/.cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
   ```

   (Replace the URL above with the exact link from your email; `wget` works too.
   If you prefer a browser download, save the file and move it into
   `~/.cache/equimo/dinov3/`.)

3. Verify the file is present:

   ```bash
   ls ~/.cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m-*.pth
   ```

## 2. Convert Meta → Equimo (PyTorch → Equinox)

Use the helper script to convert Meta’s checkpoint into an Equinox archive that
matches the Equimo layout.

> **Prerequisite:** install the optional tooling and make a local clone of
> Equimo’s conversion utilities (the published wheel omits `models/dinov3.py`):
>
> ```bash
> poetry install --with test
> git clone https://github.com/clementpoiret/Equimo.git ~/.cache/equimo/repos/Equimo
> poetry run pip install -e ~/.cache/equimo/repos/Equimo
> poetry run pip install timm torchmetrics termcolor
> ```

```bash
poetry run python scripts/convert_dinov3_from_equimo.py \
  --variant dinov3_vits16_pretrain_lvd1689m \
  --weights ~/.cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --output ~/.cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m
```

This produces `~/.cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m.tar.lz4`,
the Equinox checkpoint used by the rest of the tooling.

## 3. Map Equimo weights into the IR-only example

The IR-only DINO example lives under `jax2onnx/plugins/examples/eqx/dino.py`.
Use the mapping script to copy the Equimo parameter tree into that example:

```bash
poetry run python scripts/map_equimo_dino_weights.py \
  --variant dinov3_vits16_pretrain_lvd1689m \
  --weights ~/.cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m.tar.lz4 \
  --output  ~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx
```

By default the register/storage tokens are preserved so the example matches
Meta/Equimo semantics.

## 4. Export ONNX from the mapped example

Both static and dynamic-batch exports are available. For the static (B=1)
variant:

```bash
poetry run python scripts/export_eqx_dino_example_with_mapped_weights.py \
  --eqx ~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx \
  --output ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx \
  --img-size 224
```

For the dynamic-batch version, add `--dynamic-batch` to the command above. The
resulting ONNX files can now be used for inference or comparison.

## 5. Verify against Meta’s release

Use the comparison helper to confirm the ONNX export matches Meta’s PyTorch
checkpoint (numerical drift should be on the order of 1e-6):

```bash
curl -L -o /tmp/coco_39769.jpg \
  http://images.cocodataset.org/val2017/000000039769.jpg

poetry run python scripts/compare_meta_vs_jax2onnx.py \
  --image /tmp/coco_39769.jpg \
  --variant dinov3_vits16_pretrain_lvd1689m \
  --weights ~/.cache/torch/hub/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --onnx ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx
```

Expect cosine similarity close to 1.0 and maximum absolute differences below
`1e-5` for both CLS and pooled patch features.
