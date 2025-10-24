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

Meta publishes DINOv3 weights under the torch hub cache. The examples below use
the `dinov3_vits16_pretrain_lvd1689m` variant.

```bash
# Populate torch hub (only needed once per machine).
poetry run python - <<'PY'
import torch
torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=False)
PY

# The checkpoint will appear under ~/.cache/torch/hub/dinov3/weights/
ls ~/.cache/torch/hub/dinov3/weights/dinov3_vits16_pretrain_lvd1689m*.pth
```

If you already have the `.pth` file, skip the `torch.hub.load` step and place it
under `~/.cache/torch/hub/dinov3/weights/` (or pass `--weights` in later steps).

## 2. Convert Meta → Equimo (PyTorch → Equinox)

Use the helper script to convert Meta’s checkpoint into an Equinox archive that
matches the Equimo layout.

```bash
poetry run python scripts/convert_dinov3_from_equimo.py \
  --variant dinov3_vits16_pretrain_lvd1689m \
  --weights ~/.cache/torch/hub/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
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

## 5. Optional sanity check (JAX ⇔ ONNX)

To confirm the exported ONNX matches the example model, run:

```bash
poetry run python jax2onnx/sandbox/dino_01.py \
  --model ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx \
  --image /tmp/coco_39769.jpg \
  --print-checksum
```

This prints CLS and pooled feature checksums you can compare against the JAX
example or previous runs.
