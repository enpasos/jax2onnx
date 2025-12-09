# Flax/NNX DINOv3 ONNX exports (local-only)

This note mirrors the Equinox DINO docs but for the Flax/NNX example stack (`jax2onnx/plugins/examples/nnx/dinov3.py`). Per the repo guardrail, `.onnx` / `.onnx.data` artifacts are **not** committed; generate them locally and keep them in your workspace (e.g., under `docs/onnx/examples/nnx_dino/` which is `.gitignore`d).

## Quick export commands

Exports use random init (no pretrained weights). Adjust shapes/variants as needed.

### Static input

```bash
poetry run python - <<'PY'
from flax import nnx
from jax2onnx import to_onnx
from jax2onnx.plugins.examples.nnx.dinov3 import VisionTransformer

model = VisionTransformer(
    img_size=224,
    patch_size=14,
    embed_dim=384,
    depth=12,
    num_heads=6,
    num_storage_tokens=0,
    rngs=nnx.Rngs(0),
)

to_onnx(
    model,
    input_shapes=[(1, 3, 224, 224)],
    return_mode="file",
    output_path="docs/onnx/examples/nnx_dino/nnx_dinov3_vit_S14.onnx",
)
PY
```

### Dynamic batch

```bash
poetry run python - <<'PY'
from flax import nnx
from jax2onnx import to_onnx
from jax2onnx.plugins.examples.nnx.dinov3 import VisionTransformer

model = VisionTransformer(
    img_size=224,
    patch_size=14,
    embed_dim=384,
    depth=12,
    num_heads=6,
    num_storage_tokens=0,
    rngs=nnx.Rngs(0),
)

to_onnx(
    model,
    input_shapes=[("B", 3, 224, 224)],
    return_mode="file",
    output_path="docs/onnx/examples/nnx_dino/nnx_dinov3_vit_S14_dynamic.onnx",
)
PY
```

## Notes

- The example registry now keys by `context::component`, and the NNX DINO components use explicit names (`NnxDinoPatchEmbed`, `NnxDinoAttentionCore`, `NnxDinoAttention`, `NnxDinoBlock`, `FlaxDINOv3VisionTransformer`) so generated test files and ONNX artifacts stay unambiguous.
- To capture the submodules (PatchEmbed/Attention/Block) alongside the full ViT, run `poetry run pytest tests/examples/test_nnx_dino.py` and pick up the emitted models under `docs/onnx/examples/nnx_dino/`.
- Parity with the Equinox DINO example is covered by `tests/extra_tests/examples/test_nnx_dino_parity.py` (weight copy + forward check).
- Keep generated artifacts local; `docs/onnx/examples/nnx_dino/.gitignore` prevents accidental commits.
