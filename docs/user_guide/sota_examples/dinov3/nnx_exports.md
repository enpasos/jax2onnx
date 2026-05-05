# Flax/NNX DINOv3 ONNX exports (local-only)

This note mirrors the Equinox DINO docs for the Flax/NNX example stack and
shows how to generate local ONNX exports for inspection. Generated `.onnx` and
`.onnx.data` files are working outputs; the public [Examples](../../examples.md)
reference table links to the published sample models.

## Quick export commands

Exports use random init (no pretrained weights). Adjust shapes/variants as needed.

### Static input

```bash
poetry run python - <<'PY'
from pathlib import Path

from flax import nnx
from jax2onnx import to_onnx
from jax2onnx.plugins.examples.nnx.dinov3 import VisionTransformer

Path("artifacts/nnx_dino").mkdir(parents=True, exist_ok=True)

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
    output_path="artifacts/nnx_dino/nnx_dinov3_vit_S14.onnx",
)
PY
```

### Dynamic batch

```bash
poetry run python - <<'PY'
from pathlib import Path

from flax import nnx
from jax2onnx import to_onnx
from jax2onnx.plugins.examples.nnx.dinov3 import VisionTransformer

Path("artifacts/nnx_dino").mkdir(parents=True, exist_ok=True)

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
    output_path="artifacts/nnx_dino/nnx_dinov3_vit_S14_dynamic.onnx",
)
PY
```

## Notes

- Compare static and dynamic exports against the matching Flax/NNX model when
  validating local changes.
- Maintainer workflows for generated tests and sample-model publishing are covered in
  [SotA Example Maintenance](../../../developer_guide/maintainer/sota_example_maintenance.md).
