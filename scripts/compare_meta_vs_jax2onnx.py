#!/usr/bin/env python3
# scripts/compare_meta_vs_jax2onnx.py

"""
Compare embeddings from Meta's official PyTorch DINOv3 checkpoint against the
final ONNX export produced by jax2onnx.

Example:
    poetry run python scripts/compare_meta_vs_jax2onnx.py \
        --image /tmp/coco_39769.jpg \
        --onnx ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx \
        --variant dinov3_vits16_pretrain_lvd1689m \
        --weights ~/.cache/torch/hub/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
        --eqx ~/.cache/equimo/dinov3/eqx_dinov3_vits16_mapped.eqx --block-debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "Torch is required for this script. Install the test extras or "
        "run `poetry install --with test`."
    ) from exc

import equinox as eqx
import jax
import jax.numpy as jnp

from jax2onnx.plugins.examples.eqx.dino import VisionTransformer


_VARIANT_META = {
    "dinov3_vits16_pretrain_lvd1689m": ("dinov3_vits16", 224),
    "dinov3_vits16plus_pretrain_lvd1689m": ("dinov3_vits16plus", 224),
    "dinov3_vitb14_pretrain_lvd1689m": ("dinov3_vitb16", 224),
    "dinov3_vitl14_pretrain_lvd1689m": ("dinov3_vitl16", 224),
}


def _resolve_weights_path(variant: str, weights: Path | None) -> Path:
    if weights is not None:
        path = Path(weights).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Meta checkpoint not found at {path}")
        return path

    default_root = Path("~/.cache/torch/hub/dinov3/weights").expanduser()
    if not default_root.exists():
        raise FileNotFoundError(
            f"Default weight directory {default_root} missing. "
            "Download the official Meta checkpoints or pass --weights."
        )
    matches = sorted(default_root.glob(f"{variant}*.pth"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching '{variant}' found under {default_root}."
        )
    return matches[0]


def _load_meta_model(variant: str, weights: Path | None) -> Tuple[torch.nn.Module, int]:
    repo_root = Path("~/.cache/torch/hub/facebookresearch_dinov3_main").expanduser()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Torch hub cache not found at {repo_root}. "
            "Run `torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=False)` "
            "once to populate it, or set TORCH_HUB_DIR to the cloned repo."
        )

    builder_meta = _VARIANT_META.get(variant)
    if builder_meta is None:
        available = ", ".join(sorted(_VARIANT_META))
        raise ValueError(
            f"Unsupported variant '{variant}'. Known variants: {available}"
        )

    builder_name, img_size = builder_meta
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from dinov3.hub import backbones  # type: ignore  # pragma: no cover

    if not hasattr(backbones, builder_name):
        raise AttributeError(
            f"dinov3.hub.backbones is missing '{builder_name}'. "
            f"Check that the torch hub repo at {repo_root} matches Meta's release."
        )

    checkpoint = _resolve_weights_path(variant, weights)
    builder = getattr(backbones, builder_name)
    model = builder(pretrained=False, weights=None)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    return model, img_size


def _preprocess(image: Path, size: int) -> np.ndarray:
    img = Image.open(image).convert("RGB")
    w, h = img.size
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    img = img.crop((left, top, left + short, top + short))
    img = img.resize((size, size), Image.BICUBIC)

    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr


def _run_onnx(model_path: Path, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: x})
    if not outputs:
        raise RuntimeError("ONNX runtime returned no outputs")
    return outputs[0]


def _run_torch(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(x).to(torch.float32)
    out = model.forward_features(tensor)
    cls = out["x_norm_clstoken"].cpu().numpy()
    patches = out["x_norm_patchtokens"].cpu().numpy()
    tokens = np.concatenate([cls[:, None, :], patches], axis=1)
    return tokens


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return float("nan")
    return float(np.dot(a, b) / (a_norm * b_norm))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _load_eqx_model(variant: str, eqx_path: Path) -> VisionTransformer:
    configs = {
        "dinov3_vits16_pretrain_lvd1689m": dict(
            img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6
        ),
        "dinov3_vits16plus_pretrain_lvd1689m": dict(
            img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=8
        ),
        "dinov3_vitb14_pretrain_lvd1689m": dict(
            img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12
        ),
        "dinov3_vitl14_pretrain_lvd1689m": dict(
            img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16
        ),
    }
    cfg = configs.get(variant)
    if cfg is None:
        available = ", ".join(sorted(configs))
        raise ValueError(
            f"Unsupported variant '{variant}' for eqx loading. Known variants: {available}"
        )
    key = jax.random.PRNGKey(0)
    like = VisionTransformer(num_storage_tokens=4, key=key, **cfg)
    model = eqx.tree_deserialise_leaves(eqx_path.expanduser(), like)
    return model


def _eqx_block_debug_outputs(
    model: VisionTransformer, x_np: np.ndarray
) -> List[Dict[str, np.ndarray]]:
    x = jnp.asarray(x_np)
    debug = model.block_debug_outputs(x)
    blocks: List[Dict[str, np.ndarray]] = []
    for entry in debug:
        stage = {k: np.array(np.asarray(v)) for k, v in entry.items()}
        blocks.append(stage)
    return blocks


def _meta_block_debug_outputs(
    model: torch.nn.Module, x_np: np.ndarray
) -> List[Dict[str, np.ndarray]]:
    tensor = torch.from_numpy(x_np).to(torch.float32)

    blocks_container = getattr(model, "blocks", None)
    if blocks_container is None:
        raise AttributeError("Expected Meta model to expose a 'blocks' attribute")

    flat_blocks: List[torch.nn.Module] = []
    for entry in blocks_container:
        maybe_sub = getattr(entry, "blocks", None)
        if maybe_sub is not None:
            flat_blocks.extend(list(maybe_sub))
        else:
            flat_blocks.append(entry)

    stage_cache: Dict[int, Dict[str, np.ndarray]] = {
        i: {} for i in range(len(flat_blocks))
    }
    handles = []

    def get_optional(module: torch.nn.Module, names: List[str]):
        for name in names:
            attr = getattr(module, name, None)
            if attr is not None:
                return attr
        return None

    def to_numpy(val) -> np.ndarray:
        if isinstance(val, (list, tuple)):
            val = val[0]
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        raise TypeError(f"Unsupported value type for debug capture: {type(val)}")

    def pre_hook(idx: int):
        def hook(_, inputs):
            stage_cache[idx]["input"] = to_numpy(inputs[0])

        return hook

    def post_hook(idx: int, name: str):
        def hook(_, __, output):
            stage_cache[idx][name] = to_numpy(output)

        return hook

    for idx, block in enumerate(flat_blocks):
        handles.append(block.register_forward_pre_hook(pre_hook(idx)))
        prenorm = get_optional(block, ["prenorm", "norm1"])
        if prenorm is not None:
            handles.append(prenorm.register_forward_hook(post_hook(idx, "attn_in")))
        attn = get_optional(block, ["attn", "attention"])
        if attn is not None:
            handles.append(attn.register_forward_hook(post_hook(idx, "attn_raw")))
        postnorm = get_optional(block, ["postnorm"])
        if postnorm is not None:
            handles.append(postnorm.register_forward_hook(post_hook(idx, "attn_norm")))
        ls1 = get_optional(block, ["ls1", "layer_scale1"])
        if ls1 is not None:
            handles.append(ls1.register_forward_hook(post_hook(idx, "attn_scaled")))
        drop1 = get_optional(block, ["drop_path1", "drop_path"])
        if drop1 is not None:
            handles.append(drop1.register_forward_hook(post_hook(idx, "post_attn")))
        norm = get_optional(block, ["norm", "norm2"])
        if norm is not None:
            handles.append(norm.register_forward_hook(post_hook(idx, "mlp_in")))
        mlp = get_optional(block, ["mlp", "ffn"])
        if mlp is not None:
            handles.append(mlp.register_forward_hook(post_hook(idx, "mlp_raw")))
        ls2 = get_optional(block, ["ls2", "layer_scale2"])
        if ls2 is not None:
            handles.append(ls2.register_forward_hook(post_hook(idx, "mlp_scaled")))
        drop2 = get_optional(block, ["drop_path2", "drop_path"])
        if drop2 is not None:
            handles.append(drop2.register_forward_hook(post_hook(idx, "output")))

    with torch.no_grad():
        _ = model.forward_features(tensor)

    for handle in handles:
        handle.remove()

    return [stage_cache[i] for i in range(len(flat_blocks))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image file")
    parser.add_argument("--onnx", type=Path, help="ONNX model path")
    parser.add_argument(
        "--variant",
        type=str,
        default="dinov3_vits16_pretrain_lvd1689m",
        help="Meta DINOv3 variant identifier",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Path to Meta's .pth checkpoint (defaults to torch hub cache)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance used only for reporting",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance used only for reporting",
    )
    parser.add_argument(
        "--eqx",
        type=Path,
        help="Path to the mapped Equinox checkpoint (eqx_dinov3_vits16_mapped.eqx)",
    )
    parser.add_argument(
        "--block-debug",
        action="store_true",
        help="Capture per-block debug outputs (requires --eqx)",
    )
    args = parser.parse_args(argv)

    model, img_size = _load_meta_model(args.variant, args.weights)
    x = _preprocess(args.image.expanduser(), img_size)

    torch_tokens = _run_torch(model, x)

    if args.onnx is not None:
        onnx_tokens = _run_onnx(args.onnx.expanduser(), x)
        cls_onnx = onnx_tokens[0, 0, :]
        cls_torch = torch_tokens[0, 0, :]
        patches_onnx = onnx_tokens[0, 1:, :].mean(axis=0)
        patches_torch = torch_tokens[0, 1:, :].mean(axis=0)

        print("CLS features")
        print(f"  cosine : {_cosine(cls_onnx, cls_torch):.6f}")
        print(f"  max|Δ| : {_max_abs(cls_onnx, cls_torch):.6e}")
        print(
            f"  close  : {np.allclose(cls_onnx, cls_torch, rtol=args.rtol, atol=args.atol)}"
        )

        print("\nPooled patch features")
        print(f"  cosine : {_cosine(patches_onnx, patches_torch):.6f}")
        print(f"  max|Δ| : {_max_abs(patches_onnx, patches_torch):.6e}")
        print(
            f"  close  : {np.allclose(patches_onnx, patches_torch, rtol=args.rtol, atol=args.atol)}"
        )

    if args.eqx is not None:
        eqx_model = _load_eqx_model(args.variant, args.eqx)
        eqx_tokens = eqx_model(jnp.asarray(x))
        eqx_tokens = np.array(np.asarray(eqx_tokens))

        cls_eqx = eqx_tokens[0, 0, :]
        cls_torch = torch_tokens[0, 0, :]
        patches_eqx = eqx_tokens[0, 1:, :].mean(axis=0)
        patches_torch = torch_tokens[0, 1:, :].mean(axis=0)

        print("\nEQX vs Meta CLS")
        print(f"  cosine : {_cosine(cls_eqx, cls_torch):.6f}")
        print(f"  max|Δ| : {_max_abs(cls_eqx, cls_torch):.6e}")

        print("\nEQX vs Meta pooled patches")
        print(f"  cosine : {_cosine(patches_eqx, patches_torch):.6f}")
        print(f"  max|Δ| : {_max_abs(patches_eqx, patches_torch):.6e}")

        if args.onnx is not None:
            cls_onnx = onnx_tokens[0, 0, :]
            patches_onnx = onnx_tokens[0, 1:, :].mean(axis=0)
            print("\nEQX vs ONNX CLS")
            print(f"  cosine : {_cosine(cls_eqx, cls_onnx):.6f}")
            print(f"  max|Δ| : {_max_abs(cls_eqx, cls_onnx):.6e}")
            print("\nEQX vs ONNX pooled patches")
            print(f"  cosine : {_cosine(patches_eqx, patches_onnx):.6f}")
            print(f"  max|Δ| : {_max_abs(patches_eqx, patches_onnx):.6e}")

        if args.block_debug:
            eqx_debug = _eqx_block_debug_outputs(eqx_model, x)
            meta_debug = _meta_block_debug_outputs(model, x)
            stages = [
                "input",
                "attn_in",
                "attn_raw",
                "attn_norm",
                "attn_scaled",
                "post_attn",
                "mlp_in",
                "mlp_raw",
                "mlp_scaled",
                "output",
            ]
            print("\nPer-block max|Δ| (EQX vs Meta)")
            for idx, (eq_stage, meta_stage) in enumerate(zip(eqx_debug, meta_debug)):
                print(f"Block {idx:02d}")
                for stage in stages:
                    if stage not in eq_stage or stage not in meta_stage:
                        continue
                    diff = _max_abs(eq_stage[stage], meta_stage[stage])
                    print(f"  {stage:>10s}: {diff:.6e}")
    elif args.block_debug:
        raise ValueError("--block-debug requires --eqx")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
