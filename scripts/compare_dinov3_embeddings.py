#!/usr/bin/env python3
# scripts/compare_dinov3_embeddings.py

"""
Compare embeddings from the ONNX-exported DINOv3 model against Meta's original
PyTorch checkpoint.

Example:
    poetry run python scripts/compare_dinov3_embeddings.py \
        --image /tmp/coco_39769.jpg \
        --onnx ~/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx \
        --variant dinov3_vits16_pretrain_lvd1689m

By default the script looks for the Meta checkpoint under
``~/.cache/torch/hub/dinov3/weights/`` exactly as downloaded by the official
release instructions. Pass ``--weights`` to override.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image


_VARIANT_META = {
    # variant: (builder function, expected img_size)
    "dinov3_vits16_pretrain_lvd1689m": ("dinov3_vits16", 224),
    "dinov3_vits16plus_pretrain_lvd1689m": ("dinov3_vits16plus", 224),
    "dinov3_vitb14_pretrain_lvd1689m": ("dinov3_vitb16", 224),
    "dinov3_vitl14_pretrain_lvd1689m": ("dinov3_vitl16", 224),
}


def _resolve_weights_path(variant: str, weights: Path | None) -> Path:
    if weights is not None:
        path = Path(weights).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint for '{variant}' not found at {path}. "
                "Download Meta's release and point --weights to it."
            )
        return path

    default_root = Path("~/.cache/torch/hub/dinov3/weights").expanduser()
    if not default_root.exists():
        raise FileNotFoundError(
            f"Default weight directory {default_root} missing. "
            "Download the pretrain weights or pass --weights."
        )
    matches = sorted(default_root.glob(f"{variant}*.pth"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching '{variant}' found under {default_root}. "
            "Download Meta's release or specify --weights."
        )
    return matches[0]


def _load_torch_model(
    variant: str, weights: Path | None, seed: int
) -> Tuple[torch.nn.Module, int]:
    """
    Load Meta's DINOv3 PyTorch checkpoint using the torch.hub repository cache.

    Returns the torch model in eval mode and the image size it expects.
    """

    repo_root = Path("~/.cache/torch/hub/facebookresearch_dinov3_main").expanduser()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Torch hub repo not found at {repo_root}. "
            "Run `torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=False)` once "
            "to populate the repository cache, or clone the repo manually and point TORCH_HUB_DIR to it."
        )

    builder_meta = _VARIANT_META.get(variant)
    if builder_meta is None:
        available = ", ".join(sorted(_VARIANT_META))
        raise ValueError(
            f"Unsupported variant '{variant}'. Extend _VARIANT_META or choose one of: {available}"
        )

    builder_name, img_size = builder_meta
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from dinov3.hub import backbones  # type: ignore

    if not hasattr(backbones, builder_name):
        raise AttributeError(
            f"dinov3.hub.backbones is missing '{builder_name}'. "
            f"Check that the torch hub repo at {repo_root} matches the expected commit."
        )

    checkpoint = _resolve_weights_path(variant, weights)
    builder = getattr(backbones, builder_name)
    torch_model = builder(pretrained=True, weights=str(checkpoint))
    torch_model.eval()
    torch.manual_seed(seed)
    return torch_model, img_size


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


def _run_torch(torch_model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(x).to(torch.float32)
        out = torch_model.forward_features(tensor)
    cls = out["x_norm_clstoken"].cpu().numpy()
    patches = out["x_norm_patchtokens"].cpu().numpy()
    tokens = np.concatenate([cls[:, None, :], patches], axis=1)
    return tokens


def _cls_and_pooled(tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(f"Unexpected token shape {tokens.shape}")
    cls_vec = tokens[:, 0, :].squeeze(0)
    patch_vecs = tokens[:, 1:, :].squeeze(0)
    pooled = patch_vecs.mean(axis=0)
    return cls_vec, pooled


def _l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v)) + eps
    return v / n


def _sha256(v: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(v).tobytes()).hexdigest()


def compare(
    onnx_tokens: np.ndarray,
    torch_tokens: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    onnx_cls, onnx_pooled = _cls_and_pooled(onnx_tokens)
    torch_cls, torch_pooled = _cls_and_pooled(torch_tokens)

    cls_delta = np.max(np.abs(onnx_cls - torch_cls))
    pooled_delta = np.max(np.abs(onnx_pooled - torch_pooled))
    cls_close = np.allclose(onnx_cls, torch_cls, atol=atol, rtol=rtol)
    pooled_close = np.allclose(onnx_pooled, torch_pooled, atol=atol, rtol=rtol)

    print("CLS features")
    print(f"  shape: {onnx_cls.shape}")
    print(f"  max|Δ|: {cls_delta:.6e}")
    print(f"  close : {cls_close} (rtol={rtol}, atol={atol})")
    print(f"  sha256(ONNX):  { _sha256(_l2norm(onnx_cls)) }")
    print(f"  sha256(PyTorch): { _sha256(_l2norm(torch_cls)) }")

    print("\nPooled patch features")
    print(f"  shape: {onnx_pooled.shape}")
    print(f"  max|Δ|: {pooled_delta:.6e}")
    print(f"  close : {pooled_close} (rtol={rtol}, atol={atol})")
    print(f"  sha256(ONNX):  { _sha256(_l2norm(onnx_pooled)) }")
    print(f"  sha256(PyTorch): { _sha256(_l2norm(torch_pooled)) }")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True, help="Input image path")
    p.add_argument("--onnx", type=Path, required=True, help="Exported ONNX model")
    p.add_argument(
        "--variant",
        type=str,
        default="dinov3_vits16_pretrain_lvd1689m",
        help="Meta/Equimo DINOv3 variant identifier",
    )
    p.add_argument(
        "--weights",
        type=Path,
        help="Override path to Meta's .pth checkpoint (defaults to Equimo cache)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic module construction (mirrors conversion flow)",
    )
    p.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance")
    p.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    torch_model, img_size = _load_torch_model(
        args.variant, args.weights, seed=args.seed
    )

    x = _preprocess(args.image.expanduser(), img_size)
    onnx_tokens = _run_onnx(args.onnx.expanduser(), x)
    torch_tokens = _run_torch(torch_model, x)
    compare(onnx_tokens, torch_tokens, atol=args.atol, rtol=args.rtol)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
