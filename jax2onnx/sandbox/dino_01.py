#!/usr/bin/env python3
# jax2onnx/sandbox/dino_01.py

"""
Run a DINOv3 ONNX model on a single image and save features.

Example:
  poetry run python jax2onnx/sandbox/dino_01.py \
    --model /home/enpasos/.cache/equimo/dinov3/eqx_dinov3_vit_S16.onnx \
    --image /path/to/image.jpg \
    --out-cls /tmp/dino_cls.npy \
    --out-pooled /tmp/dino_pooled.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import hashlib
import urllib.request

import numpy as np

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - runtime dep
    raise SystemExit(
        "onnxruntime is required. Install it in your environment, e.g.\n"
        "  poetry run pip install onnxruntime\n"
        f"Original error: {exc}"
    )

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - runtime dep
    raise SystemExit(
        "Pillow is required. Install it in your environment, e.g.\n"
        "  poetry run pip install pillow\n"
        f"Original error: {exc}"
    )


def _center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    return img.crop((left, top, left + short, top + short))


def _preprocess(
    img_path: Path,
    size: int,
    *,
    do_center_crop: bool = True,
    imagenet_norm: bool = True,
) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    if do_center_crop:
        img = _center_crop_to_square(img)
    img = img.resize((size, size), Image.BICUBIC)

    x = np.asarray(img).astype("float32") / 255.0
    if imagenet_norm:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std

    # HWC -> CHW and add batch dim: (1, 3, H, W)
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x


def _l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v)) + eps
    return v / n


def run(
    model_path: Path,
    image_path: Path,
    size: int,
    *,
    save_cls: Path | None,
    save_pooled: Path | None,
    no_norm: bool,
    no_center_crop: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    x = _preprocess(
        image_path,
        size,
        do_center_crop=not no_center_crop,
        imagenet_norm=not no_norm,
    )

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {inp_name: x})
    y = outputs[0]
    # y shape: (B, tokens, dim) => (1, 1+num_patches, dim)
    if y.ndim != 3 or y.shape[0] != 1:
        raise SystemExit(
            f"Unexpected output shape {y.shape}; expected (1, tokens, dim)"
        )

    cls_feat = y[:, 0, :].squeeze(0)  # (dim,)
    patch_feats = y[:, 1:, :].squeeze(0)  # (num_patches, dim)
    pooled_feat = patch_feats.mean(axis=0)  # (dim,)

    if save_cls is not None:
        np.save(str(save_cls), _l2norm(cls_feat))
    if save_pooled is not None:
        np.save(str(save_pooled), _l2norm(pooled_feat))

    print(f"CLS: {cls_feat.shape}, POOLED: {pooled_feat.shape}")
    return cls_feat, pooled_feat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path, help="Path to ONNX model")
    p.add_argument("--image", type=Path, help="Path to input image")
    p.add_argument(
        "--img-size", type=int, default=224, help="Resize size (default: 224)"
    )
    p.add_argument(
        "--out-cls",
        type=Path,
        help="Optional path to save L2-normalized CLS features (.npy)",
    )
    p.add_argument(
        "--out-pooled",
        type=Path,
        help="Optional path to save L2-normalized pooled patch features (.npy)",
    )
    p.add_argument(
        "--no-norm", action="store_true", help="Disable ImageNet normalization"
    )
    p.add_argument(
        "--no-center-crop", action="store_true", help="Disable center crop; just resize"
    )
    p.add_argument(
        "--download-dog",
        action="store_true",
        help="Download PyTorch hub dog.jpg to /tmp and use it",
    )
    p.add_argument(
        "--ref-cls", type=Path, help="Compare CLS vector to this .npy reference"
    )
    p.add_argument(
        "--ref-pooled", type=Path, help="Compare pooled vector to this .npy reference"
    )
    p.add_argument(
        "--rtol", type=float, default=1e-4, help="Relative tolerance for comparison"
    )
    p.add_argument(
        "--atol", type=float, default=1e-6, help="Absolute tolerance for comparison"
    )
    p.add_argument(
        "--print-checksum",
        action="store_true",
        help="Print SHA256 of output vectors for easy diffing",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    img_path = args.image
    if args.download_dog:
        target = Path("/tmp/dog.jpg")
        url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:  # pragma: no cover - network
            raise SystemExit(f"Failed to download {url}: {exc}")
        img_path = target
        print(f"Downloaded dog image to {target}")

    if img_path is None:
        raise SystemExit("--image is required unless --download-dog is specified")

    run(
        model_path=args.model.expanduser(),
        image_path=img_path.expanduser(),
        size=int(args.img_size),
        save_cls=args.out_cls.expanduser() if args.out_cls else None,
        save_pooled=args.out_pooled.expanduser() if args.out_pooled else None,
        no_norm=bool(args.no_norm),
        no_center_crop=bool(args.no_center_crop),
    )

    # If references are provided, compare or print checksums.
    if args.ref_cls or args.ref_pooled or args.print_checksum:
        import numpy as _np

        def _sha256(a: _np.ndarray) -> str:
            return hashlib.sha256(_np.asarray(a).tobytes()).hexdigest()

        if args.ref_cls and args.out_cls:
            ref = _np.load(str(args.ref_cls))
            out = _np.load(str(args.out_cls))
            ok = _np.allclose(out, ref, rtol=args.rtol, atol=args.atol)
            print(f"CLS compare: {ok} (rtol={args.rtol}, atol={args.atol})")
            if args.print_checksum:
                print(f"CLS sha256: {_sha256(out)}")
        elif args.print_checksum and args.out_cls:
            out = _np.load(str(args.out_cls))
            print(f"CLS sha256: {_sha256(out)}")

        if args.ref_pooled and args.out_pooled:
            ref = _np.load(str(args.ref_pooled))
            out = _np.load(str(args.out_pooled))
            ok = _np.allclose(out, ref, rtol=args.rtol, atol=args.atol)
            print(f"Pooled compare: {ok} (rtol={args.rtol}, atol={args.atol})")
            if args.print_checksum:
                print(f"Pooled sha256: {_sha256(out)}")
        elif args.print_checksum and args.out_pooled:
            out = _np.load(str(args.out_pooled))
            print(f"Pooled sha256: {_sha256(out)}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
# file: jax2onnx/sandbox/dino_01.py
