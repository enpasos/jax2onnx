#!/usr/bin/env python3
# scripts/generate_dinov3_reference.py

"""Generate a reference output for a pretrained DINOv3 Equinox model.

This helper loads a converted DINOv3 checkpoint (produced via
``scripts/convert_dinov3_from_equimo.py``), runs a forward pass on an input array,
and saves the resulting feature map to ``.npy``. The artefact can be used as the
expected output for the optional integration test.

Example
-------
    python scripts/generate_dinov3_reference.py \
        --variant dinov3_vits16_pretrain_lvd1689m \
        --weights ~/.cache/equimo/dinov3/dinov3_vits16_pretrain_lvd1689m.tar.lz4 \
        --input my_input.npy \
        --output my_expected.npy

The input array must be stored as ``.npy`` and have shape ``(3, H, W)`` with values
in the expected preprocessing range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        default="dinov3_vits16_pretrain_lvd1689m",
        help="DINOv3 variant identifier (matches Equimo).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the converted Equinox archive (.tar.lz4).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=".npy file containing the input image tensor (shape: 3xHxW).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .npy file for the reference activations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for stochastic layers (default: 0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from jax2onnx.plugins.examples.eqx.dino import load_pretrained_dinov3
    except Exception as exc:  # pragma: no cover - import side effects
        raise RuntimeError(
            "Unable to import DINO helper. Ensure jax2onnx is on PYTHONPATH."
        ) from exc

    model = load_pretrained_dinov3(
        variant=args.variant,
        weights_path=args.weights,
        inference_mode=True,
    )

    arr = np.load(args.input).astype(np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(
            f"Expected input shape (3, H, W); received {arr.shape}. Prepare the array"
            " with channel-first layout and save as .npy."
        )

    key = jax.random.PRNGKey(args.seed)
    output = model.features(jnp.asarray(arr), inference=True, key=key)
    np.save(args.output, np.asarray(output))
    print(f"Saved reference activations to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
