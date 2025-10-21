#!/usr/bin/env python3
# scripts/export_eqx_dino_example_with_mapped_weights.py

"""
Export the simplified examples.eqx_dino VisionTransformer with mapped weights to ONNX.

This preserves the exact graph structure of the example defined in
`jax2onnx/plugins/examples/eqx/dino.py` (the same path used by testcases),
only changing the initializers to the provided pretrained values.

Usage
-----
    poetry run python scripts/export_eqx_dino_example_with_mapped_weights.py \
        --eqx docs/onnx/examples/eqx_dino/dinov3_vits16_pretrain_lvd1689m.eqx \
        --output docs/onnx/examples/eqx_dino/eqx_dinov3_vit_S16.onnx \
        --img-size 224 \
        --dynamic-b

Notes
-----
 - The `.eqx` file is produced by `scripts/map_equimo_dino_weights.py`, which
   lifts Equimo checkpoints into the simplified example VisionTransformer.
 - Use `--dynamic-b` to mirror the dynamic-batch testcase structure. Omit it to
   write a static-batch (B=1) variant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax

from jax2onnx import to_onnx
from jax2onnx.plugins.examples.eqx.dino import VisionTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eqx",
        type=Path,
        required=True,
        help="Path to mapped example .eqx file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination ONNX path",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size (HxW)",
    )
    # Model config (required to build the 'like' tree for deserialisation)
    parser.add_argument(
        "--variant",
        type=str,
        help=(
            "Optional variant hint (e.g., dinov3_vits16_pretrain_lvd1689m). "
            "Used only to infer defaults for the simplified example model."
        ),
    )
    parser.add_argument("--patch-size", type=int, help="Patch size")
    parser.add_argument("--embed-dim", type=int, help="Embedding dimension")
    parser.add_argument("--depth", type=int, help="Transformer depth (blocks)")
    parser.add_argument("--num-heads", type=int, help="Number of attention heads")
    parser.add_argument(
        "--seed", type=int, default=0, help="PRNG seed for 'like' model"
    )
    parser.add_argument(
        "--dynamic-b",
        action="store_true",
        help="Emit dynamic batch dimension (matches testcase structure)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Infer a minimal example config when not fully specified
    ps = args.patch_size
    ed = args.embed_dim
    dp = args.depth
    nh = args.num_heads
    if any(v is None for v in (ps, ed, dp, nh)):
        # Try to infer from variant string/path if provided
        hint = args.variant or args.eqx.stem  # fall back to filename
        hint_lower = str(hint).lower()
        # Known small mapping aligned with example variants
        # Ti14: patch=14, dim=192, heads=3, depth=12
        # S14:  patch=14, dim=384, heads=6, depth=12
        # B14:  patch=14, dim=768, heads=12, depth=12
        # S16:  patch=16, dim=384, heads=6, depth=12
        if "vitti14" in hint_lower or ("ti" in hint_lower and "14" in hint_lower):
            ps = ps or 14
            ed = ed or 192
            nh = nh or 3
            dp = dp or 12
        elif "vits14" in hint_lower or ("s14" in hint_lower):
            ps = ps or 14
            ed = ed or 384
            nh = nh or 6
            dp = dp or 12
        elif "vitb14" in hint_lower or ("b14" in hint_lower):
            ps = ps or 14
            ed = ed or 768
            nh = nh or 12
            dp = dp or 12
        elif "vits16" in hint_lower or ("s16" in hint_lower):
            ps = ps or 16
            ed = ed or 384
            nh = nh or 6
            dp = dp or 12

    _pairs = (("patch_size", ps), ("embed_dim", ed), ("depth", dp), ("num_heads", nh))
    missing = [name for name, val in _pairs if val is None]
    if missing:
        raise SystemExit(
            "Missing model config: "
            + ", ".join(missing)
            + ". Provide flags (e.g., --patch-size/--embed-dim/--depth/--num-heads) "
            "or a recognizable --variant hint."
        )

    # Build a 'like' model to guide deserialisation
    like = VisionTransformer(
        img_size=int(args.img_size),
        patch_size=int(ps),
        embed_dim=int(ed),
        depth=int(dp),
        num_heads=int(nh),
        key=jax.random.PRNGKey(args.seed),
    )
    model = eqx.tree_deserialise_leaves(args.eqx.expanduser(), like)

    # Export using the same callable as the example's __call__;
    # this preserves the exact operator layout used in testcases.
    def fn(x):
        return model(x)

    # Build input spec to match testcase shape semantics
    if args.dynamic_b:
        input_spec = [("B", 3, args.img_size, args.img_size)]
    else:
        input_spec = [(1, 3, args.img_size, args.img_size)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    to_onnx(
        fn,
        inputs=input_spec,
        model_name="eqx_dinov3_vit",
        return_mode="file",
        output_path=str(args.output),
    )
    print(f"Exported ONNX to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
