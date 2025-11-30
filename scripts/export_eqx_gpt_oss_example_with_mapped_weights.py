#!/usr/bin/env python3
# scripts/export_eqx_gpt_oss_example_with_mapped_weights.py

"""Export the GPT-OSS Equinox example with real checkpoint weights to ONNX."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from jax2onnx import to_onnx
from jax2onnx.plugins.examples.eqx.gpt_oss import (
    GPTOSSConfig,
    Transformer,
    load_pretrained_gpt_oss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Directory containing the GPT-OSS SafeTensors shards and config.json "
            "(e.g., downloaded from Hugging Face)."
        ),
    )
    parser.add_argument(
        "--eqx",
        type=Path,
        help=(
            "Load weights from an existing Equinox archive instead of reading a "
            "checkpoint. Useful for a two-step workflow where mapping and export run "
            "in separate processes."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination ONNX path.",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export (useful when only emitting --save-eqx artifacts).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Static batch size when not exporting with --dynamic-b.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Static sequence length when not exporting with --dynamic-seq.",
    )
    parser.add_argument(
        "--dynamic-b",
        action="store_true",
        help="Emit a dynamic batch dimension (uses the symbol 'B').",
    )
    parser.add_argument(
        "--dynamic-seq",
        action="store_true",
        help="Emit a dynamic sequence dimension (uses the symbol 'T').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device string passed through to Transformer.from_checkpoint (default: cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for constructing the Equinox example.",
    )
    parser.add_argument(
        "--param-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Parameter dtype for the Equinox model.",
    )
    parser.add_argument(
        "--save-eqx",
        type=Path,
        help="Optional path to serialise the mapped Equinox parameters (.eqx).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.eqx is None and args.checkpoint is None:
        raise SystemExit("Either --checkpoint or --eqx must be provided.")
    if not args.skip_onnx and args.output is None:
        raise SystemExit("--output is required unless --skip-onnx is set.")
    if args.skip_onnx and args.output is not None:
        print("Skipping ONNX export; ignoring --output.")

    dtype_map = {
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
    }
    metadata: dict[str, object] | None = None
    param_dtype_key = args.param_dtype
    param_dtype = dtype_map[param_dtype_key]

    if args.eqx is not None:
        eqx_path = args.eqx.expanduser()
        if not eqx_path.exists():
            raise FileNotFoundError(f"Equinox archive not found: {eqx_path}")
        meta_path = eqx_path.with_name(eqx_path.name + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file missing for Equinox archive: {meta_path}"
            )
        metadata = json.loads(meta_path.read_text())
        config_data = metadata.get("config")
        if not isinstance(config_data, dict):
            raise ValueError(f"Invalid config metadata in {meta_path}")
        config = GPTOSSConfig(**config_data)
        param_dtype_key = str(metadata.get("param_dtype", param_dtype_key))
        if param_dtype_key not in dtype_map:
            raise ValueError(
                f"Unsupported param dtype '{param_dtype_key}' recorded in {meta_path}"
            )
        param_dtype = dtype_map[param_dtype_key]
        seed_value = int(metadata.get("seed", args.seed))
        template = Transformer(
            config=config,
            key=jax.random.PRNGKey(seed_value),
            param_dtype=param_dtype,
        )
        model = eqx.tree_deserialise_leaves(eqx_path, template)
    else:
        model = load_pretrained_gpt_oss(
            checkpoint=args.checkpoint,
            device=args.device,
            param_dtype=param_dtype,
            seed=args.seed,
        )
        metadata = {
            "config": dataclasses.asdict(model.config),
            "param_dtype": str(param_dtype_key),
            "seed": int(args.seed),
        }

    if args.save_eqx is not None:
        eqx_path = args.save_eqx.expanduser()
        eqx_path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(eqx_path, model)
        print(f"Saved Equinox parameters to {eqx_path}")
        meta_path = eqx_path.with_name(eqx_path.name + ".meta.json")
        if metadata is None:
            raise RuntimeError(
                "Missing metadata while saving Equinox archive. "
                "Provide a checkpoint or ensure the source .eqx has a metadata file."
            )
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        print(f"Wrote metadata to {meta_path}")

    if args.skip_onnx:
        return 0

    def fn(tokens):
        return model(tokens)

    if args.dynamic_b and args.dynamic_seq:
        input_spec = [("B", "T")]
    elif args.dynamic_b:
        input_spec = [("B", int(args.seq_len))]
    elif args.dynamic_seq:
        input_spec = [(int(args.batch_size), "T")]
    else:
        input_spec = [(int(args.batch_size), int(args.seq_len))]

    if args.output is not None:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        to_onnx(
            fn,
            inputs=input_spec,
            model_name="eqx_gpt_oss_transformer",
            output_path=str(output_path),
            return_mode="file",
        )
        print(f"Exported ONNX to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
