#!/usr/bin/env python3
# scripts/export_eqx_gpt_oss_subset.py

"""Stage or export the GPT-OSS Equinox model with optional layer pruning."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

# Default to CPU to avoid GPU OOMs during weight mapping.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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
        help="Path to the GPT-OSS checkpoint directory (SafeTensors shards + config.json).",
    )
    parser.add_argument(
        "--eqx",
        type=Path,
        help=(
            "Load weights from an existing Equinox archive instead of a checkpoint. "
            "Expects a companion .meta.json with config/param_dtype/seed."
        ),
    )
    parser.add_argument(
        "--save-eqx",
        type=Path,
        help="Optional path to serialise the mapped Equinox parameters (.eqx).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional ONNX destination. Skipped if not provided.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Prune the model to the first N layers after loading (helps avoid OOM).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Static batch when not exporting with --dynamic-b.",
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
        help="Emit a dynamic batch dimension symbol 'B'.",
    )
    parser.add_argument(
        "--dynamic-seq",
        action="store_true",
        help="Emit a dynamic sequence dimension symbol 'T'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device string passed to load_pretrained_gpt_oss (default: cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed used when constructing the Equinox model.",
    )
    parser.add_argument(
        "--param-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Parameter dtype for the Equinox model (can override archive metadata).",
    )
    parser.add_argument(
        "--emit-debug",
        action="store_true",
        help=(
            "Emit block-level debug outputs (attn q/k/v, gate stats, MLP intermediates)"
            " alongside logits."
        ),
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export (useful when only emitting --save-eqx).",
    )
    return parser.parse_args()


def _trim_layers(model: Transformer, max_layers: int) -> Transformer:
    """Return a copy of ``model`` keeping only the first ``max_layers`` blocks."""

    total = len(model.blocks)
    if max_layers <= 0 or max_layers >= total:
        return model
    trimmed_blocks = tuple(model.blocks[:max_layers])
    new_config = dataclasses.replace(model.config, num_hidden_layers=max_layers)
    model = eqx.tree_at(lambda m: m.blocks, model, trimmed_blocks)
    # config is a static field; assign directly.
    object.__setattr__(model, "config", new_config)
    return model


def _input_spec(args: argparse.Namespace):
    if args.dynamic_b and args.dynamic_seq:
        return [("B", "T")]
    if args.dynamic_b:
        return [("B", int(args.seq_len))]
    if args.dynamic_seq:
        return [(int(args.batch_size), "T")]
    return [(int(args.batch_size), int(args.seq_len))]


DEBUG_KEYS = [
    "input",
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_out",
    "mlp_norm",
    "gate_logits",
    "expert_indices",
    "expert_weights",
    "mlp_proj1",
    "mlp_act",
    "mlp_proj2",
    "mlp_output",
    "output",
]


def main() -> int:
    args = parse_args()
    if args.eqx is None and args.checkpoint is None:
        raise SystemExit("Either --checkpoint or --eqx must be provided.")

    dtype_map = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}
    param_dtype_key = args.param_dtype
    param_dtype = dtype_map[param_dtype_key]

    metadata: dict | None = None
    if args.eqx is not None:
        eqx_path = args.eqx.expanduser()
        if not eqx_path.exists():
            raise FileNotFoundError(f"Equinox archive not found: {eqx_path}")
        meta_path = eqx_path.with_suffix(eqx_path.suffix + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file missing for Equinox archive: {meta_path}"
            )
        metadata = json.loads(meta_path.read_text())
        config_data = metadata.get("config")
        if not isinstance(config_data, dict):
            raise ValueError(f"Invalid config metadata in {meta_path}")
        config = GPTOSSConfig(**config_data)
        # Prefer the CLI override when provided; fall back to stored metadata.
        param_dtype_key = args.param_dtype or str(
            metadata.get("param_dtype", param_dtype_key)
        )
        if param_dtype_key not in dtype_map:
            raise ValueError(
                f"Unsupported param dtype '{param_dtype_key}' recorded in {meta_path}"
            )
        param_dtype = dtype_map[param_dtype_key]
        # Build a template with the recorded dtype so deserialisation matches.
        seed_value = int(metadata.get("seed", args.seed))
        template = Transformer(
            config=config,
            key=jax.random.PRNGKey(seed_value),
            param_dtype=param_dtype,
        )
        # Allow dtype differences by copying stored leaves onto the template.
        model = eqx.tree_deserialise_leaves(eqx_path, template, is_leaf=lambda _: True)
        if param_dtype == jnp.float32:
            model = jax.tree_util.tree_map(
                lambda x: (
                    x.astype(jnp.float32)
                    if isinstance(x, jnp.ndarray) and x.dtype != jnp.float32
                    else x
                ),
                model,
            )
    else:
        model = load_pretrained_gpt_oss(
            checkpoint=args.checkpoint.expanduser(),
            device=args.device,
            param_dtype=param_dtype,
            seed=args.seed,
        )
        metadata = {
            "config": dataclasses.asdict(model.config),
            "param_dtype": param_dtype_key,
            "seed": int(args.seed),
        }

    if args.max_layers is not None:
        model = _trim_layers(model, int(args.max_layers))
        if metadata is not None:
            metadata["max_layers"] = int(args.max_layers)

    if args.save_eqx is not None:
        eqx_path = args.save_eqx.expanduser()
        eqx_path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(eqx_path, model)
        print(f"Saved Equinox parameters to {eqx_path}")
        meta_path = eqx_path.with_suffix(eqx_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        print(f"Wrote metadata to {meta_path}")

    if args.skip_onnx:
        print("Skipping ONNX export (--skip-onnx set).")
    elif args.output is not None:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def fn(tokens):
            if args.emit_debug:
                logits, debug_blocks = model.debug(tokens)
            else:
                return model(tokens)
            outputs: list = [logits]
            for entry in debug_blocks:
                for key in DEBUG_KEYS:
                    if key not in entry:
                        raise RuntimeError(
                            f"Missing debug key '{key}' in block capture"
                        )
                    outputs.append(entry[key])
            return tuple(outputs)

        to_onnx(
            fn,
            inputs=_input_spec(args),
            model_name="eqx_gpt_oss_transformer",
            output_path=str(output_path),
            return_mode="file",
        )
        print(f"Exported ONNX to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
