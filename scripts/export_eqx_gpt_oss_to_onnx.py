#!/usr/bin/env python3
# scripts/export_eqx_gpt_oss_to_onnx.py

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import msgpack_restore

from jax2onnx.plugins.examples.eqx.gpt_oss import (
    GPTOSSConfig,
    Transformer,
)
from jax2onnx.user_interface import allclose, to_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the Equinox GPT-OSS transformer stack to ONNX."
    )
    parser.add_argument(
        "--params",
        required=True,
        type=Path,
        help="Path to staged Flax params (.msgpack).",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination ONNX file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional GPT-OSS config JSON. Defaults to <params>.config.json.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Sequence length to trace/export.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip running a JAX vs. ONNX numeric comparison after exporting.",
    )
    return parser.parse_args()


def _load_params(bundle: Path) -> dict:
    payload = msgpack_restore(bundle.read_bytes())
    if isinstance(payload, dict) and "params" in payload:
        payload = payload["params"]
    return jax.tree.map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
        payload,
    )


def _load_config(config_path: Path) -> GPTOSSConfig:
    data = json.loads(config_path.read_text())
    allowed = {field.name for field in fields(GPTOSSConfig)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    if "hidden_size" not in filtered:
        raise ValueError(f"Config file '{config_path}' does not define 'hidden_size'.")
    return GPTOSSConfig(**filtered)


def _transpose_kernel(kernel: jax.Array) -> jax.Array:
    # Flax Linear kernel is (in, out), Equinox Linear weight is (out, in)
    return jnp.transpose(kernel)


def _populate_eqx_from_flax_params(model: Transformer, params: dict) -> Transformer:
    # Embedding
    model = eqx.tree_at(
        lambda m: m.embedding.weight,
        model,
        params["embedding"]["embedding"],
    )

    # Blocks
    for layer_idx in range(model.config.num_hidden_layers):
        block_params = params[f"block_{layer_idx}"]

        # Attention
        attn_params = block_params["attn"]
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].attn.norm.weight,
            model,
            attn_params["norm"]["scale"],
        )
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].attn.qkv.weight,
            model,
            _transpose_kernel(attn_params["qkv"]["kernel"]),
        )
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].attn.qkv.bias,
            model,
            attn_params["qkv"]["bias"],
        )
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].attn.out.weight,
            model,
            _transpose_kernel(attn_params["out"]["kernel"]),
        )
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].attn.out.bias,
            model,
            attn_params["out"]["bias"],
        )
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].attn.sinks,
            model,
            attn_params["sinks"],
        )

        # MLP
        mlp_params = block_params["mlp"]
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].mlp.norm.weight,
            model,
            mlp_params["norm"]["scale"],
        )
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].mlp.gate.weight,
            model,
            _transpose_kernel(mlp_params["gate"]["kernel"]),
        )
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].mlp.gate.bias,
            model,
            mlp_params["gate"]["bias"],
        )
        # MLP1/MLP2 weights in Flax GPT-OSS are (experts, in, out) or similar?
        # Let's check the Flax implementation or just assume standard Linear behavior per expert.
        # In Flax GPT-OSS `mlp1_weight` is (num_experts, in_features, out_features).
        # In Equinox GPT-OSS `mlp1_weight` is (num_experts, out_features, in_features) ?
        # Let's check `jax2onnx/plugins/examples/eqx/gpt_oss.py`.
        # MLPBlock init: mlp1_weight is (num_experts, intermediate*2, hidden) -> (experts, out, in)
        # So we need to transpose the last two dims.

        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].mlp.mlp1_weight,
            model,
            jnp.swapaxes(mlp_params["mlp1_weight"], -1, -2),
        )
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].mlp.mlp1_bias,
            model,
            mlp_params["mlp1_bias"],
        )
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].mlp.mlp2_weight,
            model,
            jnp.swapaxes(mlp_params["mlp2_weight"], -1, -2),
        )
        model = eqx.tree_at(
            lambda m, i=layer_idx: m.blocks[i].mlp.mlp2_bias,
            model,
            mlp_params["mlp2_bias"],
        )

    # Final Norm
    model = eqx.tree_at(
        lambda m: m.norm.weight,
        model,
        params["norm"]["scale"],
    )
    # Unembedding
    model = eqx.tree_at(
        lambda m: m.unembedding.weight,
        model,
        _transpose_kernel(params["unembedding"]["kernel"]),
    )

    return model


def main() -> None:
    args = parse_args()
    params_path = args.params.expanduser().resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Params bundle not found: {params_path}")

    params = _load_params(params_path)

    config_path = args.config or params_path.with_suffix(".config.json")
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    config = _load_config(config_path)

    # Initialize model with dummy key (weights will be overwritten)
    # Use bfloat16 to match the params usually
    model = Transformer(
        config=config,
        key=jax.random.PRNGKey(0),
        param_dtype=jnp.bfloat16,
    )

    # Populate params
    model = _populate_eqx_from_flax_params(model, params)

    seq_len = args.sequence_length
    dummy_tokens = jnp.zeros((1, seq_len), dtype=jnp.int32)

    output_path = args.output.expanduser().resolve()
    # Use ShapeDtypeStruct to enforce static shapes
    input_spec = jax.ShapeDtypeStruct((1, seq_len), jnp.int32)

    to_onnx(
        model,
        [input_spec],
        model_name="eqx_gpt_oss_transformer",
        return_mode="file",
        output_path=str(output_path),
    )

    if not args.skip_validation:
        # For validation we need to ensure inputs match expected shape
        # The model expects (batch, seq)
        success, message = allclose(
            model,
            str(output_path),
            [np.asarray(dummy_tokens, dtype=np.int32)],
        )
        if success:
            print("[export] ONNX vs. JAX numeric check: PASS")
        else:
            print("[export] ONNX vs. JAX numeric check: FAIL")
            print(f"         details: {message}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
