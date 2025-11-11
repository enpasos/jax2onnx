#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import flax.serialization as flax_serialization
from flax.serialization import msgpack_restore
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax2onnx.plugins.examples.nnx.gpt_oss_flax import (
    GPTOSSConfig,
    Transformer,
    _causal_mask,
    _rotary_tables_for_config,
)
from jax2onnx.user_interface import to_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the Flax GPT-OSS transformer stack to ONNX."
    )
    parser.add_argument("--params", required=True, type=Path, help="Path to staged Flax params (.msgpack).")
    parser.add_argument("--output", required=True, type=Path, help="Destination ONNX file.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional GPT-OSS config JSON. Defaults to <params>.config.json (emitted by export_flax_gpt_oss_params.py).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Sequence length to trace/export. Must not exceed the rotary table length.",
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
    # hidden_size is the only required field without a default; surface a clear error if absent
    if "hidden_size" not in filtered:
        raise ValueError(
            f"Config file '{config_path}' does not define 'hidden_size'; "
            "rerun export_flax_gpt_oss_params.py or pass a config with the GPT-OSS hyper-parameters."
        )
    return GPTOSSConfig(**filtered)


def _assign_param(param: nnx.Param, value: np.ndarray | jax.Array) -> None:
    target_shape = param.value.shape
    target_dtype = param.value.dtype
    arr = jnp.asarray(value, dtype=target_dtype)
    if arr.shape != target_shape:
        raise ValueError(
            f"Parameter shape mismatch: expected {target_shape}, got {arr.shape}"
        )
    param.value = arr


def _load_model_params(model: Transformer, params: dict) -> None:
    _assign_param(model.embedding, params["embedding"]["embedding"])

    for layer_idx in range(model.config.num_hidden_layers):
        block_params = params[f"block_{layer_idx}"]
        block = model.blocks[f"block_{layer_idx}"]

        attn_params = block_params["attn"]
        _assign_param(block.attention.norm.scale, attn_params["norm"]["scale"])
        _assign_param(block.attention.qkv_kernel, attn_params["qkv"]["kernel"])
        _assign_param(block.attention.qkv_bias, attn_params["qkv"]["bias"])
        _assign_param(block.attention.out_kernel, attn_params["out"]["kernel"])
        _assign_param(block.attention.out_bias, attn_params["out"]["bias"])
        _assign_param(block.attention.sinks, attn_params["sinks"])

        mlp_params = block_params["mlp"]
        _assign_param(block.mlp.norm.scale, mlp_params["norm"]["scale"])
        _assign_param(block.mlp.gate_kernel, mlp_params["gate"]["kernel"])
        _assign_param(block.mlp.gate_bias, mlp_params["gate"]["bias"])
        _assign_param(block.mlp.mlp1_weight, mlp_params["mlp1_weight"])
        _assign_param(block.mlp.mlp1_bias, mlp_params["mlp1_bias"])
        _assign_param(block.mlp.mlp2_weight, mlp_params["mlp2_weight"])
        _assign_param(block.mlp.mlp2_bias, mlp_params["mlp2_bias"])

    _assign_param(model.norm.scale, params["norm"]["scale"])
    _assign_param(model.unembedding_kernel, params["unembedding"]["kernel"])


def main() -> None:
    args = parse_args()
    params_path = args.params.expanduser().resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Params bundle not found: {params_path}")

    params = _load_params(params_path)

    config_path = args.config or params_path.with_suffix(".config.json")
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. Provide --config or "
            "rerun export_flax_gpt_oss_params.py to emit the JSON alongside the params."
        )
    config = _load_config(config_path)

    seq_len = args.sequence_length
    cos_table, sin_table = _rotary_tables_for_config(
        config,
        min_length=seq_len,
    )
    sliding_mask = _causal_mask(seq_len, seq_len, sliding_window=config.sliding_window)
    causal_mask = _causal_mask(seq_len, seq_len, sliding_window=0)
    model = Transformer(
        config=config,
        cos_table=cos_table,
        sin_table=sin_table,
        sequence_length=seq_len,
        mask_sliding=sliding_mask,
        mask_causal=causal_mask,
        rng=jax.random.PRNGKey(0),
    )
    _load_model_params(model, params)

    def _apply(tokens):
        return model(tokens)

    dummy_tokens = jnp.zeros((seq_len,), dtype=jnp.int32)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_onnx(
        _apply,
        [dummy_tokens],
        model_name="flax_gpt_oss_transformer",
        return_mode="file",
        output_path=str(output_path),
    )


if __name__ == "__main__":
    main()
