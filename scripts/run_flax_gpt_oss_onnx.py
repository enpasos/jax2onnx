#!/usr/bin/env python3
"""Compare Flax GPT-OSS transformer outputs against its ONNX export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flax.serialization as flax_serialization
import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort

from jax2onnx.plugins.examples.nnx.gpt_oss_flax import (
    GPTOSSConfig,
    Transformer,
    _causal_mask,
    _rotary_tables_for_config,
)


def _load_params(bundle: Path) -> dict:
    payload = flax_serialization.msgpack_restore(bundle.read_bytes())
    return jax.tree.map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
        payload,
    )


def _assign_param(param, value):
    target_shape = param.value.shape
    arr = jnp.asarray(value, dtype=param.value.dtype)
    if arr.shape != target_shape:
        raise ValueError(f"Shape mismatch for {param}: expected {target_shape}, got {arr.shape}")
    param.value = arr


def _load_model(config: GPTOSSConfig, params: dict, seq_len: int) -> Transformer:
    cos, sin = _rotary_tables_for_config(config, min_length=seq_len)
    sliding_mask = _causal_mask(seq_len, seq_len, sliding_window=config.sliding_window)
    causal_mask = _causal_mask(seq_len, seq_len, sliding_window=0)
    model = Transformer(
        config=config,
        cos_table=cos,
        sin_table=sin,
        sequence_length=seq_len,
        mask_sliding=sliding_mask,
        mask_causal=causal_mask,
        rng=jax.random.PRNGKey(0),
    )

    _assign_param(model.embedding, params["embedding"]["embedding"])
    for layer_idx in range(config.num_hidden_layers):
        block = model.blocks[f"block_{layer_idx}"]
        params_block = params[f"block_{layer_idx}"]

        attn = params_block["attn"]
        _assign_param(block.attention.norm.scale, attn["norm"]["scale"])
        _assign_param(block.attention.qkv_kernel, attn["qkv"]["kernel"])
        _assign_param(block.attention.qkv_bias, attn["qkv"]["bias"])
        _assign_param(block.attention.out_kernel, attn["out"]["kernel"])
        _assign_param(block.attention.out_bias, attn["out"]["bias"])
        _assign_param(block.attention.sinks, attn["sinks"])

        mlp = params_block["mlp"]
        _assign_param(block.mlp.norm.scale, mlp["norm"]["scale"])
        _assign_param(block.mlp.gate_kernel, mlp["gate"]["kernel"])
        _assign_param(block.mlp.gate_bias, mlp["gate"]["bias"])
        _assign_param(block.mlp.mlp1_weight, mlp["mlp1_weight"])
        _assign_param(block.mlp.mlp1_bias, mlp["mlp1_bias"])
        _assign_param(block.mlp.mlp2_weight, mlp["mlp2_weight"])
        _assign_param(block.mlp.mlp2_bias, mlp["mlp2_bias"])

    _assign_param(model.norm.scale, params["norm"]["scale"])
    _assign_param(model.unembedding_kernel, params["unembedding"]["kernel"])
    return model


def _tokenize(prompt: str, vocab_size: int) -> np.ndarray:
    # Minimal byte-level fallback: map characters to pseudo vocab IDs.
    # For parity harness comparisons, replace with the actual GPT-OSS tokenizer.
    vocab_size = max(1, int(vocab_size))
    return np.array([ord(ch) % vocab_size for ch in prompt], dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Flax GPT-OSS vs ONNX outputs")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument(
        "--compare-hidden-states",
        action="store_true",
        help=(
            "Fetch/blockwise hidden states from both models. "
            "Requires an ONNX exported with --emit-hidden-states."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = _load_params(args.params.expanduser().resolve())
    config_dict = json.loads(args.config.read_text())
    allowed = GPTOSSConfig.__annotations__.keys()
    config = GPTOSSConfig(**{k: v for k, v in config_dict.items() if k in allowed})

    sess = ort.InferenceSession(
        str(args.onnx.expanduser().resolve()),
        providers=["CPUExecutionProvider"],
    )
    onnx_input = sess.get_inputs()[0]
    onnx_seq_len = onnx_input.shape[0]
    print(f"[info] ONNX input shape: {onnx_input.shape}", flush=True)
    seq_len = args.sequence_length
    if isinstance(onnx_seq_len, int) and onnx_seq_len > 0:
        seq_len = min(seq_len, int(onnx_seq_len))

    model = _load_model(config, params, seq_len)

    extra_outputs = sess.get_outputs()
    need_hidden = args.compare_hidden_states
    if need_hidden and len(extra_outputs) != 1 + config.num_hidden_layers:
        raise SystemExit(
            "ONNX model does not expose hidden states. Re-export with "
            "`scripts/export_flax_gpt_oss_to_onnx.py --emit-hidden-states ...` "
            "and rerun this script with --compare-hidden-states."
        )

    tokens = _tokenize(args.prompt, config.vocab_size)
    if tokens.shape[0] > seq_len:
        print(
            f"[info] prompt produced {tokens.shape[0]} tokens; truncating to sequence length {seq_len}",
            flush=True,
        )
        tokens = tokens[:seq_len]
    tokens_padded = np.zeros((seq_len,), dtype=np.int32)
    tokens_padded[: tokens.shape[0]] = tokens

    tokens_jax = jnp.asarray(tokens_padded)
    hidden_states: list[jax.Array] = []
    if need_hidden:
        jax_out = model(tokens_jax, capture_hidden_states=hidden_states)
    else:
        jax_out = model(tokens_jax)
    jax_out = jax.device_get(jax_out)
    jax_hidden = [np.asarray(jax.device_get(h)) for h in hidden_states]
    print(f"[info] JAX output shape: {jax_out.shape}")
    jax_logits = jax_out[-1]

    ort_outputs = sess.run(None, {sess.get_inputs()[0].name: tokens_padded})
    ort_logits_full = ort_outputs[0]
    ort_logits = ort_logits_full[-1]
    print(f"[info] ORT output shape: {ort_logits_full.shape}")
    ort_hidden = ort_outputs[1:] if need_hidden else []
    if need_hidden:
        if len(ort_hidden) != config.num_hidden_layers:
            raise SystemExit(
                f"Expected {config.num_hidden_layers} hidden outputs, "
                f"but ONNX returned {len(ort_hidden)}."
            )

    diff = np.max(np.abs(jax_logits - ort_logits))
    print(f"Prompt: {args.prompt!r}")
    print(f"JAX logits (last token): {jax_logits[:8]} ...")
    print(f"ORT logits (last token): {ort_logits[:8]} ...")
    print(f"Max |diff|: {diff}")

    if need_hidden:
        for layer_idx, (jax_state, ort_state) in enumerate(
            zip(jax_hidden, ort_hidden)
        ):
            layer_diff = np.max(np.abs(jax_state - ort_state))
            mean_diff = float(np.mean(np.abs(jax_state - ort_state)))
            print(
                f"[hidden] block_{layer_idx}: max |diff|={layer_diff:.6f}, "
                f"mean |diff|={mean_diff:.6f}"
            )


if __name__ == "__main__":
    main()
