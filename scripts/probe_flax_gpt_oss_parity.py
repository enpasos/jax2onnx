#!/usr/bin/env python3
# scripts/probe_flax_gpt_oss_parity.py

"""Probe GPT-OSS parity between the PyTorch reference and Flax/NNX modules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import flax.serialization as flax_serialization
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from jax2onnx.plugins.examples.nnx.gpt_oss_flax import (
    GPTOSSConfig,
    Transformer,
    _causal_mask,
    _rotary_tables_for_config,
)
from gpt_oss.torch.model import (
    Transformer as TorchTransformer,
    sdpa as torch_sdpa,
    swiglu as torch_swiglu,
)


def _ensure_repo_on_path(repo_path: Path) -> None:
    repo_path = repo_path.expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"gpt-oss checkout '{repo_path}' is required for torch imports"
        )
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


def _tokenize(prompt: str, vocab_size: int) -> Tuple[List[int], List[str]]:
    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        token_ids = tokenizer.encode(prompt)
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]
    except Exception:
        token_ids = [ord(ch) for ch in prompt]
        token_strs = list(prompt)

    vocab = max(1, int(vocab_size))
    token_ids = [tid % vocab for tid in token_ids]
    return token_ids, token_strs


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.dtype.is_floating_point or tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.cpu().numpy()
    if isinstance(value, jax.Array):
        arr = np.asarray(jax.device_get(value))
    else:
        arr = np.asarray(value)
    if arr.dtype.kind == "f":
        return arr.astype(np.float32)
    return arr


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
        raise ValueError(
            f"Shape mismatch for {param}: expected {target_shape}, got {arr.shape}"
        )
    param.value = arr


def _load_flax_model(
    config: GPTOSSConfig, params: dict, seq_len: int, seed: int
) -> Transformer:
    cos, sin = _rotary_tables_for_config(config, min_length=seq_len)
    sliding_mask = _causal_mask(seq_len, seq_len, sliding_window=config.sliding_window)
    causal_mask = _causal_mask(seq_len, seq_len, sliding_window=0)
    dtype = jnp.float32
    model = Transformer(
        config=config,
        cos_table=cos.astype(dtype),
        sin_table=sin.astype(dtype),
        sequence_length=seq_len,
        mask_sliding=sliding_mask.astype(dtype),
        mask_causal=causal_mask.astype(dtype),
        dtype=dtype,
        rng=jax.random.PRNGKey(seed),
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


def _select_torch_device(choice: str) -> torch.device:
    if choice == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--torch-device gpu requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cpu")


def _load_torch_model(checkpoint: Path, device: torch.device) -> TorchTransformer:
    model = TorchTransformer.from_checkpoint(str(checkpoint), device=device)
    model = model.to(torch.float32)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _truncate_torch_layers(model: TorchTransformer, max_layers: int) -> int:
    total_layers = len(model.block)
    if max_layers <= 0 or max_layers >= total_layers:
        return total_layers
    kept = list(model.block[:max_layers])
    model.block = torch.nn.ModuleList(kept)
    return len(model.block)


def _torch_block_debug(block, x: torch.Tensor) -> Dict[str, np.ndarray]:
    entry: Dict[str, np.ndarray] = {}
    entry["input"] = _to_numpy(x)

    attn = block.attn
    normed = attn.norm(x)
    entry["attn_normed"] = _to_numpy(normed)
    normed_cast = normed
    if normed.dtype != attn.qkv.weight.dtype:
        normed_cast = normed.to(attn.qkv.weight.dtype)
    qkv = attn.qkv(normed_cast)
    q_end = attn.num_attention_heads * attn.head_dim
    k_end = q_end + attn.num_key_value_heads * attn.head_dim
    q = qkv[:, :q_end]
    k = qkv[:, q_end:k_end]
    v = qkv[:, k_end:]

    q = q.view(
        -1,
        attn.num_key_value_heads,
        attn.num_attention_heads // attn.num_key_value_heads,
        attn.head_dim,
    )
    k = k.view(-1, attn.num_key_value_heads, attn.head_dim)
    v = v.view(-1, attn.num_key_value_heads, attn.head_dim)
    entry["attn_q"] = _to_numpy(q)
    entry["attn_k"] = _to_numpy(k)
    entry["attn_v"] = _to_numpy(v)
    q_rot, k_rot = attn.rope(q, k)
    attn_core = torch_sdpa(
        q_rot,
        k_rot,
        v,
        attn.sinks,
        attn.sm_scale,
        attn.sliding_window,
    )
    projected = attn.out(attn_core)
    entry["attn_projected"] = _to_numpy(projected)
    attn_out = x + projected
    entry["post_attention"] = _to_numpy(attn_out)

    mlp = block.mlp
    mlp_normed = mlp.norm(attn_out)
    entry["mlp_normed"] = _to_numpy(mlp_normed)
    gate_input = mlp_normed
    if gate_input.dtype != mlp.gate.weight.dtype:
        gate_input = gate_input.to(mlp.gate.weight.dtype)
    gate_logits = mlp.gate(gate_input)
    entry["mlp_gate_logits"] = _to_numpy(gate_logits)

    topk = torch.topk(
        gate_logits,
        k=mlp.experts_per_token,
        dim=-1,
        sorted=True,
    )
    expert_weights = F.softmax(topk.values, dim=1)
    expert_indices = topk.indices
    entry["mlp_expert_indices"] = _to_numpy(expert_indices)
    entry["mlp_expert_weights"] = _to_numpy(expert_weights)

    dense_gate_weights = torch.zeros(
        gate_logits.shape[0],
        mlp.num_experts,
        dtype=torch.float32,
        device=gate_logits.device,
    )
    dense_gate_weights.scatter_add_(
        1,
        expert_indices,
        expert_weights.to(torch.float32),
    )
    entry["mlp_dense_gate_weights"] = _to_numpy(dense_gate_weights)

    mlp1_weight = mlp.mlp1_weight[expert_indices, ...].to(torch.float32)
    mlp1_bias = mlp.mlp1_bias[expert_indices, ...].to(torch.float32)
    prelinear = torch.einsum("beck,bk->bec", mlp1_weight, mlp_normed.to(torch.float32))
    prelinear = prelinear + mlp1_bias
    entry["mlp_prelinear_outputs"] = _to_numpy(prelinear)

    act = torch_swiglu(prelinear, limit=mlp.swiglu_limit)
    entry["mlp_activated_outputs"] = _to_numpy(act)

    mlp2_weight = mlp.mlp2_weight[expert_indices, ...].to(torch.float32)
    mlp2_bias = mlp.mlp2_bias[expert_indices, ...].to(torch.float32)
    expert_outputs = torch.einsum("beck,bek->bec", mlp2_weight, act)
    if mlp.world_size > 1:
        import torch.distributed as dist

        dist.all_reduce(expert_outputs, op=dist.ReduceOp.SUM)
    expert_outputs = expert_outputs + mlp2_bias
    entry["mlp_expert_outputs"] = _to_numpy(expert_outputs)

    fused = torch.einsum("bec,be->bc", expert_outputs, expert_weights.to(torch.float32))
    entry["mlp_fused"] = _to_numpy(fused)
    out = attn_out + fused
    entry["output"] = _to_numpy(out)
    return entry


def _collect_torch_debug(
    model: TorchTransformer, tokens: torch.Tensor
) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    stages: List[Dict[str, np.ndarray]] = []
    with torch.no_grad():
        hidden = model.embedding(tokens)
        for block in model.block:
            entry = _torch_block_debug(block, hidden)
            stages.append(entry)
            hidden = block(hidden)
        normed = model.norm(hidden)
        normed_cast = normed
        if normed.dtype != model.unembedding.weight.dtype:
            normed_cast = normed.to(model.unembedding.weight.dtype)
        logits = model.unembedding(normed_cast)
    return logits.detach().to(torch.float32).cpu().numpy(), stages


def _collect_flax_debug(
    model: Transformer, tokens: np.ndarray
) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    block_debug: List[dict] = []
    logits = model(
        jnp.asarray(tokens),
        capture_block_debug=block_debug,
    )
    stages: List[Dict[str, np.ndarray]] = []
    for entry in block_debug:
        stage = {key: _to_numpy(value) for key, value in entry.items()}
        stages.append(stage)
    return np.asarray(jax.device_get(logits)), stages


def _diff_stats(torch_arr: np.ndarray, flax_arr: np.ndarray) -> Dict[str, float]:
    delta = np.abs(torch_arr - flax_arr)
    return {
        "max": float(np.max(delta)),
        "mean": float(np.mean(delta)),
        "median": float(np.median(delta)),
    }


def _summarize_stage_diffs(
    torch_stages: List[Dict[str, np.ndarray]],
    flax_stages: List[Dict[str, np.ndarray]],
) -> Tuple[List[Tuple[str, float, float]], List[str]]:
    diffs: List[Tuple[str, float, float]] = []
    issues: List[str] = []
    if len(torch_stages) != len(flax_stages):
        issues.append(
            f"Layer count mismatch: torch={len(torch_stages)} vs flax={len(flax_stages)}"
        )
    for idx in range(min(len(torch_stages), len(flax_stages))):
        torch_entry = torch_stages[idx]
        flax_entry = flax_stages[idx]
        shared_keys = sorted(set(torch_entry.keys()) & set(flax_entry.keys()))
        for key in sorted(set(torch_entry.keys()) - set(flax_entry.keys())):
            issues.append(f"block{idx}.{key} missing in Flax capture")
        for key in sorted(set(flax_entry.keys()) - set(torch_entry.keys())):
            issues.append(f"block{idx}.{key} missing in Torch capture")
        for key in shared_keys:
            torch_val = torch_entry[key]
            flax_val = flax_entry[key]
            if torch_val.shape != flax_val.shape:
                issues.append(
                    f"block{idx}.{key} shape mismatch: torch{torch_val.shape} vs flax{flax_val.shape}"
                )
                continue
            delta = np.abs(torch_val - flax_val)
            diffs.append((f"block{idx}.{key}", float(delta.max()), float(delta.mean())))
    return diffs, issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GPT-OSS PyTorch vs Flax/NNX logits and per-block debug tensors."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional GPT-OSS config JSON (defaults to <params>.config.json)",
    )
    parser.add_argument("--torch-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--gpt-oss-path",
        type=Path,
        default=Path("tmp/gpt-oss-jax-vs-torch-numerical-comparison"),
        help="Path to a checkout that exposes the gpt_oss Python package.",
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for Flax init")
    parser.add_argument(
        "--torch-device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device for the PyTorch forward pass.",
    )
    parser.add_argument(
        "--max-stage-rows",
        type=int,
        default=20,
        help="Number of stage-diff rows to print (sorted by max |Δ|).",
    )
    parser.add_argument(
        "--torch-max-layers",
        type=int,
        default=0,
        help="Limit torch blocks to N layers before comparison (0 = match Flax config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_repo_on_path(args.gpt_oss_path)

    params_path = args.params.expanduser().resolve()
    config_path = (
        args.config.expanduser().resolve()
        if args.config
        else params_path.parent / f"{params_path.name}.config.json"
    )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config JSON at {config_path}. Rerun export_flax_gpt_oss_params.py."
        )

    params = _load_params(params_path)
    config_dict = json.loads(config_path.read_text())
    allowed = GPTOSSConfig.__annotations__.keys()
    config = GPTOSSConfig(**{k: v for k, v in config_dict.items() if k in allowed})

    seq_len = min(args.sequence_length, config.initial_context_length)

    token_ids, token_strs = _tokenize(args.prompt, config.vocab_size)
    if not token_ids:
        raise SystemExit("Prompt tokenization produced zero tokens")
    if len(token_ids) > seq_len:
        print(
            f"[info] prompt has {len(token_ids)} tokens; truncating to sequence length {seq_len}",
            flush=True,
        )
        token_ids = token_ids[:seq_len]
        token_strs = token_strs[:seq_len]

    padded_tokens = np.zeros((seq_len,), dtype=np.int32)
    padded_tokens[: len(token_ids)] = token_ids

    torch_device = _select_torch_device(args.torch_device)
    torch_model = _load_torch_model(
        args.torch_checkpoint.expanduser().resolve(), torch_device
    )
    original_layers = len(torch_model.block)
    target_layers = args.torch_max_layers or config.num_hidden_layers
    effective_layers = _truncate_torch_layers(torch_model, target_layers)
    if effective_layers < original_layers:
        print(
            f"[info] truncated torch model to first {effective_layers} layers (target={target_layers})"
        )
    elif effective_layers != config.num_hidden_layers:
        print(
            f"[info] torch model exposes {effective_layers} layers; Flax config has {config.num_hidden_layers}"
        )
    torch_tokens = torch.tensor(padded_tokens, dtype=torch.long, device=torch_device)
    torch_logits, torch_stages = _collect_torch_debug(torch_model, torch_tokens)

    flax_model = _load_flax_model(config, params, seq_len, args.seed)
    flax_logits, flax_stages = _collect_flax_debug(flax_model, padded_tokens)

    valid = len(token_ids)
    torch_slice = torch_logits[:valid]
    flax_slice = flax_logits[:valid]
    logit_stats = _diff_stats(torch_slice, flax_slice)

    stage_diffs, stage_issues = _summarize_stage_diffs(torch_stages, flax_stages)
    stage_diffs.sort(key=lambda item: item[1], reverse=True)

    print(f"Prompt: {args.prompt}")
    print(f"Tokens ({valid} / {seq_len}): {token_ids}")
    print("Token strings:", token_strs)
    print(
        f"Logits diff  max |Δ|={logit_stats['max']:.6f}  "
        f"mean |Δ|={logit_stats['mean']:.6f}  median |Δ|={logit_stats['median']:.6f}"
    )

    max_rows = max(1, args.max_stage_rows)
    print("\nStage diffs (sorted by max |Δ|):")
    for name, max_diff, mean_diff in stage_diffs[:max_rows]:
        print(f"  {name:>28s}  max={max_diff:.6f}  mean={mean_diff:.6f}")
    if len(stage_diffs) > max_rows:
        print(f"  ... {len(stage_diffs) - max_rows} more entries not shown")

    if stage_issues:
        print("\n[issues]")
        for issue in stage_issues:
            print("  -", issue)

    if logit_stats["max"] > 1e-3:
        print("\n[warn] Max logits diff exceeded 1e-3; inspect per-stage diffs above.")


if __name__ == "__main__":
    main()
