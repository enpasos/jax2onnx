# scripts/probe_eqx_gpt_oss_parity.py

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from jax2onnx.plugins.examples.eqx.gpt_oss import (
    Transformer as EqxTransformer,
    _apply_linear_float32_accum,
    _apply_linear_nd,
    _apply_pointwise,
    _config_from_torch_transformer,
    _populate_eqx_from_torch,
    _sdpa_torch_style,
    _softmax_torch_approx,
    _swiglu,
)
from gpt_oss.torch.model import (
    ModelConfig,
    Transformer as TorchTransformer,
    sdpa as torch_sdpa,
    swiglu as torch_swiglu,
)


def _make_tiny_config() -> ModelConfig:
    return ModelConfig(
        num_hidden_layers=1,
        num_experts=4,
        experts_per_token=2,
        vocab_size=64,
        hidden_size=128,
        intermediate_size=128,
        swiglu_limit=7.0,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=0,
        initial_context_length=32,
        rope_theta=10_000.0,
        rope_scaling_factor=1.0,
        rope_ntk_alpha=1.0,
        rope_ntk_beta=8.0,
    )


def _randomise_torch_model(model: TorchTransformer, *, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    for name, param in model.named_parameters():
        if param.data.dtype.is_floating_point:
            noise = torch.randn(
                param.data.shape, generator=generator, dtype=torch.float32
            ) * 0.02
            param.data.copy_(noise.to(param.data.dtype))
        else:
            param.data.zero_()
    # Explicitly scale sinks (buffers) to match Flax init (std=0.02)
    with torch.no_grad():
        for block in model.block:
            if hasattr(block.attn, "sinks"):
                # For debugging parity, zero out sinks to isolate SDPA behaviour
                block.attn.sinks.zero_()


def _print_stats(name: str, torch_t: np.ndarray, eqx_t: np.ndarray) -> None:
    t_mean = float(np.mean(torch_t))
    t_max = float(np.max(np.abs(torch_t)))
    e_mean = float(np.mean(eqx_t))
    e_max = float(np.max(np.abs(eqx_t)))
    diff = float(np.max(np.abs(torch_t - eqx_t)))
    print(
        f"{name:<20} | Torch: {t_mean:+.6f} (max {t_max:.6f}) | "
        f"Eqx: {e_mean:+.6f} (max {e_max:.6f}) | Diff: {diff:.6f}"
    )


def _torch_mlp_stages(
    block, x: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    stages: Dict[str, torch.Tensor] = {}
    stages["input"] = x
    normed = block.norm(x)
    stages["norm"] = normed
    gate_logits = block.gate(normed)
    stages["gate_logits"] = gate_logits
    experts = torch.topk(gate_logits, k=block.experts_per_token, dim=-1, sorted=True)
    expert_scores = experts.values
    expert_indices = experts.indices
    stages["expert_scores"] = expert_scores
    stages["expert_indices"] = expert_indices.float()
    expert_weights = F.softmax(expert_scores, dim=1)
    stages["expert_weights"] = expert_weights

    mlp1_weight = block.mlp1_weight[expert_indices, ...]
    mlp1_bias = block.mlp1_bias[expert_indices, ...]
    proj1 = torch.einsum("beck,bk->bec", mlp1_weight, normed) + mlp1_bias
    stages["proj1"] = proj1
    act = torch_swiglu(proj1, limit=block.swiglu_limit)
    stages["act"] = act

    mlp2_weight = block.mlp2_weight[expert_indices, ...]
    mlp2_bias = block.mlp2_bias[expert_indices, ...]
    proj2 = torch.einsum("beck,bek->bec", mlp2_weight, act)
    proj2 = proj2 + mlp2_bias
    stages["proj2"] = proj2

    combined = torch.einsum("bec,be->bc", proj2, expert_weights)
    stages["combined"] = combined
    output = x + combined
    stages["output"] = output
    return stages, output


def _run_torch(model: TorchTransformer, tokens: torch.Tensor) -> Dict[str, np.ndarray]:
    stages: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        x = model.embedding(tokens)
        stages["embed"] = x.detach().to(torch.float32).cpu().numpy()
        for idx, block in enumerate(model.block):
            attn_debug, attn = _torch_attn_stages(block.attn, x)
            for name, value in attn_debug.items():
                stages[f"block{idx}.attn.{name}"] = (
                    value.detach().to(torch.float32).cpu().numpy()
                )
            stages[f"block{idx}.attn"] = stages[f"block{idx}.attn.output"]
            mlp_debug, x = _torch_mlp_stages(block.mlp, attn)
            for name, value in mlp_debug.items():
                stages[f"block{idx}.mlp.{name}"] = (
                    value.detach().to(torch.float32).cpu().numpy()
                )
            stages[f"block{idx}.mlp"] = stages[f"block{idx}.mlp.output"]
        norm = model.norm(x)
        stages["norm"] = norm.detach().to(torch.float32).cpu().numpy()
        logits = model.unembedding(norm)
        stages["logits"] = logits.detach().to(torch.float32).cpu().numpy()
    return stages


def _eqx_mlp_stages(block, x: jnp.ndarray) -> Tuple[Dict[str, np.ndarray], jnp.ndarray]:
    stages: Dict[str, np.ndarray] = {}
    stages["input"] = np.array(x[0], dtype=np.float32)
    if block.param_dtype == jnp.bfloat16:
        x_compute = x.astype(jnp.bfloat16)
        normed = _apply_pointwise(block.norm, x_compute).astype(jnp.bfloat16)
        stages["norm"] = np.array(normed[0], dtype=np.float32)
        gate_logits = _apply_linear_float32_accum(block.gate, normed).astype(
            jnp.bfloat16
        )
        stages["gate_logits"] = np.array(gate_logits[0], dtype=np.float32)
        expert_scores, expert_indices = jax.lax.top_k(
            gate_logits, block.experts_per_token
        )
        stages["expert_scores"] = np.array(expert_scores[0], dtype=np.float32)
        stages["expert_indices"] = np.array(expert_indices[0], dtype=np.float32)
        expert_weights = _softmax_torch_approx(
            expert_scores.astype(jnp.float32), axis=-1
        ).astype(jnp.bfloat16)
        stages["expert_weights"] = np.array(expert_weights[0], dtype=np.float32)

        mlp1_weight = jnp.take(block.mlp1_weight, expert_indices, axis=0).astype(
            jnp.bfloat16
        )
        mlp1_bias = jnp.take(block.mlp1_bias, expert_indices, axis=0).astype(
            jnp.bfloat16
        )
        proj1 = jnp.einsum(
            "bskoh,bsh->bsko",
            mlp1_weight.astype(jnp.float32),
            normed.astype(jnp.float32),
            optimize="optimal",
        ).astype(jnp.bfloat16)
        proj1 = proj1 + mlp1_bias
        stages["proj1"] = np.array(proj1[0], dtype=np.float32)
        act = _swiglu(proj1, limit=block.swiglu_limit).astype(jnp.bfloat16)
        stages["act"] = np.array(act[0], dtype=np.float32)

        mlp2_weight = jnp.take(block.mlp2_weight, expert_indices, axis=0).astype(
            jnp.bfloat16
        )
        mlp2_bias = jnp.take(block.mlp2_bias, expert_indices, axis=0).astype(
            jnp.bfloat16
        )
        proj2 = jnp.einsum(
            "bskhi,bski->bskh",
            mlp2_weight.astype(jnp.float32),
            act.astype(jnp.float32),
            optimize="optimal",
        ).astype(jnp.bfloat16)
        proj2 = proj2 + mlp2_bias
        stages["proj2"] = np.array(proj2[0], dtype=np.float32)

        combined = jnp.einsum(
            "bskh,bsk->bsh",
            proj2.astype(jnp.float32),
            expert_weights.astype(jnp.float32),
            optimize="optimal",
        ).astype(jnp.bfloat16)
        stages["combined"] = np.array(combined[0], dtype=np.float32)
        residual = x_compute + combined.astype(x_compute.dtype)
        output = residual.astype(x.dtype)
        stages["output"] = np.array(output[0], dtype=np.float32)
        return stages, output

    x_compute = x.astype(jnp.float32)
    normed = _apply_pointwise(block.norm, x_compute).astype(jnp.float32)
    stages["norm"] = np.array(normed[0], dtype=np.float32)
    gate_logits = _apply_linear_nd(block.gate, normed).astype(jnp.float32)
    stages["gate_logits"] = np.array(gate_logits[0], dtype=np.float32)
    expert_scores, expert_indices = jax.lax.top_k(gate_logits, block.experts_per_token)
    stages["expert_scores"] = np.array(expert_scores[0], dtype=np.float32)
    stages["expert_indices"] = np.array(expert_indices[0], dtype=np.float32)
    expert_weights = jnn.softmax(expert_scores, axis=-1)
    stages["expert_weights"] = np.array(expert_weights[0], dtype=np.float32)

    mlp1_weight = jnp.take(block.mlp1_weight, expert_indices, axis=0).astype(
        jnp.float32
    )
    mlp1_bias = jnp.take(block.mlp1_bias, expert_indices, axis=0).astype(jnp.float32)
    proj1 = jnp.einsum(
        "bskoh,bsh->bsko", mlp1_weight, normed.astype(jnp.float32), optimize="optimal"
    )
    proj1 = proj1 + mlp1_bias
    stages["proj1"] = np.array(proj1[0], dtype=np.float32)
    act = _swiglu(proj1, limit=block.swiglu_limit)
    stages["act"] = np.array(act[0], dtype=np.float32)

    mlp2_weight = jnp.take(block.mlp2_weight, expert_indices, axis=0).astype(
        jnp.float32
    )
    mlp2_bias = jnp.take(block.mlp2_bias, expert_indices, axis=0).astype(jnp.float32)
    proj2 = jnp.einsum("bskhi,bski->bskh", mlp2_weight, act, optimize="optimal")
    proj2 = proj2 + mlp2_bias
    stages["proj2"] = np.array(proj2[0], dtype=np.float32)

    combined = jnp.einsum("bskh,bsk->bsh", proj2, expert_weights, optimize="optimal")
    stages["combined"] = np.array(combined[0], dtype=np.float32)
    output = x + combined.astype(x.dtype)
    stages["output"] = np.array(output[0], dtype=np.float32)
    return stages, output


def _torch_attn_stages(
    block, x: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    stages: Dict[str, torch.Tensor] = {}
    stages["input"] = x
    normed = block.norm(x)
    stages["norm"] = normed
    qkv = block.qkv(normed)
    stages["qkv"] = qkv
    head_dim = block.head_dim
    num_q = block.num_attention_heads * head_dim
    num_k = block.num_key_value_heads * head_dim
    q = qkv[:, :num_q]
    k = qkv[:, num_q : num_q + num_k]
    v = qkv[:, num_q + num_k :]
    stages["q"] = q
    stages["k"] = k
    stages["v"] = v
    q = q.view(
        -1,
        block.num_key_value_heads,
        block.num_attention_heads // block.num_key_value_heads,
        head_dim,
    )
    k = k.view(-1, block.num_key_value_heads, head_dim)
    v = v.view(-1, block.num_key_value_heads, head_dim)
    q_rot, k_rot = block.rope(q, k)
    stages["q_rot"] = q_rot
    stages["k_rot"] = k_rot
    attn_inner = torch_sdpa(
        q_rot, k_rot, v, block.sinks, block.sm_scale, block.sliding_window
    )
    stages["attn_core"] = attn_inner
    attn = block.out(attn_inner)
    stages["projected"] = attn
    output = x + attn
    stages["output"] = output
    return stages, output


def _eqx_attn_stages(
    block, x: jnp.ndarray
) -> Tuple[Dict[str, np.ndarray], jnp.ndarray]:
    stages: Dict[str, np.ndarray] = {}
    stages["input"] = np.array(x[0], dtype=np.float32)
    if block.param_dtype == jnp.bfloat16:
        x_param = x.astype(jnp.bfloat16)
        normed = _apply_pointwise(block.norm, x_param).astype(jnp.bfloat16)
        qkv = _apply_linear_float32_accum(block.qkv, normed).astype(jnp.bfloat16)
    else:
        x_param = x.astype(jnp.float32)
        normed = _apply_pointwise(block.norm, x_param).astype(jnp.float32)
        qkv = _apply_linear_nd(block.qkv, normed).astype(jnp.float32)
    stages["norm"] = np.array(normed[0], dtype=np.float32)
    stages["qkv"] = np.array(qkv[0], dtype=np.float32)
    head_dim = block.head_dim
    num_q = block.num_attention_heads * head_dim
    num_k = block.num_key_value_heads * head_dim
    q = qkv[..., :num_q]
    k = qkv[..., num_q : num_q + num_k]
    v = qkv[..., num_q + num_k :]
    stages["q"] = np.array(q[0], dtype=np.float32)
    stages["k"] = np.array(k[0], dtype=np.float32)
    stages["v"] = np.array(v[0], dtype=np.float32)
    batch, seq_len = x.shape[:2]
    q = q.reshape(
        batch,
        seq_len,
        block.num_key_value_heads,
        block.num_attention_heads // block.num_key_value_heads,
        head_dim,
    )
    k = k.reshape(batch, seq_len, block.num_key_value_heads, head_dim)
    v = v.reshape(batch, seq_len, block.num_key_value_heads, head_dim)
    q_rot, k_rot = block.rope(q, k, seq_len=seq_len)
    stages["q_rot"] = np.array(q_rot[0], dtype=np.float32)
    stages["k_rot"] = np.array(k_rot[0], dtype=np.float32)
    sinks = block.sinks.reshape(
        block.num_key_value_heads, block.query_multiplicity
    ).astype(q_rot.dtype)
    attn_core = jax.vmap(
        lambda q_s, k_s, v_s: _sdpa_torch_style(
            q_s,
            k_s,
            v_s,
            sinks=sinks,
            sm_scale=block.sm_scale,
            sliding_window=block.sliding_window,
        )
    )(q_rot, k_rot, v)
    stages["attn_core"] = np.array(attn_core[0], dtype=np.float32)
    if block.param_dtype == jnp.bfloat16:
        projected = _apply_linear_float32_accum(
            block.out, attn_core.astype(jnp.bfloat16)
        ).astype(jnp.bfloat16)
        residual = x_param + projected.astype(x_param.dtype)
        output = residual.astype(x.dtype)
    else:
        projected = _apply_linear_nd(block.out, attn_core).astype(jnp.float32)
        residual = x_param + projected
        output = residual.astype(x.dtype)
    stages["projected"] = np.array(projected[0], dtype=np.float32)
    stages["output"] = np.array(output[0], dtype=np.float32)
    return stages, output


def _run_eqx(model: EqxTransformer, tokens: jnp.ndarray) -> Dict[str, np.ndarray]:
    stages: Dict[str, np.ndarray] = {}
    x = jnp.take(model.embedding.weight, tokens, axis=0)
    stages["embed"] = np.array(x[0], dtype=np.float32)
    for idx, block in enumerate(model.blocks):
        attn_stages, attn = _eqx_attn_stages(block.attn, x)
        for name, value in attn_stages.items():
            stages[f"block{idx}.attn.{name}"] = value
        stages[f"block{idx}.attn"] = attn_stages["output"]
        mlp_stages, x = _eqx_mlp_stages(block.mlp, attn)
        for name, value in mlp_stages.items():
            stages[f"block{idx}.mlp.{name}"] = value
        stages[f"block{idx}.mlp"] = mlp_stages["output"]
    norm = _apply_pointwise(model.norm, x)
    stages["norm"] = np.array(norm[0], dtype=np.float32)
    logits = _apply_linear_nd(model.unembedding, norm).astype(jnp.float32)
    stages["logits"] = np.array(logits[0], dtype=np.float32)
    return stages


def _compare(
    torch_model: TorchTransformer,
    eqx_model: EqxTransformer,
    tokens: torch.Tensor,
) -> Dict[str, float]:
    torch_stages = _run_torch(torch_model, tokens)
    tokens_eqx = jnp.asarray(tokens.cpu().numpy(), dtype=jnp.int32)[None, :]
    eqx_stages = _run_eqx(eqx_model, tokens_eqx)

    keys_to_trace = [
        "embed",
        "block0.attn.norm",
        "block0.attn.qkv",
        "block0.attn.q_rot",
        "block0.attn.attn_core",
        "block0.attn.projected",
    ]
    print("\n--- Layer 0 Trace ---")
    for k in keys_to_trace:
        if k in torch_stages and k in eqx_stages:
            _print_stats(k, torch_stages[k], eqx_stages[k])
    print("----------------------\n")

    diffs: Dict[str, float] = {}
    for key in torch_stages:
        torch_val = torch_stages[key]
        eqx_val = eqx_stages[key]
        diff = np.abs(torch_val - eqx_val)
        if np.isnan(diff).all():
            diffs[key] = 0.0
        else:
            diffs[key] = float(np.nanmax(diff))
    return diffs


def main(seed: int) -> None:
    config = _make_tiny_config()
    torch_model = TorchTransformer(config=config, device=torch.device("cpu"))
    _randomise_torch_model(torch_model, seed=seed)
    try:
        sinks_mean = torch_model.block[0].attn.sinks.abs().mean().item()
    except Exception:
        sinks_mean = "UNKNOWN"
    torch_eps = getattr(
        torch_model.norm, "eps", getattr(torch_model.norm, "epsilon", "UNKNOWN")
    )
    attn_eps = getattr(torch_model.block[0].attn.norm, "eps", "UNKNOWN")
    print("DEBUG DIAGNOSTICS:")
    print(f"  > Sinks Mean Abs: {sinks_mean}")
    print(f"  > Attn Norm Eps:  {attn_eps}")
    print(f"  > Torch Model Norm Eps: {torch_eps}")
    torch_model.eval()

    tokens = torch.randint(
        0,
        config.vocab_size,
        (config.initial_context_length // 2,),
        generator=torch.Generator().manual_seed(seed + 1),
        dtype=torch.int64,
    )

    config_eqx = _config_from_torch_transformer(torch_model)

    print("=== Parity diffs (param_dtype=bfloat16) ===")
    eqx_bf16 = EqxTransformer(
        config_eqx, key=jax.random.PRNGKey(seed), param_dtype=jnp.bfloat16
    )
    eqx_bf16 = _populate_eqx_from_torch(torch_model, eqx_bf16, param_dtype=jnp.bfloat16)
    diffs_bf16 = _compare(torch_model, eqx_bf16, tokens)
    for key, value in diffs_bf16.items():
        print(f"{key:>20s}: {value:.6f}")

    print("\n=== Parity diffs (param_dtype=float32) ===")
    eqx_f32 = EqxTransformer(
        config_eqx, key=jax.random.PRNGKey(seed + 1), param_dtype=jnp.float32
    )
    eqx_f32 = _populate_eqx_from_torch(torch_model, eqx_f32, param_dtype=jnp.float32)
    diffs_f32 = _compare(torch_model, eqx_f32, tokens)
    for key, value in diffs_f32.items():
        print(f"{key:>20s}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Probe torch vs. Equinox GPT-OSS parity."
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility."
    )
    args = parser.parse_args()
    main(args.seed)
