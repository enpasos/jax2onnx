#!/usr/bin/env python3
# scripts/gpt_oss_routing_parity.py

"""
Parity harness for comparing GPT-OSS JAX (Flax/NNX) vs PyTorch routing.

This script is adapted from https://github.com/atveit/gpt-oss-jax-vs-torch-numerical-comparison
and lets us reproduce the expert-routing analysis locally. Point it at a checkout of
openai/gpt-oss (or atveit's parity repo) plus the respective checkpoints and it will:

1. Tokenize the provided prompt with `tiktoken` (falls back to dummy tokens if missing).
2. Run a single forward pass through the JAX and PyTorch references while capturing
   per-layer, per-token expert IDs and gate weights.
3. Emit a textual summary plus optional markdown logs for deeper inspection.

Usage example:

    python scripts/gpt_oss_routing_parity.py \\
        --gpt-oss-path /path/to/gpt-oss \\
        --jax-checkpoint /path/to/gpt-oss-20b-orbax \\
        --torch-checkpoint /path/to/gpt-oss-20b \\
        --prompt "What is the capital of France?"
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers (import wiring + logging utilities)
# ---------------------------------------------------------------------------


def _ensure_repo_on_path(repo_path: Path) -> None:
    repo_path = repo_path.resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"gpt-oss path '{repo_path}' does not exist")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


def _setup_jax():
    import jax

    backend = jax.default_backend()
    print(f"Using JAX with {backend.upper()} backend")
    return jax


def _setup_torch(requested: str):
    import torch

    if requested == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--torch-device gpu requested but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using PyTorch with device={device}")
    return torch, device


def _load_tokenizer(prompt: str) -> Tuple[List[int], List[str]]:
    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        token_ids = tokenizer.encode(prompt)
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]
    except Exception:
        print(
            "Warning: tiktoken unavailable; falling back to dummy tokens. "
            "Install `tiktoken` for real prompt coverage."
        )
        token_ids = [1, 2, 3, 4, 5]
        token_strs = [f"token_{i}" for i in range(len(token_ids))]
    return token_ids, token_strs


# ---------------------------------------------------------------------------
# Model loading / execution
# ---------------------------------------------------------------------------


def _load_jax_model(
    checkpoint: Path,
):
    import jax
    from gpt_oss.jax.loader_orbax import OrbaxWeightLoader, load_config_from_orbax
    from gpt_oss.jax.loader_safetensors import WeightLoader
    from gpt_oss.jax.model import ModelConfig, Transformer
    from gpt_oss.jax.token_generator import (
        detect_checkpoint_format,
        load_config_from_checkpoint,
    )

    ckpt_path = Path(checkpoint)
    fmt = detect_checkpoint_format(ckpt_path)

    if fmt == "orbax":
        config_dict = load_config_from_orbax(str(ckpt_path))
        config = ModelConfig(**config_dict)
        loader = OrbaxWeightLoader(str(ckpt_path))
        params = loader.load_params(show_progress=True)
    else:
        config = load_config_from_checkpoint(ckpt_path)
        loader = WeightLoader(str(ckpt_path))
        params = loader.load_params(config, show_progress=True)

    target = jax.devices()[0]
    params = jax.tree.map(lambda x: jax.device_put(x, target), params)
    model = Transformer(config=config)
    return model, params, config


def _load_torch_model(checkpoint: Path, device):
    from gpt_oss.torch.model import Transformer

    return Transformer.from_checkpoint(str(checkpoint), device=device)


def _run_jax(model, params, token_ids: List[int], num_layers: int):
    import jax.numpy as jnp

    tokens = jnp.array(token_ids, dtype=jnp.int32)
    capture_routing = [[] for _ in range(num_layers)]
    logits = model.apply({"params": params}, tokens, capture_routing=capture_routing)
    return logits, capture_routing


def _run_torch(model, token_ids: List[int], num_layers: int, device):
    import torch

    tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
    capture_routing = [[] for _ in range(num_layers)]
    with torch.no_grad():
        logits = model(tokens, capture_routing=capture_routing)
    return logits, capture_routing


# ---------------------------------------------------------------------------
# Comparison + reporting
# ---------------------------------------------------------------------------


def _compare_routing(
    jax_routing: List[List[Dict[str, np.ndarray]]],
    torch_routing: List[List[Dict[str, np.ndarray]]],
    prompt: str,
    tokens: List[str],
) -> Dict[str, Any]:
    num_layers = len(jax_routing)
    num_tokens = len(jax_routing[0])

    layer_stats = []
    all_matches: List[bool] = []
    all_gate_diffs: List[float] = []

    for layer_idx in range(num_layers):
        jax_layer = jax_routing[layer_idx]
        torch_layer = torch_routing[layer_idx]
        layer_matches = []
        layer_diffs = []

        for token_idx in range(num_tokens):
            jax_token = jax_layer[token_idx]
            torch_token = torch_layer[token_idx]
            expert_match = np.array_equal(
                jax_token["expert_ids"], torch_token["expert_ids"]
            )
            gate_diff = np.abs(jax_token["gate_weights"] - torch_token["gate_weights"])

            layer_matches.append(expert_match)
            layer_diffs.extend(gate_diff.flatten())

        match_rate = float(np.mean(layer_matches))
        mean_diff = float(np.mean(layer_diffs))
        max_diff = float(np.max(layer_diffs))

        layer_stats.append(
            {
                "layer_idx": layer_idx,
                "match_rate": match_rate,
                "mean_gate_diff": mean_diff,
                "max_gate_diff": max_diff,
            }
        )
        all_matches.extend(layer_matches)
        all_gate_diffs.extend(layer_diffs)

    overall_match_rate = float(np.mean(all_matches))
    overall_mean_diff = float(np.mean(all_gate_diffs))
    overall_max_diff = float(np.max(all_gate_diffs))

    passed = overall_match_rate == 1.0 and overall_max_diff < 0.01
    return {
        "prompt": prompt,
        "tokens": tokens,
        "num_layers": num_layers,
        "num_tokens": num_tokens,
        "layer_stats": layer_stats,
        "overall_match_rate": overall_match_rate,
        "overall_mean_diff": overall_mean_diff,
        "overall_max_diff": overall_max_diff,
        "passed": passed,
        "jax_routing": jax_routing,
        "torch_routing": torch_routing,
    }


def _print_results(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("Expert Routing Parity Validation")
    print("=" * 60)
    print(f"Prompt: {results['prompt']}")
    print(f"Tokens: {results['num_tokens']} | Layers: {results['num_layers']}\n")

    for stats in results["layer_stats"]:
        symbol = "✓" if stats["match_rate"] == 1.0 else "✗"
        print(
            f"{symbol} Layer {stats['layer_idx']:2d} | "
            f"Expert IDs: {stats['match_rate']*100:5.1f}% | "
            f"Gate diff (mean={stats['mean_gate_diff']:.6f}, max={stats['max_gate_diff']:.6f})"
        )

    total_decisions = results["num_layers"] * results["num_tokens"]
    total_matches = int(results["overall_match_rate"] * total_decisions)
    print("\nSummary")
    print("-" * 60)
    print(f"Routing decisions: {total_decisions} ({total_matches} matches)")
    print(f"Overall match rate: {results['overall_match_rate']*100:.1f}%")
    print(f"Gate weight diff mean: {results['overall_mean_diff']:.6f}")
    print(f"Gate weight diff max:  {results['overall_max_diff']:.6f}\n")
    if results["passed"]:
        print("✓ PASS: JAX and PyTorch routing are numerically equivalent")
    else:
        print("✗ FAIL: Routing divergence detected")
    print("=" * 60)


def _write_markdown(results: Dict[str, Any], output_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / f"{timestamp}_summary.md"
    with summary.open("w") as fp:
        fp.write("# GPT-OSS Routing Parity Summary\n\n")
        fp.write(f"Prompt: {results['prompt']}\n\n")
        fp.write(f"Result: {'PASS' if results['passed'] else 'FAIL'}\n\n")
        fp.write("| Layer | Expert Match | Mean Gate Diff | Max Gate Diff | Status |\n")
        fp.write("|-------|--------------|----------------|---------------|--------|\n")
        for stats in results["layer_stats"]:
            status = "✓" if stats["match_rate"] == 1.0 else "✗"
            fp.write(
                f"| {stats['layer_idx']:2d} | {stats['match_rate']*100:5.1f}% | "
                f"{stats['mean_gate_diff']:.6f} | {stats['max_gate_diff']:.6f} | {status} |\n"
            )
    print(f"Wrote summary to {summary}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GPT-OSS JAX vs PyTorch routing."
    )
    parser.add_argument(
        "--gpt-oss-path",
        required=True,
        type=Path,
        help="Path to an openai/gpt-oss checkout (or atveit's parity repo).",
    )
    parser.add_argument(
        "--jax-checkpoint",
        required=True,
        type=Path,
        help="Directory containing the Orbax checkpoint (e.g., gpt-oss-20b-orbax).",
    )
    parser.add_argument(
        "--torch-checkpoint",
        required=True,
        type=Path,
        help="Directory containing the PyTorch safetensors checkpoint (e.g., gpt-oss-20b).",
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="Prompt to tokenize and compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/gpt_oss_routing"),
        help="Directory for markdown summaries.",
    )
    parser.add_argument(
        "--jax-platform",
        choices=["cpu", "gpu"],
        help="Force the JAX backend for the run.",
    )
    parser.add_argument(
        "--torch-device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run the PyTorch reference (defaults to cpu).",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        help="Limit comparison to the first N transformer layers.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Limit comparison to the first N prompt tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_repo_on_path(args.gpt_oss_path)

    if args.jax_platform:
        os.environ["JAX_PLATFORM_NAME"] = args.jax_platform

    token_ids, token_strs = _load_tokenizer(args.prompt)
    if args.max_tokens is not None:
        token_ids = token_ids[: args.max_tokens]
        token_strs = token_strs[: args.max_tokens]
    print(f"Prompt: {args.prompt}")
    print(f"Token IDs: {token_ids}")
    print(f"Num tokens: {len(token_ids)}\n")

    _setup_jax()
    torch, device = _setup_torch(args.torch_device)

    print("Loading JAX model/params ...")
    jax_model, jax_params, config = _load_jax_model(args.jax_checkpoint)
    num_layers = config.num_hidden_layers
    print(f"✓ Loaded JAX model with {num_layers} layers\n")

    print("Loading PyTorch model ...")
    torch_model = _load_torch_model(args.torch_checkpoint, device)
    print("✓ Loaded PyTorch model\n")

    print("Running JAX inference ...")
    _, jax_routing = _run_jax(jax_model, jax_params, token_ids, num_layers)
    print("Running PyTorch inference ...")
    _, torch_routing = _run_torch(torch_model, token_ids, num_layers, device)

    if args.max_layers is not None:
        jax_routing = jax_routing[: args.max_layers]
        torch_routing = torch_routing[: args.max_layers]

    results = _compare_routing(jax_routing, torch_routing, args.prompt, token_strs)
    _print_results(results)
    _write_markdown(results, args.output_dir)

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
