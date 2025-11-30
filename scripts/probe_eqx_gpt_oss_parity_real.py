#!/usr/bin/env python3
# scripts/probe_eqx_gpt_oss_parity_real.py

"""Parity probe using the released GPT-OSS 20B checkpoint and a harmony prompt."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch

from gpt_oss.torch.model import Transformer as TorchTransformer

from jax2onnx.plugins.examples.eqx.gpt_oss import load_pretrained_gpt_oss

from openai_harmony import (
    Author,
    Conversation,
    HarmonyEncodingName,
    Message,
    Role,
    TextContent,
    load_harmony_encoding,
)


def _render_prompt_tokens(prompt: str) -> Sequence[int]:
    """Tokenise `prompt` using the GPT-OSS harmony encoding."""

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    conversation = Conversation(
        messages=[
            Message(
                author=Author(role=Role.USER),
                content=[TextContent(text=prompt)],
            )
        ]
    )
    return encoding.render_conversation_for_completion(
        conversation, next_turn_role=Role.ASSISTANT
    )


def _load_torch_model(checkpoint: Path, device: str) -> TorchTransformer:
    """Load the official torch reference model from a checkpoint folder."""

    return TorchTransformer.from_checkpoint(str(checkpoint), device=device)


def _load_eqx_model(checkpoint: Path, *, param_dtype: jnp.dtype) -> torch.nn.Module:
    """Map the checkpoint into the Equinox Transformer."""

    return load_pretrained_gpt_oss(checkpoint, param_dtype=param_dtype, device="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parity probe for GPT-OSS 20B using real harmony prompts."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the `gpt-oss-20b` checkpoint directory (with config.json/model.safetensors).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum mechanics clearly and concisely.",
        help="User prompt (will be wrapped in the GPT-OSS harmony format).",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        default="cpu",
        help="Device to load the torch checkpoint on (default: cpu).",
    )
    parser.add_argument(
        "--param-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Equinox parameter dtype to mirror the checkpoint with.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    tokens = _render_prompt_tokens(args.prompt)
    print(f"Prompt tokens ({len(tokens)}): {tokens}")

    torch_model = _load_torch_model(checkpoint, device=args.torch_device)
    torch_inputs = torch.tensor(tokens, dtype=torch.int64, device=args.torch_device)
    with torch.no_grad():
        torch_logits = torch_model(torch_inputs)
        torch_last = torch_logits[-1].to(torch.float32).cpu().numpy()
    print("Torch logits ready:", torch_last.shape)

    param_dtype = jnp.bfloat16 if args.param_dtype == "bfloat16" else jnp.float32
    eqx_model = _load_eqx_model(checkpoint, param_dtype=param_dtype)
    eqx_tokens = jnp.asarray(tokens, dtype=jnp.int32)
    eqx_logits = jax.block_until_ready(eqx_model(eqx_tokens))
    eqx_last = np.asarray(eqx_logits[-1], dtype=np.float32)
    print("Equinox logits ready:", eqx_last.shape)

    abs_diff = np.abs(torch_last - eqx_last)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())

    print(f"Max |Δ| on final token logits: {max_diff}")
    print(f"Mean |Δ| on final token logits: {mean_diff}")

    topk = np.argsort(torch_last)[-5:][::-1]
    print("Top-5 torch token logits:")
    for idx in topk:
        print(f"  id={idx:6d} torch={torch_last[idx]:.6f} eqx={eqx_last[idx]:.6f}")


if __name__ == "__main__":
    main()
