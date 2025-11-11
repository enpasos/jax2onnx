"""Optional smoke test for the GPT-OSS Flax routing parity harness."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


_DEFAULT_PROMPT = "What is the capital of France?"
_MAX_LAYERS = "4"
_MAX_TOKENS = "2"


def _resolve_checkpoint(env_var: str) -> Path:
    raw = os.environ.get(env_var)
    if not raw:
        return Path()
    candidate = Path(raw).expanduser()
    if candidate.is_dir():
        return candidate
    orbital = candidate / "orbax"
    if orbital.is_dir():
        return orbital
    if candidate.name in {"original", "orbax"} and candidate.parent.exists():
        return candidate
    # try typical ~/.cache/gpt_oss layout
    user_cache = Path.home() / ".cache" / "gpt_oss" / "gpt-oss-20b"
    if candidate.name == "config.json" and candidate.parent.exists():
        return candidate.parent
    if user_cache.exists():
        sub = user_cache / candidate.name
        if sub.exists():
            return sub
    return candidate


@pytest.mark.slow
def test_flax_routing_parity_smoke():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "gpt_oss_routing_parity.py"

    gpt_oss_checkout = repo_root / "tmp" / "gpt-oss-jax-vs-torch-numerical-comparison"
    jax_ckpt = _resolve_checkpoint("JAX2ONNX_PARITY_JAX_CKPT")
    torch_ckpt = _resolve_checkpoint("JAX2ONNX_PARITY_TORCH_CKPT")
    cache_root = Path.home() / ".cache" / "gpt_oss" / "gpt-oss-20b"
    if not jax_ckpt.exists() and (cache_root / "original").exists():
        jax_ckpt = cache_root / "original"
    if not torch_ckpt.exists() and (cache_root / "original").exists():
        torch_ckpt = cache_root / "original"

    if not script.exists() or not gpt_oss_checkout.exists():
        pytest.skip("parity script or checkout missing")

    def _has_config(p: Path) -> bool:
        return p.is_dir() and (p / "config.json").exists()

    if not (_has_config(jax_ckpt) and _has_config(torch_ckpt)):
        pytest.skip("parity checkpoints unavailable")

    cmd = [
        "poetry",
        "run",
        "python",
        str(script),
        "--gpt-oss-path",
        str(gpt_oss_checkout),
        "--jax-checkpoint",
        str(jax_ckpt),
        "--torch-checkpoint",
        str(torch_ckpt),
        "--prompt",
        _DEFAULT_PROMPT,
        "--max-layers",
        _MAX_LAYERS,
        "--max-tokens",
        _MAX_TOKENS,
        "--torch-device",
        "cpu",
    ]

    subprocess.run(cmd, check=True, cwd=repo_root)
