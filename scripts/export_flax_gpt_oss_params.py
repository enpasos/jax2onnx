#!/usr/bin/env python3
# scripts/export_flax_gpt_oss_params.py

"""Stage GPT-OSS Flax parameters for downstream ONNX exports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import flax.serialization as flax_serialization


def _ensure_repo_on_path(repo: Path) -> None:
    repo = repo.resolve()
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def _load_params(checkpoint: Path):
    from gpt_oss.jax.loader_orbax import OrbaxWeightLoader, load_config_from_orbax
    from gpt_oss.jax.loader_safetensors import WeightLoader
    from gpt_oss.jax.token_generator import (
        detect_checkpoint_format,
        load_config_from_checkpoint,
    )

    fmt = detect_checkpoint_format(checkpoint)
    if fmt == "orbax":
        config = load_config_from_orbax(str(checkpoint))
        loader = OrbaxWeightLoader(str(checkpoint))
        params = loader.load_params(show_progress=True)
    else:
        config = load_config_from_checkpoint(checkpoint)
        loader = WeightLoader(str(checkpoint))
        params = loader.load_params(config, show_progress=True)
    return config, params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage GPT-OSS Flax params for ONNX export."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to safetensors/orbax checkpoint",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write serialized Flax params",
    )
    parser.add_argument(
        "--gpt-oss-path",
        type=Path,
        default=Path("tmp/gpt-oss-jax-vs-torch-numerical-comparison"),
        help="Path to the gpt-oss repo checkout (for loaders).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_repo_on_path(args.gpt_oss_path)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config, params = _load_params(checkpoint)
    if is_dataclass(config):
        config_payload = asdict(config)
    elif isinstance(config, dict):
        config_payload = config
    else:
        raise TypeError(
            f"Unsupported config type '{type(config)!r}'. Expected dataclass or dict."
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = flax_serialization.to_bytes(params)
    output.write_bytes(payload)

    config_path = output.with_suffix(".config.json")
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True))
    print(f"Saved Flax params to {output}")


if __name__ == "__main__":
    main()
