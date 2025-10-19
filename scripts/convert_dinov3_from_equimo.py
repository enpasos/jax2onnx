#!/usr/bin/env python3
# scripts/convert_dinov3_from_equimo.py

"""
Convert Meta/Equimo DINOv3 checkpoints (PyTorch) into Equinox trees.

The script mirrors the conversion logic in `equimo/models/dinov3.py` so we can
reuse Equimo’s parameter mappings while producing Equinox checkpoints that slot
straight into the jax2onnx examples.

Prerequisites
-------------
- Install `equimo` with its conversion extras (requires `torch`, `jax`, `equinox`):
    pip install "equimo[conversion]"
- Download the official DINOv3 checkpoints from Meta:
    https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
  Place the `.pth` files under the default cache expected by Equimo
  (`~/.cache/torch/hub/dinov3/weights/…`) or point `--weights` at a custom path.

Usage
-----
    python scripts/convert_dinov3_from_equimo.py --variant dinov3_vits16_pretrain_lvd1689m

Optional flags:
    --weights PATH   # Override the checkpoint path (defaults to Equimo’s cache location)
    --output PATH    # Directory/file prefix for the serialized Equinox model
    --seed 42        # PRNG seed used to instantiate the Equinox model before loading weights
    --skip-check     # Omit the numerical equivalence check against the PyTorch model

The resulting `.tar.lz4` archive is compatible with `equimo.io.load_model` and can be
loaded inside the jax2onnx DINO example for export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def _load_equimo_dinov3(
    variant: str,
    *,
    weights_path: Path | None,
    output_path: Path | None,
    seed: int,
    run_check: bool,
) -> Path:
    """
    Convert a single DINOv3 variant from PyTorch weights into an Equinox archive.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "PyTorch is required to convert DINOv3 checkpoints. "
            "Install PyTorch wheels (CPU-only is fine): pip install torch torchvision torchaudio"
        ) from exc

    try:
        from equimo.conversion.utils import convert_torch_to_equinox
        from equimo.io import save_model
        from equimo import models as equimo_models
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Equimo is required. Install it from GitHub:\n"
            "  pip install 'equimo @ git+https://github.com/clementpoiret/Equimo.git'\n"
            "and ensure it is installed inside the poetry environment."
        ) from exc

    def _load_dinov3_module():
        try:
            from equimo.models import dinov3 as mod  # type: ignore[attr-defined]

            return mod
        except ImportError:
            pass

        import importlib.util

        pkg_path = Path(equimo_models.__file__).resolve()
        # Search common locations: package directory and repository root.
        candidates = [
            pkg_path.parent / "dinov3.py",
            pkg_path.parents[1] / "models" / "dinov3.py",
            pkg_path.parents[2] / "models" / "dinov3.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                spec = importlib.util.spec_from_file_location(
                    "equimo.models.dinov3", candidate
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["equimo.models.dinov3"] = module
                    spec.loader.exec_module(module)  # type: ignore[misc]
                    return module
        raise ImportError(
            "Could not locate `equimo.models.dinov3`. Ensure you are using a git build "
            "of Equimo. The file should exist either inside the installed package or "
            "at <repo_root>/models/dinov3.py."
        )

    equimo_dinov3 = _load_dinov3_module()
    from equimo.models.vit import VisionTransformer

    if variant not in equimo_dinov3.configs:
        available = ", ".join(sorted(equimo_dinov3.configs))
        raise ValueError(
            f"Unknown DINOv3 variant '{variant}'. Available options: {available}"
        )

    cfg = equimo_dinov3.dinov3_config | equimo_dinov3.configs[variant]
    key = jax.random.PRNGKey(seed)
    model = VisionTransformer(**cfg, key=key)

    weight_path = (
        Path(weights_path).expanduser()
        if weights_path is not None
        else Path(equimo_dinov3.weights[variant]).expanduser()
    )
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Checkpoint for '{variant}' not found at {weight_path}.\n"
            "Download the official weights from Meta and place them at that location "
            "or pass --weights with the downloaded file."
        )

    torch_repo_dir = equimo_dinov3.DIR / "dinov3"
    torch_model_name = "_".join(variant.split("_")[:-2])
    torch_hub_cfg = {
        "repo_or_dir": str(torch_repo_dir),
        "model": torch_model_name,
        "source": "local",
        "weights": str(weight_path),
    }

    replace_cfg = {
        "reg_tokens": "storage_tokens",
        "blocks.0.blocks": "blocks",
        ".prenorm.": ".norm1.",
        ".norm.": ".norm2.",
    }
    expand_cfg = {"patch_embed.proj.bias": ["after", 2]}
    squeeze_cfg = {
        "pos_embed": 0,
        "cls_token": 0,
        "storage_tokens": 0,
    }
    torch_whitelist: list[str] = []
    jax_whitelist = ["pos_embed.periods"]

    eq_model, torch_model = convert_torch_to_equinox(
        model,
        replace_cfg,
        expand_cfg,
        squeeze_cfg,
        torch_whitelist,
        jax_whitelist,
        strict=True,
        torch_hub_cfg=torch_hub_cfg,
        return_torch=True,
    )
    eq_model = eqx.nn.inference_mode(eq_model, True)

    if run_check:
        rand = np.random.randn(3, cfg["img_size"], cfg["img_size"])
        jax_arr = jnp.array(rand)
        torch_arr = torch.tensor(rand, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            torch_out = torch_model.forward_features(torch_arr)["x_prenorm"]
        jax_out = eq_model.features(jax_arr, inference=True, key=key)
        delta = float(np.max(np.abs(np.asarray(jax_out) - torch_out.numpy())))
        if delta > 5e-4:
            raise RuntimeError(
                f"Conversion drift detected for '{variant}' (max abs diff {delta:.6f})."
            )

    if output_path is None:
        output_path = Path(f"~/.cache/equimo/dinov3/{variant}").expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_model(
        output_path,
        eq_model,
        cfg,
        torch_hub_cfg,
        compression=True,
    )
    if not output_path.name.endswith(".tar.lz4"):
        output_path = output_path.with_name(output_path.name + ".tar.lz4")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Meta/Equimo DINOv3 PyTorch checkpoints into Equinox archives "
            "that can be consumed by the jax2onnx DINO examples."
        )
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="DINOv3 identifier (e.g., dinov3_vits16_pretrain_lvd1689m).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Path to the PyTorch checkpoint (.pth). Defaults to Equimo's cache path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Target path (file or directory). Defaults to ~/.cache/equimo/dinov3/{variant}.tar.lz4."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for model initialization prior to weight loading.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the numerical equivalence check against the PyTorch model.",
    )

    args = parser.parse_args(argv)

    try:
        output = _load_equimo_dinov3(
            args.variant,
            weights_path=args.weights,
            output_path=args.output,
            seed=args.seed,
            run_check=not args.skip_check,
        )
    except Exception as exc:  # pragma: no cover - integration path
        parser.error(str(exc))
        return 2

    print(f"Saved Equinox checkpoint to {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
