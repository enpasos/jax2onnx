# jax2onnx/sandbox/issue_202/repro.py

"""Issue #202 repro helper.

This script does three things:
1. Exports a minimal Conv model with jax2onnx.
2. Prints Conv attributes to show whether kernel_shape is present.
3. Optionally runs pnnx / onnx2ncnn if those tools are on PATH.

Usage:
    poetry run python jax2onnx/sandbox/issue_202/repro.py
    poetry run python jax2onnx/sandbox/issue_202/repro.py --require-pnnx
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import onnx
from flax import nnx

from jax2onnx import to_onnx


class TinyConv(nnx.Module):
    """Minimal model that lowers to one ONNX Conv."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=3,
            out_features=4,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x):
        return self.conv(x)


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = proc.stdout
    if proc.stderr:
        if out:
            out = f"{out}\n{proc.stderr}"
        else:
            out = proc.stderr
    return proc.returncode, out.strip()


def export_model(onnx_path: Path) -> None:
    model = TinyConv(rngs=nnx.Rngs(0))
    to_onnx(
        model,
        [(1, 8, 8, 3)],  # NHWC
        return_mode="file",
        output_path=str(onnx_path),
    )


def inspect_conv_attrs(onnx_path: Path) -> list[list[str]]:
    model = onnx.load(str(onnx_path))
    conv_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
    attrs = [sorted(attr.name for attr in node.attribute) for node in conv_nodes]
    return attrs


def maybe_run_pnnx(onnx_path: Path) -> tuple[bool, int | None, str]:
    if shutil.which("pnnx") is None:
        return False, None, "pnnx not found on PATH"
    cmd = ["pnnx", str(onnx_path), "inputshape=[1,8,8,3]"]
    rc, out = _run(cmd)
    return True, rc, out


def maybe_run_onnx2ncnn(onnx_path: Path) -> tuple[bool, int | None, str]:
    if shutil.which("onnx2ncnn") is None:
        return False, None, "onnx2ncnn not found on PATH"
    param_path = onnx_path.with_suffix(".ncnn.param")
    bin_path = onnx_path.with_suffix(".ncnn.bin")
    cmd = ["onnx2ncnn", str(onnx_path), str(param_path), str(bin_path)]
    rc, out = _run(cmd)
    return True, rc, out


def _pnnx_expected_artifacts(onnx_path: Path) -> list[Path]:
    stem = onnx_path.with_suffix("")
    return [
        stem.with_suffix(".pnnx.param"),
        stem.with_suffix(".pnnx.bin"),
        Path(f"{stem}_pnnx.py"),
        stem.with_suffix(".pnnx.onnx"),
        stem.with_suffix(".ncnn.param"),
        stem.with_suffix(".ncnn.bin"),
        Path(f"{stem}_ncnn.py"),
    ]


def _validate_artifacts(paths: list[Path]) -> list[Path]:
    missing: list[Path] = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            missing.append(path)
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Issue #202 repro helper.")
    parser.add_argument(
        "--require-pnnx",
        action="store_true",
        help="Fail (non-zero exit) if pnnx is not installed or conversion fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    tmp_dir = repo_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = tmp_dir / "issue202_tinyconv.onnx"

    print(f"[1/4] Exporting model to {onnx_path}")
    export_model(onnx_path)

    print("[2/4] Inspecting Conv attributes")
    conv_attrs = inspect_conv_attrs(onnx_path)
    print(f"Found {len(conv_attrs)} Conv node(s).")
    for i, attrs in enumerate(conv_attrs):
        has_kernel_shape = "kernel_shape" in attrs
        print(f"  Conv[{i}] attrs={attrs} has_kernel_shape={has_kernel_shape}")

    print("[3/4] Trying pnnx (if installed)")
    has_pnnx, pnnx_rc, pnnx_out = maybe_run_pnnx(onnx_path)
    if not has_pnnx:
        print(f"  {pnnx_out}")
        if args.require_pnnx:
            print("  --require-pnnx set: failing because pnnx is unavailable.")
            return 2
    else:
        print(f"  pnnx exit code: {pnnx_rc}")
        if pnnx_out:
            print("  pnnx output:")
            print(pnnx_out)
        if pnnx_rc != 0:
            if args.require_pnnx:
                print("  --require-pnnx set: failing because pnnx conversion failed.")
                return 3
        else:
            artifacts = _pnnx_expected_artifacts(onnx_path)
            missing = _validate_artifacts(artifacts)
            if missing:
                print("  pnnx artifacts missing/empty:")
                for path in missing:
                    print(f"    - {path}")
                if args.require_pnnx:
                    print(
                        "  --require-pnnx set: failing because artifacts are invalid."
                    )
                    return 4
            else:
                print("  pnnx artifact check: ok")

    print("[4/4] Trying onnx2ncnn (if installed)")
    has_onnx2ncnn, onnx2ncnn_rc, onnx2ncnn_out = maybe_run_onnx2ncnn(onnx_path)
    if not has_onnx2ncnn:
        print(f"  {onnx2ncnn_out}")
    else:
        print(f"  onnx2ncnn exit code: {onnx2ncnn_rc}")
        if onnx2ncnn_out:
            print("  onnx2ncnn output:")
            print(onnx2ncnn_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
