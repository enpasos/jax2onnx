# jax2onnx/sandbox/ort_nested_function_failure.py

"""
Repro for the ONNX Runtime 1.23.2 bug triggered by nested custom FunctionProtos.

Running this module will attempt to execute ONNX shape inference on the
ops-et 23 DINOv3 export that still contains stacked `custom.*` functions.
Even though every FunctionProto carries explicit value_info with dtype/shape
metadata, ONNX Runtime loses that information once a FunctionProto calls
another FunctionProto (VisionTransformer -> Block -> LayerScale/Attention),
and shape inference fails with the familiar "Input 0 expected to have type
but instead is null" error.
"""

from __future__ import annotations

import pathlib
import sys
import textwrap

import onnx
from onnx import shape_inference
import onnxruntime as ort


def _report(msg: str) -> None:
    print(f"[sandbox] {msg}")


def main() -> int:
    _report(f"ONNX Runtime version: {ort.__version__}")

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    model_path = (
        repo_root
        / "docs"
        / "onnx"
        / "examples"
        / "eqx_dino"
        / "eqx_dinov3_vit_S16_opset23.onnx"
    )
    _report(f"Loading model: {model_path}")

    model = onnx.load(str(model_path))

    try:
        shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:  # noqa: BLE001 - we want the exact diagnostic
        _report("Shape inference failed (expected) with:")
        print(textwrap.indent(str(exc), prefix="    "))
    else:  # pragma: no cover - we do not expect this branch
        _report("Unexpected success: shape inference no longer reproduces the issue.")
        return 1

    try:
        ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        _report("InferenceSession construction failed with:")
        print(textwrap.indent(str(exc), prefix="    "))
    else:
        _report(
            "InferenceSession succeeded; the nested-function failure only hits shape inference."
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
