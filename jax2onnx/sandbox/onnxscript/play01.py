#!/usr/bin/env python
# jax2onnx/sandbox/onnxscript/play01.py

"""
Small playground script to exercise the onnxscript authoring API.

It defines a tiny `add_relu` function using the `@script` decorator, converts it
to an ONNX model, saves the model next to this file, and finally reloads it via
onnxscript's IR utilities to print a quick summary. Use this as a starting
point for richer experiments (custom rewrites, model inspection, etc.).
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.ir import serde

DIM_N = "N"


@script()
def add_relu(x: FLOAT[DIM_N], y: FLOAT[DIM_N]) -> FLOAT[DIM_N]:
    """
    Return Relu(x + y).
    """

    summed = op.Add(x, y)
    return op.Relu(summed)


def main() -> None:
    model = add_relu.to_model_proto()
    out_path = Path(__file__).with_suffix(".onnx")
    onnx.save(model, out_path)

    print(f"Model written to: {out_path}")

    model_ir = serde.deserialize_model(model)
    graph = model_ir.graph
    print("Graph name:", graph.name)
    print("Inputs:", [vi.name for vi in graph.inputs])
    print("Outputs:", [vi.name for vi in graph.outputs])
    print("Ops in graph:", [graph.node(i).op_type for i in range(graph.num_nodes())])


if __name__ == "__main__":
    main()
