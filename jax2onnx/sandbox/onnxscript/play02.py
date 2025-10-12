#!/usr/bin/env python
# jax2onnx/sandbox/onnxscript/play02.py

"""
Playground #2: load the issue52 payload and prototype an onnxscript rewrite.

This script shows how to:
  * load an existing ONNX model into ONNX Script IR,
  * traverse nodes to spot the problematic constant-based Concat,
  * replace one of its inputs with a dynamic axis gathered from Shape,
  * serialize the modified model and print a sanity summary.

We only rewrite the first Concat that builds the reshape target for the top-level
broadcast; extend as needed for loop bodies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import onnx
from onnx import helper, numpy_helper


ISSUE52_MODEL = Path(__file__).resolve().parents[3] / "sod_issue52_payload.onnx"
OUTPUT_MODEL = Path(__file__).with_name("sod_issue52_payload_rewritten.onnx")


def _find_concat_with_constants(graph) -> Optional[int]:
    for idx, node in enumerate(graph.node):
        if node.op_type != "Concat":
            continue
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), None)
        if axis != 0:
            continue
        if any(inp in {init.name for init in graph.initializer} for inp in node.input):
            return idx
    return None


def _inject_dynamic_dim(model, concat_idx: int) -> None:
    graph = model.graph
    concat_node = graph.node[concat_idx]
    concat_input = concat_node.input[0]

    axis_zero_name = "play02_axis_zero"
    if not any(init.name == axis_zero_name for init in graph.initializer):
        graph.initializer.append(
            numpy_helper.from_array(np.array([0], dtype=np.int64), name=axis_zero_name)
        )

    unsqueeze_axes_name = "play02_unsqueeze_axes"
    if not any(init.name == unsqueeze_axes_name for init in graph.initializer):
        graph.initializer.append(
            numpy_helper.from_array(
                np.array([0], dtype=np.int64), name=unsqueeze_axes_name
            )
        )

    shape_out = "play02_dyn_shape"
    gather_out = "play02_dyn_dim_scalar"
    unsqueeze_out = "play02_dyn_dim_vec"

    shape_node = helper.make_node(
        "Shape",
        inputs=[concat_input],
        outputs=[shape_out],
        name="play02_shape",
    )
    gather_node = helper.make_node(
        "Gather",
        inputs=[shape_out, axis_zero_name],
        outputs=[gather_out],
        name="play02_gather",
        axis=0,
    )
    unsqueeze_node = helper.make_node(
        "Unsqueeze",
        inputs=[gather_out, unsqueeze_axes_name],
        outputs=[unsqueeze_out],
        name="play02_unsqueeze",
    )

    graph.node.extend([shape_node, gather_node, unsqueeze_node])

    for idx, inp in enumerate(concat_node.input):
        if inp in {init.name for init in graph.initializer}:
            concat_node.input[idx] = unsqueeze_out
            break


def main() -> None:
    if not ISSUE52_MODEL.exists():
        raise SystemExit(f"Missing payload at {ISSUE52_MODEL}")

    model = onnx.load(str(ISSUE52_MODEL))

    idx = _find_concat_with_constants(model.graph)
    if idx is None:
        print("No matching Concat found; exiting.")
        return

    _inject_dynamic_dim(model, idx)

    onnx.save(model, str(OUTPUT_MODEL))
    print("Rewritten model saved to", OUTPUT_MODEL)
    concat_node = model.graph.node[idx]
    print("Updated Concat inputs:", list(concat_node.input))


if __name__ == "__main__":
    main()
