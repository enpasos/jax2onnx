# -*- coding: utf-8 -*-
"""Shared utilities for all Reduce* plugins"""

from typing import Sequence, Optional, List
from onnx import helper, TensorProto


def add_reduce_node(
    builder,
    op_type: str,
    inp: str,
    out: str,
    axes: Optional[Sequence[int]],
    keepdims: int = 0,
):
    """
    Emit a single ONNX Reduce* node.

    Parameters
    ----------
    builder     : `jax2onnx.converter.onnx_builder.OnnxBuilder`
    op_type     : 'ReduceSum' | 'ReduceMin' | …
    inp, out    : input / output value-names
    axes        : None   →  reduce over *all* dims (leave 1 input)
                  list   →  INT64 initializer is created automatically
    keepdims    : 0 or 1 – reflected in the node’s attribute
    """
    inputs: List[str] = [inp]

    if axes is not None:
        axes_name = builder.get_unique_name("axes")
        axes_tensor = helper.make_tensor(
            name=axes_name,
            data_type=TensorProto.INT64,
            dims=[len(axes)],
            vals=axes,
        )
        builder.initializers.append(axes_tensor)
        inputs.append(axes_name)

    node = helper.make_node(
        op_type,
        inputs=inputs,
        outputs=[out],
        keepdims=keepdims,
        name=builder.get_unique_name(op_type),
    )
    builder.add_node(node)
