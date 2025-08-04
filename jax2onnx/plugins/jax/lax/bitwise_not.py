"""ONNX plug-in for the unary **NOT** primitive:
  * ``jax.numpy.logical_not``               (bool inputs)
  * ``jax.lax.bitwise_not``                 (integer inputs)

JAX primitive name           : ``"not"``   (object ``lax.not_p``)
Mapped ONNX operator          : ``BitwiseNot`` (opset ≥ 18)
"""

from __future__ import annotations

from typing import Any, Sequence

import logging
import numpy as np
from jax import lax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

logger = logging.getLogger("jax2onnx.plugins.jax.lax.bitwise_not")


@register_primitive(
    jaxpr_primitive=lax.not_p.name,  # ▶  primitive is literally called "not"
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.bitwise_not.html",
    onnx=[{"component": "BitwiseNot", "doc": "https://onnx.ai/onnx/operators/onnx__BitwiseNot.html"}],
    since="v0.7.5",
    context="primitives.lax",
    component="bitwise_not",
    testcases=[
        # simple smoke-tests to exercise both bool and int paths
        {
            "testcase": "bitwise_not_bool",
            "callable": lambda x: lax.bitwise_not(x),
            "input_values": [np.array(True, dtype=np.bool_)],
            "expected_output_dtypes": [np.bool_],
        },
        {
            "testcase": "bitwise_not_i32",
            "callable": lambda x: lax.bitwise_not(x),
            "input_values": [np.array(7, dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
        },
    ],
)
class BitwiseNotPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.not_p`` to ONNX ``BitwiseNot`` or logical ``Not``."""

    # ① abstract eval — shape & dtype unchanged
    @staticmethod
    def abstract_eval(x):
        return x

    # ② lowering
    def to_onnx(
        self,
        s,                       # Jaxpr2OnnxConverter
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        inp_name  = s.get_name(node_inputs[0])
        out_name  = s.get_name(node_outputs[0])
        dtype = node_inputs[0].aval.dtype

        # ONNX rule:  bool → Not,  ints → BitwiseNot
        is_bool = np.dtype(dtype).kind == "b"
        op_type = "Not" if is_bool else "BitwiseNot"

        s.add_node(
            helper.make_node(
                op_type,
                inputs=[inp_name],
                outputs=[out_name],
                name=s.get_unique_name(op_type.lower()),
            )
        )

        aval = node_inputs[0].aval
        s.add_shape_info(out_name, aval.shape, aval.dtype) 

        logger.debug("Lowered lax.not_p  →  ONNX %s  (%s → %s)", op_type, inp_name, out_name)
