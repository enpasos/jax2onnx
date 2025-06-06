# file: jax2onnx/plugins/jax/lax/reduce_and.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import jax.numpy as jnp
import numpy as np
from onnx import helper, TensorProto
from jax import lax

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from ._reduce_utils import add_reduce_node  # shared helper

module_logger = logging.getLogger("jax2onnx.plugins.jax.lax.reduce_and")

reduce_and_p = lax.reduce_and_p


@register_primitive(
    jaxpr_primitive=reduce_and_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.all.html",
    onnx=[
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
        {
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        },
    ],
    since="v0.6.1",
    context="primitives.lax",
    component="reduce_and",
    testcases=[
        {
            "testcase": "reduce_and_all_true",
            "callable": lambda: jnp.all(np.array([True, True, True])),
            "input_shapes": [],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.bool_],
        },
        {
            "testcase": "reduce_and_one_false",
            "callable": lambda: jnp.all(np.array([True, False, True])),
            "input_shapes": [],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.bool_],
        },
        {
            "testcase": "reduce_and_keepdims",
            "callable": lambda: jnp.all(
                np.array([[True, True], [True, False]]), axis=1, keepdims=True
            ),
            "input_shapes": [],
            "expected_output_shapes": [(2, 1)],
            "expected_output_dtypes": [np.bool_],
        },
    ],
)
class ReduceAndPlugin(PrimitiveLeafPlugin):
    """Plugin for the `reduce_and` primitive."""

    primitive = reduce_and_p

    def to_onnx(
        self,
        conv: Jaxpr2OnnxConverter,
        invars: List[Any],
        outvars: List[Any],
        params: Dict[str, Any],
    ):
        """
        Maps the JAX reduce_and primitive to ONNX operators.

        This is implemented by:
        1. Casting the boolean input tensor to INT32.
        2. Applying ReduceMin to the integer tensor.
        3. Casting the integer result back to a boolean.
        """
        inp_var = invars[0]
        out_var = outvars[0]
        inp_name = conv.get_name(inp_var)
        out_name = conv.get_name(out_var)
        axes: Optional[Sequence[int]] = params.get("axes", None)

        # Determine keepdims flag based on input vs output rank
        inp_rank = len(inp_var.aval.shape)
        out_rank = len(out_var.aval.shape)
        keepdims_flag = True if inp_rank == out_rank else False

        # 1) Cast boolean input to INT32 (False -> 0, True -> 1)
        cast_int_out = conv.builder.get_unique_name("cast_to_int")
        # Declare value_info for the cast output
        conv.builder.add_value_info(
            name=cast_int_out,
            shape=tuple(conv._dim_to_symbol_safe(d) for d in inp_var.aval.shape),
            dtype=TensorProto.INT32,
        )
        cast_int_node = helper.make_node(
            "Cast",
            inputs=[inp_name],
            outputs=[cast_int_out],
            to=TensorProto.INT32,
            name=conv.get_unique_name("Cast_to_int"),
        )
        conv.builder.add_node(cast_int_node)

        # 2) Apply ReduceMin on the INT32 tensor
        reduce_min_out = conv.builder.get_unique_name("reduce_min")
        # Declare value_info for the reduce_min output (before casting back to bool)
        conv.builder.add_value_info(
            name=reduce_min_out,
            shape=tuple(conv._dim_to_symbol_safe(d) for d in out_var.aval.shape),
            dtype=TensorProto.INT32,
        )
        add_reduce_node(
            conv.builder,
            "ReduceMin",
            cast_int_out,
            reduce_min_out,
            axes=list(axes) if axes is not None else None,
            keepdims=1 if keepdims_flag else 0,
        )

        # 3) Cast the integer result back to BOOL
        cast_bool_node = helper.make_node(
            "Cast",
            inputs=[reduce_min_out],
            outputs=[out_name],
            to=TensorProto.BOOL,
            name=conv.get_unique_name("Cast_to_bool"),
        )
        conv.builder.add_node(cast_bool_node)

        # 4) Declare the final output's value_info (shape & dtype)
        conv.builder.add_value_info(
            name=out_name,
            shape=tuple(conv._dim_to_symbol_safe(d) for d in out_var.aval.shape),
            dtype=TensorProto.BOOL,
        )
