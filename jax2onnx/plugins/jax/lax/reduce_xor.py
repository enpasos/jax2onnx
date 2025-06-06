# file: jax2onnx/plugins/jax/lax/reduce_xor.py

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

module_logger = logging.getLogger("jax2onnx.plugins.jax.lax.reduce_xor")

reduce_xor_p = lax.reduce_xor_p


@register_primitive(
    jaxpr_primitive=reduce_xor_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logical_xor.html",
    onnx=[
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {"component": "Mod", "doc": "https://onnx.ai/onnx/operators/onnx__Mod.html"},
    ],
    since="v0.6.1",
    context="primitives.lax",
    component="reduce_xor",
    testcases=[
        {
            "testcase": "reduce_xor_all_false",
            "callable": lambda: jnp.logical_xor.reduce(np.array([False, False, False])),
            "input_shapes": [],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.bool_],
        },
        {
            "testcase": "reduce_xor_one_true",
            "callable": lambda: jnp.logical_xor.reduce(np.array([False, True, False])),
            "input_shapes": [],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.bool_],
        },
        {
            "testcase": "reduce_xor_two_true",
            "callable": lambda: jnp.logical_xor.reduce(np.array([True, True, False])),
            "input_shapes": [],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.bool_],
        },
        {
            "testcase": "reduce_xor_keepdims",
            "callable": lambda: jnp.logical_xor.reduce(
                np.array([[False, True], [True, False]]), axis=1, keepdims=True
            ),
            "input_shapes": [],
            "expected_output_shapes": [(2, 1)],
            "expected_output_dtypes": [np.bool_],
        },
    ],
)
class ReduceXorPlugin(PrimitiveLeafPlugin):
    """Plugin for the `reduce_xor` primitive."""

    primitive = reduce_xor_p

    def to_onnx(
        self,
        conv: Jaxpr2OnnxConverter,
        invars: List[Any],
        outvars: List[Any],
        params: Dict[str, Any],
    ):
        """
        Maps the JAX reduce_xor primitive to ONNX operators.

        Implementation steps:
        1) Cast boolean input to INT32 (False->0, True->1).
        2) ReduceSum over the specified axes.
        3) Take Mod of the sum with 2 (to compute parity).
        4) Cast the result back to BOOL.
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

        # 1) Cast boolean input to INT32
        cast_int_out = conv.builder.get_unique_name("cast_to_int")
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

        # 2) Apply ReduceSum on the INT32 tensor
        reduced_sum = conv.builder.get_unique_name("reduce_sum")
        conv.builder.add_value_info(
            name=reduced_sum,
            shape=tuple(conv._dim_to_symbol_safe(d) for d in out_var.aval.shape),
            dtype=TensorProto.INT32,
        )
        add_reduce_node(
            conv.builder,
            "ReduceSum",
            cast_int_out,
            reduced_sum,
            axes=list(axes) if axes is not None else None,
            keepdims=1 if keepdims_flag else 0,
        )

        # 3) Create constant '2' (INT32) for Mod
        const_two_name = conv.builder.get_constant_name(np.array(2, dtype=np.int32))

        # 4) Compute sum mod 2 to get parity (XOR)
        mod_out = conv.builder.get_unique_name("mod_parity")
        conv.builder.add_value_info(
            name=mod_out,
            shape=tuple(conv._dim_to_symbol_safe(d) for d in out_var.aval.shape),
            dtype=TensorProto.INT32,
        )
        mod_node = helper.make_node(
            "Mod",
            inputs=[reduced_sum, const_two_name],
            outputs=[mod_out],
            name=conv.get_unique_name("Mod_parity"),
        )
        conv.builder.add_node(mod_node)

        # 5) Cast the result back to BOOL
        cast_bool_node = helper.make_node(
            "Cast",
            inputs=[mod_out],
            outputs=[out_name],
            to=TensorProto.BOOL,
            name=conv.get_unique_name("Cast_to_bool"),
        )
        conv.builder.add_node(cast_bool_node)

        # 6) Declare the final output's value_info (shape & dtype)
        conv.builder.add_value_info(
            name=out_name,
            shape=tuple(conv._dim_to_symbol_safe(d) for d in out_var.aval.shape),
            dtype=TensorProto.BOOL,
        )
