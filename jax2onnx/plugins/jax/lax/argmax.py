# file: jax2onnx/plugins/jax/lax/argmax.py

from typing import TYPE_CHECKING, List, Dict, Any

import jax
import jax.numpy as jnp
from onnx import TensorProto, helper
from onnx import mapping  # For np_dtype_to_tensor_dtype

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
    from jax.core import Var  # For type hinting


@register_primitive(
    jaxpr_primitive=jax.lax.argmax_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmax.html",
    onnx=[
        {
            "component": "ArgMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMax.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="argmax",
    testcases=[
        {
            "testcase": "argmax_float_axis0",
            "callable": lambda x: jax.lax.argmax(x, axis=0, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
            "input_dtypes": [jnp.float32],
            "description": "Test lax.argmax with float32 input tensor along axis 0.",
        },
        {
            "testcase": "argmax_float_axis1",
            "callable": lambda x: jax.lax.argmax(x, axis=1, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
            "input_dtypes": [jnp.float32],
            "description": "Test lax.argmax with float32 input tensor along axis 1.",
        },
        {
            "testcase": "argmax_boolean_input_axis0_specific_values",
            "callable": lambda x_bool: jax.lax.argmax(
                x_bool, axis=0, index_dtype=jnp.int32
            ),
            "input_values": [
                jnp.array(
                    [[False, True, False], [True, False, True], [False, False, False]],
                    dtype=jnp.bool_,
                )
            ],
            "description": "Test lax.argmax with a specific boolean input tensor along axis 0.",
        },
        {
            "testcase": "argmax_boolean_input_axis1_specific_values",
            "callable": lambda x_bool: jax.lax.argmax(
                x_bool, axis=1, index_dtype=jnp.int32
            ),
            "input_values": [
                jnp.array(
                    [[False, True, False], [True, False, True], [False, True, True]],
                    dtype=jnp.bool_,
                )
            ],
            "description": "Test lax.argmax with a specific boolean input tensor along axis 1.",
        },
        {
            "testcase": "argmax_boolean_random_input_axis0",
            "callable": lambda x: jax.lax.argmax(x, axis=0, index_dtype=jnp.int32),
            "input_shapes": [(4, 5)],
            "input_dtypes": [jnp.bool_],
            "description": "Test lax.argmax with random boolean input tensor along axis 0.",
        },
    ],
)
class ArgMaxPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.argmax to ONNX.
    """

    def _get_onnx_dtype_from_jax_dtype(
        self, jax_dtype_obj, converter_instance: "Jaxpr2OnnxConverter"
    ):
        """
        Converts a JAX dtype object (like jnp.int32) to ONNX TensorProto type.
        """
        try:
            # jax_dtype_obj is like jnp.int32, its .dtype is numpy.dtype('int32')
            return mapping.np_dtype_to_tensor_dtype(jax_dtype_obj.dtype)
        except Exception as e:
            converter_instance.logger.warning(
                f"Failed to map JAX dtype '{jax_dtype_obj}' to ONNX using onnx.mapping.np_dtype_to_tensor_dtype: {e}. "
                f"Defaulting to INT32."
            )
            return TensorProto.INT32

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: List["Var"],
        node_outputs: List["Var"],
        params: Dict[str, Any],
    ):
        """Handle JAX argmax primitive."""
        input_var = node_inputs[0]
        output_var = node_outputs[0]
        # params dict is used directly

        input_aval = input_var.aval
        input_name = s.get_name(input_var)

        onnx_argmax_input_name = input_name

        if input_aval.dtype == jnp.bool_:
            casted_input_name = s.get_unique_name(f"{input_name}_casted_to_int32")
            cast_node_bool_to_int = helper.make_node(
                "Cast",
                inputs=[input_name],
                outputs=[casted_input_name],
                to=TensorProto.INT32,
            )
            s.add_node(cast_node_bool_to_int)
            s.add_shape_info(
                casted_input_name, input_aval.shape, jnp.int32
            )  # dtype after cast
            onnx_argmax_input_name = casted_input_name

        intermediate_name = s.get_unique_name("argmax_intermediate_output")
        output_name = s.get_name(output_var)
        axis = params["axes"][0]
        keepdims = 0
        select_last_index_for_onnx = 0

        node_1_argmax = helper.make_node(
            "ArgMax",
            inputs=[onnx_argmax_input_name],
            outputs=[intermediate_name],
            name=s.get_unique_name("argmax_op"),
            axis=axis,
            keepdims=keepdims,
            select_last_index=select_last_index_for_onnx,
        )
        s.add_node(node_1_argmax)

        intermediate_shape = output_var.aval.shape
        s.add_shape_info(
            intermediate_name, intermediate_shape, jnp.int64
        )  # ONNX ArgMax outputs int64

        jax_target_index_dtype_obj = params["index_dtype"]

        onnx_target_dtype = self._get_onnx_dtype_from_jax_dtype(
            jax_target_index_dtype_obj, s
        )

        if onnx_target_dtype != TensorProto.INT64:
            node_2_cast = helper.make_node(
                "Cast",
                inputs=[intermediate_name],
                outputs=[output_name],
                name=s.get_unique_name("cast_to_jax_index_dtype"),
                to=onnx_target_dtype,
            )
            s.add_node(node_2_cast)
        else:
            s._emit_result(
                node_outputs[0], wanted_sym=output_name, src_sym=intermediate_name
            )
