# file: jax2onnx/plugins/jax/lax/squeeze.py
from typing import TYPE_CHECKING, List, Tuple, Union, Sequence

import jax
import numpy as np
from onnx import helper, TensorProto

from jax.extend import core as jax_core_extend

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.squeeze_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.squeeze.html",
    onnx=[
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="squeeze",
    testcases=[  # Test cases use jax.lax.squeeze as this is the lax plugin
        {
            "testcase": "lax_squeeze_specific_axis_0",
            "callable": lambda x: jax.lax.squeeze(x, dimensions=(0,)),
            "input_shapes": [(1, 3)],
            "expected_output_shapes": [(3,)],
        },
        {
            "testcase": "lax_squeeze_multiple_axes",
            "callable": lambda x: jax.lax.squeeze(x, dimensions=(0, 2, 4)),
            "input_shapes": [(1, 3, 1, 4, 1)],
            "expected_output_shapes": [(3, 4)],
        },
        {
            "testcase": "lax_squeeze_no_op_empty_dims",
            "callable": lambda x: jax.lax.squeeze(x, dimensions=()),
            "input_shapes": [(1, 3, 1)],
            "expected_output_shapes": [(1, 3, 1)],
        },
        {
            "testcase": "lax_squeeze_problem_case_input_squeeze_only_axis_0",
            "callable": lambda x: jax.lax.squeeze(x, dimensions=(0,)),
            "input_shapes": [(1, 201, 1, 1)],
            "expected_output_shapes": [(201, 1, 1)],
        },
        {
            "testcase": "lax_squeeze_problem_case_input_squeeze_axes_0_2",
            "callable": lambda x: jax.lax.squeeze(x, dimensions=(0, 2)),
            "input_shapes": [(1, 201, 1, 1)],
            "expected_output_shapes": [(201, 1)],
        },
        {
            "testcase": "lax_squeeze_problem_case_input_squeeze_all_dims_explicitly",
            "callable": lambda x: jax.lax.squeeze(x, dimensions=(0, 2, 3)),
            "input_shapes": [(1, 201, 1, 1)],
            "expected_output_shapes": [(201,)],
        },
        # Test cases involving jnp.squeeze are better placed in a test_jnp.py
        # as they test the JAX API frontend, which then calls lax.squeeze_p.
    ],
)
class SqueezePlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.squeeze to ONNX Squeeze."""

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[jax_core_extend.Var],
        node_outputs: Sequence[jax_core_extend.Var],
        params: dict,
    ) -> None:
        input_var = node_inputs[0]
        input_name = s.get_name(input_var)
        output_name = s.get_var_name(node_outputs[0])
        output_aval_dtype = node_outputs[0].aval.dtype  # Storing original dtype

        input_shape: Tuple[Union[int, str], ...] = input_var.aval.shape
        input_rank: int = len(input_shape)

        jax_dimensions_to_squeeze: Sequence[int] = params["dimensions"]

        normalized_axes_to_remove: List[int] = []
        if jax_dimensions_to_squeeze:  # If the tuple is not empty
            for axis_from_jax in jax_dimensions_to_squeeze:
                actual_axis = (
                    axis_from_jax if axis_from_jax >= 0 else axis_from_jax + input_rank
                )
                if not (0 <= actual_axis < input_rank):
                    # This should ideally be caught by JAX.
                    raise ValueError(
                        f"Squeeze dimension {axis_from_jax} (normalized: {actual_axis}) is out of bounds "
                        f"for input rank {input_rank} with shape {input_shape}."
                    )
                # JAX ensures dimensions to be squeezed are of size 1.
                normalized_axes_to_remove.append(actual_axis)

        # This list will be used for the ONNX 'axes' attribute.
        # Sorting and making unique is good practice for canonical representation.
        onnx_axes_for_node_attribute = sorted(list(set(normalized_axes_to_remove)))

        # Calculate the final output shape for the Squeezed data tensor
        calculated_output_shape_list: List[Union[int, str]] = []
        for i, dim_val in enumerate(input_shape):
            if i not in onnx_axes_for_node_attribute:
                calculated_output_shape_list.append(dim_val)

        if not calculated_output_shape_list and input_rank > 0:
            final_output_shape: Tuple[Union[int, str], ...] = ()
        else:
            final_output_shape = tuple(calculated_output_shape_list)

        # --- Build the ONNX Squeeze node ---
        squeeze_node_inputs = [input_name]

        axes_tensor_name = s.builder.get_unique_name(f"{output_name}_squeeze_axes")

        # Create the numpy array for the 'axes' initializer
        axes_np_array = np.array(onnx_axes_for_node_attribute, dtype=np.int64)

        # **CRITICAL FIX for ValueError**: Explicitly pass the correct 'dims' for the axes tensor.
        # The shape of axes_np_array is (N,) where N is the number of axes to squeeze.
        # For an empty list of axes (no-op), shape is (0,).
        axes_tensor_dims = list(axes_np_array.shape)

        s.builder.add_initializer(
            name=axes_tensor_name,
            vals=axes_np_array,  # Pass the numpy array
            data_type=TensorProto.INT64,
            dims=axes_tensor_dims,  # Explicitly pass the correct dimensions
        )
        squeeze_node_inputs.append(axes_tensor_name)

        squeeze_node = helper.make_node(
            "Squeeze",
            inputs=squeeze_node_inputs,
            outputs=[output_name],
            name=s.builder.get_unique_name("SqueezeOpInstance"),
        )
        s.add_node(
            squeeze_node
        )  # This should use the corrected add_node in jaxpr_converter
        s.add_shape_info(output_name, final_output_shape, output_aval_dtype)
