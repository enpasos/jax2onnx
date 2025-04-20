from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import TensorProto, helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.dynamic_slice_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html",
    onnx=[
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        }
    ],
    since="v0.1.0",
    context="primitives.lax",
    component="dynamic_slice",
    testcases=[
        {
            "testcase": "dynamic_slice_test1",
            "callable": lambda x: jax.lax.dynamic_slice(x, [1], [2]),
            "input_shapes": [(5,)],
        },
        {
            "testcase": "dynamic_slice_2d",
            "callable": lambda x: jax.lax.dynamic_slice(x, (1, 2), (2, 3)),
            "input_shapes": [(4, 6)],
        },
        {
            "testcase": "dynamic_slice_3d",
            "callable": lambda x: jax.lax.dynamic_slice(x, (1, 0, 2), (2, 3, 1)),
            "input_shapes": [(3, 4, 5)],
        },
        # {
        #     "testcase": "dynamic_slice_vit",
        #     "callable": lambda x: jax.lax.dynamic_slice(x, (0, 0, 0), (1, 50, 256)),
        #     "input_shapes": [(3, 4, 5)],
        # },
    ],
)
class DynamicSlicePlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.dynamic_slice to ONNX Slice."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        operand_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        d = len(node_inputs[0].aval.shape)

        start_names = []
        for i in range(1, 1 + d):
            start_i_name = s.get_name(node_inputs[i])

            # First Cast the scalar index to int64
            casted_scalar = s.get_unique_name(f"cast_start_scalar_{i}")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[start_i_name],
                    outputs=[casted_scalar],
                    name=s.get_unique_name("cast_start_scalar"),
                    to=TensorProto.INT64,
                )
            )
            s.add_shape_info(casted_scalar, tuple([]), dtype=np.int64)

            # Then Unsqueeze it to shape [1]
            axes_const = s.get_constant_name(np.array([0], dtype=np.int64))
            unsqueezed_name = s.get_unique_name(f"unsqueezed_start_{i}")
            s.add_node(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[casted_scalar, axes_const],
                    outputs=[unsqueezed_name],
                    name=s.get_unique_name("unsqueeze_start"),
                )
            )
            s.add_shape_info(unsqueezed_name, tuple([1]), dtype=np.int64)

            start_names.append(unsqueezed_name)

        # Concatenate start indices into 1D tensor
        starts_concat_name = s.get_unique_name("dynamic_starts")
        s.add_node(
            helper.make_node(
                "Concat",
                inputs=start_names,
                outputs=[starts_concat_name],
                name=s.get_unique_name("concat_starts"),
                axis=0,
            )
        )
        s.add_shape_info(starts_concat_name, tuple([d]), dtype=np.int64)

        # Create constant for slice sizes
        slice_sizes = params["slice_sizes"]
        slice_sizes_const = s.get_constant_name(np.array(slice_sizes, dtype=np.int64))

        # Compute ends = starts + slice_sizes
        ends_name = s.get_unique_name("dynamic_ends")
        s.add_node(
            helper.make_node(
                "Add",
                inputs=[starts_concat_name, slice_sizes_const],
                outputs=[ends_name],
                name=s.get_unique_name("add_slice_ends"),
            )
        )
        s.add_shape_info(ends_name, tuple([d]), dtype=np.int64)

        # Axes: [0, 1, ..., d-1]
        axes_const = s.get_constant_name(np.arange(d, dtype=np.int64))

        inputs_list = [operand_name, starts_concat_name, ends_name, axes_const]
        if "strides" in params and params["strides"]:
            strides = params["strides"]
            strides_const = s.get_constant_name(np.array(strides, dtype=np.int64))
            inputs_list.append(strides_const)

        s.add_node(
            helper.make_node(
                "Slice",
                inputs=inputs_list,
                outputs=[output_name],
                name=s.get_unique_name("dynamic_slice"),
            )
        )
