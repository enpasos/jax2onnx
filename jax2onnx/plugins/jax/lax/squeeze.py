from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.dynamic_utils import encode_dims

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
    testcases=[
        {
            "testcase": "squeeze",
            "callable": lambda x: jax.lax.squeeze(x, (0,)),
            "input_shapes": [(1, 3)],
        }
    ],
)
class SqueezePlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.squeeze to ONNX Squeeze."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX squeeze primitive."""
        from onnx import helper
        from jax2onnx.converter.dynamic_utils import encode_dims

        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])

        # Use symbolic shape if available (e.g. batch dim "B")
        var = node_inputs[0]
        var_name = s.get_var_name(var)
        input_shape = s.symbolic_shapes.get(var_name, var.aval.shape)

        axes = params.get("axes", None)

        # Normalize axes into positive indices; collect any symbolic axes
        normalized_axes = []
        symbolic_axes = []
        for axis in axes or []:
            axis_val = axis if axis >= 0 else axis + len(input_shape)
            if 0 <= axis_val < len(input_shape):
                if isinstance(input_shape[axis_val], int):
                    normalized_axes.append(axis_val)
                else:
                    symbolic_axes.append(axis_val)

        # Identify size-1 dims among those axes
        static_axes = [i for i in normalized_axes if input_shape[i] == 1]

        # If user specified axes but none are static size-1, identity
        if axes is not None and not static_axes:
            identity = helper.make_node(
                "Identity",
                inputs=[input_name],
                outputs=[output_name],
                name=s.get_unique_name("identity"),
            )
            s.add_node(identity)
            s.add_shape_info(output_name, tuple(input_shape))
            return

        # Build the ONNX Squeeze node
        if static_axes:
            axes_name = s.get_unique_name("squeeze_axes")
            s.add_initializer(name=axes_name, vals=encode_dims(static_axes))
            squeeze_inputs = [input_name, axes_name]
            output_shape = tuple(
                dim for i, dim in enumerate(input_shape) if i not in static_axes
            )
        else:
            squeeze_inputs = [input_name]
            output_shape = tuple(
                dim for dim in input_shape if not (isinstance(dim, int) and dim == 1)
            )

        squeeze_node = helper.make_node(
            "Squeeze",
            inputs=squeeze_inputs,
            outputs=[output_name],
            name=s.get_unique_name("squeeze"),
        )
        s.add_node(squeeze_node)
        s.add_shape_info(output_name, output_shape)
