from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.concatenate_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="concatenate",
    testcases=[
        {
            "testcase": "concatenate",
            "callable": lambda a, b: jax.lax.concatenate(
                (a, b), dimension=0
            ),  # Corrected callable
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "concatenate_axis1",
            "callable": lambda a, b: jax.lax.concatenate(
                (a, b), dimension=1
            ),  # Corrected callable
            "input_shapes": [("B", 3), ("B", 4)],
        },
        {
            "testcase": "concatenate_axis0",
            "callable": lambda a, b: jax.lax.concatenate(
                (a, b), dimension=0
            ),  # Corrected callable
            "input_shapes": [(7, 3), (4, 3)],
        },
        {  # 3D inputs
            "testcase": "concatenate_3d",
            "callable": lambda a, b: jax.lax.concatenate(
                (a, b), dimension=1
            ),  # Corrected callable
            "input_shapes": [(2, 3, 4), (2, 5, 4)],
        },
    ],
)
class ConcatenatePlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.concatenate to ONNX.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX concatenate primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_name(node_outputs[0])
        dimension = params["dimension"]

        # Calculate and propagate shape information
        # Defensive: always use .aval.shape (JAX tracing) or .shape (ShapeDtypeStruct), fallback to tuple(inp) for raw tuples
        input_shapes = []
        for inp in node_inputs:
            if hasattr(inp, "aval") and hasattr(inp.aval, "shape"):
                input_shapes.append(inp.aval.shape)
            elif hasattr(inp, "shape"):
                input_shapes.append(inp.shape)
            elif isinstance(inp, (tuple, list)):
                input_shapes.append(tuple(inp))
            else:
                raise ValueError(f"Input to concatenate has no shape: {inp}")
        output_shape = list(input_shapes[0])  # Start with the shape of the first input
        # Normalize the axis
        rank = len(output_shape)
        dimension = dimension % rank if dimension < 0 else dimension

        for shape in input_shapes[1:]:
            if len(shape) != rank:
                raise ValueError("Inputs must have the same rank.")
            for i in range(rank):
                if i != dimension:
                    if output_shape[i] != shape[i] and not (
                        isinstance(output_shape[i], str) or isinstance(shape[i], str)
                    ):
                        raise ValueError(
                            f"Shapes are incompatible along dimension {i} for Concatenate: {output_shape} vs {shape}"
                        )
                    else:
                        if isinstance(shape[i], str):
                            output_shape[i] = shape[i]

        # Accumulate on the concat axis
        for shape in input_shapes[1:]:
            output_shape[dimension] = (
                output_shape[dimension] + shape[dimension]
                if isinstance(output_shape[dimension], int)
                and isinstance(shape[dimension], int)
                else f"{output_shape[dimension]}+{shape[dimension]}"  # Keep strings
            )

        node = helper.make_node(
            "Concat",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("concat"),
            axis=dimension,
        )
        s.add_node(node)
        s.add_shape_info(output_name, tuple(output_shape))
