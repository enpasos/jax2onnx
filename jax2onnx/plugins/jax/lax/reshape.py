from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.reshape_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reshape.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="reshape",
    testcases=[
        {
            "testcase": "reshape",
            "callable": lambda x: jax.lax.reshape(x, (9,)),
            "input_shapes": [(3, 3)],
        }
    ],
)
class ReshapePlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.reshape to ONNX Reshape."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX reshape primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        new_shape = params["new_sizes"]
        input_shape = node_inputs[0].aval.shape
        dtype = node_inputs[0].aval.dtype

        def _process_newshape(newshape):
            if isinstance(newshape, (int, str)):
                newshape = [newshape]
            else:
                newshape = list(newshape)
            neg_one_count = sum(1 for dim in newshape if dim == -1)
            if neg_one_count > 1:
                raise ValueError("Only one dimension can be -1 (inferred).")
            return newshape

        def _concretize_shape(shape, concrete_value=2):
            return tuple(
                concrete_value if isinstance(dim, str) else dim for dim in shape
            )

        processed_newshape = _process_newshape(new_shape)
        concrete_shape = _concretize_shape(processed_newshape)

        if len(new_shape) == 2 and new_shape[0] == 1 and input_shape == (new_shape[1],):
            s.var_to_name[node_outputs[0]] = input_name
            return

        # ✅ FIX HERE: Use get_constant_name to reliably register metadata
        shape_name = s.builder.get_constant_name(
            np.array(concrete_shape, dtype=np.int64)
        )

        node = helper.make_node(
            "Reshape",
            inputs=[input_name, shape_name],
            outputs=[output_name],
            name=s.get_unique_name("reshape"),
        )
        s.add_node(node)

        s.builder.register_value_info_metadata(
            output_name, shape=tuple(processed_newshape), dtype=dtype
        )
