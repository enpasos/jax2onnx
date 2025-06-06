# file: jax2onnx/plugins/jax/lax/sort.py
from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.sort_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.sort.html",
    onnx=[
        {
            "component": "TopK",
            "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="sort",
    testcases=[
        {
            "testcase": "sort_1d",
            "callable": lambda x: jax.lax.sort(x),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "sort_2d",
            "callable": lambda x: jax.lax.sort(x, dimension=0),
            "input_shapes": [(3, 4)],
        },
    ],
)
class SortPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.sort to ONNX TopK."""

    def to_onnx(
        self,
        conv: "Jaxpr2OnnxConverter",
        invars: List,
        outvars: List,
        params: Dict[str, Any],
    ):
        """Handle JAX sort primitive."""
        x_var = invars[0]
        x_name = conv.get_name(x_var)
        x_aval = x_var.aval

        # Get the sorting dimension. Defaults to the last dimension.
        axis = params.get("dimension", -1)
        if axis < 0:
            axis += len(x_aval.shape)

        # Get the size of the dimension to sort along, which is the 'k' for TopK.
        # Create a 'Shape' node to get the shape of the input tensor.
        shape_name = conv.get_unique_name("shape_of")
        conv.add_node(helper.make_node("Shape", inputs=[x_name], outputs=[shape_name]))
        conv.add_shape_info(shape_name, (len(x_aval.shape),), np.int64)

        # Create a 'Gather' node to extract the size of the sorting dimension.
        axis_const_name = conv.builder.get_constant_name(np.array(axis, dtype=np.int64))
        k_scalar = conv.get_unique_name("dim_size")
        conv.add_node(
            helper.make_node(
                "Gather",
                inputs=[shape_name, axis_const_name],
                outputs=[k_scalar],
                axis=0,
            )
        )
        conv.add_shape_info(k_scalar, (), np.int64)

        # The 'k' input to TopK must be a 1D tensor. Unsqueeze the scalar to make it 1D.
        axes_const_name = conv.builder.get_constant_name(np.array([0], dtype=np.int64))
        k_name = conv.get_unique_name("k_unsqueezed")
        conv.add_node(
            helper.make_node(
                "Unsqueeze", inputs=[k_scalar, axes_const_name], outputs=[k_name]
            )
        )
        conv.add_shape_info(k_name, (1,), np.int64)

        # Create the TopK node for sorting.
        values_out_name = conv.get_name(outvars[0])
        indices_tmp_name = conv.get_unique_name("topk_indices")
        conv.add_node(
            helper.make_node(
                "TopK",
                inputs=[x_name, k_name],
                outputs=[values_out_name, indices_tmp_name],
                axis=axis,
                largest=0,  # jax.lax.sort is ascending, so we want the smallest values.
                sorted=1,  # jax.lax.sort returns sorted elements.
            )
        )

        # Add shape info for the outputs of the TopK node.
        out_shape = tuple(conv._dim_to_symbol_safe(d) for d in x_aval.shape)
        out_dtype = conv._ensure_onnx_dtype(x_aval.dtype)
        conv.add_shape_info(values_out_name, out_shape, out_dtype)
        conv.add_shape_info(indices_tmp_name, out_shape, np.int64)
