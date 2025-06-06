# file: jax2onnx/plugins/jax/numpy/sort.py


from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from onnx import TensorProto, helper
from jax import lax
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# --------------------------------------------------------------------------- #
# Primitive registration                                                      #
# --------------------------------------------------------------------------- #

sort_p = lax.sort_p  # the underlying JAX primitive


@register_primitive(
    jaxpr_primitive=sort_p.name,
    context="primitives.jnp",
    component="sort",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sort.html",
    onnx=[
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"}
    ],
    since="v0.6.1",
    testcases=[
        # 1-D
        {
            "testcase": "sort_1d",
            "callable": lambda x: lax.sort(x, dimension=-1),
            "input_shapes": [(7,)],
            "input_dtypes": [np.float32],
            "expected_output_shapes": [(7,)],
        },
        # 2-D along axis 0
        {
            "testcase": "sort_2d_axis0",
            "callable": lambda x: lax.sort(x, dimension=0),
            "input_shapes": [("B", 4)],
            "input_dtypes": [np.float32],
            "expected_output_shapes": [("B", 4)],
        },
    ],
)
class SortPlugin(PrimitiveLeafPlugin):
    """Lower `lax.sort` (single-tensor flavour) to ONNX TopK."""

    primitive = sort_p  # for clarity

    # --------------------------------------------------------------------- #
    # Public entry point called by the converter                            #
    # --------------------------------------------------------------------- #
    def to_onnx(
        self,
        conv: Jaxpr2OnnxConverter,
        invars: List,  # one input tensor
        outvars: List,
        params: Dict[str, Any],  # contains the target axis
    ):
        x_var = invars[0]
        x_name = conv.get_name(x_var)

        axis: int = params.get("dimension", -1)
        if axis < 0:  # normalise negative axis w.r.t. rank
            axis += len(x_var.aval.shape)

        # -------------------------------------------------------------- #
        # Create an ONNX scalar tensor `k = input.shape[axis]`           #
        # -------------------------------------------------------------- #
        shape_name = conv.builder.get_unique_name("shape_of")
        gather_name = conv.builder.get_unique_name("gather_dim")
        k_name = conv.builder.get_unique_name("k_unsqueezed")

        # Shape(x)
        conv.builder.add_node(
            helper.make_node(
                "Shape",
                inputs=[x_name],
                outputs=[shape_name],
                name=conv.get_unique_name("Shape"),
            )
        )
        # constant(index)
        axis_const = np.array([axis], dtype=np.int64)
        axis_const_name = conv.builder.get_unique_name("axis_const")
        conv.builder.add_initializer(
            axis_const_name, axis_const.shape, TensorProto.INT64, axis_const
        )
        # Gather(shape, axis)
        conv.builder.add_node(
            helper.make_node(
                "Gather",
                inputs=[shape_name, axis_const_name],
                outputs=[gather_name],
                axis=0,
                name=conv.get_unique_name("Gather"),
            )
        )
        # Unsqueeze → TopK expects k to be a tensor of rank 1
        unsq_axes_const = np.array([0], dtype=np.int64)
        unsq_axes_name = conv.builder.get_unique_name("unsq_axes")
        conv.builder.add_initializer(
            unsq_axes_name, unsq_axes_const.shape, TensorProto.INT64, unsq_axes_const
        )
        conv.builder.add_node(
            helper.make_node(
                "Unsqueeze",
                inputs=[gather_name, unsq_axes_name],
                outputs=[k_name],
                name=conv.get_unique_name("Unsqueeze"),
            )
        )

        # -------------------------------------------------------------- #
        # TopK(x, k)   (largest = 0 → smallest; sorted = 1)             #
        # -------------------------------------------------------------- #
        values_name = conv.get_name(outvars[0])  # the plugin reserves the final name
        indices_dummy = conv.builder.get_unique_name("topk_indices")

        topk_node = helper.make_node(
            "TopK",
            inputs=[x_name, k_name],
            outputs=[values_name, indices_dummy],
            axis=axis,
            largest=0,  # 0 == take the smallest values
            sorted=1,  # ensure ascending order
            name=conv.get_unique_name("TopK_sort"),
        )
        conv.builder.add_node(topk_node)

        # Register output’s shape & dtype
        out_shape = tuple(conv._dim_to_symbol_safe(d) for d in x_var.aval.shape)
        out_dtype = conv._ensure_onnx_dtype(x_var.aval.dtype)
        conv.builder.add_output(values_name, out_shape, out_dtype)
