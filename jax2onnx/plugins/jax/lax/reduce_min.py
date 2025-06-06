# file: jax2onnx/plugins/jax/lax/reduce_min.py

from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import jax.numpy as jnp
from jax import lax

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from ._reduce_utils import add_reduce_node  # shared helper

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


reduce_min_p = lax.reduce_min_p


@register_primitive(
    jaxpr_primitive=reduce_min_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_min.html",
    onnx=[
        {
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="reduce_min",
    testcases=[
        {
            "testcase": "reduce_min",
            "callable": lambda x: jnp.min(x, axis=(0,)),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "reduce_min_allaxes",
            "callable": lambda x: jnp.min(x),  # min over all axes
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "reduce_min_keepdims",
            "callable": lambda x: jnp.min(x, axis=(1,), keepdims=True),
            "input_shapes": [(3, 4)],
        },
    ],
)
class ReduceMinPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.reduce_min (invoked via jnp.min) to ONNX ReduceMin."""

    primitive = reduce_min_p

    def to_onnx(
        self,
        conv: Jaxpr2OnnxConverter,
        node_inputs: List[Any],
        node_outputs: List[Any],
        params: dict[str, Any],
    ):
        # 1) Grab the JAX inputs and target ONNX names:
        x_var = node_inputs[0]
        input_name = conv.get_name(x_var)
        out_var = node_outputs[0]
        output_name = conv.get_name(out_var)

        # 2) Extract axes / keepdims from JAX parameters:
        axes: Optional[Sequence[int]] = params.get("axes", None)
        keepdims_flag: bool = params.get("keepdims", False)

        # 3) Emit the ReduceMin node via our shared helper:
        add_reduce_node(
            conv.builder,
            "ReduceMin",
            input_name,
            output_name,
            axes=list(axes) if axes is not None else None,
            keepdims=1 if keepdims_flag else 0,
        )

        # 4) Declare the output's value_info (shape & dtype):
        aval = out_var.aval
        out_dtype_enum = conv._ensure_onnx_dtype(aval.dtype)
        conv.builder.add_value_info(
            output_name,
            tuple(conv._dim_to_symbol_safe(d) for d in aval.shape),
            out_dtype_enum,
        )
