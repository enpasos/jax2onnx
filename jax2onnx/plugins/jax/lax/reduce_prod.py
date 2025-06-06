# file: jax2onnx/plugins/jax/lax/reduce_prod.py

from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import jax.numpy as jnp
import numpy as np
from onnx import helper, TensorProto
from jax import lax

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from ._reduce_utils import add_reduce_node  # shared helper

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


reduce_prod_p = lax.reduce_prod_p


@register_primitive(
    jaxpr_primitive=reduce_prod_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_prod.html",
    onnx=[
        {
            "component": "ReduceProd",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceProd.html",
        }
    ],
    since="v0.6.1",
    context="primitives.lax",
    component="reduce_prod",
    testcases=[
        {
            "testcase": "reduce_prod",
            "callable": lambda x: jnp.prod(x, axis=(0,)),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "reduce_prod_allaxes",
            "callable": lambda x: jnp.prod(x),  # prod over all axes
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "reduce_prod_keepdims",
            "callable": lambda x: jnp.prod(x, axis=(1,), keepdims=True),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "reduce_prod_dtype_f64",
            "callable": lambda x: jnp.prod(x, axis=(0, 1), dtype=jnp.float64),
            "input_shapes": [(2, 2)],
            "input_dtypes": [np.int64],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "reduce_prod_dtype",
            "callable": lambda x: jnp.prod(x, axis=(0, 1), dtype=jnp.float32),
            "input_shapes": [(2, 2)],
            "input_dtypes": [np.int32],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
        },
    ],
)
class ReduceProdPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.reduce_prod (invoked via jnp.prod) to ONNX ReduceProd."""

    primitive = reduce_prod_p

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

        # 2) Extract axes / keepdims / dtype from JAX parameters:
        axes: Optional[Sequence[int]] = params.get("axes", None)
        keepdims_flag: bool = params.get("keepdims", False)

        # If dtype promotion was requested explicitly, insert a Cast→(dtype) first:
        requested_dtype = params.get("dtype", None)
        if requested_dtype is not None:
            onnx_requested = conv._ensure_onnx_dtype(requested_dtype)
            casted_name = conv.builder.get_unique_name("cast_promote")
            cast_node = helper.make_node(
                "Cast",
                inputs=[input_name],
                outputs=[casted_name],
                to=onnx_requested,
                name=conv.get_unique_name("Cast_to_dtype"),
            )
            conv.builder.add_node(cast_node)
            input_name = casted_name

        # 3) Emit the ReduceProd node via our shared helper:
        add_reduce_node(
            conv.builder,
            "ReduceProd",
            input_name,
            output_name,
            axes=list(axes) if axes is not None else None,
            keepdims=1 if keepdims_flag else 0,
        )

        # 4) If the *original input* was integer, force‐promote the ReduceProd output to DOUBLE.
        int_input_dtype = x_var.aval.dtype
        if np.issubdtype(int_input_dtype, np.integer):
            promoted_name = conv.builder.get_unique_name("reduce_prod_out_double")
            cast_out = helper.make_node(
                "Cast",
                inputs=[output_name],
                outputs=[promoted_name],
                to=TensorProto.DOUBLE,
                name=conv.get_unique_name("Cast_prod_output"),
            )
            conv.builder.add_node(cast_out)
            output_name = promoted_name

        # 5) Finally, declare the output's value_info (shape & dtype):
        aval = out_var.aval
        # If we forced a Cast→DOUBLE above, out_name refers to the DOUBLE‐typed tensor.
        # Otherwise, use requested_dtype (if any) or the JAX‐inferred dtype.
        if np.issubdtype(int_input_dtype, np.integer) and requested_dtype is None:
            # We promoted to DOUBLE unconditionally when input was integer and no dtype was requested.
            out_dtype_enum = TensorProto.DOUBLE
        else:
            final_dtype = requested_dtype if requested_dtype is not None else aval.dtype
            out_dtype_enum = conv._ensure_onnx_dtype(final_dtype)

        conv.builder.add_value_info(
            output_name,
            tuple(conv._dim_to_symbol_safe(d) for d in aval.shape),
            out_dtype_enum,
        )
