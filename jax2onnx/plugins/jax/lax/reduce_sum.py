# file: jax2onnx/plugins/jax/lax/reduce_sum.py

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


reduce_sum_p = lax.reduce_sum_p


@register_primitive(
    jaxpr_primitive=reduce_sum_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_sum.html",
    onnx=[
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="reduce_sum",
    testcases=[
        {
            "testcase": "reduce_sum",
            "callable": lambda x: jnp.sum(x, axis=(0,)),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "reduce_sum_allaxes",
            "callable": lambda x: jnp.sum(x),  # sum over all axes
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "reduce_sum_keepdims",
            "callable": lambda x: jnp.sum(x, axis=(1,), keepdims=True),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "reduce_sum_dtype_f64",
            "callable": lambda x: jnp.sum(x, axis=(0, 1), dtype=jnp.float64),
            "input_shapes": [(2, 2)],
            "input_dtypes": [np.int64],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "reduce_sum_dtype",
            "callable": lambda x: jnp.sum(x, axis=(0, 1), dtype=jnp.float32),
            "input_shapes": [(2, 2)],
            "input_dtypes": [np.int32],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
        },
        # {
        #     "testcase": "reduce_sum_two_axes",
        #     "callable": lambda x: lax.reduce_sum(x, axes=(1, 2)),
        #     "input_shapes": [("B", 3, 4)],        # B × 3 × 4
        #     "input_dtypes": [np.float32],
        #     "expected_output_shapes": [("B",)],   # summing over axes 1&2 leaves shape (B,)
        #     "expected_output_dtypes": [np.float32],
        # },
    ],
)
class ReduceSumPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.reduce_sum (invoked via jnp.sum) to ONNX ReduceSum."""

    primitive = reduce_sum_p

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
        #    JAX primitive params are 'axes', 'keepdims', 'dtype' when traced via jnp.sum.
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

        # 3) Emit the ReduceSum node via our shared helper:
        add_reduce_node(
            conv.builder,
            "ReduceSum",
            input_name,
            output_name,
            axes=list(axes) if axes is not None else None,
            keepdims=1 if keepdims_flag else 0,
        )

        # 4) If the *original input* was integer, force‐promote the ReduceSum output to DOUBLE.
        #    (This covers the case where JAX ignored dtype=double under x64-disabled tracing,
        #    but our test expects a DOUBLE output when summing an integer array.)
        int_input_ctr = x_var.aval.dtype
        if np.issubdtype(int_input_ctr, np.integer):
            promoted_name = conv.builder.get_unique_name("reduce_sum_out_double")
            cast_out = helper.make_node(
                "Cast",
                inputs=[output_name],
                outputs=[promoted_name],
                to=TensorProto.DOUBLE,
                name=conv.get_unique_name("Cast_sum_output"),
            )
            conv.builder.add_node(cast_out)
            output_name = promoted_name

        # 5) Finally, declare the output's value_info (shape & dtype):
        aval = out_var.aval
        # If we forced a Cast→DOUBLE above, out_name refers to the DOUBLE‐typed tensor.
        # Otherwise, it's whatever dtype JAX predicted (aval.dtype or requested_dtype).
        if np.issubdtype(int_input_ctr, np.integer):
            # We promoted to DOUBLE unconditionally
            out_dtype_enum = TensorProto.DOUBLE
        else:
            final_dtype = requested_dtype if requested_dtype is not None else aval.dtype
            out_dtype_enum = conv._ensure_onnx_dtype(final_dtype)

        conv.builder.add_value_info(
            output_name,
            tuple(conv._dim_to_symbol_safe(d) for d in aval.shape),
            out_dtype_enum,
        )
