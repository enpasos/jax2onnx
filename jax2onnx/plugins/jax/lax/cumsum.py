# file: jax2onnx/plugins/jax/lax/cumsum.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import jax.numpy as jnp
from jax import lax, core
from onnx import helper as onnx_helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.cumsum")


def _cumsum_last_axis_reverse(x):
    # Use last axis; avoids lax’s “negative axis” restriction during tracing.
    return lax.cumsum(x, axis=x.ndim - 1, reverse=True)


@register_primitive(
    jaxpr_primitive=lax.cumsum_p.name,  # primitive name: "cumsum"
    context="primitives.lax",
    component="cumsum",
    since="v0.7.4",
    testcases=[
        {
            "testcase": "cumsum_i32_axis2",
            "callable": lambda x: lax.cumsum(x, axis=2, reverse=False),
            "input_shapes": [(2, 3, 4)],
            "input_dtypes": [jnp.int32],
            "expected_output_shapes": [(2, 3, 4)],
            "expected_output_dtypes": [jnp.int32],
            "post_check_onnx_graph": lambda m: any(
                n.op_type == "CumSum" for n in m.graph.node
            ),
        },
        {
            "testcase": "cumsum_f32_axism1_reverse",
            "callable": _cumsum_last_axis_reverse,  # avoids negative axis at trace time
            "input_shapes": [(1, 2, 3, 4)],
            "input_dtypes": [jnp.float32],
            "expected_output_shapes": [(1, 2, 3, 4)],
            "expected_output_dtypes": [jnp.float32],
            "post_check_onnx_graph": lambda m: any(
                n.op_type == "CumSum"
                and any(a.name == "reverse" and a.i == 1 for a in n.attribute)
                for n in m.graph.node
            ),
        },
    ],
)
class LaxCumsumPlugin(PrimitiveLeafPlugin):
    """
    Lower jax.lax.cumsum to ONNX CumSum.

    ONNX CumSum takes:
      - inputs: data, axis (scalar tensor)
      - attributes: exclusive(bool), reverse(bool)

    We map:
      - axis (normalize negative to non-negative using rank)
      - reverse -> reverse
      - dtype (if provided) via a Cast before CumSum
      - exclusive is always False (inclusive scan)
    """

    @staticmethod
    def abstract_eval(
        x_av: core.ShapedArray,
        *,
        axis: int = 0,
        reverse: bool = False,
        dtype: Any | None = None,
        **kwargs: Any,
    ) -> core.ShapedArray:
        # Shape unchanged; dtype is either the requested one or the input’s.
        out_dtype = dtype if dtype is not None else x_av.dtype
        return core.ShapedArray(x_av.shape, out_dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[core.Var],
        node_outputs: Sequence[core.Var],
        params: dict[str, Any],
    ) -> None:
        builder = s.builder

        x_var = node_inputs[0]
        out_var = node_outputs[0]

        x_name = s.get_name(x_var)
        out_name = s.get_name(out_var)

        # Params from JAX primitive
        axis_param = int(params.get("axis", 0))
        reverse = bool(params.get("reverse", False))
        req_dtype = params.get("dtype", None)

        # Normalize negative axes (defensive; current lax rejects negatives).
        rank = x_var.aval.ndim
        if axis_param < 0:
            axis_param = axis_param % rank

        # If a dtype override is provided, Cast the input first to match JAX semantics.
        x_for_cumsum = x_name
        if req_dtype is not None:
            onnx_dtype = builder._numpy_dtype_to_onnx(jnp.dtype(req_dtype))
            cast_out = builder.get_unique_name("Cast_CumSumInput")
            s.add_node(
                onnx_helper.make_node(
                    "Cast",
                    inputs=[x_name],
                    outputs=[cast_out],
                    to=onnx_dtype,
                    name=cast_out,
                )
            )
            x_for_cumsum = cast_out

        # ONNX wants axis as a scalar tensor (use INT64 for consistency).
        axis_const_name = builder.get_constant_name(
            np.asarray(axis_param, dtype=np.int64)
        )

        node = onnx_helper.make_node(
            "CumSum",
            inputs=[x_for_cumsum, axis_const_name],
            outputs=[out_name],
            name=builder.get_unique_name("CumSum"),
            exclusive=0,
            reverse=1 if reverse else 0,
        )
        s.add_node(node)

        # Output metadata (shape same as input; dtype is aval’s dtype)
        aval_out = out_var.aval
        builder.register_value_info_metadata(
            out_name,
            tuple(
                int(d) if isinstance(d, (int, np.integer)) else d
                for d in aval_out.shape
            ),
            builder._numpy_dtype_to_onnx(aval_out.dtype),
        )
