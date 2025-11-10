# jax2onnx/plugins/jax/lax/cos.py

from typing import Any

from jax import core
import jax
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.cos_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cos.html",
    onnx=[
        {
            "component": "Cos",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cos.html",
        }
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="cos",
    testcases=[
        {
            "testcase": "cos",
            "callable": lambda x: jax.lax.cos(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Cos:3", "Add:3 -> Sin:3"],
                mode="any",
                no_unused_inputs=True,
            ),
        }
    ],
)
class CosPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("cos_in"))

        x_dtype = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        if x_dtype == np.float64:
            # ONNX runtime lacks a double kernel for Cos; use sin(x + pi/2) instead.
            pi_over_two = ctx.bind_const_for_var(
                object(), np.asarray(np.pi / 2, dtype=np.float64)
            )

            shifted_name = ctx.fresh_name("cos_shifted")
            result_add = ctx.builder.Add(x_val, pi_over_two, _outputs=[shifted_name])
            result_add.type = x_val.type
            result_add.shape = x_val.shape

            sin_out = ctx.builder.Sin(
                result_add, _outputs=[ctx.fresh_name("cos_via_sin")]
            )
            sin_out.type = x_val.type
            sin_out.shape = x_val.shape
            _stamp_type_and_shape(sin_out, getattr(x_var.aval, "shape", ()))
            ctx.bind_value_for_var(out_var, sin_out)
        else:
            out_spec = ctx.get_value_for_var(
                out_var, name_hint=ctx.fresh_name("cos_out")
            )
            result = ctx.builder.Cos(x_val, _outputs=[out_spec.name])
            result.type = out_spec.type
            result.shape = out_spec.shape
            ctx.bind_value_for_var(out_var, result)
