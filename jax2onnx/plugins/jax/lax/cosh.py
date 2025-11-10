# jax2onnx/plugins/jax/lax/cosh.py

from typing import Any

from jax import core
import jax
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.cosh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cosh.html",
    onnx=[
        {
            "component": "Cosh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cosh.html",
        }
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="cosh",
    testcases=[
        {
            "testcase": "cosh",
            "callable": lambda x: jax.lax.cosh(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                [
                    "Cosh:3",
                    {
                        "path": "Exp:3 -> Add:3 -> Mul:3",
                        "inputs": {1: {"const": 0.5}},
                    },
                ],
                mode="any",
                no_unused_inputs=True,
            ),
        }
    ],
)
class CoshPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("cosh_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("cosh_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("cosh_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("cosh_out")

        x_dtype = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        if x_dtype == np.float32:
            result = ctx.builder.Cosh(x_val, _outputs=[desired_name])
            if getattr(out_spec, "type", None) is not None:
                result.type = out_spec.type
            if getattr(out_spec, "shape", None) is not None:
                result.shape = out_spec.shape
            ctx.bind_value_for_var(out_var, result)
            return

        exp_x = ctx.builder.Exp(x_val, _outputs=[ctx.fresh_name("cosh_exp")])
        exp_x.type = x_val.type
        exp_x.shape = x_val.shape

        neg_x = ctx.builder.Neg(x_val, _outputs=[ctx.fresh_name("cosh_neg")])
        neg_x.type = x_val.type
        neg_x.shape = x_val.shape

        exp_neg_x = ctx.builder.Exp(neg_x, _outputs=[ctx.fresh_name("cosh_exp_neg")])
        exp_neg_x.type = x_val.type
        exp_neg_x.shape = x_val.shape

        sum_val = ctx.builder.Add(
            exp_x, exp_neg_x, _outputs=[ctx.fresh_name("cosh_sum")]
        )
        sum_val.type = x_val.type
        sum_val.shape = x_val.shape

        half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=x_dtype))

        result = ctx.builder.Mul(sum_val, half, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
