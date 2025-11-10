# jax2onnx/plugins/jax/lax/gt.py

from typing import Any

from jax import core
import jax
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.gt_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.gt.html",
    onnx=[
        {
            "component": "Greater",
            "doc": "https://onnx.ai/onnx/operators/onnx__Greater.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="gt",
    testcases=[
        {
            "testcase": "gt",
            "callable": lambda x1, x2: jax.lax.gt(x1, x2),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Greater:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class GtPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("gt_lhs"))
        prefer_dtype = np.dtype(getattr(lhs_var.aval, "dtype", np.float32))
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("gt_rhs"),
            prefer_np_dtype=prefer_dtype,
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("gt_out"))

        lhs_dtype_enum = getattr(getattr(lhs_val, "type", None), "dtype", None)
        rhs_dtype_enum = getattr(getattr(rhs_val, "type", None), "dtype", None)
        if (
            lhs_dtype_enum is not None
            and rhs_dtype_enum is not None
            and lhs_dtype_enum != rhs_dtype_enum
        ):
            rhs_val = ctx.cast_like(rhs_val, lhs_val, name_hint="gt_rhs")

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("gt_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("gt_out")

        result = ctx.builder.Greater(lhs_val, rhs_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
