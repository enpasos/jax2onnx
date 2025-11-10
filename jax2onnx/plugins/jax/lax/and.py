# jax2onnx/plugins/jax/lax/and.py

from typing import Any

from jax import core
import jax
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.and_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.html#jax.lax.bitwise_and",
    onnx=[
        {
            "component": "And",
            "doc": "https://onnx.ai/onnx/operators/onnx__And.html",
        },
        {
            "component": "BitwiseAnd",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitwiseAnd.html",
        },
    ],
    since="v0.6.5",
    context="primitives.lax",
    component="and",
    testcases=[
        {
            "testcase": "and_bool",
            "callable": lambda x, y: jax.lax.bitwise_and(x, y),
            "input_values": [
                np.array([True, True, False, False], dtype=np.bool_),
                np.array([True, False, True, False], dtype=np.bool_),
            ],
            "post_check_onnx_graph": EG(
                ["And:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "and_int",
            "callable": lambda x, y: jax.lax.bitwise_and(x, y),
            "input_values": [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([3, 1, 2], dtype=np.int32),
            ],
            "post_check_onnx_graph": EG(
                ["BitwiseAnd:3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class AndPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("and_lhs"))
        prefer_dtype = np.dtype(getattr(lhs_var.aval, "dtype", np.bool_))
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("and_rhs"),
            prefer_np_dtype=prefer_dtype,
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("and_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("and_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("and_out")

        is_bool = np.issubdtype(prefer_dtype, np.bool_)
        op_type = "And" if is_bool else "BitwiseAnd"
        builder_fn = getattr(ctx.builder, op_type)

        result = builder_fn(lhs_val, rhs_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
