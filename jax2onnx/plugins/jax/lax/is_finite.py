# jax2onnx/plugins/jax/lax/is_finite.py

from typing import Any

import jax
import numpy as np
from jax import core

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.is_finite_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.is_finite.html",
    onnx=[
        {
            "component": "IsInf",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsInf.html",
        },
        {
            "component": "IsNaN",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsNaN.html",
        },
        {"component": "Or", "doc": "https://onnx.ai/onnx/operators/onnx__Or.html"},
        {"component": "Not", "doc": "https://onnx.ai/onnx/operators/onnx__Not.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="is_finite",
    testcases=[
        {
            "testcase": "is_finite_vec",
            "callable": lambda x: jax.lax.is_finite(x),
            "input_values": [
                np.array([-np.inf, -1.0, 0.0, np.inf, np.nan], dtype=np.float32)
            ],
            "expected_output_shapes": [(5,)],
            "expected_output_dtypes": [np.bool_],
            "post_check_onnx_graph": EG(
                ["IsInf:5 -> Or:5 -> Not:5", "IsNaN:5 -> Or:5 -> Not:5"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class IsFinitePlugin(PrimitiveLeafPlugin):
    """Lower ``lax.is_finite`` to ONNX via IsInf + IsNaN + Or + Not."""

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("is_finite_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("is_finite_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "is_finite_out"
        )
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("is_finite_out")

        is_inf_val = ctx.builder.IsInf(
            x_val,
            _outputs=[ctx.fresh_name("is_finite_is_inf")],
        )
        if getattr(out_spec, "type", None) is not None:
            is_inf_val.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            is_inf_val.shape = out_spec.shape

        is_nan_val = ctx.builder.IsNaN(
            x_val,
            _outputs=[ctx.fresh_name("is_finite_is_nan")],
        )
        if getattr(out_spec, "type", None) is not None:
            is_nan_val.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            is_nan_val.shape = out_spec.shape

        any_non_finite = ctx.builder.Or(
            is_inf_val,
            is_nan_val,
            _outputs=[ctx.fresh_name("is_finite_or")],
        )
        if getattr(out_spec, "type", None) is not None:
            any_non_finite.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            any_non_finite.shape = out_spec.shape

        result = ctx.builder.Not(any_non_finite, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
