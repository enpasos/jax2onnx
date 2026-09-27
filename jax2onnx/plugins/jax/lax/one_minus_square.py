# jax2onnx/plugins/jax/lax/one_minus_square.py

"""Lower JAX's precision-preserving ``one_minus_square`` primitive."""

from __future__ import annotations

import jax
import numpy as np

from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _one_minus_square(x: jax.Array) -> jax.Array:
    if hasattr(jax.lax, "one_minus_square"):
        return jax.lax.one_minus_square(x)
    return (1 + x) * (1 - x)


@register_primitive(
    jaxpr_primitive=getattr(getattr(jax.lax, "one_minus_square_p", None), "name", ""),
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.one_minus_square.html",
    onnx=[
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {
            "component": "CastLike",
            "doc": "https://onnx.ai/onnx/operators/onnx__CastLike.html",
        },
    ],
    since="0.17.0",
    context="primitives.lax",
    component="one_minus_square",
    testcases=[
        {
            "testcase": "one_minus_square_near_one",
            "callable": _one_minus_square,
            "input_values": [
                np.array([-0.99999994, -0.5, 0.0, 0.5, 0.99999994], dtype=np.float32)
            ],
            "post_check_onnx_graph": EG(
                ["Add:5 -> Mul:5", "Sub:5 -> Mul:5"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class OneMinusSquarePlugin(PrimitiveLeafPlugin):
    """Compute ``1 - x²`` as ``(1 + x) * (1 - x)`` near ``|x| = 1``."""

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        np_dtype = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        if np.issubdtype(np_dtype, np.complexfloating):
            raise NotImplementedError(
                "Complex one_minus_square requires packed-complex arithmetic."
            )

        x_val = ctx.get_value_for_var(
            x_var, name_hint=ctx.fresh_name("one_minus_square_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("one_minus_square_out")
        )
        output_name = out_spec.name or ctx.fresh_name("one_minus_square_out")
        if out_spec.producer() is not None:
            output_name = ctx.fresh_name("one_minus_square_out")

        one = ctx.bind_const_for_var(object(), np.asarray(1, dtype=np_dtype))
        # Loop graph processing may promote constants while retaining float32 inputs.
        one = ctx.cast_like(one, x_val, name_hint="one_minus_square_one")
        plus = ctx.builder.Add(one, x_val, _outputs=[ctx.fresh_name("one_plus_x")])
        minus = ctx.builder.Sub(one, x_val, _outputs=[ctx.fresh_name("one_minus_x")])
        result = ctx.builder.Mul(plus, minus, _outputs=[output_name])

        output_type = out_spec.type or x_val.type
        output_shape = tuple(getattr(out_var.aval, "shape", ()))
        for value in (plus, minus, result):
            if output_type is not None:
                value.type = output_type
            _stamp_type_and_shape(value, output_shape)
            _ensure_value_metadata(ctx, value)

        ctx.bind_value_for_var(out_var, result)
