# jax2onnx/plugins/jax/lax/cbrt.py


import jax
import numpy as np
from jax import core

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._axis0_utils import _np_dtype_for_enum
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive=jax.lax.cbrt_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cbrt.html",
    onnx=[
        {"component": "Abs", "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html"},
        {"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"},
        {"component": "Sign", "doc": "https://onnx.ai/onnx/operators/onnx__Sign.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="cbrt",
    testcases=[
        {
            "testcase": "cbrt",
            "callable": lambda x: jax.lax.cbrt(x),
            "input_values": [np.array([-8.0, -1.0, 0.0, 1.0, 27.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Abs:5", "Sign:5", "Pow:5", "Mul:5"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class CbrtPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.cbrt`` to ``sign(x) * pow(abs(x), 1/3)`` in ONNX."""

    def lower(self, ctx: LoweringContextProtocol, eqn: "core.JaxprEqn") -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("cbrt_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("cbrt_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("cbrt_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("cbrt_out")

        dtype_enum = getattr(getattr(x_val, "type", None), "dtype", None)
        np_dtype = _np_dtype_for_enum(dtype_enum)
        if np_dtype is None:
            aval = getattr(x_var, "aval", None)
            np_dtype = np.dtype(getattr(aval, "dtype", np.float32))

        one_third = ctx.bind_const_for_var(
            object(), np.asarray(1.0 / 3.0, dtype=np_dtype)
        )

        abs_val = ctx.builder.Abs(x_val, _outputs=[ctx.fresh_name("cbrt_abs")])
        sign_val = ctx.builder.Sign(x_val, _outputs=[ctx.fresh_name("cbrt_sign")])
        pow_val = ctx.builder.Pow(
            abs_val, one_third, _outputs=[ctx.fresh_name("cbrt_pow")]
        )

        for intermediate in (abs_val, sign_val, pow_val):
            if getattr(x_val, "type", None) is not None:
                intermediate.type = x_val.type
            if getattr(x_val, "shape", None) is not None:
                intermediate.shape = x_val.shape

        result = ctx.builder.Mul(sign_val, pow_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif getattr(x_val, "type", None) is not None:
            result.type = x_val.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        elif getattr(x_val, "shape", None) is not None:
            result.shape = x_val.shape
        ctx.bind_value_for_var(out_var, result)
