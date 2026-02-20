# jax2onnx/plugins/jax/lax/exp2.py

from typing import Any

import jax
import numpy as np
from jax import core

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._axis0_utils import _np_dtype_for_enum
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.exp2_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.exp2.html",
    onnx=[{"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"}],
    since="0.12.1",
    context="primitives.lax",
    component="exp2",
    testcases=[
        {
            "testcase": "exp2",
            "callable": lambda x: jax.lax.exp2(x),
            "input_values": [np.array([-2.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Pow:5"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class Exp2Plugin(PrimitiveLeafPlugin):
    """Lower ``lax.exp2`` to ONNX ``Pow`` as ``2^x``."""

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("exp2_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("exp2_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("exp2_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("exp2_out")

        dtype_enum = getattr(getattr(x_val, "type", None), "dtype", None)
        np_dtype = _np_dtype_for_enum(dtype_enum)
        if np_dtype is None:
            aval = getattr(x_var, "aval", None)
            np_dtype = np.dtype(getattr(aval, "dtype", np.float32))

        two_scalar = ctx.bind_const_for_var(object(), np.asarray(2.0, dtype=np_dtype))

        result = ctx.builder.Pow(two_scalar, x_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif getattr(x_val, "type", None) is not None:
            result.type = x_val.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        elif getattr(x_val, "shape", None) is not None:
            result.shape = x_val.shape
        ctx.bind_value_for_var(out_var, result)
