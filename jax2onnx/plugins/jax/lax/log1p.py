# jax2onnx/plugins/jax/lax/log1p.py

from typing import Any

from jax import core
import jax
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._axis0_utils import _np_dtype_for_enum
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.log1p_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.log1p.html",
    onnx=[
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
    ],
    since="v0.11.0",
    context="primitives.lax",
    component="log1p",
    testcases=[
        {
            "testcase": "log1p",
            "callable": lambda x: jax.lax.log1p(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Add:3 -> Log:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class Log1pPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.log1p`` to ONNX via Add + Log."""

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("log1p_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("log1p_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("log1p_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("log1p_out")

        dtype_enum = getattr(getattr(x_val, "type", None), "dtype", None)
        np_dtype = _np_dtype_for_enum(dtype_enum)
        if np_dtype is None:
            aval = getattr(x_var, "aval", None)
            np_dtype = np.dtype(getattr(aval, "dtype", np.float32))

        one_scalar = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))

        sum_val = ctx.builder.Add(
            x_val,
            one_scalar,
            _outputs=[ctx.fresh_name("log1p_add")],
        )
        if getattr(x_val, "type", None) is not None:
            sum_val.type = x_val.type
        elif getattr(out_spec, "type", None) is not None:
            sum_val.type = out_spec.type
        if getattr(x_val, "shape", None) is not None:
            sum_val.shape = x_val.shape
        elif getattr(out_spec, "shape", None) is not None:
            sum_val.shape = out_spec.shape

        result = ctx.builder.Log(sum_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif getattr(sum_val, "type", None) is not None:
            result.type = sum_val.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        elif getattr(sum_val, "shape", None) is not None:
            result.shape = sum_val.shape

        ctx.bind_value_for_var(out_var, result)
