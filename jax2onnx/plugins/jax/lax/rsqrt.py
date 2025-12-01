# jax2onnx/plugins/jax/lax/rsqrt.py

"""Lower JAX's ``lax.rsqrt`` primitive to ONNX."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._axis0_utils import _np_dtype_for_enum
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.rsqrt_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.rsqrt.html",
    onnx=[
        {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="v0.10.2",
    context="primitives.lax",
    component="rsqrt",
    testcases=[
        {
            "testcase": "rsqrt",
            "callable": lambda x: jax.lax.rsqrt(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Sqrt -> Div"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class RsqrtPlugin(PrimitiveLeafPlugin):
    def lower(
        self, ctx: "IRContext", eqn
    ) -> None:  # pragma: no cover - exercised via tests
        builder = ctx.builder

        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("rsqrt_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("rsqrt_out"))

        sqrt_val = builder.Sqrt(
            x_val,
            _outputs=[ctx.fresh_name("rsqrt_sqrt")],
        )

        x_type = getattr(x_val, "type", None)
        x_shape = getattr(x_val, "shape", None)
        if x_type is not None and getattr(x_type, "dtype", None) is not None:
            sqrt_val.type = ir.TensorType(x_type.dtype)
        if x_shape is not None:
            sqrt_val.shape = x_shape

        dtype_enum = getattr(getattr(sqrt_val, "type", None), "dtype", None)
        np_dtype = _np_dtype_for_enum(dtype_enum) or np.float32
        one_scalar = ctx.bind_const_for_var(
            object(),
            np.asarray(1.0, dtype=np_dtype),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("rsqrt_out")
        result = builder.Div(
            one_scalar,
            sqrt_val,
            _outputs=[desired_name],
        )

        out_type = getattr(out_spec, "type", None)
        out_shape = getattr(out_spec, "shape", None)
        if out_type is not None:
            result.type = out_type
        elif x_type is not None:
            result.type = x_type
        if out_shape is not None:
            result.shape = out_shape
        elif x_shape is not None:
            result.shape = x_shape

        ctx.bind_value_for_var(out_var, result)
