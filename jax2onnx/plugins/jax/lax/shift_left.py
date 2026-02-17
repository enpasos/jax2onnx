# jax2onnx/plugins/jax/lax/shift_left.py

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from jax import core

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.shift_left_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.shift_left.html",
    onnx=[
        {
            "component": "BitShift",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitShift.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="shift_left",
    testcases=[
        {
            "testcase": "shift_left_vec",
            "callable": lambda x, s: jax.lax.shift_left(x, s),
            "input_values": [
                np.array([1, 2, 3, 4], dtype=np.uint32),
                np.array([1, 2, 3, 4], dtype=np.uint32),
            ],
            "expected_output_shapes": [(4,)],
            "expected_output_dtypes": [np.uint32],
            "post_check_onnx_graph": EG(
                ["BitShift:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "shift_left_scalar",
            "callable": lambda x, s: jax.lax.shift_left(x, s),
            "input_values": [
                np.array(3, dtype=np.uint32),
                np.array(5, dtype=np.uint32),
            ],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.uint32],
            "post_check_onnx_graph": EG(
                ["BitShift"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ShiftLeftPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.shift_left`` to ONNX ``BitShift(direction='LEFT')``."""

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("shl_input"))
        rhs_val = ctx.get_value_for_var(
            rhs_var, name_hint=ctx.fresh_name("shl_shift"), prefer_np_dtype=None
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("shl_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("shl_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("shl_out")

        result = ctx.builder.BitShift(
            lhs_val,
            rhs_val,
            direction="LEFT",
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
