# jax2onnx/plugins/jax/lax/add_any.py

from __future__ import annotations

from typing import cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._axis0_utils import (
    maybe_expand_binary_axis0,
    stamp_axis0_binary_result,
)
from jax2onnx.plugins._loop_extent_meta import (
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.add import AddPlugin, lower_add
from jax2onnx.plugins.plugin_system import register_primitive


@register_primitive(
    jaxpr_primitive="add_any",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.add.html",
    onnx=[
        {"component": "Sum", "doc": "https://onnx.ai/onnx/operators/onnx__Sum.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    ],
    since="0.8.0",
    context="primitives.lax",
    component="add_any",
    testcases=[
        {
            "testcase": "add_any_via_jvp_on_mul",
            "callable": lambda x1, x2: jax.jvp(lambda a, b: a * b, (x1, x2), (x1, x2))[
                1
            ],
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Mul:3 -> Sum:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class AddAnyPlugin(AddPlugin):
    """Alias for JAX's internal ``add_any`` primitive."""

    def lower(self, ctx, eqn):
        x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        prefer_dt = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        a_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("add_any_lhs"))
        b_val = ctx.get_value_for_var(
            y_var,
            name_hint=ctx.fresh_name("add_any_rhs"),
            prefer_np_dtype=prefer_dt,
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("add_any_out")
        )
        a_val, b_val, override = maybe_expand_binary_axis0(
            ctx, a_val, b_val, out_spec, out_var
        )
        output_name = out_spec.name or ctx.fresh_name("add_any_out")
        out_spec.name = output_name

        out_dtype = getattr(getattr(out_var, "aval", None), "dtype", None)
        if out_dtype is not None and np.issubdtype(
            np.dtype(out_dtype), np.complexfloating
        ):
            # Keep Add for packed-complex lowering compatibility.
            lower_add(ctx, eqn)
            return

        result = cast(
            ir.Value,
            ctx.builder.Sum(a_val, b_val, _outputs=[output_name]),
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        stamp_axis0_binary_result(result, out_var, out_spec, override)
        if override is not None:
            set_axis0_override(result, override)
        propagate_axis0_override(a_val, result)
        propagate_axis0_override(b_val, result)
        ctx.bind_value_for_var(out_var, result)
