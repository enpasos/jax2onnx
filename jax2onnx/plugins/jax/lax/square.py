# jax2onnx/plugins/jax/lax/square.py

from typing import Any

from jax import core
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._axis0_utils import ensure_axis0_extent, _axis0_debug
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._loop_extent_meta import (
    get_axis0_override,
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.square_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.square.html",
    onnx=[
        {
            "component": "Mul",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="square",
    testcases=[
        {
            "testcase": "square",
            "callable": lambda x: jax.lax.square(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Mul:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class SquarePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("square_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("square_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("square_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("square_out")

        x_override = get_axis0_override(x_val)
        spec_override = get_axis0_override(out_spec)
        ctx_override = getattr(ctx, "_static_loop_extent_axis0", None)
        override_candidates = [
            int(candidate)
            for candidate in (x_override, spec_override, ctx_override)
            if isinstance(candidate, int) and candidate > 1
        ]
        axis0_override = max(override_candidates, default=None)
        _axis0_debug(
            "square override resolution "
            f"value={desired_name} "
            f"x={getattr(x_val, 'name', None)} "
            f"sources={(x_override, spec_override, ctx_override)} "
            f"candidates={override_candidates} "
            f"selected={axis0_override}"
        )
        x_val = ensure_axis0_extent(ctx, x_val, axis0_override)

        result = ctx.builder.Mul(x_val, x_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        result = ensure_axis0_extent(ctx, result, axis0_override, reference=x_val)

        target_shape = tuple(getattr(out_var.aval, "shape", ()))
        if axis0_override is not None and target_shape:
            target_shape = (axis0_override,) + target_shape[1:]
        if target_shape:
            _stamp_type_and_shape(result, target_shape)
        _ensure_value_metadata(ctx, result)
        propagate_axis0_override(x_val, result)
        if axis0_override is not None:
            set_axis0_override(result, axis0_override)
        ctx.bind_value_for_var(out_var, result)
