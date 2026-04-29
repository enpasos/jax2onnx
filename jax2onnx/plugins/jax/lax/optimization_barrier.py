# jax2onnx/plugins/jax/lax/optimization_barrier.py

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive=jax.lax.optimization_barrier_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="optimization_barrier",
    testcases=[
        {
            "testcase": "optimization_barrier_single",
            "callable": lambda x: jax.lax.optimization_barrier(x),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                ["Identity:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "optimization_barrier_tuple",
            "callable": lambda x, y: jax.lax.optimization_barrier((x, y)),
            "input_values": [
                np.asarray([1.0, -2.0, 3.0], dtype=np.float32),
                np.asarray([2, 0, -1], dtype=np.int32),
            ],
        },
    ],
)
class OptimizationBarrierPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.optimization_barrier`` as one-to-one ``Identity`` nodes."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        if len(eqn.invars) != len(eqn.outvars):
            raise ValueError(
                "optimization_barrier expects equal numbers of inputs and outputs"
            )

        for in_var, out_var in zip(eqn.invars, eqn.outvars):
            in_val = ctx.get_value_for_var(
                in_var, name_hint=ctx.fresh_name("optimization_barrier_in")
            )
            out_spec = ctx.get_value_for_var(
                out_var, name_hint=ctx.fresh_name("optimization_barrier_out")
            )
            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
                "optimization_barrier"
            )

            result = cast(
                ir.Value, ctx.builder.Identity(in_val, _outputs=[desired_name])
            )
            if getattr(out_spec, "type", None) is not None:
                result.type = out_spec.type
            elif getattr(in_val, "type", None) is not None:
                result.type = in_val.type
            if getattr(out_spec, "shape", None) is not None:
                result.shape = out_spec.shape
            elif getattr(in_val, "shape", None) is not None:
                result.shape = in_val.shape
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)
