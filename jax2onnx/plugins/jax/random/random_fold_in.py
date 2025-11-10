# jax2onnx/plugins/jax/random/random_fold_in.py

"""Minimal lowering for ``random_fold_in``.

The IR pipeline currently only needs deterministic PRNG plumbing for
modules that never consume random bits (e.g. nnx.Dropout in inference
mode).  We therefore forward the incoming key unchanged, while still
producing an ONNX Value so downstream passes see a concrete tensor.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import jax

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _identity(
    ctx: LoweringContextProtocol, value: ir.Value, name_hint: str
) -> ir.Value:
    result = ctx.builder.Identity(value, _outputs=[ctx.fresh_name(name_hint)])
    if getattr(value, "type", None) is not None:
        result.type = value.type
    if getattr(value, "shape", None) is not None:
        result.shape = value.shape
    return result


@register_primitive(
    jaxpr_primitive="random_fold_in",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.random.fold_in.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.2.0",
    context="primitives.random",
    component="random_fold_in",
    testcases=[
        {
            "testcase": "random_fold_in_passthrough",
            "callable": lambda key, msg: key,
            "input_shapes": [(2,), ()],
            "input_dtypes": [np.uint32, np.uint32],
            "input_values": [
                np.array([0x1234_5678, 0x9ABC_DEF0], dtype=np.uint32),
                np.array(7, dtype=np.uint32),
            ],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.dtype(np.uint32)],
            "post_check_onnx_graph": EG([]),
        }
    ],
)
class RandomFoldInPlugin(PrimitiveLeafPlugin):
    """Forward the incoming key; sufficient for deterministic inference paths."""

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:  # type: ignore[override]
        key_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        key_value = ctx.get_value_for_var(
            key_var, name_hint=ctx.fresh_name("prng_fold_in_in")
        )
        out_value = _identity(ctx, key_value, name_hint="prng_fold_in_out")
        ctx.bind_value_for_var(out_var, out_value)
