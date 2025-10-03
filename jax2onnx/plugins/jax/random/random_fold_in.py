"""Minimal lowering for ``random_fold_in``.

The IR pipeline currently only needs deterministic PRNG plumbing for
modules that never consume random bits (e.g. nnx.Dropout in inference
mode).  We therefore forward the incoming key unchanged, while still
producing an ONNX Value so downstream passes see a concrete tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _identity(ctx: "IRContext", value: ir.Value, name_hint: str) -> ir.Value:
    out = ir.Value(name=ctx.fresh_name(name_hint), type=value.type, shape=value.shape)
    ctx.add_node(
        ir.Node(
            op_type="Identity",
            domain="",
            inputs=[value],
            outputs=[out],
            name=ctx.fresh_name("Identity"),
        )
    )
    return out


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
        }
    ],
)
class RandomFoldInPlugin(PrimitiveLeafPlugin):
    """Forward the incoming key; sufficient for deterministic inference paths."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        key_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        key_value = ctx.get_value_for_var(
            key_var, name_hint=ctx.fresh_name("prng_fold_in_in")
        )
        out_value = _identity(ctx, key_value, name_hint="prng_fold_in_out")
        ctx.bind_value_for_var(out_var, out_value)
