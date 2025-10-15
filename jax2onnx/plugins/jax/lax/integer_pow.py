# jax2onnx/plugins/jax/lax/integer_pow.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.integer_pow_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.integer_pow.html",
    onnx=[
        {
            "component": "Pow",
            "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="integer_pow",
    testcases=[
        {
            "testcase": "integer_pow",
            "callable": lambda x: jax.lax.integer_pow(x, 2),
            "input_shapes": [(5,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 2.0}},
                        "path": "Pow:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        }
    ],
)
class IntegerPowPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.integer_pow`` to ONNX ``Pow`` with constant exponent."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        base_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        exponent = int(params.get("y", 2))

        base_val = ctx.get_value_for_var(
            base_var, name_hint=ctx.fresh_name("ipow_base")
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("ipow_out"))

        target_dtype = np.dtype(getattr(base_var.aval, "dtype", np.float32))
        exp_const = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name("ipow_exp"),
            value=np.array(exponent, dtype=target_dtype),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Pow")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Pow")

        result = ctx.builder.Pow(base_val, exp_const, _outputs=[desired_name])

        out_dtype_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        result.type = ir.TensorType(out_dtype_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
