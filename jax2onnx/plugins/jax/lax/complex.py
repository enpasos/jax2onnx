# jax2onnx/plugins/jax/lax/complex.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._complex_utils import (
    cast_real_tensor,
    pack_real_imag_pair,
    resolve_common_real_dtype,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.complex_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.complex.html",
    onnx=[
        {
            "component": "Unsqueeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unsqueeze.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.12.2",
    context="primitives.lax",
    component="complex",
    testcases=[
        {
            "testcase": "complex_scalar",
            "callable": lambda x, y: jax.lax.complex(x, y),
            "input_values": [
                np.array(1.0, dtype=np.float32),
                np.array(2.0, dtype=np.float32),
            ],
            "expected_output_dtypes": [np.float32],
        },
        {
            "testcase": "complex_array",
            "callable": lambda x, y: jax.lax.complex(x, y),
            "input_values": [
                np.array([1.0, -0.5], dtype=np.float32),
                np.array([2.0, 0.25], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.float32],
        },
        {
            "testcase": "complex_double_precision",
            "callable": lambda x, y: jax.lax.complex(x, y),
            "input_values": [
                np.array([1.0, -0.5], dtype=np.float64),
                np.array([2.0, 0.25], dtype=np.float64),
            ],
            "enable_double_precision": True,
            "expected_output_dtypes": [np.float64],
        },
    ],
)
class ComplexPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.complex`` to ONNX ops, returning a packed [..., 2] tensor."""

    def lower(self, ctx: "IRContext", eqn):
        (real_var, imag_var) = eqn.invars
        (out_var,) = eqn.outvars

        real_val = ctx.get_value_for_var(
            real_var, name_hint=ctx.fresh_name("complex_real_in")
        )
        imag_val = ctx.get_value_for_var(
            imag_var, name_hint=ctx.fresh_name("complex_imag_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("complex_out")
        )

        real_dtype = real_val.dtype or ir.DataType.FLOAT
        imag_dtype = imag_val.dtype or ir.DataType.FLOAT

        # Resolve to a common float dtype (mostly handles float vs double)
        base_dtype = resolve_common_real_dtype(real_dtype, imag_dtype)
        if base_dtype not in (ir.DataType.FLOAT, ir.DataType.DOUBLE):
            base_dtype = ir.DataType.FLOAT

        real_cast = cast_real_tensor(
            ctx, real_val, base_dtype, name_hint="complex_real_cast"
        )
        imag_cast = cast_real_tensor(
            ctx, imag_val, base_dtype, name_hint="complex_imag_cast"
        )

        output_name = getattr(out_spec, "name", None) or ctx.fresh_name("complex_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            output_name = ctx.fresh_name("complex_out")

        packed = pack_real_imag_pair(
            ctx,
            real_cast,
            imag_cast,
            base_dtype,
            name_hint="complex",
            output_name=output_name,
        )

        _ensure_value_metadata(ctx, packed)
        ctx.bind_value_for_var(out_var, packed)
