# jax2onnx/plugins/jax/lax/real.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._complex_utils import (
    COMPLEX_DTYPES,
    ensure_packed_real_pair,
    is_packed_complex_tensor,
    split_packed_real_imag,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.real_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.real.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="0.10.2",
    context="primitives.lax",
    component="real",
    testcases=[
        {
            "testcase": "real_complex64_input",
            "callable": lambda x: jax.lax.real(x),
            "input_values": [np.array([1.0 + 2.0j, -0.5 + 0.25j], dtype=np.complex64)],
            "expected_output_dtypes": [np.float32],
        },
    ],
)
class RealPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.real`` to ONNX ops."""

    def lower(self, ctx: "IRContext", eqn):
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("real_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("real_out"))

        def _is_complex_var(var) -> bool:
            aval_dtype = getattr(getattr(var, "aval", None), "dtype", None)
            if aval_dtype is None:
                return False
            try:
                return np.issubdtype(np.dtype(aval_dtype), np.complexfloating)
            except TypeError:
                return False

        dtype = getattr(x_val, "dtype", None)
        complex_hint = _is_complex_var(x_var) or dtype in COMPLEX_DTYPES
        packed_hint = is_packed_complex_tensor(x_val) if complex_hint else False

        if complex_hint or packed_hint:
            packed, base_dtype = ensure_packed_real_pair(ctx, x_val, name_hint="real")
            real_part, _ = split_packed_real_imag(
                ctx, packed, base_dtype, prefix="real_split"
            )
            output_name = getattr(out_spec, "name", None) or ctx.fresh_name("real_out")
            if getattr(real_part, "name", None) != output_name:
                result = ctx.builder.Identity(real_part, _outputs=[output_name])
                result.type = ir.TensorType(base_dtype)
                result.dtype = base_dtype
                if getattr(real_part, "shape", None) is not None:
                    result.shape = real_part.shape
                _ensure_value_metadata(ctx, result)
                ctx.bind_value_for_var(out_var, result)
                out_spec.type = ir.TensorType(base_dtype)
                out_spec.dtype = base_dtype
                out_spec.shape = result.shape
                return
            _ensure_value_metadata(ctx, real_part)
            ctx.bind_value_for_var(out_var, real_part)
            out_spec.type = ir.TensorType(base_dtype)
            out_spec.dtype = base_dtype
            if getattr(real_part, "shape", None) is not None:
                out_spec.shape = real_part.shape
            return

        output_name = getattr(out_spec, "name", None) or ctx.fresh_name("real_out")
        result = ctx.builder.Identity(x_val, _outputs=[output_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif getattr(x_val, "type", None) is not None:
            result.type = x_val.type
        output_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(result, output_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
