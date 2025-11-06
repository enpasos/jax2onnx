# jax2onnx/plugins/jax/lax/conj.py

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import jax

from typing import TYPE_CHECKING

from jax2onnx.plugins._complex_utils import (
    COMPLEX_DTYPES,
    conjugate_packed_tensor,
    ensure_packed_real_pair,
    is_packed_complex_tensor,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.conj_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.conj.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.10.1",
    context="primitives.lax",
    component="conj",
    testcases=[
        {
            "testcase": "conj_real",
            "callable": lambda x: jax.lax.conj(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(["Identity:3"], no_unused_inputs=True),
        },
        {
            "testcase": "conj_complex64",
            "callable": lambda x: jax.lax.conj(x),
            "input_values": [
                np.array(
                    [1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 4.0j],
                    dtype=np.complex64,
                )
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Neg", "counts": {"Neg": 1}},
                    {"path": "Concat", "counts": {"Concat": 1}},
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ConjPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.conj`` to ONNX ops."""

    def lower(self, ctx: "IRContext", eqn):
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("conj_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("conj_out"))

        def _is_complex_var(var) -> bool:
            aval_dtype = getattr(getattr(var, "aval", None), "dtype", None)
            if aval_dtype is None:
                return False
            try:
                return np.issubdtype(np.dtype(aval_dtype), np.complexfloating)
            except TypeError:
                return False

        dtype = getattr(x_val, "dtype", None)
        complex_hint = (
            dtype in COMPLEX_DTYPES
            or is_packed_complex_tensor(x_val)
            or _is_complex_var(x_var)
        )

        if complex_hint:
            packed, base_dtype = ensure_packed_real_pair(ctx, x_val, name_hint="conj")
            target_dtype = packed.dtype or base_dtype
            output_name = getattr(out_spec, "name", None) or ctx.fresh_name("conj_out")
            conj_val = conjugate_packed_tensor(
                ctx,
                packed,
                target_dtype,
                prefix="conj",
                output_name=output_name,
            )
            out_spec.type = ir.TensorType(target_dtype)
            out_spec.dtype = target_dtype
            if getattr(conj_val, "shape", None) is not None:
                out_spec.shape = conj_val.shape
            _ensure_value_metadata(ctx, conj_val)
            ctx.bind_value_for_var(out_var, conj_val)
            return

        identity_val = ctx.builder.Identity(
            x_val,
            _outputs=[getattr(out_spec, "name", None) or ctx.fresh_name("conj_out")],
        )
        identity_val.type = getattr(out_spec, "type", None) or getattr(
            x_val, "type", None
        )
        if identity_val.type is None and dtype is not None:
            identity_val.type = ir.TensorType(dtype)
        output_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(identity_val, output_shape)
        _ensure_value_metadata(ctx, identity_val)
        ctx.bind_value_for_var(out_var, identity_val)
