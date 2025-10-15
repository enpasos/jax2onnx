# jax2onnx/plugins/jax/lax/rev.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import onnx_ir as ir

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import (
    _gather_int_scalar,
    _scalar_i64,
    _shape_of,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.rev_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.rev.html",
    onnx=[
        {"component": "Flip", "doc": "https://onnx.ai/onnx/operators/onnx__Flip.html"}
    ],
    since="v0.7.5",
    context="primitives.lax",
    component="rev",
    testcases=[
        {
            "testcase": "rev_vector",
            "callable": lambda x: jax.lax.rev(x, (0,)),
            "input_shapes": [(5,)],
            "post_check_onnx_graph": EG(
                ["Gather:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "rev_matrix_axes01",
            "callable": lambda x: jax.lax.rev(x, (0, 1)),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Gather:3x4 -> Gather:3x4"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class RevPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.rev`` to a sequence of Gather ops (no Flip dependency)."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        axes_param = tuple(int(a) for a in eqn.params.get("dimensions", ()))
        input_shape = tuple(getattr(x_var.aval, "shape", ()))
        rank = len(input_shape)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("rev_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("rev_out"))

        if rank == 0 or not axes_param:
            identity_name = getattr(out_spec, "name", None) or ctx.fresh_name(
                "Identity"
            )
            identity = ctx.builder.Identity(
                x_val,
                _outputs=[identity_name],
            )
            src_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
            if src_dtype is not None:
                identity.type = ir.TensorType(src_dtype)
            _stamp_type_and_shape(identity, input_shape)
            _ensure_value_metadata(ctx, identity)
            ctx.bind_value_for_var(out_var, identity)
            return

        canonical_axes: list[int] = []
        for axis in axes_param:
            canonical = axis % rank if axis < 0 else axis
            if canonical < 0 or canonical >= rank:
                raise ValueError(
                    f"Axis {axis} out of range for lax.rev with rank {rank}"
                )
            if canonical not in canonical_axes:
                canonical_axes.append(canonical)

        one = _scalar_i64(ctx, 1, "rev_one")
        neg_one = _scalar_i64(ctx, -1, "rev_neg_one")

        current_val = x_val
        final_name = getattr(out_spec, "name", None) or ctx.fresh_name("rev_out")

        for idx, axis in enumerate(canonical_axes):
            shape_val = _shape_of(ctx, current_val, ctx.fresh_name("rev_shape"))

            dim_len = _gather_int_scalar(
                ctx, shape_val, axis, ctx.fresh_name("rev_dim_len")
            )

            start_val = ctx.builder.Sub(
                dim_len,
                one,
                _outputs=[ctx.fresh_name("rev_start")],
            )
            start_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(start_val, ())
            _ensure_value_metadata(ctx, start_val)

            range_val = ctx.builder.Range(
                start_val,
                neg_one,
                neg_one,
                _outputs=[ctx.fresh_name("rev_range")],
            )
            range_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(range_val, (None,))
            _ensure_value_metadata(ctx, range_val)

            gather_name = (
                final_name
                if idx == len(canonical_axes) - 1
                else ctx.fresh_name("rev_out")
            )

            target_val = ctx.builder.Gather(
                current_val,
                range_val,
                axis=axis,
                _outputs=[gather_name],
            )
            target_dtype = getattr(getattr(current_val, "type", None), "dtype", None)
            if target_dtype is not None:
                target_val.type = ir.TensorType(target_dtype)
            _stamp_type_and_shape(target_val, input_shape)
            _ensure_value_metadata(ctx, target_val)

            current_val = target_val

        ctx.bind_value_for_var(out_var, current_val)
