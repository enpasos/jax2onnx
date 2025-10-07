# jax2onnx/plugins/jax/lax/argmin.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import (
    _ensure_value_info,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins.jax.lax._control_flow_utils import (
    builder_cast,
    builder_identity,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.argmin_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmin.html",
    onnx=[
        {
            "component": "ArgMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMin.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="argmin",
    testcases=[
        {
            "testcase": "argmin_test1",
            "callable": lambda x: jax.lax.argmin(x, axis=0, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "argmin_test2",
            "callable": lambda x: jax.lax.argmin(x, axis=1, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
        },
    ],
)
class ArgMinPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.argmin`` to ONNX ``ArgMin`` with optional index casts."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        axes = params.get("axes") or (0,)
        axis = int(axes[0])
        select_last = int(params.get("select_last_index", 0))
        index_dtype = np.dtype(params.get("index_dtype", np.int64))

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("argmin_in")
        )
        operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
        rank = len(operand_shape)
        if rank > 0 and axis < 0:
            axis = axis % rank

        tmp_out = ctx.builder.ArgMin(
            operand_val,
            axis=axis,
            keepdims=0,
            select_last_index=select_last,
            _outputs=[ctx.fresh_name("argmin_tmp")],
        )
        tmp_out.type = ir.TensorType(ir.DataType.INT64)
        tmp_out.shape = ir.Shape(
            tuple(_to_ir_dim_for_shape(d) for d in getattr(out_var.aval, "shape", ()))
        )
        _ensure_value_info(ctx, tmp_out)

        target_enum = _dtype_to_ir(index_dtype, ctx.builder.enable_double_precision)
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("argmin_out"))
        if target_enum == ir.DataType.INT64:
            out_val = builder_identity(
                ctx,
                tmp_out,
                name_hint="argmin_out",
            )
        else:
            out_val = builder_cast(
                ctx,
                tmp_out,
                target_enum,
                name_hint="argmin_out",
            )

        _stamp_type_and_shape(out_val, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_info(ctx, out_val)
        out_val.type = ir.TensorType(target_enum)
        ctx.bind_value_for_var(out_var, out_val)
