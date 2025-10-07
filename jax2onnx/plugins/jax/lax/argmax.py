# jax2onnx/plugins/jax/lax/argmax.py

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
    jaxpr_primitive=jax.lax.argmax_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmax.html",
    onnx=[
        {
            "component": "ArgMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMax.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="argmax",
    testcases=[
        {
            "testcase": "argmax_float_axis0",
            "callable": lambda x: jax.lax.argmax(x, axis=0, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
            "input_dtypes": [jnp.float32],
        },
        {
            "testcase": "argmax_float_axis1",
            "callable": lambda x: jax.lax.argmax(x, axis=1, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
            "input_dtypes": [jnp.float32],
        },
        {
            "testcase": "argmax_boolean_input_axis0_specific_values",
            "callable": lambda x: jax.lax.argmax(x, axis=0, index_dtype=jnp.int32),
            "input_values": [
                np.array(
                    [[False, True, False], [True, False, True], [False, False, False]],
                    dtype=np.bool_,
                )
            ],
        },
        {
            "testcase": "argmax_boolean_input_axis1_specific_values",
            "callable": lambda x: jax.lax.argmax(x, axis=1, index_dtype=jnp.int32),
            "input_values": [
                np.array(
                    [[False, True, False], [True, False, True], [False, True, True]],
                    dtype=np.bool_,
                )
            ],
        },
        {
            "testcase": "argmax_boolean_random_input_axis0",
            "callable": lambda x: jax.lax.argmax(x, axis=0, index_dtype=jnp.int32),
            "input_shapes": [(4, 5)],
            "input_dtypes": [np.bool_],
        },
    ],
)
class ArgMaxPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.argmax`` to ONNX ``ArgMax`` with optional dtype casts."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        axes = params.get("axes") or (0,)
        axis = int(axes[0])
        select_last = int(params.get("select_last_index", 0))
        index_dtype = np.dtype(params.get("index_dtype", np.int64))

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("argmax_in")
        )
        operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
        operand_dtype = np.dtype(getattr(operand_var.aval, "dtype", np.float32))
        rank = len(operand_shape)
        if rank > 0 and axis < 0:
            axis = axis % rank

        input_for_argmax = operand_val
        if operand_dtype == np.bool_:
            input_for_argmax = builder_cast(
                ctx,
                operand_val,
                _dtype_to_ir(np.dtype(np.int64), ctx.builder.enable_double_precision),
                name_hint="argmax_cast",
            )
            _stamp_type_and_shape(input_for_argmax, operand_shape)
            _ensure_value_info(ctx, input_for_argmax)

        tmp_out = ctx.builder.ArgMax(
            input_for_argmax,
            axis=int(axis),
            keepdims=0,
            select_last_index=select_last,
            _outputs=[ctx.fresh_name("argmax_tmp")],
        )
        tmp_out.type = ir.TensorType(ir.DataType.INT64)
        tmp_out.shape = ir.Shape(
            tuple(_to_ir_dim_for_shape(d) for d in getattr(out_var.aval, "shape", ()))
        )
        _ensure_value_info(ctx, tmp_out)

        target_enum = _dtype_to_ir(index_dtype, ctx.builder.enable_double_precision)
        if target_enum == ir.DataType.INT64:
            out_val = builder_identity(
                ctx,
                tmp_out,
                name_hint="argmax_out",
            )
        else:
            out_val = builder_cast(
                ctx,
                tmp_out,
                target_enum,
                name_hint="argmax_out",
            )

        _stamp_type_and_shape(out_val, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_info(ctx, out_val)
        out_val.type = ir.TensorType(target_enum)
        ctx.bind_value_for_var(out_var, out_val)
