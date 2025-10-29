# jax2onnx/plugins/jax/lax/sub.py

from typing import TYPE_CHECKING, Optional, cast

import onnx_ir as ir
import jax
import numpy as np
from jax2onnx.plugins._axis0_utils import (
    maybe_expand_binary_axis0,
    stamp_axis0_binary_result,
)
from jax2onnx.plugins._loop_extent_meta import (
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins._complex_utils import (
    COMPLEX_DTYPES,
    cast_real_tensor,
    ensure_packed_real_pair,
    is_packed_complex_tensor,
    resolve_common_real_dtype,
    coerce_dim_values,
    _shape_tuple,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.ir_context import IRContext


def _lower_complex_sub(
    ctx: "IRContext",
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    output_name: str,
) -> tuple[ir.Value, ir.DataType]:
    lhs_packed, lhs_dtype = ensure_packed_real_pair(ctx, lhs, name_hint="sub_lhs")
    rhs_packed, rhs_dtype = ensure_packed_real_pair(ctx, rhs, name_hint="sub_rhs")

    target_dtype = resolve_common_real_dtype(lhs_dtype, rhs_dtype)
    lhs_ready = (
        lhs_packed
        if lhs_packed.dtype == target_dtype
        else cast_real_tensor(ctx, lhs_packed, target_dtype, name_hint="sub_lhs_cast")
    )
    rhs_ready = (
        rhs_packed
        if rhs_packed.dtype == target_dtype
        else cast_real_tensor(ctx, rhs_packed, target_dtype, name_hint="sub_rhs_cast")
    )

    result = cast(
        ir.Value,
        ctx.builder.Sub(
            lhs_ready,
            rhs_ready,
            _outputs=[output_name],
        ),
    )
    result.type = ir.TensorType(target_dtype)
    result.dtype = target_dtype
    dims = coerce_dim_values(_shape_tuple(lhs_ready))
    _stamp_type_and_shape(result, dims)
    _ensure_value_metadata(ctx, result)
    return result, target_dtype


@register_primitive(
    jaxpr_primitive=jax.lax.sub_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sub.html",
    onnx=[{"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"}],
    since="v0.1.0",
    context="primitives.lax",
    component="sub",
    testcases=[
        {
            "testcase": "sub_test1",
            "callable": lambda x1, x2: x1 - x2,
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Sub:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sub_test2",
            "callable": lambda x1, x2: jax.lax.sub(x1, x2),
            "input_shapes": [(2, 2), (2, 2)],
            "post_check_onnx_graph": EG(
                ["Sub:2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sub_const",
            "callable": lambda x: x - 1.0,
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Sub:3",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sub_complex64",
            "callable": lambda x, y: x - y,
            "input_values": [
                np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64),
                np.array([-1.5 + 0.5j, 2.0 + 1.25j], dtype=np.complex64),
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Sub:2x2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SubPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        prefer_dt: Optional[np.dtype] = np.dtype(
            getattr(x_var.aval, "dtype", np.float32)
        )
        a_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("sub_lhs"))
        b_val = ctx.get_value_for_var(
            y_var, name_hint=ctx.fresh_name("sub_rhs"), prefer_np_dtype=prefer_dt
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("sub_out"))
        a_val, b_val, override = maybe_expand_binary_axis0(
            ctx, a_val, b_val, out_spec, out_var
        )
        output_name = out_spec.name or ctx.fresh_name("sub_out")
        out_spec.name = output_name

        def _is_complex_var(var) -> bool:
            aval_dtype = getattr(getattr(var, "aval", None), "dtype", None)
            if aval_dtype is None:
                return False
            try:
                return np.issubdtype(np.dtype(aval_dtype), np.complexfloating)
            except TypeError:
                return False

        complex_var_hint = (
            _is_complex_var(out_var) or _is_complex_var(x_var) or _is_complex_var(y_var)
        )
        complex_dtype_hint = (
            a_val.dtype in COMPLEX_DTYPES or b_val.dtype in COMPLEX_DTYPES
        )
        packed_hint = False
        if complex_var_hint or complex_dtype_hint:
            packed_hint = is_packed_complex_tensor(a_val) or is_packed_complex_tensor(
                b_val
            )
        complex_route = complex_var_hint or complex_dtype_hint or packed_hint

        if complex_route:
            result, result_dtype = _lower_complex_sub(
                ctx,
                a_val,
                b_val,
                output_name=output_name,
            )
            out_spec.type = ir.TensorType(result_dtype)
            out_spec.dtype = result_dtype
            if getattr(result, "shape", None) is not None:
                out_spec.shape = result.shape
            stamp_axis0_binary_result(result, out_var, out_spec, override)
            if override is not None:
                set_axis0_override(result, override)
            propagate_axis0_override(a_val, result)
            propagate_axis0_override(b_val, result)
            ctx.bind_value_for_var(out_var, result)
            try:
                outputs = ctx.builder.outputs
            except AttributeError:
                outputs = []
            result_name = getattr(result, "name", None)
            for out_val in outputs:
                if out_val is result or getattr(out_val, "name", None) == result_name:
                    out_val.type = ir.TensorType(result_dtype)
                    out_val.dtype = result_dtype
                    if getattr(result, "shape", None) is not None:
                        out_val.shape = result.shape
                    _ensure_value_metadata(ctx, out_val)
                    break
            return

        result = cast(
            ir.Value,
            ctx.builder.Sub(a_val, b_val, _outputs=[output_name]),
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        stamp_axis0_binary_result(result, out_var, out_spec, override)
        if override is not None:
            set_axis0_override(result, override)
        propagate_axis0_override(a_val, result)
        propagate_axis0_override(b_val, result)
        ctx.bind_value_for_var(out_var, result)
