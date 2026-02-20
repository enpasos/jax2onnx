# jax2onnx/plugins/jax/lax/div.py

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
    split_packed_real_imag,
    pack_real_imag_pair,
    _shape_tuple,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _lower_complex_div(
    ctx: "IRContext",
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    output_name: str,
) -> tuple[ir.Value, ir.DataType]:
    lhs_packed, lhs_dtype = ensure_packed_real_pair(ctx, lhs, name_hint="div_lhs")
    rhs_packed, rhs_dtype = ensure_packed_real_pair(ctx, rhs, name_hint="div_rhs")

    target_dtype = resolve_common_real_dtype(lhs_dtype, rhs_dtype)
    lhs_ready = (
        lhs_packed
        if lhs_packed.dtype == target_dtype
        else cast_real_tensor(ctx, lhs_packed, target_dtype, name_hint="div_lhs_cast")
    )
    rhs_ready = (
        rhs_packed
        if rhs_packed.dtype == target_dtype
        else cast_real_tensor(ctx, rhs_packed, target_dtype, name_hint="div_rhs_cast")
    )

    base_dims = _shape_tuple(lhs_ready)[:-1]
    dim_values = coerce_dim_values(base_dims)

    def _binary(op_name: str, a: ir.Value, b: ir.Value, name: str) -> ir.Value:
        op = getattr(ctx.builder, op_name)
        result = cast(
            ir.Value,
            op(a, b, _outputs=[ctx.fresh_name(name)]),
        )
        result.type = ir.TensorType(target_dtype)
        result.dtype = target_dtype
        _stamp_type_and_shape(result, dim_values)
        _ensure_value_metadata(ctx, result)
        return result

    lhs_real, lhs_imag = split_packed_real_imag(
        ctx, lhs_ready, target_dtype, prefix="div_lhs"
    )
    rhs_real, rhs_imag = split_packed_real_imag(
        ctx, rhs_ready, target_dtype, prefix="div_rhs"
    )

    rhs_real_sq = _binary("Mul", rhs_real, rhs_real, "div_rhs_real_sq")
    rhs_imag_sq = _binary("Mul", rhs_imag, rhs_imag, "div_rhs_imag_sq")
    denom = _binary("Add", rhs_real_sq, rhs_imag_sq, "div_denom")

    ar_br = _binary("Mul", lhs_real, rhs_real, "div_ar_br")
    ai_bi = _binary("Mul", lhs_imag, rhs_imag, "div_ai_bi")
    real_num = _binary("Add", ar_br, ai_bi, "div_real_num")

    ai_br = _binary("Mul", lhs_imag, rhs_real, "div_ai_br")
    ar_bi = _binary("Mul", lhs_real, rhs_imag, "div_ar_bi")
    imag_num = _binary("Sub", ai_br, ar_bi, "div_imag_num")

    real_part = _binary("Div", real_num, denom, "div_real_part")
    imag_part = _binary("Div", imag_num, denom, "div_imag_part")

    packed = pack_real_imag_pair(
        ctx,
        real_part,
        imag_part,
        target_dtype,
        name_hint="div_output",
        output_name=output_name,
    )
    return packed, target_dtype


def _const_scalar_as_float(value: ir.Value) -> Optional[float]:
    const_val = getattr(value, "const_value", None)
    if const_val is None:
        return None
    try:
        arr = np.asarray(const_val.numpy())
    except Exception:
        try:
            arr = np.asarray(const_val)
        except Exception:
            return None
    if arr.size != 1:
        return None
    try:
        return float(arr.reshape(()))
    except (TypeError, ValueError):
        return None


def _const_axes(value: ir.Value) -> Optional[list[int]]:
    const_val = getattr(value, "const_value", None)
    if const_val is None:
        return None
    try:
        arr = np.asarray(const_val.numpy())
    except Exception:
        try:
            arr = np.asarray(const_val)
        except Exception:
            return None
    if arr.size == 0:
        return []
    try:
        return [int(v) for v in arr.reshape(-1).tolist()]
    except (TypeError, ValueError):
        return None


def _same_value(lhs: ir.Value, rhs: ir.Value) -> bool:
    if lhs is rhs:
        return True
    lhs_name = getattr(lhs, "name", None)
    rhs_name = getattr(rhs, "name", None)
    return lhs_name is not None and lhs_name == rhs_name


def _producer(value: ir.Value) -> object | None:
    getter = getattr(value, "producer", None)
    if callable(getter):
        return getter()
    return None


def _unwrap_passthrough(value: ir.Value) -> ir.Value:
    current = value
    passthrough_ops = {"Expand", "Reshape", "Identity", "Cast", "CastLike"}
    while True:
        node = _producer(current)
        if getattr(node, "op_type", "") not in passthrough_ops:
            return current
        inputs = list(getattr(node, "inputs", ()))
        if not inputs:
            return current
        next_val = inputs[0]
        if not isinstance(next_val, ir.Value):
            return current
        current = next_val


def _match_lpnormalization_pattern(
    lhs_val: ir.Value,
    rhs_val: ir.Value,
) -> tuple[int, int] | None:
    """Match ``x / ||x||_p`` style denominators and return ``(p, axis)``."""

    rhs_core = _unwrap_passthrough(rhs_val)
    rhs_node = _producer(rhs_core)
    op_type = getattr(rhs_node, "op_type", "")

    reduce_node = None
    p = None
    if op_type == "ReduceL1":
        reduce_node = rhs_node
        p = 1
    elif op_type == "ReduceL2":
        reduce_node = rhs_node
        p = 2
    elif op_type == "Sqrt":
        sqrt_inputs = list(getattr(rhs_node, "inputs", ()))
        if not sqrt_inputs:
            return None
        sqrt_in = sqrt_inputs[0]
        if not isinstance(sqrt_in, ir.Value):
            return None
        reduce_val = _unwrap_passthrough(sqrt_in)
        candidate = _producer(reduce_val)
        candidate_type = getattr(candidate, "op_type", "")
        if candidate_type == "ReduceSumSquare":
            reduce_node = candidate
            p = 2
        else:
            return None
    else:
        return None

    reduce_inputs = list(getattr(reduce_node, "inputs", ()))
    if not reduce_inputs:
        return None
    reduce_base = reduce_inputs[0]
    if not isinstance(reduce_base, ir.Value):
        return None
    if not _same_value(reduce_base, lhs_val):
        return None

    axes = _const_axes(reduce_inputs[1]) if len(reduce_inputs) > 1 else None
    if axes is None or len(axes) != 1:
        return None
    return int(p), int(axes[0])


@register_primitive(
    jaxpr_primitive=jax.lax.div_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.div.html",
    onnx=[
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {
            "component": "LpNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__LpNormalization.html",
        },
        {
            "component": "Mean",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mean.html",
        },
    ],
    since="0.2.0",
    context="primitives.lax",
    component="div",
    testcases=[
        {
            "testcase": "div",
            "callable": lambda x1, x2: x1 / x2,
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Div:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "div_const",
            "callable": lambda x: x / 2.0,
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Div:3",
                        "inputs": {1: {"const": 2.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "div_add_half_fuses_to_mean",
            "callable": lambda x1, x2: (x1 + x2) / 2.0,
            "input_shapes": [(2, 3), (2, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Mean:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "div_add_third_no_mean",
            "callable": lambda x1, x2: (x1 + x2) / 3.0,
            "input_shapes": [(2, 3), (2, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Div:2x3"],
                must_absent=["Mean"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "div_add_half_f64_no_mean",
            "callable": lambda x1, x2: (x1 + x2) / 2.0,
            "input_shapes": [(2, 3), (2, 3)],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Div:2x3"],
                must_absent=["Mean"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "div_lpnorm_l2_axis1",
            "callable": lambda x: x
            / jax.numpy.sqrt(jax.numpy.sum(jax.numpy.square(x), axis=1, keepdims=True)),
            "input_values": [
                np.asarray(
                    [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
                    dtype=np.float32,
                )
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LpNormalization:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "div_lpnorm_l1_axis1",
            "callable": lambda x: x
            / jax.numpy.sum(jax.numpy.abs(x), axis=1, keepdims=True),
            "input_values": [
                np.asarray(
                    [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
                    dtype=np.float32,
                )
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LpNormalization:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "div_lpnorm_l2_axis2",
            "callable": lambda x: x
            / jax.numpy.sqrt(jax.numpy.sum(jax.numpy.square(x), axis=2, keepdims=True)),
            "input_shapes": [(2, 3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LpNormalization:2x3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "div_sqrt_of_norm_no_lpnorm_fusion",
            "callable": lambda x: x
            / jax.numpy.sqrt(jax.numpy.linalg.norm(x, ord=2, axis=1, keepdims=True)),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Div:2x3"],
                must_absent=["LpNormalization"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "div_complex64",
            "callable": lambda x, y: x / y,
            "input_values": [
                np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64),
                np.array([-1.5 + 0.5j, 2.0 + 1.25j], dtype=np.complex64),
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Div", "counts": {"Div": 2}},
                    "Concat:2x2",
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class DivPlugin(PrimitiveLeafPlugin):
    """IR-first lowering of ``lax.div`` to ONNX ``Div``."""

    def lower(self, ctx: "IRContext", eqn):
        x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        prefer_dt: Optional[np.dtype] = np.dtype(
            getattr(x_var.aval, "dtype", np.float32)
        )

        lhs_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("div_lhs"))
        rhs_val = ctx.get_value_for_var(
            y_var, name_hint=ctx.fresh_name("div_rhs"), prefer_np_dtype=prefer_dt
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("div_out"))

        lhs_val, rhs_val, override = maybe_expand_binary_axis0(
            ctx, lhs_val, rhs_val, out_spec, out_var
        )
        output_name = out_spec.name or ctx.fresh_name("div_out")
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
            lhs_val.dtype in COMPLEX_DTYPES or rhs_val.dtype in COMPLEX_DTYPES
        )
        packed_hint = False
        if complex_var_hint or complex_dtype_hint:
            packed_hint = is_packed_complex_tensor(lhs_val) or is_packed_complex_tensor(
                rhs_val
            )
        complex_route = complex_var_hint or complex_dtype_hint or packed_hint

        if complex_route:
            result, result_dtype = _lower_complex_div(
                ctx,
                lhs_val,
                rhs_val,
                output_name=output_name,
            )
            out_spec.type = ir.TensorType(result_dtype)
            out_spec.dtype = result_dtype
            if getattr(result, "shape", None) is not None:
                out_spec.shape = result.shape
            stamp_axis0_binary_result(result, out_var, out_spec, override)
            if override is not None:
                set_axis0_override(result, override)
            propagate_axis0_override(lhs_val, result)
            propagate_axis0_override(rhs_val, result)
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

        out_aval_dtype = getattr(getattr(out_var, "aval", None), "dtype", None)
        out_is_f64 = False
        if out_aval_dtype is not None:
            try:
                out_is_f64 = np.dtype(out_aval_dtype) == np.dtype(np.float64)
            except TypeError:
                out_is_f64 = False

        lhs_producer = None
        producer = getattr(lhs_val, "producer", None)
        if callable(producer):
            lhs_producer = producer()
        rhs_scalar = _const_scalar_as_float(rhs_val)
        lpnorm_match = _match_lpnormalization_pattern(lhs_val, rhs_val)
        if lpnorm_match is not None:
            p, axis = lpnorm_match
            result = cast(
                ir.Value,
                ctx.builder.LpNormalization(
                    lhs_val,
                    axis=axis,
                    p=p,
                    _outputs=[output_name],
                ),
            )
            if getattr(out_spec, "type", None) is not None:
                result.type = out_spec.type
            stamp_axis0_binary_result(result, out_var, out_spec, override)
            if override is not None:
                set_axis0_override(result, override)
            propagate_axis0_override(lhs_val, result)
            propagate_axis0_override(rhs_val, result)
            ctx.bind_value_for_var(out_var, result)
            return

        if (
            not out_is_f64
            and rhs_scalar is not None
            and np.isclose(rhs_scalar, 2.0)
            and getattr(lhs_producer, "op_type", "") == "Add"
            and len(getattr(lhs_producer, "inputs", ())) == 2
        ):
            add_lhs, add_rhs = lhs_producer.inputs
            result = cast(
                ir.Value,
                ctx.builder.Mean(add_lhs, add_rhs, _outputs=[output_name]),
            )
        else:
            result = cast(
                ir.Value,
                ctx.builder.Div(lhs_val, rhs_val, _outputs=[output_name]),
            )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        stamp_axis0_binary_result(result, out_var, out_spec, override)
        if override is not None:
            set_axis0_override(result, override)
        propagate_axis0_override(lhs_val, result)
        propagate_axis0_override(rhs_val, result)
        ctx.bind_value_for_var(out_var, result)
