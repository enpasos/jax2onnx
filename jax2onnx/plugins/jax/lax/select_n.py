# jax2onnx/plugins/jax/lax/select_n.py

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import _Site, _tensor, _validate_complex_select
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import (
    DimInput,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins.jax.lax._index_utils import _cast_to_i64, _const_i64
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)
import jax.numpy as jnp


def _lower_source_complex_select(ctx: LoweringContextProtocol, eqn: Any) -> bool:
    """Copy the selected complete pair after proving every physical operand."""
    if ctx.precision_policy is not PrecisionPolicy.SOURCE:
        return False
    if not any(
        isinstance(var.aval.dtype, np.dtype)
        and var.aval.dtype in (np.dtype(np.complex64), np.dtype(np.complex128))
        for var in (*eqn.invars, *eqn.outvars)
    ):
        return False
    site = _Site(
        "static complex select lowering", ctx.builder.opset, eqn.primitive.name
    )
    _validate_complex_select(
        eqn,
        [_tensor(var, site) for var in eqn.invars],
        [_tensor(var, site) for var in eqn.outvars],
        site,
    )
    values = []
    for var in eqn.invars:
        value = ctx.require_value_for_var(var)
        if not representation_for_aval(var.aval).matches_ir_value(value):
            raise TypeError(
                "SOURCE complex select physical input differs from its declared representation"
            )
        values.append(value)
    out_var = eqn.outvars[0]
    representation = representation_for_aval(out_var.aval)
    output = ctx.allocate_value_for_var(
        out_var, name_hint=ctx.fresh_name("source_complex_select")
    )
    if not representation.matches_ir_value(output):
        raise TypeError(
            "SOURCE complex select physical output differs from its declared pair"
        )
    logical = tuple(int(dim) for dim in out_var.aval.shape)
    physical = (*logical, 2)
    dtype = representation.ir_dtype
    builder = ctx.builder

    def stamp(value: ir.Value, shape: tuple[int, ...], kind: ir.DataType) -> ir.Value:
        value.type = ir.TensorType(kind)
        _stamp_type_and_shape(value, shape)
        _ensure_value_metadata(ctx, value)
        return value

    axis = _const_i64(ctx, np.asarray([0], dtype=np.int64), "complex_select_axis")
    arms = [
        stamp(builder.op("Unsqueeze", [value, axis]), (1, *physical), dtype)
        for value in values[1:]
    ]
    stacked = stamp(
        builder.op("Concat", arms, {"axis": 0}), (len(arms), *physical), dtype
    )
    index = values[0]
    if index.dtype != ir.DataType.INT64:
        index = stamp(
            builder.op("Cast", [index], {"to": int(ir.DataType.INT64.value)}),
            tuple(int(dim) for dim in eqn.invars[0].aval.shape),
            ir.DataType.INT64,
        )
    # A single logical selector is duplicated across both real/imaginary words.
    index_shape = (
        (1, *logical, 1) if eqn.invars[0].aval.shape else (1,) * (len(logical) + 2)
    )
    reshape_shape = _const_i64(
        ctx, np.asarray(index_shape, dtype=np.int64), "complex_select_index_shape"
    )
    index = stamp(
        builder.op("Reshape", [index, reshape_shape], {"allowzero": 1}),
        index_shape,
        ir.DataType.INT64,
    )
    gathered_shape = (1, *physical)
    expand_shape = _const_i64(
        ctx, np.asarray(gathered_shape, dtype=np.int64), "complex_select_expand_shape"
    )
    index = stamp(
        builder.op("Expand", [index, expand_shape]), gathered_shape, ir.DataType.INT64
    )
    selected = stamp(
        builder.op("GatherElements", [stacked, index], {"axis": 0}),
        gathered_shape,
        dtype,
    )
    result = builder.op("Squeeze", [selected, axis], output=output)
    if not representation.matches_ir_value(result):
        raise TypeError("SOURCE complex select changed the declared physical result")
    _ensure_value_metadata(ctx, result)
    ctx.bind_value_for_var(out_var, result)
    return True


@register_native_primitive(
    jaxpr_primitive=jax.lax.select_n_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.select_n.html",
    onnx=[
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        },
        {
            "component": "GatherElements",
            "doc": "https://onnx.ai/onnx/operators/onnx__GatherElements.html",
        },
    ],
    since="0.2.0",
    context="primitives.lax",
    component="select_n",
    testcases=[
        {
            "testcase": "select_n_bool_predicate_two_cases_float",
            "callable": lambda pred, on_false, on_true: jax.lax.select_n(
                pred, on_false, on_true
            ),
            "input_values": [
                jnp.array([True, False, True], dtype=jnp.bool_),
                jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
                jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),
            ],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Where:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_n_bool_predicate_two_cases_int",
            "callable": lambda pred, on_false, on_true: jax.lax.select_n(
                pred, on_false, on_true
            ),
            "input_values": [
                jnp.array([[True, False], [False, True]], dtype=jnp.bool_),
                jnp.array([[10, 20], [30, 40]], dtype=jnp.int32),
                jnp.array([[50, 60], [70, 80]], dtype=jnp.int32),
            ],
            "expected_output_shapes": [(2, 2)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Where:2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_n_bool_predicate_scalar_broadcast",
            "callable": lambda pred, on_false, on_true: jax.lax.select_n(
                pred, on_false, on_true
            ),
            "input_values": [
                jnp.array(True, dtype=jnp.bool_),
                jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
                jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),
            ],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Where:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_n_int_indices_three_cases",
            "callable": lambda indices, c0, c1, c2: jax.lax.select_n(
                indices, c0, c1, c2
            ),
            "input_values": [
                jnp.array([0, 1, 2, 0], dtype=jnp.int32),
                jnp.array([10, 11, 12, 13], dtype=jnp.float32),
                jnp.array([20, 21, 22, 23], dtype=jnp.float32),
                jnp.array([30, 31, 32, 33], dtype=jnp.float32),
            ],
            "expected_output_shapes": [(4,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:4 -> Equal:4 -> Where:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_n_int_indices_four_cases",
            "callable": lambda indices, c0, c1, c2, c3: jax.lax.select_n(
                indices, c0, c1, c2, c3
            ),
            "input_values": [
                jnp.array([0, 1, 2, 3, 1, 0], dtype=jnp.int32),
                jnp.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=jnp.float32),
                jnp.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5], dtype=jnp.float32),
                jnp.array([3.0, 3.1, 3.2, 3.3, 3.4, 3.5], dtype=jnp.float32),
                jnp.array([4.0, 4.1, 4.2, 4.3, 4.4, 4.5], dtype=jnp.float32),
            ],
            "expected_output_shapes": [(6,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Cast:6 -> Equal:6 -> Where:6"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SelectNPlugin(PrimitiveLeafPlugin):
    """IR-only selection with exact SOURCE floating-point copies."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        if _lower_source_complex_select(ctx, eqn):
            return
        builder = ctx.builder

        invars = list(eqn.invars)
        out_var = eqn.outvars[0]
        source = (
            ctx.precision_policy is PrecisionPolicy.SOURCE
            and len(invars) == 3
            and np.issubdtype(invars[0].aval.dtype, np.bool_)
        )
        out_spec = (ctx.allocate_value_for_var if source else ctx.get_value_for_var)(
            out_var, name_hint=ctx.fresh_name("select_out")
        )

        if len(invars) < 2:
            raise ValueError("select_n requires at least one choice")

        selector_var = invars[0]
        case_vars = invars[1:]

        selector_val = ctx.get_value_for_var(
            selector_var, name_hint=ctx.fresh_name("select_idx")
        )

        out_shape: tuple[DimInput, ...] = tuple(getattr(out_var.aval, "shape", ()))
        selector_shape: tuple[DimInput, ...] = tuple(
            getattr(selector_var.aval, "shape", ())
        )

        def _ensure_tensor_metadata(
            value: ir.Value, shape: tuple[DimInput, ...], dtype: ir.DataType | None
        ) -> None:
            if dtype is not None:
                value.type = ir.TensorType(dtype)
            _stamp_type_and_shape(value, shape)
            _ensure_value_metadata(ctx, value)

        # --- Boolean two-case path -------------------------------------------------
        if len(case_vars) == 2 and np.issubdtype(selector_var.aval.dtype, np.bool_):
            cond_val = selector_val
            if selector_var.aval.dtype != np.bool_:
                cond_val = cast(
                    ir.Value,
                    builder.Cast(
                        cond_val,
                        _outputs=[ctx.fresh_name("select_cond_bool")],
                        to=int(ir.DataType.BOOL.value),
                    ),
                )
                _ensure_tensor_metadata(cond_val, selector_shape, ir.DataType.BOOL)

            true_val = ctx.get_value_for_var(
                case_vars[1], name_hint=ctx.fresh_name("select_true")
            )
            false_val = ctx.get_value_for_var(
                case_vars[0], name_hint=ctx.fresh_name("select_false")
            )

            if source and np.dtype(out_var.aval.dtype) == np.dtype(np.bool_):
                not_cond = builder.op("Not", [cond_val])
                _ensure_tensor_metadata(not_cond, selector_shape, ir.DataType.BOOL)
                selected_true = builder.op("And", [cond_val, true_val])
                _ensure_tensor_metadata(selected_true, out_shape, ir.DataType.BOOL)
                selected_false = builder.op("And", [not_cond, false_val])
                _ensure_tensor_metadata(selected_false, out_shape, ir.DataType.BOOL)
                result = builder.op(
                    "Or", [selected_true, selected_false], output=out_spec
                )
                _ensure_tensor_metadata(result, out_shape, ir.DataType.BOOL)
                ctx.bind_value_for_var(out_var, result)
                return

            if source and np.dtype(out_var.aval.dtype) in (
                np.dtype(np.float16),
                np.dtype(np.float32),
                np.dtype(np.float64),
            ):
                # Copy selected floating bytes: Where on the measured CPU
                # stacks loses negative zero and some FP16 NaN payloads.
                # Native validation already proves equal-shaped case arms.
                result_dtype = true_val.dtype
                assert result_dtype is not None
                axis = builder.const_i64(name=ctx.fresh_name("select_axis"), values=[0])
                index = builder.op(
                    "Cast", [cond_val], {"to": int(ir.DataType.INT32.value)}
                )
                _ensure_tensor_metadata(index, selector_shape, ir.DataType.INT32)
                if selector_shape != out_shape:
                    shape = builder.op("Shape", [true_val])
                    _ensure_tensor_metadata(shape, (len(out_shape),), ir.DataType.INT64)
                    index = builder.op("Expand", [index, shape])
                    _ensure_tensor_metadata(index, out_shape, ir.DataType.INT32)
                expanded_shape = (1, *out_shape)
                false_arm = builder.op("Unsqueeze", [false_val, axis])
                _ensure_tensor_metadata(false_arm, expanded_shape, result_dtype)
                true_arm = builder.op("Unsqueeze", [true_val, axis])
                _ensure_tensor_metadata(true_arm, expanded_shape, result_dtype)
                cases = builder.op("Concat", [false_arm, true_arm], {"axis": 0})
                _ensure_tensor_metadata(cases, (2, *out_shape), result_dtype)
                index = builder.op("Unsqueeze", [index, axis])
                _ensure_tensor_metadata(index, expanded_shape, ir.DataType.INT32)
                selected = builder.op("GatherElements", [cases, index], {"axis": 0})
                _ensure_tensor_metadata(selected, expanded_shape, result_dtype)
                result = builder.op("Squeeze", [selected, axis], output=out_spec)
                _ensure_tensor_metadata(result, out_shape, result_dtype)
                ctx.bind_value_for_var(out_var, result)
                return

            if source and np.dtype(out_var.aval.dtype) == np.dtype(np.int32):
                from jax2onnx.converter.source_profile import (
                    _Site,
                    _tensor,
                    _validate_select,
                )
                from jax2onnx.converter.source_representation import (
                    representation_for_aval,
                )

                site = _Site(
                    "native INT32 select emission", ctx.opset, eqn.primitive.name
                )
                _validate_select(
                    eqn,
                    [_tensor(v, site) for v in eqn.invars],
                    [_tensor(out_var, site)],
                    site,
                )
                for variable, value in (
                    (selector_var, cond_val),
                    (case_vars[0], false_val),
                    (case_vars[1], true_val),
                ):
                    if not representation_for_aval(variable.aval).matches_ir_value(
                        value
                    ):
                        raise TypeError(
                            "SOURCE INT32 select physical input differs from its native declaration"
                        )
                # Both supported CPU providers implement exact INT32 Where.
                # Wider carriers remain necessary for the narrower/missing
                # integer kernels below; index arithmetic stays native INT32.
                result = builder.op(
                    "Where", [cond_val, true_val, false_val], output=out_spec
                )
                _ensure_tensor_metadata(result, out_shape, ir.DataType.INT32)
                ctx.bind_value_for_var(out_var, result)
                return

            if (
                source
                and np.issubdtype(out_var.aval.dtype, np.integer)
                and np.dtype(out_var.aval.dtype) != np.dtype(np.int64)
            ):
                # CPU Where has integer kernel gaps. Integer Cast preserves
                # narrower values exactly and maps UINT64 modulo 2**64 into
                # INT64; the inverse Cast recovers every selected bit. Keep
                # canonical arms and all their sibling consumers untouched.
                true_carrier = builder.op(
                    "Cast", [true_val], {"to": int(ir.DataType.INT64.value)}
                )
                false_carrier = builder.op(
                    "Cast", [false_val], {"to": int(ir.DataType.INT64.value)}
                )
                _ensure_tensor_metadata(true_carrier, out_shape, ir.DataType.INT64)
                _ensure_tensor_metadata(false_carrier, out_shape, ir.DataType.INT64)
                selected = builder.op("Where", [cond_val, true_carrier, false_carrier])
                _ensure_tensor_metadata(selected, out_shape, ir.DataType.INT64)
                result_dtype = true_val.dtype
                assert result_dtype is not None
                result = builder.op(
                    "Cast", [selected], {"to": int(result_dtype.value)}, output=out_spec
                )
                _ensure_tensor_metadata(result, out_shape, result_dtype)
                ctx.bind_value_for_var(out_var, result)
                return

            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Where")
            where_out = cast(
                ir.Value,
                builder.Where(
                    cond_val,
                    true_val,
                    false_val,
                    _outputs=[desired_name],
                ),
            )
            result_dtype = getattr(getattr(true_val, "type", None), "dtype", None)
            _ensure_tensor_metadata(where_out, out_shape, result_dtype)
            ctx.bind_value_for_var(out_var, where_out)
            return

        # --- Integer selector path -------------------------------------------------
        selector_i64 = selector_val
        if not np.issubdtype(selector_var.aval.dtype, np.integer):
            raise TypeError(
                "select_n with more than two cases requires integer selector"
            )
        if selector_var.aval.dtype != np.int64:
            selector_i64 = _cast_to_i64(ctx, selector_val, ctx.fresh_name("select_i64"))

        current_val = ctx.get_value_for_var(
            case_vars[0], name_hint=ctx.fresh_name("select_case0")
        )
        current_dtype = getattr(getattr(current_val, "type", None), "dtype", None)

        if len(case_vars) == 1:
            identity_out = cast(
                ir.Value,
                builder.Identity(
                    current_val,
                    _outputs=[ctx.fresh_name("select_identity")],
                ),
            )
            _ensure_tensor_metadata(identity_out, out_shape, current_dtype)
            ctx.bind_value_for_var(out_var, identity_out)
            return

        result_val = current_val
        for idx, case_var in enumerate(case_vars[1:], start=1):
            case_val = ctx.get_value_for_var(
                case_var, name_hint=ctx.fresh_name(f"select_case{idx}")
            )
            case_dtype = getattr(
                getattr(case_val, "type", None), "dtype", current_dtype
            )

            const_idx = _const_i64(
                ctx, np.asarray(idx, dtype=np.int64), ctx.fresh_name("select_const")
            )
            eq_val = cast(
                ir.Value,
                builder.Equal(
                    selector_i64,
                    const_idx,
                    _outputs=[ctx.fresh_name("select_eq")],
                ),
            )
            _ensure_tensor_metadata(eq_val, selector_shape, ir.DataType.BOOL)

            result_val = cast(
                ir.Value,
                builder.Where(
                    eq_val,
                    case_val,
                    result_val,
                    _outputs=[ctx.fresh_name("select_where")],
                ),
            )
            _ensure_tensor_metadata(result_val, out_shape, case_dtype)
            current_dtype = case_dtype

        ctx.bind_value_for_var(out_var, result_val)
