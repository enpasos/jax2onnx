# jax2onnx/plugins/jax/numpy/linalg_inv.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, cast

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_LINALG_INV_PRIM: Final = make_jnp_primitive("jax.numpy.linalg.inv")
_JAX_LINALG_INV_ORIG: Final = jnp.linalg.inv


def _all_static_ints(shape: tuple[object, ...]) -> bool:
    return all(isinstance(dim, (int, np.integer)) for dim in shape)


def _as_output_name(
    ctx: LoweringContextProtocol,
    spec: ir.Value,
    name_hint: str,
) -> str:
    desired_name = cast(str, getattr(spec, "name", None) or ctx.fresh_name(name_hint))
    producer = getattr(spec, "producer", None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name(name_hint)
    return desired_name


def _const_scalar(
    ctx: LoweringContextProtocol,
    *,
    dtype: np.dtype[Any],
    value: float,
    name_hint: str,
) -> ir.Value:
    dtype_enum = _dtype_to_ir(dtype, ctx.builder.enable_double_precision)
    result: ir.Value = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(name_hint),
        array=np.asarray(value, dtype=dtype),
    )
    result.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(result, ())
    _ensure_value_metadata(ctx, result)
    return result


def _gather_matrix_elem(
    ctx: LoweringContextProtocol,
    matrix: ir.Value,
    *,
    matrix_shape: tuple[int, ...],
    row: int,
    col: int,
    dtype_enum: ir.DataType,
    name_hint: str,
) -> ir.Value:
    batch_shape = matrix_shape[:-2]
    row_axis = len(matrix_shape) - 2
    row_idx = _const_i64(ctx, np.asarray(row, dtype=np.int64), f"{name_hint}_row")
    row_val = cast(
        ir.Value,
        ctx.builder.Gather(
            matrix,
            row_idx,
            axis=row_axis,
            _outputs=[ctx.fresh_name(f"{name_hint}_row")],
        ),
    )
    row_val.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(row_val, batch_shape + (matrix_shape[-1],))
    _ensure_value_metadata(ctx, row_val)

    col_axis = len(batch_shape)
    col_idx = _const_i64(ctx, np.asarray(col, dtype=np.int64), f"{name_hint}_col")
    elem = cast(
        ir.Value,
        ctx.builder.Gather(
            row_val,
            col_idx,
            axis=col_axis,
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    elem.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(elem, batch_shape)
    _ensure_value_metadata(ctx, elem)
    return elem


def _binary_op(
    ctx: LoweringContextProtocol,
    op_type: str,
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    dtype_enum: ir.DataType,
    shape: tuple[int, ...],
    name_hint: str,
    output_name: str | None = None,
) -> ir.Value:
    outputs = [output_name or ctx.fresh_name(name_hint)]
    if op_type == "Add":
        result = cast(ir.Value, ctx.builder.Add(lhs, rhs, _outputs=outputs))
    elif op_type == "Sub":
        result = cast(ir.Value, ctx.builder.Sub(lhs, rhs, _outputs=outputs))
    elif op_type == "Mul":
        result = cast(ir.Value, ctx.builder.Mul(lhs, rhs, _outputs=outputs))
    elif op_type == "Div":
        result = cast(ir.Value, ctx.builder.Div(lhs, rhs, _outputs=outputs))
    else:  # pragma: no cover - internal guard
        raise ValueError(f"Unsupported linalg.inv op: {op_type}")
    result.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _neg(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    dtype_enum: ir.DataType,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    result = cast(
        ir.Value,
        ctx.builder.Neg(val, _outputs=[ctx.fresh_name(name_hint)]),
    )
    result.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _unsqueeze(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    axis: int,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    axes = _const_i64(ctx, np.asarray([axis], dtype=np.int64), f"{name_hint}_axes")
    result = cast(
        ir.Value,
        ctx.builder.Unsqueeze(
            val,
            axes,
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    if getattr(val, "type", None) is not None:
        result.type = val.type
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _concat(
    ctx: LoweringContextProtocol,
    values: tuple[ir.Value, ir.Value],
    *,
    axis: int,
    dtype_enum: ir.DataType,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    result = cast(
        ir.Value,
        ctx.builder.Concat(
            *values,
            axis=axis,
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    result.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _abstract_eval_via_orig(x: core.AbstractValue) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    x_dtype = np.dtype(getattr(x, "dtype", np.float32))
    if np.issubdtype(x_dtype, np.complexfloating):
        raise TypeError("jnp.linalg.inv lowering does not support complex inputs")
    orig = get_orig_impl(_LINALG_INV_PRIM, "inv")
    out = jax.eval_shape(
        lambda value: orig(value), jax.ShapeDtypeStruct(x_shape, x_dtype)
    )
    out_dtype = np.dtype(getattr(out, "dtype", np.float32))
    if np.issubdtype(out_dtype, np.complexfloating):
        raise TypeError("jnp.linalg.inv lowering does not support complex outputs")
    return core.ShapedArray(tuple(getattr(out, "shape", ())), out_dtype)


@register_primitive(
    jaxpr_primitive=_LINALG_INV_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.inv.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "Mul",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html",
        },
        {
            "component": "Sub",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html",
        },
        {"component": "Neg", "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html"},
        {
            "component": "Div",
            "doc": "https://onnx.ai/onnx/operators/onnx__Div.html",
        },
        {
            "component": "Unsqueeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unsqueeze.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.13.0",
    context="primitives.jnp",
    component="linalg_inv",
    testcases=[
        {
            "testcase": "linalg_inv_1x1",
            "callable": lambda x: jnp.linalg.inv(x),
            "input_values": [np.asarray([[4.0]], dtype=np.float32)],
            "expected_output_shapes": [(1, 1)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(["Div:1x1"], no_unused_inputs=True),
        },
        {
            "testcase": "linalg_inv_2x2",
            "callable": lambda x: jnp.linalg.inv(x),
            "input_values": [np.asarray([[4.0, 7.0], [2.0, 6.0]], dtype=np.float32)],
            "expected_output_shapes": [(2, 2)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gather", "Mul", "Sub", "Neg", "Concat", "Div:2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linalg_inv_batched_2x2",
            "callable": lambda x: jnp.linalg.inv(x),
            "input_values": [
                np.asarray(
                    [
                        [[4.0, 7.0], [2.0, 6.0]],
                        [[3.0, 1.0], [2.0, 5.0]],
                    ],
                    dtype=np.float32,
                )
            ],
            "expected_output_shapes": [(2, 2, 2)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gather", "Sub:2", "Unsqueeze", "Concat:2x2x2", "Div:2x2x2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpLinalgInvPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _LINALG_INV_PRIM
    _FUNC_NAME: ClassVar[str] = "inv"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        return _abstract_eval_via_orig(x)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_shape_raw = tuple(getattr(x_var.aval, "shape", ()))
        if len(x_shape_raw) < 2:
            raise ValueError("jnp.linalg.inv lowering requires rank >= 2")
        if not _all_static_ints(x_shape_raw[-2:]):
            raise TypeError(
                "jnp.linalg.inv lowering requires static trailing matrix dimensions"
            )
        x_shape = tuple(int(dim) for dim in x_shape_raw)
        rows = int(x_shape[-2])
        cols = int(x_shape[-1])
        if rows != cols:
            raise ValueError("jnp.linalg.inv lowering requires square matrices")
        if rows not in (1, 2):
            raise NotImplementedError(
                "jnp.linalg.inv lowering currently supports 1x1 and 2x2 matrices"
            )

        x_dtype = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        out_dtype = np.dtype(getattr(out_var.aval, "dtype", x_dtype))
        if np.issubdtype(x_dtype, np.complexfloating) or np.issubdtype(
            out_dtype, np.complexfloating
        ):
            raise TypeError("jnp.linalg.inv lowering does not support complex dtypes")

        dtype_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("linalg_inv_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("linalg_inv_out")
        )
        desired_name = _as_output_name(ctx, out_spec, "linalg_inv_out")

        if getattr(getattr(x_val, "type", None), "dtype", None) != dtype_enum:
            x_val = ctx.builder.Cast(
                x_val,
                to=int(dtype_enum.value),
                _outputs=[ctx.fresh_name("linalg_inv_cast")],
            )
            x_val.type = ir.TensorType(dtype_enum)
        else:
            x_val.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(x_val, x_shape)
        _ensure_value_metadata(ctx, x_val)

        if rows == 1:
            one = _const_scalar(
                ctx,
                dtype=out_dtype,
                value=1.0,
                name_hint="linalg_inv_one",
            )
            result = _binary_op(
                ctx,
                "Div",
                one,
                x_val,
                dtype_enum=dtype_enum,
                shape=x_shape,
                name_hint="linalg_inv_1x1",
                output_name=desired_name,
            )
            ctx.bind_value_for_var(out_var, result)
            return

        batch_shape = x_shape[:-2]
        elem_shape = batch_shape
        a = _gather_matrix_elem(
            ctx,
            x_val,
            matrix_shape=x_shape,
            row=0,
            col=0,
            dtype_enum=dtype_enum,
            name_hint="linalg_inv_a",
        )
        b = _gather_matrix_elem(
            ctx,
            x_val,
            matrix_shape=x_shape,
            row=0,
            col=1,
            dtype_enum=dtype_enum,
            name_hint="linalg_inv_b",
        )
        c = _gather_matrix_elem(
            ctx,
            x_val,
            matrix_shape=x_shape,
            row=1,
            col=0,
            dtype_enum=dtype_enum,
            name_hint="linalg_inv_c",
        )
        d = _gather_matrix_elem(
            ctx,
            x_val,
            matrix_shape=x_shape,
            row=1,
            col=1,
            dtype_enum=dtype_enum,
            name_hint="linalg_inv_d",
        )

        ad = _binary_op(
            ctx,
            "Mul",
            a,
            d,
            dtype_enum=dtype_enum,
            shape=elem_shape,
            name_hint="linalg_inv_ad",
        )
        bc = _binary_op(
            ctx,
            "Mul",
            b,
            c,
            dtype_enum=dtype_enum,
            shape=elem_shape,
            name_hint="linalg_inv_bc",
        )
        det = _binary_op(
            ctx,
            "Sub",
            ad,
            bc,
            dtype_enum=dtype_enum,
            shape=elem_shape,
            name_hint="linalg_inv_det",
        )

        neg_b = _neg(
            ctx,
            b,
            dtype_enum=dtype_enum,
            shape=elem_shape,
            name_hint="linalg_inv_neg_b",
        )
        neg_c = _neg(
            ctx,
            c,
            dtype_enum=dtype_enum,
            shape=elem_shape,
            name_hint="linalg_inv_neg_c",
        )

        elem_axis = len(batch_shape)
        row_shape = batch_shape + (2,)
        top = _concat(
            ctx,
            (
                _unsqueeze(
                    ctx,
                    d,
                    axis=elem_axis,
                    shape=batch_shape + (1,),
                    name_hint="linalg_inv_d_col",
                ),
                _unsqueeze(
                    ctx,
                    neg_b,
                    axis=elem_axis,
                    shape=batch_shape + (1,),
                    name_hint="linalg_inv_neg_b_col",
                ),
            ),
            axis=elem_axis,
            dtype_enum=dtype_enum,
            shape=row_shape,
            name_hint="linalg_inv_top_row",
        )
        bottom = _concat(
            ctx,
            (
                _unsqueeze(
                    ctx,
                    neg_c,
                    axis=elem_axis,
                    shape=batch_shape + (1,),
                    name_hint="linalg_inv_neg_c_col",
                ),
                _unsqueeze(
                    ctx,
                    a,
                    axis=elem_axis,
                    shape=batch_shape + (1,),
                    name_hint="linalg_inv_a_col",
                ),
            ),
            axis=elem_axis,
            dtype_enum=dtype_enum,
            shape=row_shape,
            name_hint="linalg_inv_bottom_row",
        )

        matrix_axis = len(batch_shape)
        top_matrix = _unsqueeze(
            ctx,
            top,
            axis=matrix_axis,
            shape=batch_shape + (1, 2),
            name_hint="linalg_inv_top_matrix",
        )
        bottom_matrix = _unsqueeze(
            ctx,
            bottom,
            axis=matrix_axis,
            shape=batch_shape + (1, 2),
            name_hint="linalg_inv_bottom_matrix",
        )
        adjugate = _concat(
            ctx,
            (top_matrix, bottom_matrix),
            axis=matrix_axis,
            dtype_enum=dtype_enum,
            shape=x_shape,
            name_hint="linalg_inv_adjugate",
        )

        det_matrix = _unsqueeze(
            ctx,
            _unsqueeze(
                ctx,
                det,
                axis=elem_axis,
                shape=batch_shape + (1,),
                name_hint="linalg_inv_det_col",
            ),
            axis=elem_axis + 1,
            shape=batch_shape + (1, 1),
            name_hint="linalg_inv_det_matrix",
        )
        result = _binary_op(
            ctx,
            "Div",
            adjugate,
            det_matrix,
            dtype_enum=dtype_enum,
            shape=x_shape,
            name_hint="linalg_inv_result",
            output_name=desired_name,
        )
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.linalg.inv not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a: ArrayLike) -> jax.Array:
                rank = getattr(a, "ndim", None)
                shape = getattr(a, "shape", None)
                if (
                    isinstance(rank, int)
                    and rank >= 2
                    and shape is not None
                    and len(shape) >= 2
                    and isinstance(shape[-1], (int, np.integer))
                    and isinstance(shape[-2], (int, np.integer))
                    and int(shape[-1]) == int(shape[-2])
                    and int(shape[-1]) in (1, 2)
                ):
                    return cls._PRIM.bind(jnp.asarray(a))
                return orig(a)

            return _patched

        return [
            MonkeyPatchSpec(
                target="jax.numpy.linalg",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            )
        ]


@JnpLinalgInvPlugin._PRIM.def_impl
def _linalg_inv_impl(x: ArrayLike) -> jax.Array:
    try:
        orig = get_orig_impl(JnpLinalgInvPlugin._PRIM, JnpLinalgInvPlugin._FUNC_NAME)
    except RuntimeError:
        orig = _JAX_LINALG_INV_ORIG
    return orig(x)


JnpLinalgInvPlugin._PRIM.def_abstract_eval(JnpLinalgInvPlugin.abstract_eval)
