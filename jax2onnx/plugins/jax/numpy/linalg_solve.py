# jax2onnx/plugins/jax/numpy/linalg_solve.py

from __future__ import annotations

from typing import Callable, ClassVar, Final, cast

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
from jax2onnx.plugins.jax.numpy.linalg_inv import (
    _all_static_ints,
    _as_output_name,
    _binary_op,
    _gather_matrix_elem,
    _unsqueeze,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_LINALG_SOLVE_PRIM: Final = make_jnp_primitive("jax.numpy.linalg.solve")
_JAX_LINALG_SOLVE_ORIG: Final = jnp.linalg.solve


def _abstract_eval_via_orig(
    a: core.AbstractValue,
    b: core.AbstractValue,
) -> core.ShapedArray:
    a_shape = tuple(getattr(a, "shape", ()))
    b_shape = tuple(getattr(b, "shape", ()))
    a_dtype = np.dtype(getattr(a, "dtype", np.float32))
    b_dtype = np.dtype(getattr(b, "dtype", np.float32))
    if np.issubdtype(a_dtype, np.complexfloating) or np.issubdtype(
        b_dtype, np.complexfloating
    ):
        raise TypeError("jnp.linalg.solve lowering does not support complex inputs")

    orig = get_orig_impl(_LINALG_SOLVE_PRIM, "solve")
    out = jax.eval_shape(
        lambda a_val, b_val: orig(a_val, b_val),
        jax.ShapeDtypeStruct(a_shape, a_dtype),
        jax.ShapeDtypeStruct(b_shape, b_dtype),
    )
    out_dtype = np.dtype(getattr(out, "dtype", np.float32))
    if np.issubdtype(out_dtype, np.complexfloating):
        raise TypeError("jnp.linalg.solve lowering does not support complex outputs")
    return core.ShapedArray(tuple(getattr(out, "shape", ())), out_dtype)


def _cast_to_output_dtype(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    dtype_enum: ir.DataType,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    if getattr(getattr(val, "type", None), "dtype", None) == dtype_enum:
        val.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(val, shape)
        _ensure_value_metadata(ctx, val)
        return val
    result = cast(
        ir.Value,
        ctx.builder.Cast(
            val,
            to=int(dtype_enum.value),
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    result.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _gather_rhs_row(
    ctx: LoweringContextProtocol,
    rhs: ir.Value,
    *,
    rhs_shape: tuple[int, ...],
    row: int,
    dtype_enum: ir.DataType,
    name_hint: str,
) -> ir.Value:
    row_axis = len(rhs_shape) - 2 if len(rhs_shape) >= 2 else len(rhs_shape) - 1
    idx = _const_i64(ctx, np.asarray(row, dtype=np.int64), f"{name_hint}_idx")
    result = cast(
        ir.Value,
        ctx.builder.Gather(
            rhs,
            idx,
            axis=row_axis,
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    result.type = ir.TensorType(dtype_enum)
    out_shape = rhs_shape[:row_axis] + rhs_shape[row_axis + 1 :]
    _stamp_type_and_shape(result, out_shape)
    _ensure_value_metadata(ctx, result)
    return result


def _unsqueeze_to_shape(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    current_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    result = val
    shape = current_shape
    while len(shape) < len(target_shape):
        axis = len(shape)
        shape = shape + (1,)
        result = _unsqueeze(
            ctx,
            result,
            axis=axis,
            shape=shape,
            name_hint=f"{name_hint}_unsq",
        )
    return result


def _concat_two(
    ctx: LoweringContextProtocol,
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    axis: int,
    dtype_enum: ir.DataType,
    shape: tuple[int, ...],
    name_hint: str,
    output_name: str | None = None,
) -> ir.Value:
    result = cast(
        ir.Value,
        ctx.builder.Concat(
            lhs,
            rhs,
            axis=axis,
            _outputs=[output_name or ctx.fresh_name(name_hint)],
        ),
    )
    result.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _mul_coeff_rhs(
    ctx: LoweringContextProtocol,
    coeff: ir.Value,
    rhs: ir.Value,
    *,
    coeff_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    dtype_enum: ir.DataType,
    name_hint: str,
) -> ir.Value:
    coeff_ready = _unsqueeze_to_shape(
        ctx,
        coeff,
        current_shape=coeff_shape,
        target_shape=rhs_shape,
        name_hint=f"{name_hint}_coeff",
    )
    return _binary_op(
        ctx,
        "Mul",
        coeff_ready,
        rhs,
        dtype_enum=dtype_enum,
        shape=rhs_shape,
        name_hint=name_hint,
    )


def _stack_solution_rows(
    ctx: LoweringContextProtocol,
    x0: ir.Value,
    x1: ir.Value,
    *,
    row_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    dtype_enum: ir.DataType,
    output_name: str,
) -> ir.Value:
    row_axis = (
        len(output_shape) - 2 if len(output_shape) >= 2 else len(output_shape) - 1
    )
    row0 = _unsqueeze(
        ctx,
        x0,
        axis=row_axis,
        shape=output_shape[:row_axis] + (1,) + output_shape[row_axis + 1 :],
        name_hint="linalg_solve_x0_row",
    )
    row1 = _unsqueeze(
        ctx,
        x1,
        axis=row_axis,
        shape=output_shape[:row_axis] + (1,) + output_shape[row_axis + 1 :],
        name_hint="linalg_solve_x1_row",
    )
    del row_shape
    return _concat_two(
        ctx,
        row0,
        row1,
        axis=row_axis,
        dtype_enum=dtype_enum,
        shape=output_shape,
        name_hint="linalg_solve_result",
        output_name=output_name,
    )


@register_primitive(
    jaxpr_primitive=_LINALG_SOLVE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.solve.html",
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
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
    ],
    since="0.13.0",
    context="primitives.jnp",
    component="linalg_solve",
    testcases=[
        {
            "testcase": "linalg_solve_1x1_vector",
            "callable": lambda a, b: jnp.linalg.solve(a, b),
            "input_values": [
                np.asarray([[4.0]], dtype=np.float32),
                np.asarray([8.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(1,)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(["Div:1"], no_unused_inputs=True),
        },
        {
            "testcase": "linalg_solve_2x2_vector",
            "callable": lambda a, b: jnp.linalg.solve(a, b),
            "input_values": [
                np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=np.float32),
                np.asarray([9.0, 8.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gather", "Mul", "Sub", "Div", "Unsqueeze", "Concat:2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linalg_solve_batched_2x2_matrix_rhs",
            "callable": lambda a, b: jnp.linalg.solve(a, b),
            "input_values": [
                np.asarray(
                    [
                        [[3.0, 1.0], [1.0, 2.0]],
                        [[4.0, 7.0], [2.0, 6.0]],
                    ],
                    dtype=np.float32,
                ),
                np.asarray(
                    [
                        [[9.0, 4.0], [8.0, 3.0]],
                        [[1.0, 2.0], [3.0, 4.0]],
                    ],
                    dtype=np.float32,
                ),
            ],
            "expected_output_shapes": [(2, 2, 2)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gather", "Sub:2", "Div:2x2", "Unsqueeze", "Concat:2x2x2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpLinalgSolvePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _LINALG_SOLVE_PRIM
    _FUNC_NAME: ClassVar[str] = "solve"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        a: core.AbstractValue,
        b: core.AbstractValue,
    ) -> core.ShapedArray:
        return _abstract_eval_via_orig(a, b)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        a_var, b_var = eqn.invars
        (out_var,) = eqn.outvars

        a_shape_raw = tuple(getattr(a_var.aval, "shape", ()))
        b_shape_raw = tuple(getattr(b_var.aval, "shape", ()))
        out_shape_raw = tuple(getattr(out_var.aval, "shape", ()))
        if len(a_shape_raw) < 2 or len(b_shape_raw) < 1:
            raise ValueError("jnp.linalg.solve lowering requires matrix and RHS ranks")
        if not _all_static_ints(a_shape_raw[-2:]) or not _all_static_ints(b_shape_raw):
            raise TypeError(
                "jnp.linalg.solve lowering requires static matrix/RHS dimensions"
            )
        a_shape = tuple(int(dim) for dim in a_shape_raw)
        b_shape = tuple(int(dim) for dim in b_shape_raw)
        out_shape = tuple(int(dim) for dim in out_shape_raw)

        rows = a_shape[-2]
        cols = a_shape[-1]
        if rows != cols:
            raise ValueError("jnp.linalg.solve lowering requires square matrices")
        if rows not in (1, 2):
            raise NotImplementedError(
                "jnp.linalg.solve lowering currently supports 1x1 and 2x2 systems"
            )

        a_batch = a_shape[:-2]
        vector_rhs = len(b_shape) == 1 and len(a_batch) == 0
        matrix_rhs = len(b_shape) >= 2 and b_shape[-2] == rows
        if not vector_rhs and not matrix_rhs:
            raise ValueError("jnp.linalg.solve RHS shape is not supported")
        if matrix_rhs and b_shape[:-2] != a_batch:
            raise ValueError(
                "jnp.linalg.solve lowering requires matching batch dimensions"
            )

        a_dtype = np.dtype(getattr(a_var.aval, "dtype", np.float32))
        b_dtype = np.dtype(getattr(b_var.aval, "dtype", a_dtype))
        out_dtype = np.dtype(getattr(out_var.aval, "dtype", b_dtype))
        if (
            np.issubdtype(a_dtype, np.complexfloating)
            or np.issubdtype(b_dtype, np.complexfloating)
            or np.issubdtype(out_dtype, np.complexfloating)
        ):
            raise TypeError("jnp.linalg.solve lowering does not support complex dtypes")

        dtype_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("linalg_solve_a"))
        b_val = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("linalg_solve_b"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("linalg_solve_out")
        )
        desired_name = _as_output_name(ctx, out_spec, "linalg_solve_out")

        a_val = _cast_to_output_dtype(
            ctx,
            a_val,
            dtype_enum=dtype_enum,
            shape=a_shape,
            name_hint="linalg_solve_a_cast",
        )
        b_val = _cast_to_output_dtype(
            ctx,
            b_val,
            dtype_enum=dtype_enum,
            shape=b_shape,
            name_hint="linalg_solve_b_cast",
        )

        if rows == 1:
            a00 = _gather_matrix_elem(
                ctx,
                a_val,
                matrix_shape=a_shape,
                row=0,
                col=0,
                dtype_enum=dtype_enum,
                name_hint="linalg_solve_a00",
            )
            divisor = _unsqueeze_to_shape(
                ctx,
                a00,
                current_shape=a_batch,
                target_shape=out_shape,
                name_hint="linalg_solve_a00",
            )
            result = _binary_op(
                ctx,
                "Div",
                b_val,
                divisor,
                dtype_enum=dtype_enum,
                shape=out_shape,
                name_hint="linalg_solve_1x1",
                output_name=desired_name,
            )
            ctx.bind_value_for_var(out_var, result)
            return

        a = _gather_matrix_elem(
            ctx,
            a_val,
            matrix_shape=a_shape,
            row=0,
            col=0,
            dtype_enum=dtype_enum,
            name_hint="linalg_solve_a00",
        )
        b = _gather_matrix_elem(
            ctx,
            a_val,
            matrix_shape=a_shape,
            row=0,
            col=1,
            dtype_enum=dtype_enum,
            name_hint="linalg_solve_a01",
        )
        c = _gather_matrix_elem(
            ctx,
            a_val,
            matrix_shape=a_shape,
            row=1,
            col=0,
            dtype_enum=dtype_enum,
            name_hint="linalg_solve_a10",
        )
        d = _gather_matrix_elem(
            ctx,
            a_val,
            matrix_shape=a_shape,
            row=1,
            col=1,
            dtype_enum=dtype_enum,
            name_hint="linalg_solve_a11",
        )

        ad = _binary_op(
            ctx,
            "Mul",
            a,
            d,
            dtype_enum=dtype_enum,
            shape=a_batch,
            name_hint="linalg_solve_ad",
        )
        bc = _binary_op(
            ctx,
            "Mul",
            b,
            c,
            dtype_enum=dtype_enum,
            shape=a_batch,
            name_hint="linalg_solve_bc",
        )
        det = _binary_op(
            ctx,
            "Sub",
            ad,
            bc,
            dtype_enum=dtype_enum,
            shape=a_batch,
            name_hint="linalg_solve_det",
        )

        rhs0 = _gather_rhs_row(
            ctx,
            b_val,
            rhs_shape=b_shape,
            row=0,
            dtype_enum=dtype_enum,
            name_hint="linalg_solve_rhs0",
        )
        rhs1 = _gather_rhs_row(
            ctx,
            b_val,
            rhs_shape=b_shape,
            row=1,
            dtype_enum=dtype_enum,
            name_hint="linalg_solve_rhs1",
        )
        rhs_row_shape = tuple(getattr(rhs0, "shape", ()))
        if isinstance(rhs0.shape, ir.Shape):
            if not all(isinstance(dim, int) for dim in rhs0.shape.dims):
                raise TypeError("jnp.linalg.solve requires static RHS row shape")
            rhs_row_shape = cast(tuple[int, ...], tuple(rhs0.shape.dims))

        d_rhs0 = _mul_coeff_rhs(
            ctx,
            d,
            rhs0,
            coeff_shape=a_batch,
            rhs_shape=rhs_row_shape,
            dtype_enum=dtype_enum,
            name_hint="linalg_solve_d_rhs0",
        )
        b_rhs1 = _mul_coeff_rhs(
            ctx,
            b,
            rhs1,
            coeff_shape=a_batch,
            rhs_shape=rhs_row_shape,
            dtype_enum=dtype_enum,
            name_hint="linalg_solve_b_rhs1",
        )
        numerator0 = _binary_op(
            ctx,
            "Sub",
            d_rhs0,
            b_rhs1,
            dtype_enum=dtype_enum,
            shape=rhs_row_shape,
            name_hint="linalg_solve_num0",
        )

        a_rhs1 = _mul_coeff_rhs(
            ctx,
            a,
            rhs1,
            coeff_shape=a_batch,
            rhs_shape=rhs_row_shape,
            dtype_enum=dtype_enum,
            name_hint="linalg_solve_a_rhs1",
        )
        c_rhs0 = _mul_coeff_rhs(
            ctx,
            c,
            rhs0,
            coeff_shape=a_batch,
            rhs_shape=rhs_row_shape,
            dtype_enum=dtype_enum,
            name_hint="linalg_solve_c_rhs0",
        )
        numerator1 = _binary_op(
            ctx,
            "Sub",
            a_rhs1,
            c_rhs0,
            dtype_enum=dtype_enum,
            shape=rhs_row_shape,
            name_hint="linalg_solve_num1",
        )

        det_ready = _unsqueeze_to_shape(
            ctx,
            det,
            current_shape=a_batch,
            target_shape=rhs_row_shape,
            name_hint="linalg_solve_det",
        )
        x0 = _binary_op(
            ctx,
            "Div",
            numerator0,
            det_ready,
            dtype_enum=dtype_enum,
            shape=rhs_row_shape,
            name_hint="linalg_solve_x0",
        )
        x1 = _binary_op(
            ctx,
            "Div",
            numerator1,
            det_ready,
            dtype_enum=dtype_enum,
            shape=rhs_row_shape,
            name_hint="linalg_solve_x1",
        )

        result = _stack_solution_rows(
            ctx,
            x0,
            x1,
            row_shape=rhs_row_shape,
            output_shape=out_shape,
            dtype_enum=dtype_enum,
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
                raise RuntimeError("Original jnp.linalg.solve not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a: ArrayLike, b: ArrayLike) -> jax.Array:
                a_shape = getattr(a, "shape", None)
                b_shape = getattr(b, "shape", None)
                a_rank = getattr(a, "ndim", None)
                b_rank = getattr(b, "ndim", None)
                if (
                    isinstance(a_rank, int)
                    and isinstance(b_rank, int)
                    and a_rank >= 2
                    and b_rank >= 1
                    and a_shape is not None
                    and b_shape is not None
                    and isinstance(a_shape[-1], (int, np.integer))
                    and isinstance(a_shape[-2], (int, np.integer))
                    and int(a_shape[-1]) == int(a_shape[-2])
                    and int(a_shape[-1]) in (1, 2)
                ):
                    n = int(a_shape[-1])
                    a_batch = tuple(a_shape[:-2])
                    vector_rhs = a_rank == 2 and b_rank == 1 and int(b_shape[-1]) == n
                    matrix_rhs = (
                        b_rank >= 2
                        and tuple(b_shape[:-2]) == a_batch
                        and isinstance(b_shape[-2], (int, np.integer))
                        and int(b_shape[-2]) == n
                    )
                    if vector_rhs or matrix_rhs:
                        return cls._PRIM.bind(jnp.asarray(a), jnp.asarray(b))
                return orig(a, b)

            return _patched

        return [
            MonkeyPatchSpec(
                target="jax.numpy.linalg",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            )
        ]


@JnpLinalgSolvePlugin._PRIM.def_impl
def _linalg_solve_impl(a: ArrayLike, b: ArrayLike) -> jax.Array:
    try:
        orig = get_orig_impl(
            JnpLinalgSolvePlugin._PRIM,
            JnpLinalgSolvePlugin._FUNC_NAME,
        )
    except RuntimeError:
        orig = _JAX_LINALG_SOLVE_ORIG
    return orig(a, b)


JnpLinalgSolvePlugin._PRIM.def_abstract_eval(JnpLinalgSolvePlugin.abstract_eval)
