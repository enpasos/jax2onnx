# jax2onnx/plugins/jax/lax/triangular_solve.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _needs_transpose(transpose_a: object) -> bool:
    if isinstance(transpose_a, bool):
        return transpose_a
    name = str(getattr(transpose_a, "name", "")).upper()
    if name in {"TRANSPOSE", "ADJOINT"}:
        return True
    if isinstance(transpose_a, (int, np.integer)):
        return int(transpose_a) != 0
    return False


def _swap_last_two_shape(shape: tuple[object, ...]) -> tuple[object, ...]:
    if len(shape) < 2:
        return shape
    dims = list(shape)
    dims[-1], dims[-2] = dims[-2], dims[-1]
    return tuple(dims)


def _swap_last_two(
    ctx: "IRContext",
    val: ir.Value,
    shape: tuple[object, ...],
    name: str,
    *,
    output_name: str | None = None,
) -> ir.Value:
    if len(shape) < 2:
        raise ValueError("triangular_solve transpose requires rank >= 2")
    perm = list(range(len(shape)))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    out = ctx.builder.Transpose(
        val,
        perm=perm,
        _outputs=[output_name or ctx.fresh_name(name)],
    )
    if getattr(val, "type", None) is not None:
        out.type = val.type
    _stamp_type_and_shape(out, _swap_last_two_shape(shape))
    _ensure_value_metadata(ctx, out)
    return out


def _gather_scalar(
    ctx: "IRContext",
    data: ir.Value,
    *,
    axis: int,
    index: int,
    out_shape: tuple[object, ...],
    name: str,
) -> ir.Value:
    idx = _const_i64(ctx, np.asarray(index, dtype=np.int64), f"{name}_idx")
    out = ctx.builder.Gather(
        data,
        idx,
        axis=int(axis),
        _outputs=[ctx.fresh_name(name)],
    )
    if getattr(data, "type", None) is not None:
        out.type = data.type
    _stamp_type_and_shape(out, out_shape)
    _ensure_value_metadata(ctx, out)
    return out


def _unsqueeze(
    ctx: "IRContext",
    data: ir.Value,
    *,
    axis: int,
    out_shape: tuple[object, ...],
    name: str,
) -> ir.Value:
    axes = _const_i64(ctx, np.asarray([axis], dtype=np.int64), f"{name}_axes")
    out = ctx.builder.Unsqueeze(data, axes, _outputs=[ctx.fresh_name(name)])
    if getattr(data, "type", None) is not None:
        out.type = data.type
    _stamp_type_and_shape(out, out_shape)
    _ensure_value_metadata(ctx, out)
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.triangular_solve_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.triangular_solve.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "Div",
            "doc": "https://onnx.ai/onnx/operators/onnx__Div.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="triangular_solve",
    testcases=[
        {
            "testcase": "triangular_solve_left_basic",
            "callable": lambda a, b: jax.lax.linalg.triangular_solve(
                a, b, left_side=True, lower=True
            ),
            "input_values": [
                np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float32),
                np.array([[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], dtype=np.float32),
            ],
        },
        {
            "testcase": "triangular_solve_batched_unit_diag",
            "callable": lambda a, b: jax.lax.linalg.triangular_solve(
                a, b, left_side=True, lower=True, unit_diagonal=True
            ),
            "input_shapes": [(1, 1, 2, 2), (1, 1, 2, 2)],
            "run_only_f32_variant": True,
        },
    ],
)
class TriangularSolvePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        (a_var, b_var) = eqn.invars
        (out_var,) = eqn.outvars
        params = dict(getattr(eqn, "params", {}) or {})

        left_side = bool(params.get("left_side", True))
        lower = bool(params.get("lower", False))
        transpose_a = _needs_transpose(params.get("transpose_a", False))
        conjugate_a = bool(params.get("conjugate_a", False))
        unit_diagonal = bool(params.get("unit_diagonal", False))
        if conjugate_a:
            raise NotImplementedError(
                "triangular_solve with conjugate_a=True is not supported yet"
            )

        a_shape = tuple(getattr(getattr(a_var, "aval", None), "shape", ()))
        b_shape = tuple(getattr(getattr(b_var, "aval", None), "shape", ()))
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))
        if len(a_shape) < 2:
            raise ValueError("triangular_solve requires rank >= 2 for matrix input")

        n_dim = a_shape[-1]
        if not isinstance(n_dim, (int, np.integer)):
            raise NotImplementedError(
                "triangular_solve currently requires static matrix dimension"
            )
        n = int(n_dim)
        if n < 0:
            raise ValueError("triangular_solve matrix dimension must be non-negative")
        n_rows = a_shape[-2]
        if isinstance(n_rows, (int, np.integer)) and int(n_rows) != n:
            raise ValueError("triangular_solve requires square matrix input")

        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("tri_solve_a"))
        b_val = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("tri_solve_b"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("tri_solve_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "tri_solve_out"
        )
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("tri_solve_out")

        a_work = a_val
        a_work_shape = a_shape
        b_work = b_val
        b_work_shape = b_shape

        if not left_side:
            b_work = _swap_last_two(ctx, b_work, b_work_shape, "tri_solve_b_t")
            b_work_shape = _swap_last_two_shape(b_work_shape)
            transpose_a = not transpose_a

        if transpose_a:
            a_work = _swap_last_two(ctx, a_work, a_work_shape, "tri_solve_a_t")
            a_work_shape = _swap_last_two_shape(a_work_shape)
            lower = not lower

        work_out_shape = out_shape if left_side else _swap_last_two_shape(out_shape)
        if len(work_out_shape) < 2:
            raise ValueError("triangular_solve output rank must be >= 2")

        if n == 0:
            work_name = desired_name if left_side else ctx.fresh_name("tri_solve_work")
            empty_work = ctx.builder.Identity(b_work, _outputs=[work_name])
            if getattr(b_work, "type", None) is not None:
                empty_work.type = b_work.type
            _stamp_type_and_shape(empty_work, work_out_shape)
            _ensure_value_metadata(ctx, empty_work)

            result = empty_work
            if not left_side:
                result = _swap_last_two(
                    ctx,
                    empty_work,
                    work_out_shape,
                    "tri_solve_out",
                    output_name=desired_name,
                )
                if getattr(out_spec, "type", None) is not None:
                    result.type = out_spec.type
                _stamp_type_and_shape(result, out_shape)
                _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)
            return

        a_row_shape = a_work_shape[:-2] + (a_work_shape[-1],)
        a_elem_shape = a_work_shape[:-2]
        work_row_shape = work_out_shape[:-2] + (work_out_shape[-1],)
        work_row_with_axis_shape = work_out_shape[:-2] + (1, work_out_shape[-1])

        work_rank = len(work_out_shape)
        b_row_axis = work_rank - 2
        a_row_axis = len(a_work_shape) - 2
        a_col_axis = len(a_work_shape) - 2  # row gather removes one axis
        unsqueeze_a_axis = len(a_elem_shape)
        unsqueeze_row_axis = work_rank - 2

        def gather_b_row(i: int) -> ir.Value:
            return _gather_scalar(
                ctx,
                b_work,
                axis=b_row_axis,
                index=i,
                out_shape=work_row_shape,
                name=f"tri_solve_b_row_{i}",
            )

        def gather_a_elem(i: int, j: int) -> ir.Value:
            row = _gather_scalar(
                ctx,
                a_work,
                axis=a_row_axis,
                index=i,
                out_shape=a_row_shape,
                name=f"tri_solve_a_row_{i}",
            )
            return _gather_scalar(
                ctx,
                row,
                axis=a_col_axis,
                index=j,
                out_shape=a_elem_shape,
                name=f"tri_solve_a_elem_{i}_{j}",
            )

        x_rows: list[ir.Value | None] = [None] * n
        index_order: range | list[int]
        if lower:
            index_order = range(n)
        else:
            index_order = list(range(n - 1, -1, -1))

        for i in index_order:
            rhs = gather_b_row(i)
            acc: ir.Value | None = None
            k_iter = range(i) if lower else range(i + 1, n)
            for k in k_iter:
                xk = x_rows[k]
                if xk is None:
                    raise RuntimeError(
                        "triangular_solve internal ordering error: dependency missing"
                    )
                aik = gather_a_elem(i, k)
                aik_expanded = _unsqueeze(
                    ctx,
                    aik,
                    axis=unsqueeze_a_axis,
                    out_shape=a_elem_shape + (1,),
                    name=f"tri_solve_aik_unsq_{i}_{k}",
                )
                term = ctx.builder.Mul(
                    aik_expanded,
                    xk,
                    _outputs=[ctx.fresh_name(f"tri_solve_term_{i}_{k}")],
                )
                if getattr(xk, "type", None) is not None:
                    term.type = xk.type
                _stamp_type_and_shape(term, work_row_shape)
                _ensure_value_metadata(ctx, term)

                if acc is None:
                    acc = term
                else:
                    acc = ctx.builder.Add(
                        acc,
                        term,
                        _outputs=[ctx.fresh_name(f"tri_solve_acc_{i}_{k}")],
                    )
                    if getattr(term, "type", None) is not None:
                        acc.type = term.type
                    _stamp_type_and_shape(acc, work_row_shape)
                    _ensure_value_metadata(ctx, acc)

            rhs_adjusted = rhs
            if acc is not None:
                rhs_adjusted = ctx.builder.Sub(
                    rhs,
                    acc,
                    _outputs=[ctx.fresh_name(f"tri_solve_rhs_{i}")],
                )
                if getattr(rhs, "type", None) is not None:
                    rhs_adjusted.type = rhs.type
                _stamp_type_and_shape(rhs_adjusted, work_row_shape)
                _ensure_value_metadata(ctx, rhs_adjusted)

            if unit_diagonal:
                x_rows[i] = rhs_adjusted
            else:
                aii = gather_a_elem(i, i)
                aii_expanded = _unsqueeze(
                    ctx,
                    aii,
                    axis=unsqueeze_a_axis,
                    out_shape=a_elem_shape + (1,),
                    name=f"tri_solve_aii_unsq_{i}",
                )
                xi = ctx.builder.Div(
                    rhs_adjusted,
                    aii_expanded,
                    _outputs=[ctx.fresh_name(f"tri_solve_x_{i}")],
                )
                if getattr(rhs_adjusted, "type", None) is not None:
                    xi.type = rhs_adjusted.type
                _stamp_type_and_shape(xi, work_row_shape)
                _ensure_value_metadata(ctx, xi)
                x_rows[i] = xi

        stacked_rows: list[ir.Value] = []
        for i, row in enumerate(x_rows):
            if row is None:
                raise RuntimeError(
                    "triangular_solve internal error: unresolved output row"
                )
            expanded = _unsqueeze(
                ctx,
                row,
                axis=unsqueeze_row_axis,
                out_shape=work_row_with_axis_shape,
                name=f"tri_solve_row_unsq_{i}",
            )
            stacked_rows.append(expanded)

        work_name = desired_name if left_side else ctx.fresh_name("tri_solve_work")
        result_work = ctx.builder.Concat(
            *stacked_rows,
            axis=unsqueeze_row_axis,
            _outputs=[work_name],
        )
        if left_side and getattr(out_spec, "type", None) is not None:
            result_work.type = out_spec.type
        elif getattr(b_work, "type", None) is not None:
            result_work.type = b_work.type
        _stamp_type_and_shape(result_work, work_out_shape)
        _ensure_value_metadata(ctx, result_work)

        result = result_work
        if not left_side:
            result = _swap_last_two(
                ctx,
                result_work,
                work_out_shape,
                "tri_solve_out",
                output_name=desired_name,
            )
            if getattr(out_spec, "type", None) is not None:
                result.type = out_spec.type
            _stamp_type_and_shape(result, out_shape)
            _ensure_value_metadata(ctx, result)

        ctx.bind_value_for_var(out_var, result)
