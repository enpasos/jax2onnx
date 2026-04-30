# jax2onnx/plugins/jax/lax/svd.py

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _as_value(value: Any) -> ir.Value:
    return cast(ir.Value, value)


def _stamp_like(value: ir.Value, ref: ir.Value) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


def _gather_mat_elem(
    ctx: LoweringContextProtocol, mat: ir.Value, i: int, j: int, name: str
) -> ir.Value:
    i_idx = _const_i64(ctx, np.asarray([i], dtype=np.int64), f"{name}_i")
    row = _as_value(
        ctx.builder.Gather(
            mat,
            i_idx,
            axis=0,
            _outputs=[ctx.fresh_name(f"{name}_row")],
        )
    )
    if getattr(mat, "type", None) is not None:
        row.type = mat.type
    j_idx = _const_i64(ctx, np.asarray([j], dtype=np.int64), f"{name}_j")
    elem = _as_value(
        ctx.builder.Gather(
            row,
            j_idx,
            axis=1,
            _outputs=[ctx.fresh_name(name)],
        )
    )
    if getattr(mat, "type", None) is not None:
        elem.type = mat.type
    elem.shape = ir.Shape((1, 1))
    return elem


def _cast_if_needed(
    ctx: LoweringContextProtocol, value: ir.Value, target: ir.DataType, name_hint: str
) -> ir.Value:
    current = getattr(value, "dtype", None)
    if current is None:
        value_type = getattr(value, "type", None)
        if isinstance(value_type, ir.TensorType):
            current = value_type.dtype
    if current == target:
        return value
    casted = _as_value(
        ctx.builder.Cast(
            value,
            to=int(target.value),
            _outputs=[ctx.fresh_name(name_hint)],
        )
    )
    casted.type = ir.TensorType(target)
    if getattr(value, "shape", None) is not None:
        casted.shape = value.shape
    return casted


def _reshape_to_1d_len1(
    ctx: LoweringContextProtocol, value: ir.Value, name_hint: str
) -> ir.Value:
    one_shape = _const_i64(ctx, np.asarray([1], dtype=np.int64), f"{name_hint}_shape")
    out = _as_value(
        ctx.builder.Reshape(
            value,
            one_shape,
            _outputs=[ctx.fresh_name(name_hint)],
        )
    )
    if getattr(value, "type", None) is not None:
        out.type = value.type
    out.shape = ir.Shape((1,))
    return out


def _gather_col(
    ctx: LoweringContextProtocol, mat: ir.Value, j: int, name: str
) -> ir.Value:
    j_idx = _const_i64(ctx, np.asarray([j], dtype=np.int64), f"{name}_j")
    col = _as_value(
        ctx.builder.Gather(
            mat,
            j_idx,
            axis=1,
            _outputs=[ctx.fresh_name(name)],
        )
    )
    if getattr(mat, "type", None) is not None:
        col.type = mat.type
    return col


def _gather_row(
    ctx: LoweringContextProtocol, mat: ir.Value, i: int, name: str
) -> ir.Value:
    i_idx = _const_i64(ctx, np.asarray([i], dtype=np.int64), f"{name}_i")
    row = _as_value(
        ctx.builder.Gather(
            mat,
            i_idx,
            axis=0,
            _outputs=[ctx.fresh_name(name)],
        )
    )
    if getattr(mat, "type", None) is not None:
        row.type = mat.type
    return row


def _sum_2d_to_scalar(
    ctx: LoweringContextProtocol, value: ir.Value, name: str
) -> ir.Value:
    axes = _const_i64(ctx, np.asarray([0, 1], dtype=np.int64), f"{name}_axes")
    out = _as_value(
        ctx.builder.ReduceSum(
            value,
            axes,
            keepdims=1,
            _outputs=[ctx.fresh_name(name)],
        )
    )
    if getattr(value, "type", None) is not None:
        out.type = value.type
    out.shape = ir.Shape((1, 1))
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.svd_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.svd.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {"component": "Abs", "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html"},
        {"component": "Sign", "doc": "https://onnx.ai/onnx/operators/onnx__Sign.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Max", "doc": "https://onnx.ai/onnx/operators/onnx__Max.html"},
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="svd",
    testcases=[
        {
            "testcase": "svd_1x1_default",
            "callable": lambda x: jax.lax.linalg.svd(x),
            "input_values": [np.asarray([[-2.0]], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Abs", "Sign", "Where"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "svd_1x1_values_only",
            "callable": lambda x: jax.lax.linalg.svd(x, compute_uv=False),
            "input_values": [np.asarray([[0.5]], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Abs", "Reshape"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "svd_2x2_values_only",
            "callable": lambda x: jax.lax.linalg.svd(x, compute_uv=False),
            "input_values": [
                np.asarray(
                    [
                        [3.0, 1.0],
                        [0.0, 2.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["Sqrt", "Concat"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "svd_3x2_values_only",
            "callable": lambda x: jax.lax.linalg.svd(x, compute_uv=False),
            "input_values": [
                np.asarray(
                    [
                        [3.0, 1.0],
                        [0.0, 2.0],
                        [1.0, -1.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["ReduceSum", "Sqrt", "Concat"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "svd_2x4_values_only",
            "callable": lambda x: jax.lax.linalg.svd(x, compute_uv=False),
            "input_values": [
                np.asarray(
                    [
                        [2.0, -1.0, 0.5, 3.0],
                        [1.0, 0.0, -2.0, 4.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["ReduceSum", "Sqrt", "Concat"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SvdPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.svd`` for static real ``1x1`` and values-only when ``min(m, n)==2``."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        params = dict(getattr(eqn, "params", {}) or {})
        compute_uv = bool(params.get("compute_uv", True))
        subset_by_index = params.get("subset_by_index", None)
        if subset_by_index is not None:
            raise NotImplementedError("svd subset_by_index is not supported yet")

        (x_var,) = eqn.invars
        outvars = list(eqn.outvars)
        expected_n = 3 if compute_uv else 1
        if len(outvars) != expected_n:
            raise NotImplementedError(
                f"svd output arity mismatch: expected {expected_n}, got {len(outvars)}"
            )

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("svd_in"))
        x_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )
        if np.issubdtype(x_dtype, np.complexfloating):
            raise NotImplementedError("svd complex input is not supported yet")
        if not np.issubdtype(x_dtype, np.floating):
            raise TypeError(f"svd requires floating input dtype, got '{x_dtype}'")

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if len(x_shape) != 2:
            raise NotImplementedError("svd currently supports rank-2 inputs only")
        m_raw, n_raw = x_shape
        if not isinstance(m_raw, (int, np.integer)) or not isinstance(
            n_raw, (int, np.integer)
        ):
            raise NotImplementedError("svd requires static matrix dimensions")
        m = int(m_raw)
        n = int(n_raw)
        if min(m, n) == 2 and not compute_uv:
            s_var = outvars[0]
            s_spec = ctx.get_value_for_var(s_var, name_hint=ctx.fresh_name("svd_s_out"))
            s_dtype = np.dtype(getattr(getattr(s_var, "aval", None), "dtype", x_dtype))
            s_dtype_enum = _dtype_to_ir(s_dtype, ctx.builder.enable_double_precision)
            if s_dtype_enum is None:
                raise TypeError(f"Unsupported svd singular value dtype '{s_dtype}'")

            scalar_np_dtype = (
                np.float64 if s_dtype_enum == ir.DataType.DOUBLE else np.float32
            )
            zero = ctx.bind_const_for_var(
                object(), np.asarray([[0.0]], dtype=scalar_np_dtype)
            )
            two = ctx.bind_const_for_var(
                object(), np.asarray([[2.0]], dtype=scalar_np_dtype)
            )
            four = ctx.bind_const_for_var(
                object(), np.asarray([[4.0]], dtype=scalar_np_dtype)
            )
            for const in (zero, two, four):
                const.type = ir.TensorType(s_dtype_enum)
                const.shape = ir.Shape((1, 1))

            if n == 2:
                # Use A^T A when there are exactly two columns.
                col0 = _cast_if_needed(
                    ctx,
                    _gather_col(ctx, x, 0, "svd_col0"),
                    s_dtype_enum,
                    "svd_col0_cast",
                )
                col1 = _cast_if_needed(
                    ctx,
                    _gather_col(ctx, x, 1, "svd_col1"),
                    s_dtype_enum,
                    "svd_col1_cast",
                )
                p_prod = _as_value(
                    ctx.builder.Mul(
                        col0, col0, _outputs=[ctx.fresh_name("svd2_p_prod")]
                    )
                )
                q_prod = _as_value(
                    ctx.builder.Mul(
                        col0, col1, _outputs=[ctx.fresh_name("svd2_q_prod")]
                    )
                )
                r_prod = _as_value(
                    ctx.builder.Mul(
                        col1, col1, _outputs=[ctx.fresh_name("svd2_r_prod")]
                    )
                )
                for val in (p_prod, q_prod, r_prod):
                    val.type = ir.TensorType(s_dtype_enum)
                p = _sum_2d_to_scalar(ctx, p_prod, "svd2_p")
                q = _sum_2d_to_scalar(ctx, q_prod, "svd2_q")
                r = _sum_2d_to_scalar(ctx, r_prod, "svd2_r")
            else:
                # Use A A^T when there are exactly two rows.
                row0 = _cast_if_needed(
                    ctx,
                    _gather_row(ctx, x, 0, "svd_row0"),
                    s_dtype_enum,
                    "svd_row0_cast",
                )
                row1 = _cast_if_needed(
                    ctx,
                    _gather_row(ctx, x, 1, "svd_row1"),
                    s_dtype_enum,
                    "svd_row1_cast",
                )
                p_prod = _as_value(
                    ctx.builder.Mul(
                        row0, row0, _outputs=[ctx.fresh_name("svd2_p_prod")]
                    )
                )
                q_prod = _as_value(
                    ctx.builder.Mul(
                        row0, row1, _outputs=[ctx.fresh_name("svd2_q_prod")]
                    )
                )
                r_prod = _as_value(
                    ctx.builder.Mul(
                        row1, row1, _outputs=[ctx.fresh_name("svd2_r_prod")]
                    )
                )
                for val in (p_prod, q_prod, r_prod):
                    val.type = ir.TensorType(s_dtype_enum)
                p = _sum_2d_to_scalar(ctx, p_prod, "svd2_p")
                q = _sum_2d_to_scalar(ctx, q_prod, "svd2_q")
                r = _sum_2d_to_scalar(ctx, r_prod, "svd2_r")
            p.type = ir.TensorType(s_dtype_enum)
            q.type = ir.TensorType(s_dtype_enum)
            r.type = ir.TensorType(s_dtype_enum)

            tr = _as_value(ctx.builder.Add(p, r, _outputs=[ctx.fresh_name("svd2_tr")]))
            pr_diff = _as_value(
                ctx.builder.Sub(p, r, _outputs=[ctx.fresh_name("svd2_pr_diff")])
            )
            tr.type = ir.TensorType(s_dtype_enum)
            pr_diff.type = ir.TensorType(s_dtype_enum)

            pr_diff2 = _as_value(
                ctx.builder.Mul(
                    pr_diff, pr_diff, _outputs=[ctx.fresh_name("svd2_pr_diff2")]
                )
            )
            q2 = _as_value(ctx.builder.Mul(q, q, _outputs=[ctx.fresh_name("svd2_q2")]))
            four_q2 = _as_value(
                ctx.builder.Mul(four, q2, _outputs=[ctx.fresh_name("svd2_four_q2")])
            )
            pr_diff2.type = ir.TensorType(s_dtype_enum)
            q2.type = ir.TensorType(s_dtype_enum)
            four_q2.type = ir.TensorType(s_dtype_enum)

            disc2 = _as_value(
                ctx.builder.Add(
                    pr_diff2,
                    four_q2,
                    _outputs=[ctx.fresh_name("svd2_disc2")],
                )
            )
            disc2.type = ir.TensorType(s_dtype_enum)
            disc = _as_value(
                ctx.builder.Sqrt(disc2, _outputs=[ctx.fresh_name("svd2_disc")])
            )
            disc.type = ir.TensorType(s_dtype_enum)

            mu_max_num = _as_value(
                ctx.builder.Add(
                    tr,
                    disc,
                    _outputs=[ctx.fresh_name("svd2_mu_max_num")],
                )
            )
            mu_min_num = _as_value(
                ctx.builder.Sub(
                    tr,
                    disc,
                    _outputs=[ctx.fresh_name("svd2_mu_min_num")],
                )
            )
            mu_max_num.type = ir.TensorType(s_dtype_enum)
            mu_min_num.type = ir.TensorType(s_dtype_enum)

            mu_max = _as_value(
                ctx.builder.Div(
                    mu_max_num,
                    two,
                    _outputs=[ctx.fresh_name("svd2_mu_max")],
                )
            )
            mu_min = _as_value(
                ctx.builder.Div(
                    mu_min_num,
                    two,
                    _outputs=[ctx.fresh_name("svd2_mu_min")],
                )
            )
            mu_max.type = ir.TensorType(s_dtype_enum)
            mu_min.type = ir.TensorType(s_dtype_enum)

            mu_max_clip = _as_value(
                ctx.builder.Max(
                    mu_max,
                    zero,
                    _outputs=[ctx.fresh_name("svd2_mu_max_clip")],
                )
            )
            mu_min_clip = _as_value(
                ctx.builder.Max(
                    mu_min,
                    zero,
                    _outputs=[ctx.fresh_name("svd2_mu_min_clip")],
                )
            )
            mu_max_clip.type = ir.TensorType(s_dtype_enum)
            mu_min_clip.type = ir.TensorType(s_dtype_enum)

            s_max = _as_value(
                ctx.builder.Sqrt(mu_max_clip, _outputs=[ctx.fresh_name("svd2_s_max")])
            )
            s_min = _as_value(
                ctx.builder.Sqrt(mu_min_clip, _outputs=[ctx.fresh_name("svd2_s_min")])
            )
            s_max.type = ir.TensorType(s_dtype_enum)
            s_min.type = ir.TensorType(s_dtype_enum)

            s_max_1d = _reshape_to_1d_len1(ctx, s_max, "svd2_s_max_1d")
            s_min_1d = _reshape_to_1d_len1(ctx, s_min, "svd2_s_min_1d")
            s_val = _as_value(
                ctx.builder.Concat(
                    s_max_1d,
                    s_min_1d,
                    axis=0,
                    _outputs=[ctx.fresh_name("svd2_s")],
                )
            )
            s_val.type = ir.TensorType(s_dtype_enum)
            s_val.shape = ir.Shape((2,))

            s_name = getattr(s_spec, "name", None) or ctx.fresh_name("svd_s")
            s_out = s_val
            if getattr(s_out, "name", None) != s_name:
                s_out = _as_value(ctx.builder.Identity(s_val, _outputs=[s_name]))
            _stamp_like(s_out, s_spec if getattr(s_spec, "type", None) else s_val)
            if getattr(s_spec, "shape", None) is not None:
                s_out.shape = s_spec.shape
            ctx.bind_value_for_var(s_var, s_out)
            return

        if m != 1 or n != 1:
            raise NotImplementedError(
                "svd currently supports only 1x1 (full) and values-only with min(m, n)==2"
            )

        x00 = _gather_mat_elem(ctx, x, 0, 0, "svd_x00")
        s00 = _as_value(ctx.builder.Abs(x00, _outputs=[ctx.fresh_name("svd_s00")]))
        _stamp_like(s00, x00)
        s_shape = _const_i64(ctx, np.asarray([1], dtype=np.int64), "svd_s_shape")
        s_val = _as_value(
            ctx.builder.Reshape(
                s00,
                s_shape,
                _outputs=[ctx.fresh_name("svd_s")],
            )
        )
        if getattr(s00, "type", None) is not None:
            s_val.type = s00.type
        s_val.shape = ir.Shape((1,))

        s_var = outvars[0]
        s_spec = ctx.get_value_for_var(s_var, name_hint=ctx.fresh_name("svd_s_out"))
        s_name = getattr(s_spec, "name", None) or ctx.fresh_name("svd_s")
        s_out = s_val
        if getattr(s_out, "name", None) != s_name:
            s_out = _as_value(ctx.builder.Identity(s_val, _outputs=[s_name]))
        _stamp_like(s_out, s_spec if getattr(s_spec, "type", None) else s_val)
        if getattr(s_spec, "shape", None) is not None:
            s_out.shape = s_spec.shape
        ctx.bind_value_for_var(s_var, s_out)

        if not compute_uv:
            return

        u_var = outvars[1]
        vh_var = outvars[2]
        u_spec = ctx.get_value_for_var(u_var, name_hint=ctx.fresh_name("svd_u_out"))
        vh_spec = ctx.get_value_for_var(vh_var, name_hint=ctx.fresh_name("svd_vh_out"))

        sign = _as_value(ctx.builder.Sign(x00, _outputs=[ctx.fresh_name("svd_sign")]))
        _stamp_like(sign, x00)
        zero = ctx.bind_const_for_var(object(), np.asarray([[0.0]], dtype=x_dtype))
        one = ctx.bind_const_for_var(object(), np.asarray([[1.0]], dtype=x_dtype))
        if getattr(x, "type", None) is not None:
            zero.type = x.type
            one.type = x.type
        zero.shape = ir.Shape((1, 1))
        one.shape = ir.Shape((1, 1))

        sign_is_zero = _as_value(
            ctx.builder.Equal(
                sign,
                zero,
                _outputs=[ctx.fresh_name("svd_sign_is_zero")],
            )
        )
        sign_is_zero.type = ir.TensorType(ir.DataType.BOOL)
        sign_is_zero.shape = ir.Shape((1, 1))
        u_val = _as_value(
            ctx.builder.Where(
                sign_is_zero,
                one,
                sign,
                _outputs=[ctx.fresh_name("svd_u")],
            )
        )
        _stamp_like(u_val, sign)

        vh_const = ctx.bind_const_for_var(object(), np.asarray([[1.0]], dtype=x_dtype))
        if getattr(x, "type", None) is not None:
            vh_const.type = x.type
        vh_const.shape = ir.Shape((1, 1))
        vh_val = _as_value(
            ctx.builder.Identity(vh_const, _outputs=[ctx.fresh_name("svd_vh")])
        )
        _stamp_like(vh_val, vh_const)

        u_name = getattr(u_spec, "name", None) or ctx.fresh_name("svd_u")
        u_out = u_val
        if getattr(u_out, "name", None) != u_name:
            u_out = _as_value(ctx.builder.Identity(u_val, _outputs=[u_name]))
        _stamp_like(u_out, u_spec if getattr(u_spec, "type", None) else u_val)
        if getattr(u_spec, "shape", None) is not None:
            u_out.shape = u_spec.shape

        vh_name = getattr(vh_spec, "name", None) or ctx.fresh_name("svd_vh")
        vh_out = vh_val
        if getattr(vh_out, "name", None) != vh_name:
            vh_out = _as_value(ctx.builder.Identity(vh_val, _outputs=[vh_name]))
        _stamp_like(vh_out, vh_spec if getattr(vh_spec, "type", None) else vh_val)
        if getattr(vh_spec, "shape", None) is not None:
            vh_out.shape = vh_spec.shape

        ctx.bind_value_for_var(u_var, u_out)
        ctx.bind_value_for_var(vh_var, vh_out)
