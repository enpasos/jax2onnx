# jax2onnx/plugins/jax/lax/eig.py

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._complex_utils import (
    ensure_packed_real_pair,
    pack_real_imag_pair,
    split_packed_real_imag,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _as_value(value: Any) -> ir.Value:
    return cast(ir.Value, value)


def _base_dtype_for_complex_out(var: Any) -> ir.DataType:
    out_dtype: np.dtype[Any] = np.dtype(
        getattr(getattr(var, "aval", None), "dtype", np.complex64)
    )
    if out_dtype == np.dtype(np.complex128):
        return ir.DataType.DOUBLE
    return ir.DataType.FLOAT


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


def _complex_add(
    ctx: LoweringContextProtocol,
    lhs: tuple[ir.Value, ir.Value],
    rhs: tuple[ir.Value, ir.Value],
    *,
    dtype: ir.DataType,
    prefix: str,
) -> tuple[ir.Value, ir.Value]:
    lhs_r, lhs_i = lhs
    rhs_r, rhs_i = rhs
    out_r = _as_value(
        ctx.builder.Add(lhs_r, rhs_r, _outputs=[ctx.fresh_name(f"{prefix}_r")])
    )
    out_i = _as_value(
        ctx.builder.Add(lhs_i, rhs_i, _outputs=[ctx.fresh_name(f"{prefix}_i")])
    )
    out_r.type = ir.TensorType(dtype)
    out_i.type = ir.TensorType(dtype)
    return out_r, out_i


def _complex_sub(
    ctx: LoweringContextProtocol,
    lhs: tuple[ir.Value, ir.Value],
    rhs: tuple[ir.Value, ir.Value],
    *,
    dtype: ir.DataType,
    prefix: str,
) -> tuple[ir.Value, ir.Value]:
    lhs_r, lhs_i = lhs
    rhs_r, rhs_i = rhs
    out_r = _as_value(
        ctx.builder.Sub(lhs_r, rhs_r, _outputs=[ctx.fresh_name(f"{prefix}_r")])
    )
    out_i = _as_value(
        ctx.builder.Sub(lhs_i, rhs_i, _outputs=[ctx.fresh_name(f"{prefix}_i")])
    )
    out_r.type = ir.TensorType(dtype)
    out_i.type = ir.TensorType(dtype)
    return out_r, out_i


def _complex_mul(
    ctx: LoweringContextProtocol,
    lhs: tuple[ir.Value, ir.Value],
    rhs: tuple[ir.Value, ir.Value],
    *,
    dtype: ir.DataType,
    prefix: str,
) -> tuple[ir.Value, ir.Value]:
    lhs_r, lhs_i = lhs
    rhs_r, rhs_i = rhs

    pr = _as_value(
        ctx.builder.Mul(lhs_r, rhs_r, _outputs=[ctx.fresh_name(f"{prefix}_pr")])
    )
    qs = _as_value(
        ctx.builder.Mul(lhs_i, rhs_i, _outputs=[ctx.fresh_name(f"{prefix}_qs")])
    )
    ps = _as_value(
        ctx.builder.Mul(lhs_r, rhs_i, _outputs=[ctx.fresh_name(f"{prefix}_ps")])
    )
    qr = _as_value(
        ctx.builder.Mul(lhs_i, rhs_r, _outputs=[ctx.fresh_name(f"{prefix}_qr")])
    )
    for val in (pr, qs, ps, qr):
        val.type = ir.TensorType(dtype)

    out_r = _as_value(ctx.builder.Sub(pr, qs, _outputs=[ctx.fresh_name(f"{prefix}_r")]))
    out_i = _as_value(ctx.builder.Add(ps, qr, _outputs=[ctx.fresh_name(f"{prefix}_i")]))
    out_r.type = ir.TensorType(dtype)
    out_i.type = ir.TensorType(dtype)
    return out_r, out_i


def _complex_real_scale(
    ctx: LoweringContextProtocol,
    value: tuple[ir.Value, ir.Value],
    scalar: ir.Value,
    *,
    dtype: ir.DataType,
    prefix: str,
) -> tuple[ir.Value, ir.Value]:
    val_r, val_i = value
    out_r = _as_value(
        ctx.builder.Mul(val_r, scalar, _outputs=[ctx.fresh_name(f"{prefix}_r")])
    )
    out_i = _as_value(
        ctx.builder.Mul(val_i, scalar, _outputs=[ctx.fresh_name(f"{prefix}_i")])
    )
    out_r.type = ir.TensorType(dtype)
    out_i.type = ir.TensorType(dtype)
    return out_r, out_i


def _complex_div_real(
    ctx: LoweringContextProtocol,
    value: tuple[ir.Value, ir.Value],
    scalar: ir.Value,
    *,
    dtype: ir.DataType,
    prefix: str,
) -> tuple[ir.Value, ir.Value]:
    val_r, val_i = value
    out_r = _as_value(
        ctx.builder.Div(val_r, scalar, _outputs=[ctx.fresh_name(f"{prefix}_r")])
    )
    out_i = _as_value(
        ctx.builder.Div(val_i, scalar, _outputs=[ctx.fresh_name(f"{prefix}_i")])
    )
    out_r.type = ir.TensorType(dtype)
    out_i.type = ir.TensorType(dtype)
    return out_r, out_i


def _complex_sqrt_principal(
    ctx: LoweringContextProtocol,
    value: tuple[ir.Value, ir.Value],
    *,
    dtype: ir.DataType,
    two: ir.Value,
    zero: ir.Value,
    prefix: str,
) -> tuple[ir.Value, ir.Value]:
    x, y = value
    x2 = _as_value(ctx.builder.Mul(x, x, _outputs=[ctx.fresh_name(f"{prefix}_x2")]))
    y2 = _as_value(ctx.builder.Mul(y, y, _outputs=[ctx.fresh_name(f"{prefix}_y2")]))
    x2.type = ir.TensorType(dtype)
    y2.type = ir.TensorType(dtype)

    radius2 = _as_value(
        ctx.builder.Add(x2, y2, _outputs=[ctx.fresh_name(f"{prefix}_r2")])
    )
    radius2.type = ir.TensorType(dtype)
    radius = _as_value(
        ctx.builder.Sqrt(radius2, _outputs=[ctx.fresh_name(f"{prefix}_r")])
    )
    radius.type = ir.TensorType(dtype)

    r_plus_x = _as_value(
        ctx.builder.Add(radius, x, _outputs=[ctx.fresh_name(f"{prefix}_rp")])
    )
    r_minus_x = _as_value(
        ctx.builder.Sub(radius, x, _outputs=[ctx.fresh_name(f"{prefix}_rm")])
    )
    r_plus_x.type = ir.TensorType(dtype)
    r_minus_x.type = ir.TensorType(dtype)

    half_plus = _as_value(
        ctx.builder.Div(r_plus_x, two, _outputs=[ctx.fresh_name(f"{prefix}_half_plus")])
    )
    half_minus = _as_value(
        ctx.builder.Div(
            r_minus_x, two, _outputs=[ctx.fresh_name(f"{prefix}_half_minus")]
        )
    )
    half_plus.type = ir.TensorType(dtype)
    half_minus.type = ir.TensorType(dtype)

    half_plus_clip = _as_value(
        ctx.builder.Max(
            half_plus,
            zero,
            _outputs=[ctx.fresh_name(f"{prefix}_half_plus_clip")],
        )
    )
    half_minus_clip = _as_value(
        ctx.builder.Max(
            half_minus,
            zero,
            _outputs=[ctx.fresh_name(f"{prefix}_half_minus_clip")],
        )
    )
    half_plus_clip.type = ir.TensorType(dtype)
    half_minus_clip.type = ir.TensorType(dtype)

    out_r = _as_value(
        ctx.builder.Sqrt(
            half_plus_clip, _outputs=[ctx.fresh_name(f"{prefix}_sqrt_real")]
        )
    )
    imag_abs = _as_value(
        ctx.builder.Sqrt(
            half_minus_clip, _outputs=[ctx.fresh_name(f"{prefix}_sqrt_imag_abs")]
        )
    )
    out_r.type = ir.TensorType(dtype)
    imag_abs.type = ir.TensorType(dtype)

    sign_y = _as_value(
        ctx.builder.Sign(y, _outputs=[ctx.fresh_name(f"{prefix}_sign_y")])
    )
    sign_y.type = ir.TensorType(dtype)
    out_i = _as_value(
        ctx.builder.Mul(sign_y, imag_abs, _outputs=[ctx.fresh_name(f"{prefix}_i")])
    )
    out_i.type = ir.TensorType(dtype)
    return out_r, out_i


def _concat_1d_pair(
    ctx: LoweringContextProtocol,
    first: ir.Value,
    second: ir.Value,
    *,
    dtype: ir.DataType,
    name: str,
) -> ir.Value:
    out = _as_value(
        ctx.builder.Concat(
            first,
            second,
            axis=0,
            _outputs=[name],
        )
    )
    out.type = ir.TensorType(dtype)
    out.shape = ir.Shape((2,))
    return out


def _bind_complex_output(
    ctx: LoweringContextProtocol,
    out_var: Any,
    value: ir.Value,
    *,
    base_dtype: ir.DataType,
    name_hint: str,
) -> None:
    out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name(name_hint))
    out_name = getattr(out_spec, "name", None) or ctx.fresh_name(name_hint)
    out_val = value
    if getattr(out_val, "name", None) != out_name:
        out_val = _as_value(ctx.builder.Identity(value, _outputs=[out_name]))

    packed_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ())) + (2,)
    out_val.type = ir.TensorType(base_dtype)
    out_val.shape = ir.Shape(packed_shape)
    out_spec.type = ir.TensorType(base_dtype)
    out_spec.shape = ir.Shape(packed_shape)
    ctx.bind_value_for_var(out_var, out_val)


def _unit_vector_packed_1x1(
    ctx: LoweringContextProtocol,
    *,
    base_dtype: ir.DataType,
    output_name: str,
    name_hint: str,
) -> ir.Value:
    np_dtype = np.float64 if base_dtype == ir.DataType.DOUBLE else np.float32
    real = ctx.bind_const_for_var(object(), np.asarray([[1.0]], dtype=np_dtype))
    imag = ctx.bind_const_for_var(object(), np.asarray([[0.0]], dtype=np_dtype))
    real.type = ir.TensorType(base_dtype)
    imag.type = ir.TensorType(base_dtype)
    real.shape = ir.Shape((1, 1))
    imag.shape = ir.Shape((1, 1))
    return pack_real_imag_pair(
        ctx,
        real,
        imag,
        base_dtype,
        name_hint=name_hint,
        output_name=output_name,
    )


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.eig_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.eig.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Unsqueeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unsqueeze.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="eig",
    testcases=[
        {
            "testcase": "eig_1x1_values_only",
            "callable": lambda x: jax.lax.linalg.eig(
                x,
                compute_left_eigenvectors=False,
                compute_right_eigenvectors=False,
            ),
            "input_values": [np.asarray([[2.5]], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(["Gather", "Concat"], no_unused_inputs=True),
        },
        {
            "testcase": "eig_1x1_left_only",
            "callable": lambda x: jax.lax.linalg.eig(
                x,
                compute_left_eigenvectors=True,
                compute_right_eigenvectors=False,
            ),
            "input_values": [np.asarray([[-1.25]], dtype=np.float32)],
            "expected_output_dtypes": [np.float32, np.float32],
            "post_check_onnx_graph": EG(["Gather", "Concat"], no_unused_inputs=True),
        },
        {
            "testcase": "eig_1x1_right_only",
            "callable": lambda x: jax.lax.linalg.eig(
                x,
                compute_left_eigenvectors=False,
                compute_right_eigenvectors=True,
            ),
            "input_values": [np.asarray([[0.75]], dtype=np.float32)],
            "expected_output_dtypes": [np.float32, np.float32],
            "post_check_onnx_graph": EG(["Gather", "Concat"], no_unused_inputs=True),
        },
        {
            "testcase": "eig_1x1_full",
            "callable": lambda x: jax.lax.linalg.eig(
                x,
                compute_left_eigenvectors=True,
                compute_right_eigenvectors=True,
            ),
            "input_values": [np.asarray([[3.0]], dtype=np.float32)],
            "expected_output_dtypes": [np.float32, np.float32, np.float32],
            "post_check_onnx_graph": EG(["Gather", "Concat"], no_unused_inputs=True),
        },
        {
            "testcase": "eig_2x2_values_only_real",
            "callable": lambda x: jax.lax.linalg.eig(
                x,
                compute_left_eigenvectors=False,
                compute_right_eigenvectors=False,
            ),
            "input_values": [
                np.asarray(
                    [
                        [3.0, 1.0],
                        [0.0, 2.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Sqrt", "Sign", "Concat"], no_unused_inputs=True
            ),
        },
        {
            "testcase": "eig_2x2_values_only_complex128",
            "callable": lambda x: jax.lax.linalg.eig(
                x,
                compute_left_eigenvectors=False,
                compute_right_eigenvectors=False,
            ),
            "input_values": [
                np.asarray(
                    [
                        [2.0 + 3.0j, 1.0 - 1.0j],
                        [0.0 + 0.0j, 2.0 + 3.0j],
                    ],
                    dtype=np.complex128,
                )
            ],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Sqrt", "Sign", "Concat"], no_unused_inputs=True
            ),
        },
    ],
)
class EigPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.eig`` for static square ``1x1`` matrices."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        params = dict(getattr(eqn, "params", {}) or {})
        compute_left = bool(params.get("compute_left_eigenvectors", True))
        compute_right = bool(params.get("compute_right_eigenvectors", True))

        (x_var,) = eqn.invars
        outvars = list(eqn.outvars)
        expected_n = 1 + int(compute_left) + int(compute_right)
        if len(outvars) != expected_n:
            raise NotImplementedError(
                f"eig output arity mismatch: expected {expected_n}, got {len(outvars)}"
            )

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if len(x_shape) != 2:
            raise NotImplementedError("eig currently supports rank-2 inputs only")
        n_rows_raw, n_cols_raw = x_shape
        if not isinstance(n_rows_raw, (int, np.integer)) or not isinstance(
            n_cols_raw, (int, np.integer)
        ):
            raise NotImplementedError("eig requires static matrix dimensions")
        n_rows = int(n_rows_raw)
        n_cols = int(n_cols_raw)
        if n_rows != n_cols:
            raise ValueError("eig requires square matrices")
        if n_rows > 2:
            raise NotImplementedError(
                "eig currently supports only 1x1 (full) and 2x2 values-only matrices"
            )

        eigvals_var = outvars[0]
        eigvals_spec = ctx.get_value_for_var(
            eigvals_var, name_hint=ctx.fresh_name("eig_vals")
        )
        eigvals_base = _base_dtype_for_complex_out(eigvals_var)
        eigvals_name = getattr(eigvals_spec, "name", None) or ctx.fresh_name("eig_vals")
        x_input_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("eig_in"))
        if np.issubdtype(x_input_dtype, np.complexfloating):
            packed_x, packed_base = ensure_packed_real_pair(ctx, x, name_hint="eig_in")
            x_real, x_imag = split_packed_real_imag(
                ctx,
                packed_x,
                packed_base,
                prefix="eig_in_split",
            )
            x_real = _cast_if_needed(ctx, x_real, eigvals_base, "eig_in_real_cast")
            x_imag = _cast_if_needed(ctx, x_imag, eigvals_base, "eig_in_imag_cast")
        else:
            x_real = _cast_if_needed(ctx, x, eigvals_base, "eig_in_real_cast")
            scalar_dtype = (
                np.float64 if eigvals_base == ir.DataType.DOUBLE else np.float32
            )
            zero = ctx.bind_const_for_var(object(), np.asarray(0.0, dtype=scalar_dtype))
            zero.type = ir.TensorType(eigvals_base)
            x_imag = _as_value(
                ctx.builder.Mul(
                    x_real,
                    zero,
                    _outputs=[ctx.fresh_name("eig_in_imag_zero")],
                )
            )
            x_imag.type = ir.TensorType(eigvals_base)
            if getattr(x_real, "shape", None) is not None:
                x_imag.shape = x_real.shape

        if n_rows == 2:
            if compute_left or compute_right:
                raise NotImplementedError(
                    "eig 2x2 currently supports eigenvalues-only mode "
                    "(compute_left_eigenvectors=False, compute_right_eigenvectors=False)"
                )
            a = (
                _cast_if_needed(
                    ctx,
                    _gather_mat_elem(ctx, x_real, 0, 0, "eig2_a_r"),
                    eigvals_base,
                    "eig2_a_r_cast",
                ),
                _cast_if_needed(
                    ctx,
                    _gather_mat_elem(ctx, x_imag, 0, 0, "eig2_a_i"),
                    eigvals_base,
                    "eig2_a_i_cast",
                ),
            )
            b = (
                _cast_if_needed(
                    ctx,
                    _gather_mat_elem(ctx, x_real, 0, 1, "eig2_b_r"),
                    eigvals_base,
                    "eig2_b_r_cast",
                ),
                _cast_if_needed(
                    ctx,
                    _gather_mat_elem(ctx, x_imag, 0, 1, "eig2_b_i"),
                    eigvals_base,
                    "eig2_b_i_cast",
                ),
            )
            c = (
                _cast_if_needed(
                    ctx,
                    _gather_mat_elem(ctx, x_real, 1, 0, "eig2_c_r"),
                    eigvals_base,
                    "eig2_c_r_cast",
                ),
                _cast_if_needed(
                    ctx,
                    _gather_mat_elem(ctx, x_imag, 1, 0, "eig2_c_i"),
                    eigvals_base,
                    "eig2_c_i_cast",
                ),
            )
            d = (
                _cast_if_needed(
                    ctx,
                    _gather_mat_elem(ctx, x_real, 1, 1, "eig2_d_r"),
                    eigvals_base,
                    "eig2_d_r_cast",
                ),
                _cast_if_needed(
                    ctx,
                    _gather_mat_elem(ctx, x_imag, 1, 1, "eig2_d_i"),
                    eigvals_base,
                    "eig2_d_i_cast",
                ),
            )

            scalar_dtype = (
                np.float64 if eigvals_base == ir.DataType.DOUBLE else np.float32
            )
            zero = ctx.bind_const_for_var(object(), np.asarray(0.0, dtype=scalar_dtype))
            two = ctx.bind_const_for_var(object(), np.asarray(2.0, dtype=scalar_dtype))
            four = ctx.bind_const_for_var(object(), np.asarray(4.0, dtype=scalar_dtype))
            zero.type = ir.TensorType(eigvals_base)
            two.type = ir.TensorType(eigvals_base)
            four.type = ir.TensorType(eigvals_base)

            tr = _complex_add(ctx, a, d, dtype=eigvals_base, prefix="eig2_tr")
            ad = _complex_mul(ctx, a, d, dtype=eigvals_base, prefix="eig2_ad")
            bc = _complex_mul(ctx, b, c, dtype=eigvals_base, prefix="eig2_bc")
            det = _complex_sub(ctx, ad, bc, dtype=eigvals_base, prefix="eig2_det")
            tr2 = _complex_mul(ctx, tr, tr, dtype=eigvals_base, prefix="eig2_tr2")
            four_det = _complex_real_scale(
                ctx, det, four, dtype=eigvals_base, prefix="eig2_four_det"
            )
            disc = _complex_sub(
                ctx,
                tr2,
                four_det,
                dtype=eigvals_base,
                prefix="eig2_disc",
            )
            disc_sqrt = _complex_sqrt_principal(
                ctx,
                disc,
                dtype=eigvals_base,
                two=two,
                zero=zero,
                prefix="eig2_disc_sqrt",
            )

            tr_plus = _complex_add(
                ctx,
                tr,
                disc_sqrt,
                dtype=eigvals_base,
                prefix="eig2_tr_plus",
            )
            tr_minus = _complex_sub(
                ctx,
                tr,
                disc_sqrt,
                dtype=eigvals_base,
                prefix="eig2_tr_minus",
            )
            lam1_real, lam1_imag = _complex_div_real(
                ctx, tr_plus, two, dtype=eigvals_base, prefix="eig2_lam1"
            )
            lam2_real, lam2_imag = _complex_div_real(
                ctx, tr_minus, two, dtype=eigvals_base, prefix="eig2_lam2"
            )

            lam1_real_1d = _reshape_to_1d_len1(ctx, lam1_real, "eig2_lam1_real_1d")
            lam2_real_1d = _reshape_to_1d_len1(ctx, lam2_real, "eig2_lam2_real_1d")
            lam1_imag_1d = _reshape_to_1d_len1(ctx, lam1_imag, "eig2_lam1_imag_1d")
            lam2_imag_1d = _reshape_to_1d_len1(ctx, lam2_imag, "eig2_lam2_imag_1d")

            eig_real_vec = _concat_1d_pair(
                ctx,
                lam1_real_1d,
                lam2_real_1d,
                dtype=eigvals_base,
                name=ctx.fresh_name("eig2_real_vec"),
            )
            eig_imag_vec = _concat_1d_pair(
                ctx,
                lam1_imag_1d,
                lam2_imag_1d,
                dtype=eigvals_base,
                name=ctx.fresh_name("eig2_imag_vec"),
            )
            eigvals_out = pack_real_imag_pair(
                ctx,
                eig_real_vec,
                eig_imag_vec,
                eigvals_base,
                name_hint="eig2_vals_pack",
                output_name=eigvals_name,
            )
            _bind_complex_output(
                ctx,
                eigvals_var,
                eigvals_out,
                base_dtype=eigvals_base,
                name_hint="eig_vals",
            )
            return

        x00_real = _gather_mat_elem(ctx, x_real, 0, 0, "eig_x00_real")
        x00_imag = _gather_mat_elem(ctx, x_imag, 0, 0, "eig_x00_imag")
        x00_real = _cast_if_needed(ctx, x00_real, eigvals_base, "eig_x00_real_cast")
        x00_imag = _cast_if_needed(ctx, x00_imag, eigvals_base, "eig_x00_imag_cast")
        eig_real_1d = _reshape_to_1d_len1(ctx, x00_real, "eig_real_1d")
        eig_imag_1d = _reshape_to_1d_len1(ctx, x00_imag, "eig_imag_1d")
        eigvals_out = pack_real_imag_pair(
            ctx,
            eig_real_1d,
            eig_imag_1d,
            eigvals_base,
            name_hint="eig_vals_pack",
            output_name=eigvals_name,
        )
        _bind_complex_output(
            ctx,
            eigvals_var,
            eigvals_out,
            base_dtype=eigvals_base,
            name_hint="eig_vals",
        )

        output_index = 1
        if compute_left:
            left_var = outvars[output_index]
            output_index += 1
            left_base = _base_dtype_for_complex_out(left_var)
            left_spec = ctx.get_value_for_var(
                left_var, name_hint=ctx.fresh_name("eig_left")
            )
            left_name = getattr(left_spec, "name", None) or ctx.fresh_name("eig_left")
            left_out = _unit_vector_packed_1x1(
                ctx,
                base_dtype=left_base,
                output_name=left_name,
                name_hint="eig_left_pack",
            )
            _bind_complex_output(
                ctx,
                left_var,
                left_out,
                base_dtype=left_base,
                name_hint="eig_left",
            )

        if compute_right:
            right_var = outvars[output_index]
            right_base = _base_dtype_for_complex_out(right_var)
            right_spec = ctx.get_value_for_var(
                right_var, name_hint=ctx.fresh_name("eig_right")
            )
            right_name = getattr(right_spec, "name", None) or ctx.fresh_name(
                "eig_right"
            )
            right_out = _unit_vector_packed_1x1(
                ctx,
                base_dtype=right_base,
                output_name=right_name,
                name_hint="eig_right_pack",
            )
            _bind_complex_output(
                ctx,
                right_var,
                right_out,
                base_dtype=right_base,
                name_hint="eig_right",
            )
