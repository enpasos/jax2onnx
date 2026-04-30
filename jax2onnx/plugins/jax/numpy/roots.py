# jax2onnx/plugins/jax/numpy/roots.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, ClassVar, Final, cast

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._complex_utils import cast_real_tensor, pack_real_imag_pair
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_ROOTS_PRIM: Final = make_jnp_primitive("jax.numpy.roots")
_JAX_ROOTS_ORIG: Final = jnp.roots


def _np_dtype_for_base(base_dtype: ir.DataType) -> type[np.floating]:
    return np.float64 if base_dtype == ir.DataType.DOUBLE else np.float32


def _base_dtype_for_output(var: object) -> ir.DataType:
    out_dtype = np.dtype(getattr(getattr(var, "aval", None), "dtype", np.complex64))
    return (
        ir.DataType.DOUBLE
        if out_dtype == np.dtype(np.complex128)
        else ir.DataType.FLOAT
    )


def _stamp(value: ir.Value, dtype: ir.DataType, shape: Sequence[int]) -> ir.Value:
    value.type = ir.TensorType(dtype)
    value.dtype = dtype
    _stamp_type_and_shape(value, tuple(int(dim) for dim in shape))
    return value


def _const(
    ctx: LoweringContextProtocol,
    value: float,
    *,
    base_dtype: ir.DataType,
    name_hint: str,
) -> ir.Value:
    np_dtype = _np_dtype_for_base(base_dtype)
    out = ctx.bind_const_for_var(object(), np.asarray(value, dtype=np_dtype))
    out.name = out.name or ctx.fresh_name(name_hint)
    return _stamp(out, base_dtype, ())


def _gather_coeff(
    ctx: LoweringContextProtocol,
    coeffs: ir.Value,
    index: int,
    *,
    base_dtype: ir.DataType,
    name_hint: str,
) -> ir.Value:
    idx = _const_i64(ctx, np.asarray([index], dtype=np.int64), f"{name_hint}_idx")
    out = ctx.builder.Gather(
        coeffs,
        idx,
        axis=0,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    return _stamp(out, base_dtype, (1,))


def _unary(
    ctx: LoweringContextProtocol,
    op_type: str,
    value: ir.Value,
    *,
    base_dtype: ir.DataType,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    out = getattr(ctx.builder, op_type)(
        value,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    return _stamp(out, base_dtype, shape)


def _binary(
    ctx: LoweringContextProtocol,
    op_type: str,
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    base_dtype: ir.DataType,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    out = getattr(ctx.builder, op_type)(
        lhs,
        rhs,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    return _stamp(out, base_dtype, shape)


def _equal_zero(
    ctx: LoweringContextProtocol,
    value: ir.Value,
    zero: ir.Value,
    *,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    out = cast(
        ir.Value,
        ctx.builder.Equal(value, zero, _outputs=[ctx.fresh_name(name_hint)]),
    )
    out.type = ir.TensorType(ir.DataType.BOOL)
    out.dtype = ir.DataType.BOOL
    _stamp_type_and_shape(out, shape)
    _ensure_value_metadata(ctx, out)
    return out


def _less_than_zero(
    ctx: LoweringContextProtocol,
    value: ir.Value,
    zero: ir.Value,
    *,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    out = cast(
        ir.Value,
        ctx.builder.Less(value, zero, _outputs=[ctx.fresh_name(name_hint)]),
    )
    out.type = ir.TensorType(ir.DataType.BOOL)
    out.dtype = ir.DataType.BOOL
    _stamp_type_and_shape(out, shape)
    _ensure_value_metadata(ctx, out)
    return out


def _bool_binary(
    ctx: LoweringContextProtocol,
    op_type: str,
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    out = cast(
        ir.Value,
        getattr(ctx.builder, op_type)(
            lhs,
            rhs,
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    out.type = ir.TensorType(ir.DataType.BOOL)
    out.dtype = ir.DataType.BOOL
    _stamp_type_and_shape(out, shape)
    _ensure_value_metadata(ctx, out)
    return out


def _where(
    ctx: LoweringContextProtocol,
    cond: ir.Value,
    true_value: ir.Value,
    false_value: ir.Value,
    *,
    base_dtype: ir.DataType,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    out = ctx.builder.Where(
        cond,
        true_value,
        false_value,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    return _stamp(out, base_dtype, shape)


def _concat_pair(
    ctx: LoweringContextProtocol,
    first: ir.Value,
    second: ir.Value,
    *,
    base_dtype: ir.DataType,
    name_hint: str,
) -> ir.Value:
    out = ctx.builder.Concat(
        first,
        second,
        axis=0,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    return _stamp(out, base_dtype, (2,))


def _complex_sqrt_principal(
    ctx: LoweringContextProtocol,
    real: ir.Value,
    imag: ir.Value,
    *,
    base_dtype: ir.DataType,
    zero: ir.Value,
    one: ir.Value,
    two: ir.Value,
    shape: tuple[int, ...],
) -> tuple[ir.Value, ir.Value]:
    x2 = _binary(
        ctx,
        "Mul",
        real,
        real,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_x2",
    )
    y2 = _binary(
        ctx,
        "Mul",
        imag,
        imag,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_y2",
    )
    radius2 = _binary(
        ctx,
        "Add",
        x2,
        y2,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_r2",
    )
    radius = _unary(
        ctx,
        "Sqrt",
        radius2,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_r",
    )

    r_plus_x = _binary(
        ctx,
        "Add",
        radius,
        real,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_rp",
    )
    r_minus_x = _binary(
        ctx,
        "Sub",
        radius,
        real,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_rm",
    )
    half_plus = _binary(
        ctx,
        "Div",
        r_plus_x,
        two,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_half_plus",
    )
    half_minus = _binary(
        ctx,
        "Div",
        r_minus_x,
        two,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_half_minus",
    )

    half_plus_clip = _binary(
        ctx,
        "Max",
        half_plus,
        zero,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_half_plus_clip",
    )
    half_minus_clip = _binary(
        ctx,
        "Max",
        half_minus,
        zero,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_half_minus_clip",
    )
    out_real = _unary(
        ctx,
        "Sqrt",
        half_plus_clip,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_real",
    )
    imag_abs = _unary(
        ctx,
        "Sqrt",
        half_minus_clip,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_imag_abs",
    )
    sign_y = _unary(
        ctx,
        "Sign",
        imag,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_sign_y",
    )
    imag_is_zero = _equal_zero(
        ctx,
        imag,
        zero,
        shape=shape,
        name_hint="roots_sqrt_imag_is_zero",
    )
    real_is_negative = _less_than_zero(
        ctx,
        real,
        zero,
        shape=shape,
        name_hint="roots_sqrt_real_is_negative",
    )
    on_negative_real_axis = _bool_binary(
        ctx,
        "And",
        imag_is_zero,
        real_is_negative,
        shape=shape,
        name_hint="roots_sqrt_negative_real_axis",
    )
    sign_y = _where(
        ctx,
        on_negative_real_axis,
        one,
        sign_y,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_sign_y_adjusted",
    )
    out_imag = _binary(
        ctx,
        "Mul",
        sign_y,
        imag_abs,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_sqrt_imag",
    )
    return out_real, out_imag


def _linear_roots(
    ctx: LoweringContextProtocol,
    leading: ir.Value,
    constant: ir.Value,
    *,
    base_dtype: ir.DataType,
    zero: ir.Value,
    nan: ir.Value,
) -> tuple[ir.Value, ir.Value]:
    shape = (1,)
    neg_constant = _unary(
        ctx,
        "Neg",
        constant,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_linear_neg_constant",
    )
    root = _binary(
        ctx,
        "Div",
        neg_constant,
        leading,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_linear_root",
    )
    leading_zero = _equal_zero(
        ctx,
        leading,
        zero,
        shape=shape,
        name_hint="roots_linear_leading_zero",
    )
    real = _where(
        ctx,
        leading_zero,
        nan,
        root,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_linear_real",
    )
    imag = _where(
        ctx,
        leading_zero,
        nan,
        zero,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_linear_imag",
    )
    return real, imag


def _quadratic_roots(
    ctx: LoweringContextProtocol,
    a: ir.Value,
    b: ir.Value,
    c: ir.Value,
    *,
    base_dtype: ir.DataType,
    zero: ir.Value,
    one: ir.Value,
    two: ir.Value,
    four: ir.Value,
    nan: ir.Value,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
    shape = (1,)
    b2 = _binary(
        ctx, "Mul", b, b, base_dtype=base_dtype, shape=shape, name_hint="roots_quad_b2"
    )
    ac = _binary(
        ctx, "Mul", a, c, base_dtype=base_dtype, shape=shape, name_hint="roots_quad_ac"
    )
    four_ac = _binary(
        ctx,
        "Mul",
        four,
        ac,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_four_ac",
    )
    disc_real = _binary(
        ctx,
        "Sub",
        b2,
        four_ac,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_disc",
    )
    disc_imag = _binary(
        ctx,
        "Mul",
        disc_real,
        zero,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_disc_imag",
    )
    sqrt_real, sqrt_imag = _complex_sqrt_principal(
        ctx,
        disc_real,
        disc_imag,
        base_dtype=base_dtype,
        zero=zero,
        one=one,
        two=two,
        shape=shape,
    )

    neg_b = _unary(
        ctx, "Neg", b, base_dtype=base_dtype, shape=shape, name_hint="roots_quad_neg_b"
    )
    denom = _binary(
        ctx,
        "Mul",
        two,
        a,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_denom",
    )
    plus_real = _binary(
        ctx,
        "Add",
        neg_b,
        sqrt_real,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_plus_real",
    )
    minus_real = _binary(
        ctx,
        "Sub",
        neg_b,
        sqrt_real,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_minus_real",
    )
    plus_imag = sqrt_imag
    minus_imag = _unary(
        ctx,
        "Neg",
        sqrt_imag,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_minus_imag",
    )

    root1_real = _binary(
        ctx,
        "Div",
        plus_real,
        denom,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_root1_real",
    )
    root1_imag = _binary(
        ctx,
        "Div",
        plus_imag,
        denom,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_root1_imag",
    )
    root2_real = _binary(
        ctx,
        "Div",
        minus_real,
        denom,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_root2_real",
    )
    root2_imag = _binary(
        ctx,
        "Div",
        minus_imag,
        denom,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_root2_imag",
    )

    lin_real, lin_imag = _linear_roots(
        ctx,
        b,
        c,
        base_dtype=base_dtype,
        zero=zero,
        nan=nan,
    )
    a_zero = _equal_zero(ctx, a, zero, shape=shape, name_hint="roots_quad_a_zero")
    root1_real = _where(
        ctx,
        a_zero,
        lin_real,
        root1_real,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_root1_real_select",
    )
    root1_imag = _where(
        ctx,
        a_zero,
        lin_imag,
        root1_imag,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_root1_imag_select",
    )
    root2_real = _where(
        ctx,
        a_zero,
        nan,
        root2_real,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_root2_real_select",
    )
    root2_imag = _where(
        ctx,
        a_zero,
        nan,
        root2_imag,
        base_dtype=base_dtype,
        shape=shape,
        name_hint="roots_quad_root2_imag_select",
    )
    return root1_real, root1_imag, root2_real, root2_imag


def _bind_packed_output(
    ctx: LoweringContextProtocol,
    out_var: core.Var,
    value: ir.Value,
    *,
    base_dtype: ir.DataType,
    name_hint: str,
) -> None:
    out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name(name_hint))
    desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(name_hint)
    out_val = value
    if getattr(out_val, "name", None) != desired_name:
        out_val = ctx.builder.Identity(value, _outputs=[desired_name])
    packed_shape = tuple(int(dim) for dim in getattr(out_var.aval, "shape", ())) + (2,)
    _stamp(out_val, base_dtype, packed_shape)
    _stamp(out_spec, base_dtype, packed_shape)
    _ensure_value_metadata(ctx, out_val)
    ctx.bind_value_for_var(out_var, out_val)


def _abstract_eval_via_orig(
    p: core.AbstractValue,
    *,
    strip_zeros: bool,
) -> core.ShapedArray:
    p_shape = tuple(getattr(p, "shape", ()))
    p_dtype = np.dtype(getattr(p, "dtype", np.float32))
    orig = get_orig_impl(_ROOTS_PRIM, "roots")
    out = jax.eval_shape(
        lambda value: orig(value, strip_zeros=strip_zeros),
        jax.ShapeDtypeStruct(p_shape, p_dtype),
    )
    return core.ShapedArray(
        tuple(getattr(out, "shape", ())), getattr(out, "dtype", np.complex64)
    )


@register_primitive(
    jaxpr_primitive=_ROOTS_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.roots.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
        {"component": "Sign", "doc": "https://onnx.ai/onnx/operators/onnx__Sign.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Less", "doc": "https://onnx.ai/onnx/operators/onnx__Less.html"},
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        },
        {"component": "And", "doc": "https://onnx.ai/onnx/operators/onnx__And.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.13.0",
    context="primitives.jnp",
    component="roots",
    testcases=[
        {
            "testcase": "jnp_roots_linear",
            "callable": lambda x: jnp.roots(x, strip_zeros=False),
            "input_values": [np.asarray([2.0, -4.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Gather", "Div:1", "Where:1", "Concat:1x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_roots_quadratic_real",
            "callable": lambda x: jnp.roots(x, strip_zeros=False),
            "input_values": [np.asarray([1.0, -3.0, 2.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Gather", "Sqrt:1", "Where:1", "Concat:2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_roots_quadratic_complex_pair",
            "callable": lambda x: jnp.roots(x, strip_zeros=False),
            "input_values": [np.asarray([1.0, 0.0, 1.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Gather", "Sqrt:1", "Sign:1", "Concat:2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_roots_quadratic_leading_zero",
            "callable": lambda x: jnp.roots(x, strip_zeros=False),
            "input_values": [np.asarray([0.0, 1.0, -2.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Equal:1", "Where:1", "Concat:2x2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpRootsPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ROOTS_PRIM
    _FUNC_NAME: ClassVar[str] = "roots"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        p: core.AbstractValue,
        *,
        strip_zeros: bool = True,
    ) -> core.ShapedArray:
        return _abstract_eval_via_orig(p, strip_zeros=bool(strip_zeros))

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (p_var,) = eqn.invars
        (out_var,) = eqn.outvars
        params = dict(getattr(eqn, "params", {}) or {})
        strip_zeros = bool(params.get("strip_zeros", True))
        if strip_zeros:
            raise NotImplementedError(
                "jnp.roots lowering supports strip_zeros=False only"
            )

        p_shape = tuple(getattr(p_var.aval, "shape", ()))
        if len(p_shape) != 1 or not isinstance(p_shape[0], (int, np.integer)):
            raise NotImplementedError("jnp.roots lowering requires static 1D input")
        coeff_count = int(p_shape[0])
        if coeff_count not in (2, 3):
            raise NotImplementedError(
                "jnp.roots lowering currently supports degree 1 and degree 2 only"
            )

        out_shape = tuple(int(dim) for dim in getattr(out_var.aval, "shape", ()))
        if out_shape != (coeff_count - 1,):
            raise ValueError("jnp.roots output shape mismatch")

        base_dtype = _base_dtype_for_output(out_var)
        coeffs = ctx.get_value_for_var(p_var, name_hint=ctx.fresh_name("roots_in"))
        coeffs = cast_real_tensor(ctx, coeffs, base_dtype, name_hint="roots_coeff_cast")
        zero = _const(ctx, 0.0, base_dtype=base_dtype, name_hint="roots_zero")
        one = _const(ctx, 1.0, base_dtype=base_dtype, name_hint="roots_one")
        two = _const(ctx, 2.0, base_dtype=base_dtype, name_hint="roots_two")
        four = _const(ctx, 4.0, base_dtype=base_dtype, name_hint="roots_four")
        nan = _const(ctx, np.nan, base_dtype=base_dtype, name_hint="roots_nan")

        if coeff_count == 2:
            a = _gather_coeff(
                ctx, coeffs, 0, base_dtype=base_dtype, name_hint="roots_linear_a"
            )
            b = _gather_coeff(
                ctx, coeffs, 1, base_dtype=base_dtype, name_hint="roots_linear_b"
            )
            real, imag = _linear_roots(
                ctx,
                a,
                b,
                base_dtype=base_dtype,
                zero=zero,
                nan=nan,
            )
            packed = pack_real_imag_pair(
                ctx,
                real,
                imag,
                base_dtype,
                name_hint="roots_linear_pack",
            )
            _bind_packed_output(
                ctx,
                out_var,
                packed,
                base_dtype=base_dtype,
                name_hint="roots_out",
            )
            return

        a = _gather_coeff(
            ctx, coeffs, 0, base_dtype=base_dtype, name_hint="roots_quad_a"
        )
        b = _gather_coeff(
            ctx, coeffs, 1, base_dtype=base_dtype, name_hint="roots_quad_b"
        )
        c = _gather_coeff(
            ctx, coeffs, 2, base_dtype=base_dtype, name_hint="roots_quad_c"
        )
        root1_real, root1_imag, root2_real, root2_imag = _quadratic_roots(
            ctx,
            a,
            b,
            c,
            base_dtype=base_dtype,
            zero=zero,
            one=one,
            two=two,
            four=four,
            nan=nan,
        )
        real = _concat_pair(
            ctx,
            root1_real,
            root2_real,
            base_dtype=base_dtype,
            name_hint="roots_quad_real",
        )
        imag = _concat_pair(
            ctx,
            root1_imag,
            root2_imag,
            base_dtype=base_dtype,
            name_hint="roots_quad_imag",
        )
        packed = pack_real_imag_pair(
            ctx,
            real,
            imag,
            base_dtype,
            name_hint="roots_quad_pack",
        )
        _bind_packed_output(
            ctx,
            out_var,
            packed,
            base_dtype=base_dtype,
            name_hint="roots_out",
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.roots not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(p: ArrayLike, *, strip_zeros: bool = True) -> jax.Array:
                shape = getattr(p, "shape", None)
                if (
                    not strip_zeros
                    and shape is not None
                    and len(tuple(shape)) == 1
                    and int(tuple(shape)[0]) in (2, 3)
                ):
                    return cls._PRIM.bind(jnp.asarray(p), strip_zeros=False)
                return orig(p, strip_zeros=strip_zeros)

            return _patched

        return [
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            )
        ]


@JnpRootsPlugin._PRIM.def_impl
def _roots_impl(
    p: ArrayLike,
    *,
    strip_zeros: bool = True,
) -> jax.Array:
    try:
        orig = get_orig_impl(JnpRootsPlugin._PRIM, JnpRootsPlugin._FUNC_NAME)
    except RuntimeError:
        orig = _JAX_ROOTS_ORIG
    return orig(p, strip_zeros=strip_zeros)


JnpRootsPlugin._PRIM.def_abstract_eval(JnpRootsPlugin.abstract_eval)
