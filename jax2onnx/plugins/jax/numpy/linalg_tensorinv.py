# jax2onnx/plugins/jax/numpy/linalg_tensorinv.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax.numpy.linalg_inv import (
    _all_static_ints,
    _as_output_name,
    _binary_op,
    _concat,
    _const_scalar,
    _gather_matrix_elem,
    _neg,
    _unsqueeze,
)
from jax2onnx.plugins.jax.numpy.linalg_solve import _cast_to_output_dtype
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_TENSORINV_PRIM: Final = make_jnp_primitive("jax.numpy.linalg.tensorinv")
_JAX_TENSORINV_ORIG: Final = jnp.linalg.tensorinv


def _prod(shape: Sequence[int]) -> int:
    return int(np.prod(tuple(int(dim) for dim in shape), dtype=np.int64))


def _reshape(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    shape: tuple[int, ...],
    dtype_enum: ir.DataType,
    name_hint: str,
    output_name: str | None = None,
) -> ir.Value:
    shape_val = _const_i64(
        ctx,
        np.asarray(shape, dtype=np.int64),
        f"{name_hint}_shape",
    )
    result = ctx.builder.Reshape(
        val,
        shape_val,
        _outputs=[output_name or ctx.fresh_name(name_hint)],
    )
    result.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _normalise_shapes(
    shape_raw: tuple[object, ...],
    *,
    ind: int,
) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    if not _all_static_ints(shape_raw):
        raise TypeError("jnp.linalg.tensorinv lowering requires static shapes")
    shape = tuple(int(dim) for dim in shape_raw)
    if ind <= 0 or ind >= len(shape):
        raise ValueError("jnp.linalg.tensorinv requires 0 < ind < a.ndim")
    rows = _prod(shape[:ind])
    cols = _prod(shape[ind:])
    if rows != cols:
        raise ValueError("jnp.linalg.tensorinv reshaped matrix must be square")
    if rows not in (1, 2):
        raise NotImplementedError(
            "jnp.linalg.tensorinv lowering currently supports 1x1 and 2x2 systems"
        )
    return shape, rows, shape[ind:] + shape[:ind]


def _abstract_eval_via_orig(
    x: core.AbstractValue,
    *,
    ind: int,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    x_dtype = np.dtype(getattr(x, "dtype", np.float32))
    if np.issubdtype(x_dtype, np.complexfloating):
        raise TypeError("jnp.linalg.tensorinv lowering does not support complex inputs")
    orig = get_orig_impl(_TENSORINV_PRIM, "tensorinv")
    out = jax.eval_shape(
        lambda value: orig(value, ind=ind),
        jax.ShapeDtypeStruct(x_shape, x_dtype),
    )
    out_dtype = np.dtype(getattr(out, "dtype", np.float32))
    if np.issubdtype(out_dtype, np.complexfloating):
        raise TypeError(
            "jnp.linalg.tensorinv lowering does not support complex outputs"
        )
    return core.ShapedArray(tuple(getattr(out, "shape", ())), out_dtype)


def _invert_1x1(
    ctx: LoweringContextProtocol,
    matrix: ir.Value,
    *,
    dtype_enum: ir.DataType,
    out_dtype: np.dtype,
    output_name: str | None = None,
) -> ir.Value:
    one = _const_scalar(
        ctx,
        dtype=out_dtype,
        value=1.0,
        name_hint="linalg_tensorinv_one",
    )
    return _binary_op(
        ctx,
        "Div",
        one,
        matrix,
        dtype_enum=dtype_enum,
        shape=(1, 1),
        name_hint="linalg_tensorinv_1x1",
        output_name=output_name,
    )


def _invert_2x2(
    ctx: LoweringContextProtocol,
    matrix: ir.Value,
    *,
    dtype_enum: ir.DataType,
    output_name: str | None = None,
) -> ir.Value:
    a = _gather_matrix_elem(
        ctx,
        matrix,
        matrix_shape=(2, 2),
        row=0,
        col=0,
        dtype_enum=dtype_enum,
        name_hint="linalg_tensorinv_a",
    )
    b = _gather_matrix_elem(
        ctx,
        matrix,
        matrix_shape=(2, 2),
        row=0,
        col=1,
        dtype_enum=dtype_enum,
        name_hint="linalg_tensorinv_b",
    )
    c = _gather_matrix_elem(
        ctx,
        matrix,
        matrix_shape=(2, 2),
        row=1,
        col=0,
        dtype_enum=dtype_enum,
        name_hint="linalg_tensorinv_c",
    )
    d = _gather_matrix_elem(
        ctx,
        matrix,
        matrix_shape=(2, 2),
        row=1,
        col=1,
        dtype_enum=dtype_enum,
        name_hint="linalg_tensorinv_d",
    )
    ad = _binary_op(
        ctx,
        "Mul",
        a,
        d,
        dtype_enum=dtype_enum,
        shape=(),
        name_hint="linalg_tensorinv_ad",
    )
    bc = _binary_op(
        ctx,
        "Mul",
        b,
        c,
        dtype_enum=dtype_enum,
        shape=(),
        name_hint="linalg_tensorinv_bc",
    )
    det = _binary_op(
        ctx,
        "Sub",
        ad,
        bc,
        dtype_enum=dtype_enum,
        shape=(),
        name_hint="linalg_tensorinv_det",
    )
    neg_b = _neg(
        ctx,
        b,
        dtype_enum=dtype_enum,
        shape=(),
        name_hint="linalg_tensorinv_neg_b",
    )
    neg_c = _neg(
        ctx,
        c,
        dtype_enum=dtype_enum,
        shape=(),
        name_hint="linalg_tensorinv_neg_c",
    )
    top = _concat(
        ctx,
        (
            _unsqueeze(
                ctx,
                d,
                axis=0,
                shape=(1,),
                name_hint="linalg_tensorinv_d_col",
            ),
            _unsqueeze(
                ctx,
                neg_b,
                axis=0,
                shape=(1,),
                name_hint="linalg_tensorinv_neg_b_col",
            ),
        ),
        axis=0,
        dtype_enum=dtype_enum,
        shape=(2,),
        name_hint="linalg_tensorinv_top_row",
    )
    bottom = _concat(
        ctx,
        (
            _unsqueeze(
                ctx,
                neg_c,
                axis=0,
                shape=(1,),
                name_hint="linalg_tensorinv_neg_c_col",
            ),
            _unsqueeze(
                ctx,
                a,
                axis=0,
                shape=(1,),
                name_hint="linalg_tensorinv_a_col",
            ),
        ),
        axis=0,
        dtype_enum=dtype_enum,
        shape=(2,),
        name_hint="linalg_tensorinv_bottom_row",
    )
    adjugate = _concat(
        ctx,
        (
            _unsqueeze(
                ctx,
                top,
                axis=0,
                shape=(1, 2),
                name_hint="linalg_tensorinv_top_matrix",
            ),
            _unsqueeze(
                ctx,
                bottom,
                axis=0,
                shape=(1, 2),
                name_hint="linalg_tensorinv_bottom_matrix",
            ),
        ),
        axis=0,
        dtype_enum=dtype_enum,
        shape=(2, 2),
        name_hint="linalg_tensorinv_adjugate",
    )
    det_matrix = _unsqueeze(
        ctx,
        _unsqueeze(
            ctx,
            det,
            axis=0,
            shape=(1,),
            name_hint="linalg_tensorinv_det_col",
        ),
        axis=1,
        shape=(1, 1),
        name_hint="linalg_tensorinv_det_matrix",
    )
    return _binary_op(
        ctx,
        "Div",
        adjugate,
        det_matrix,
        dtype_enum=dtype_enum,
        shape=(2, 2),
        name_hint="linalg_tensorinv_result",
        output_name=output_name,
    )


@register_primitive(
    jaxpr_primitive=_TENSORINV_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.tensorinv.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
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
    component="linalg_tensorinv",
    testcases=[
        {
            "testcase": "linalg_tensorinv_2x2_tensor",
            "callable": lambda x: jnp.linalg.tensorinv(x, ind=1),
            "input_values": [
                np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=np.float32).reshape(2, 1, 2)
            ],
            "expected_output_shapes": [(1, 2, 2)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Reshape:2x2", "Gather", "Sub", "Div:2x2", "Reshape:1x2x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linalg_tensorinv_1x1_tensor",
            "callable": lambda x: jnp.linalg.tensorinv(x, ind=1),
            "input_values": [np.asarray([[4.0]], dtype=np.float32).reshape(1, 1, 1)],
            "expected_output_shapes": [(1, 1, 1)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Reshape:1x1", "Div:1x1", "Reshape:1x1x1"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpLinalgTensorInvPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _TENSORINV_PRIM
    _FUNC_NAME: ClassVar[str] = "tensorinv"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        ind: int = 2,
    ) -> core.ShapedArray:
        return _abstract_eval_via_orig(x, ind=int(ind))

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars
        params = getattr(eqn, "params", {})
        ind = int(params.get("ind", 2))

        x_shape, rows, output_shape = _normalise_shapes(
            tuple(getattr(x_var.aval, "shape", ())),
            ind=ind,
        )
        out_shape = tuple(int(dim) for dim in getattr(out_var.aval, "shape", ()))
        if out_shape != output_shape:
            raise ValueError("jnp.linalg.tensorinv output shape mismatch")

        x_dtype = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        out_dtype = np.dtype(getattr(out_var.aval, "dtype", x_dtype))
        if np.issubdtype(x_dtype, np.complexfloating) or np.issubdtype(
            out_dtype, np.complexfloating
        ):
            raise TypeError(
                "jnp.linalg.tensorinv lowering does not support complex dtypes"
            )

        dtype_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        x_val = ctx.get_value_for_var(
            x_var, name_hint=ctx.fresh_name("linalg_tensorinv_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("linalg_tensorinv_out")
        )
        desired_name = _as_output_name(ctx, out_spec, "linalg_tensorinv_out")
        x_val = _cast_to_output_dtype(
            ctx,
            x_val,
            dtype_enum=dtype_enum,
            shape=x_shape,
            name_hint="linalg_tensorinv_cast",
        )
        matrix = _reshape(
            ctx,
            x_val,
            shape=(rows, rows),
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorinv_matrix",
        )
        inv_matrix = (
            _invert_1x1(
                ctx,
                matrix,
                dtype_enum=dtype_enum,
                out_dtype=out_dtype,
                output_name=desired_name if output_shape == (1, 1) else None,
            )
            if rows == 1
            else _invert_2x2(
                ctx,
                matrix,
                dtype_enum=dtype_enum,
                output_name=desired_name if output_shape == (2, 2) else None,
            )
        )
        if output_shape != (rows, rows):
            inv_matrix = _reshape(
                ctx,
                inv_matrix,
                shape=output_shape,
                dtype_enum=dtype_enum,
                name_hint="linalg_tensorinv_result",
                output_name=desired_name,
            )
        ctx.bind_value_for_var(out_var, inv_matrix)

    @classmethod
    def binding_specs(cls) -> list[MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.linalg.tensorinv not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a: ArrayLike, ind: int = 2) -> jax.Array:
                shape = getattr(a, "shape", None)
                if shape is not None:
                    try:
                        _, rows, _ = _normalise_shapes(tuple(shape), ind=int(ind))
                    except (TypeError, ValueError, NotImplementedError):
                        rows = 0
                    if rows in (1, 2):
                        return cls._PRIM.bind(jnp.asarray(a), ind=int(ind))
                return orig(a, ind=ind)

            return _patched

        return [
            MonkeyPatchSpec(
                target="jax.numpy.linalg",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            )
        ]


@JnpLinalgTensorInvPlugin._PRIM.def_impl
def _linalg_tensorinv_impl(
    x: ArrayLike,
    *,
    ind: int = 2,
) -> jax.Array:
    try:
        orig = get_orig_impl(
            JnpLinalgTensorInvPlugin._PRIM,
            JnpLinalgTensorInvPlugin._FUNC_NAME,
        )
    except RuntimeError:
        orig = _JAX_TENSORINV_ORIG
    return orig(x, ind=ind)


JnpLinalgTensorInvPlugin._PRIM.def_abstract_eval(JnpLinalgTensorInvPlugin.abstract_eval)
