# jax2onnx/plugins/jax/numpy/linalg_tensorsolve.py

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
    _gather_matrix_elem,
)
from jax2onnx.plugins.jax.numpy.linalg_solve import (
    _cast_to_output_dtype,
    _gather_rhs_row,
    _mul_coeff_rhs,
    _stack_solution_rows,
    _unsqueeze_to_shape,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_TENSORSOLVE_PRIM: Final = make_jnp_primitive("jax.numpy.linalg.tensorsolve")
_JAX_TENSORSOLVE_ORIG: Final = jnp.linalg.tensorsolve


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


def _abstract_eval_via_orig(
    a: core.AbstractValue,
    b: core.AbstractValue,
    *,
    axes: tuple[int, ...] | None,
) -> core.ShapedArray:
    a_shape = tuple(getattr(a, "shape", ()))
    b_shape = tuple(getattr(b, "shape", ()))
    a_dtype = np.dtype(getattr(a, "dtype", np.float32))
    b_dtype = np.dtype(getattr(b, "dtype", np.float32))
    if np.issubdtype(a_dtype, np.complexfloating) or np.issubdtype(
        b_dtype, np.complexfloating
    ):
        raise TypeError(
            "jnp.linalg.tensorsolve lowering does not support complex inputs"
        )
    orig = get_orig_impl(_TENSORSOLVE_PRIM, "tensorsolve")
    out = jax.eval_shape(
        lambda a_val, b_val: orig(a_val, b_val, axes=axes),
        jax.ShapeDtypeStruct(a_shape, a_dtype),
        jax.ShapeDtypeStruct(b_shape, b_dtype),
    )
    out_dtype = np.dtype(getattr(out, "dtype", np.float32))
    if np.issubdtype(out_dtype, np.complexfloating):
        raise TypeError(
            "jnp.linalg.tensorsolve lowering does not support complex outputs"
        )
    return core.ShapedArray(tuple(getattr(out, "shape", ())), out_dtype)


def _normalise_shapes(
    a_shape_raw: tuple[object, ...],
    b_shape_raw: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, ...]]:
    if not _all_static_ints(a_shape_raw) or not _all_static_ints(b_shape_raw):
        raise TypeError("jnp.linalg.tensorsolve lowering requires static shapes")
    a_shape = tuple(int(dim) for dim in a_shape_raw)
    b_shape = tuple(int(dim) for dim in b_shape_raw)
    split = len(b_shape)
    if split <= 0 or len(a_shape) <= split:
        raise ValueError("jnp.linalg.tensorsolve requires 0 < b.ndim < a.ndim")
    rows = _prod(a_shape[:split])
    cols = _prod(a_shape[split:])
    if rows != cols:
        raise ValueError("jnp.linalg.tensorsolve reshaped matrix must be square")
    if rows not in (1, 2):
        raise NotImplementedError(
            "jnp.linalg.tensorsolve lowering currently supports 1x1 and 2x2 systems"
        )
    if _prod(b_shape) != rows:
        raise ValueError("jnp.linalg.tensorsolve RHS size mismatch")
    output_shape = a_shape[split:]
    return a_shape, b_shape, rows, output_shape


@register_primitive(
    jaxpr_primitive=_TENSORSOLVE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.tensorsolve.html",
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
    component="linalg_tensorsolve",
    testcases=[
        {
            "testcase": "linalg_tensorsolve_2x2_vector",
            "callable": lambda a, b: jnp.linalg.tensorsolve(a, b),
            "input_values": [
                np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=np.float32).reshape(2, 1, 2),
                np.asarray([9.0, 8.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(1, 2)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Reshape:2x2", "Gather", "Sub", "Div", "Concat:2", "Reshape:1x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linalg_tensorsolve_1x1_vector",
            "callable": lambda a, b: jnp.linalg.tensorsolve(a, b),
            "input_values": [
                np.asarray([[4.0]], dtype=np.float32).reshape(1, 1, 1),
                np.asarray([8.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(1, 1)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Reshape:1x1", "Div:1", "Reshape:1x1"], no_unused_inputs=True
            ),
        },
    ],
)
class JnpLinalgTensorSolvePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _TENSORSOLVE_PRIM
    _FUNC_NAME: ClassVar[str] = "tensorsolve"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        a: core.AbstractValue,
        b: core.AbstractValue,
        *,
        axes: tuple[int, ...] | None = None,
    ) -> core.ShapedArray:
        return _abstract_eval_via_orig(a, b, axes=axes)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        a_var, b_var = eqn.invars
        (out_var,) = eqn.outvars
        params = getattr(eqn, "params", {})
        axes = params.get("axes", None)
        if axes is not None:
            raise NotImplementedError(
                "jnp.linalg.tensorsolve lowering does not support axes yet"
            )

        a_shape, b_shape, rows, output_shape = _normalise_shapes(
            tuple(getattr(a_var.aval, "shape", ())),
            tuple(getattr(b_var.aval, "shape", ())),
        )
        out_shape = tuple(int(dim) for dim in getattr(out_var.aval, "shape", ()))
        if out_shape != output_shape:
            raise ValueError("jnp.linalg.tensorsolve output shape mismatch")

        a_dtype = np.dtype(getattr(a_var.aval, "dtype", np.float32))
        b_dtype = np.dtype(getattr(b_var.aval, "dtype", a_dtype))
        out_dtype = np.dtype(getattr(out_var.aval, "dtype", b_dtype))
        if (
            np.issubdtype(a_dtype, np.complexfloating)
            or np.issubdtype(b_dtype, np.complexfloating)
            or np.issubdtype(out_dtype, np.complexfloating)
        ):
            raise TypeError(
                "jnp.linalg.tensorsolve lowering does not support complex dtypes"
            )

        dtype_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        a_val = ctx.get_value_for_var(
            a_var, name_hint=ctx.fresh_name("linalg_tensorsolve_a")
        )
        b_val = ctx.get_value_for_var(
            b_var, name_hint=ctx.fresh_name("linalg_tensorsolve_b")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("linalg_tensorsolve_out")
        )
        desired_name = _as_output_name(ctx, out_spec, "linalg_tensorsolve_out")

        a_val = _cast_to_output_dtype(
            ctx,
            a_val,
            dtype_enum=dtype_enum,
            shape=a_shape,
            name_hint="linalg_tensorsolve_a_cast",
        )
        b_val = _cast_to_output_dtype(
            ctx,
            b_val,
            dtype_enum=dtype_enum,
            shape=b_shape,
            name_hint="linalg_tensorsolve_b_cast",
        )
        matrix = _reshape(
            ctx,
            a_val,
            shape=(rows, rows),
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_matrix",
        )
        rhs = _reshape(
            ctx,
            b_val,
            shape=(rows,),
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_rhs",
        )

        if rows == 1:
            a00 = _gather_matrix_elem(
                ctx,
                matrix,
                matrix_shape=(1, 1),
                row=0,
                col=0,
                dtype_enum=dtype_enum,
                name_hint="linalg_tensorsolve_a00",
            )
            divisor = _unsqueeze_to_shape(
                ctx,
                a00,
                current_shape=(),
                target_shape=(1,),
                name_hint="linalg_tensorsolve_a00",
            )
            solved = _binary_op(
                ctx,
                "Div",
                rhs,
                divisor,
                dtype_enum=dtype_enum,
                shape=(1,),
                name_hint="linalg_tensorsolve_1x1",
                output_name=desired_name if output_shape == (1,) else None,
            )
            if output_shape != (1,):
                solved = _reshape(
                    ctx,
                    solved,
                    shape=output_shape,
                    dtype_enum=dtype_enum,
                    name_hint="linalg_tensorsolve_result",
                    output_name=desired_name,
                )
            ctx.bind_value_for_var(out_var, solved)
            return

        a = _gather_matrix_elem(
            ctx,
            matrix,
            matrix_shape=(2, 2),
            row=0,
            col=0,
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_a00",
        )
        b = _gather_matrix_elem(
            ctx,
            matrix,
            matrix_shape=(2, 2),
            row=0,
            col=1,
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_a01",
        )
        c = _gather_matrix_elem(
            ctx,
            matrix,
            matrix_shape=(2, 2),
            row=1,
            col=0,
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_a10",
        )
        d = _gather_matrix_elem(
            ctx,
            matrix,
            matrix_shape=(2, 2),
            row=1,
            col=1,
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_a11",
        )

        ad = _binary_op(
            ctx,
            "Mul",
            a,
            d,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="linalg_tensorsolve_ad",
        )
        bc = _binary_op(
            ctx,
            "Mul",
            b,
            c,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="linalg_tensorsolve_bc",
        )
        det = _binary_op(
            ctx,
            "Sub",
            ad,
            bc,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="linalg_tensorsolve_det",
        )

        rhs0 = _gather_rhs_row(
            ctx,
            rhs,
            rhs_shape=(2,),
            row=0,
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_rhs0",
        )
        rhs1 = _gather_rhs_row(
            ctx,
            rhs,
            rhs_shape=(2,),
            row=1,
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_rhs1",
        )

        d_rhs0 = _mul_coeff_rhs(
            ctx,
            d,
            rhs0,
            coeff_shape=(),
            rhs_shape=(),
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_d_rhs0",
        )
        b_rhs1 = _mul_coeff_rhs(
            ctx,
            b,
            rhs1,
            coeff_shape=(),
            rhs_shape=(),
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_b_rhs1",
        )
        numerator0 = _binary_op(
            ctx,
            "Sub",
            d_rhs0,
            b_rhs1,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="linalg_tensorsolve_num0",
        )

        a_rhs1 = _mul_coeff_rhs(
            ctx,
            a,
            rhs1,
            coeff_shape=(),
            rhs_shape=(),
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_a_rhs1",
        )
        c_rhs0 = _mul_coeff_rhs(
            ctx,
            c,
            rhs0,
            coeff_shape=(),
            rhs_shape=(),
            dtype_enum=dtype_enum,
            name_hint="linalg_tensorsolve_c_rhs0",
        )
        numerator1 = _binary_op(
            ctx,
            "Sub",
            a_rhs1,
            c_rhs0,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="linalg_tensorsolve_num1",
        )

        x0 = _binary_op(
            ctx,
            "Div",
            numerator0,
            det,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="linalg_tensorsolve_x0",
        )
        x1 = _binary_op(
            ctx,
            "Div",
            numerator1,
            det,
            dtype_enum=dtype_enum,
            shape=(),
            name_hint="linalg_tensorsolve_x1",
        )
        solved = _stack_solution_rows(
            ctx,
            x0,
            x1,
            row_shape=(),
            output_shape=(2,),
            dtype_enum=dtype_enum,
            output_name=(
                desired_name if output_shape == (2,) else ctx.fresh_name("solve")
            ),
        )
        if output_shape != (2,):
            solved = _reshape(
                ctx,
                solved,
                shape=output_shape,
                dtype_enum=dtype_enum,
                name_hint="linalg_tensorsolve_result",
                output_name=desired_name,
            )
        ctx.bind_value_for_var(out_var, solved)

    @classmethod
    def binding_specs(cls) -> list[MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.linalg.tensorsolve not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                b: ArrayLike,
                axes: tuple[int, ...] | None = None,
            ) -> jax.Array:
                if axes is not None:
                    return orig(a, b, axes=axes)
                a_shape = getattr(a, "shape", None)
                b_shape = getattr(b, "shape", None)
                if a_shape is not None and b_shape is not None:
                    try:
                        _, _, rows, _ = _normalise_shapes(
                            tuple(a_shape), tuple(b_shape)
                        )
                    except (TypeError, ValueError, NotImplementedError):
                        rows = 0
                    if rows in (1, 2):
                        return cls._PRIM.bind(
                            jnp.asarray(a),
                            jnp.asarray(b),
                            axes=None,
                        )
                return orig(a, b, axes=axes)

            return _patched

        return [
            MonkeyPatchSpec(
                target="jax.numpy.linalg",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            )
        ]


@JnpLinalgTensorSolvePlugin._PRIM.def_impl
def _linalg_tensorsolve_impl(
    a: ArrayLike,
    b: ArrayLike,
    *,
    axes: tuple[int, ...] | None = None,
) -> jax.Array:
    try:
        orig = get_orig_impl(
            JnpLinalgTensorSolvePlugin._PRIM,
            JnpLinalgTensorSolvePlugin._FUNC_NAME,
        )
    except RuntimeError:
        orig = _JAX_TENSORSOLVE_ORIG
    return orig(a, b, axes=axes)


JnpLinalgTensorSolvePlugin._PRIM.def_abstract_eval(
    JnpLinalgTensorSolvePlugin.abstract_eval
)
