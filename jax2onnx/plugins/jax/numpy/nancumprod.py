# jax2onnx/plugins/jax/numpy/nancumprod.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, TypeAlias

import jax
from jax import core
from jax.interpreters import batching
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


_NANCUMPROD_PRIM: Final = make_jnp_primitive("jax.numpy.nancumprod")
_JAX_NANCUMPROD_ORIG: Final = jnp.nancumprod

BatchDim: TypeAlias = int | None


def _all_static_ints(shape: tuple[object, ...]) -> bool:
    return all(isinstance(dim, (int, np.integer)) for dim in shape)


def _num_elements(shape: tuple[object, ...]) -> int:
    if not _all_static_ints(shape):
        raise TypeError("jnp.nancumprod lowering requires static input shape")
    return int(np.prod(tuple(int(dim) for dim in shape), dtype=np.int64))


def _const_scalar(
    ctx: LoweringContextProtocol,
    *,
    dtype: np.dtype[Any],
    value: int | float,
    name_hint: str,
) -> ir.Value:
    dtype_enum = _dtype_to_ir(dtype, ctx.builder.enable_double_precision)
    result = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(name_hint),
        array=np.asarray(value, dtype=dtype),
    )
    result.type = ir.TensorType(dtype_enum)
    _stamp_type_and_shape(result, ())
    _ensure_value_metadata(ctx, result)
    return result


def _const_bool(
    ctx: LoweringContextProtocol,
    values: np.ndarray[Any, np.dtype[np.bool_]],
    *,
    name_hint: str,
) -> ir.Value:
    result = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(name_hint),
        array=np.asarray(values, dtype=np.bool_),
    )
    result.type = ir.TensorType(ir.DataType.BOOL)
    _stamp_type_and_shape(result, tuple(int(dim) for dim in values.shape))
    _ensure_value_metadata(ctx, result)
    return result


def _reshape(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    shape_val = _const_i64(
        ctx,
        np.asarray(shape, dtype=np.int64),
        f"{name_hint}_shape",
    )
    result = ctx.builder.Reshape(
        val,
        shape_val,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    if getattr(val, "type", None) is not None:
        result.type = val.type
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _cast_to_dtype(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    to_dtype: np.dtype[Any],
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    target_enum = _dtype_to_ir(to_dtype, ctx.builder.enable_double_precision)
    if getattr(getattr(val, "type", None), "dtype", None) == target_enum:
        return val
    result = ctx.builder.Cast(
        val,
        to=int(target_enum.value),
        _outputs=[ctx.fresh_name(name_hint)],
    )
    result.type = ir.TensorType(target_enum)
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _abstract_eval_via_orig(
    x: core.AbstractValue,
    *,
    axis: int | None,
    dtype: np.dtype[Any] | type | None,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    x_dtype = np.dtype(getattr(x, "dtype", np.float32))
    if np.issubdtype(x_dtype, np.complexfloating):
        raise TypeError("jnp.nancumprod lowering does not support complex inputs")

    out = jax.eval_shape(
        lambda value: _JAX_NANCUMPROD_ORIG(value, axis=axis, dtype=dtype),
        jax.ShapeDtypeStruct(x_shape, x_dtype),
    )
    out_dtype = np.dtype(getattr(out, "dtype", np.float32))
    if np.issubdtype(out_dtype, np.complexfloating):
        raise TypeError("jnp.nancumprod lowering does not support complex outputs")
    return core.ShapedArray(tuple(getattr(out, "shape", ())), out_dtype)


@register_primitive(
    jaxpr_primitive=_NANCUMPROD_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nancumprod.html",
    onnx=[
        {
            "component": "IsNaN",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsNaN.html",
        },
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Expand",
            "doc": "https://onnx.ai/onnx/operators/onnx__Expand.html",
        },
        {
            "component": "ReduceProd",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceProd.html",
        },
        {
            "component": "Cast",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html",
        },
    ],
    since="0.13.0",
    context="primitives.jnp",
    component="nancumprod",
    testcases=[
        {
            "testcase": "jnp_nancumprod_axis1",
            "callable": lambda x: jnp.nancumprod(x, axis=1),
            "input_values": [
                np.asarray(
                    [[2.0, np.nan, 3.0], [np.nan, 4.0, 5.0]],
                    dtype=np.float32,
                )
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["IsNaN:2x3", "Where:2x3", "Expand:2x3x3", "ReduceProd:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_nancumprod_axis_none_flatten",
            "callable": lambda x: jnp.nancumprod(x),
            "input_values": [
                np.asarray(
                    [[1.0, np.nan, 2.0], [3.0, np.nan, 4.0]],
                    dtype=np.float32,
                )
            ],
            "expected_output_shapes": [(6,)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Reshape:6", "IsNaN:6", "Expand:6x6", "ReduceProd:6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_nancumprod_dtype_cast_i32",
            "callable": lambda x: jnp.nancumprod(x, axis=0, dtype=jnp.float32),
            "input_values": [
                np.asarray(
                    [[1, 2, 3], [4, 5, 6]],
                    dtype=np.int32,
                )
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast:2x3", "Expand:2x2x3", "ReduceProd:2x3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpNanCumProdPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _NANCUMPROD_PRIM
    _FUNC_NAME: ClassVar[str] = "nancumprod"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        axis: int | None = None,
        dtype: np.dtype[Any] | type | None = None,
    ) -> core.ShapedArray:
        return _abstract_eval_via_orig(x, axis=axis, dtype=dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (operand_var,) = eqn.invars
        (out_var,) = eqn.outvars

        params = getattr(eqn, "params", {})
        axis_param = params.get("axis", None)

        operand_shape_raw = tuple(getattr(operand_var.aval, "shape", ()))
        if not _all_static_ints(operand_shape_raw):
            raise TypeError("jnp.nancumprod lowering requires static input shape")
        operand_shape = tuple(int(dim) for dim in operand_shape_raw)

        if axis_param is None:
            work_shape = (_num_elements(operand_shape_raw),)
            axis = 0
        else:
            work_shape = operand_shape
            rank = len(work_shape)
            axis = int(axis_param)
            if axis < 0 and rank:
                axis = axis % rank
            if axis < 0 or axis >= rank:
                raise ValueError(f"nancumprod axis {axis_param} out of bounds")

        axis_size = int(work_shape[axis])
        if axis_size < 1:
            raise ValueError("jnp.nancumprod lowering requires non-empty axis")

        operand_dtype = np.dtype(getattr(operand_var.aval, "dtype", np.float32))
        out_dtype = np.dtype(getattr(out_var.aval, "dtype", operand_dtype))
        if np.issubdtype(operand_dtype, np.complexfloating) or np.issubdtype(
            out_dtype, np.complexfloating
        ):
            raise TypeError("jnp.nancumprod lowering does not support complex dtypes")

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("jnp_nancumprod_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_nancumprod_out")
        )

        work_val = operand_val
        if axis_param is None:
            work_val = _reshape(
                ctx,
                work_val,
                shape=work_shape,
                name_hint="jnp_nancumprod_flatten",
            )
        else:
            dtype_enum = _dtype_to_ir(
                operand_dtype, ctx.builder.enable_double_precision
            )
            work_val.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(work_val, work_shape)
            _ensure_value_metadata(ctx, work_val)

        if np.issubdtype(operand_dtype, np.floating):
            is_nan = ctx.builder.IsNaN(
                work_val,
                _outputs=[ctx.fresh_name("jnp_nancumprod_is_nan")],
            )
            is_nan.type = ir.TensorType(ir.DataType.BOOL)
            _stamp_type_and_shape(is_nan, work_shape)
            _ensure_value_metadata(ctx, is_nan)

            source_one = _const_scalar(
                ctx,
                dtype=operand_dtype,
                value=1,
                name_hint="jnp_nancumprod_source_one",
            )
            clean = ctx.builder.Where(
                is_nan,
                source_one,
                work_val,
                _outputs=[ctx.fresh_name("jnp_nancumprod_nan_to_one")],
            )
            clean.type = work_val.type
            _stamp_type_and_shape(clean, work_shape)
            _ensure_value_metadata(ctx, clean)
            work_val = clean

        product_input = _cast_to_dtype(
            ctx,
            work_val,
            to_dtype=out_dtype,
            shape=work_shape,
            name_hint="jnp_nancumprod_cast",
        )
        out_dtype_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        product_input.type = ir.TensorType(out_dtype_enum)
        _stamp_type_and_shape(product_input, work_shape)
        _ensure_value_metadata(ctx, product_input)

        rank = len(work_shape)
        pre_shape = work_shape[:axis]
        post_shape = work_shape[axis + 1 :]
        expanded_input_shape = pre_shape + (1, axis_size) + post_shape
        expanded_shape = pre_shape + (axis_size, axis_size) + post_shape

        reshaped = _reshape(
            ctx,
            product_input,
            shape=expanded_input_shape,
            name_hint="jnp_nancumprod_expand_base",
        )
        target_shape = _const_i64(
            ctx,
            np.asarray(expanded_shape, dtype=np.int64),
            "jnp_nancumprod_expand_shape",
        )
        expanded = ctx.builder.Expand(
            reshaped,
            target_shape,
            _outputs=[ctx.fresh_name("jnp_nancumprod_expanded")],
        )
        expanded.type = ir.TensorType(out_dtype_enum)
        _stamp_type_and_shape(expanded, expanded_shape)
        _ensure_value_metadata(ctx, expanded)

        mask_matrix = (
            np.arange(axis_size)[None, :] <= np.arange(axis_size)[:, None]
        ).astype(np.bool_)
        mask_shape = (1,) * axis + (axis_size, axis_size) + (1,) * (rank - axis - 1)
        mask = _const_bool(
            ctx,
            mask_matrix.reshape(mask_shape),
            name_hint="jnp_nancumprod_prefix_mask",
        )
        target_one = _const_scalar(
            ctx,
            dtype=out_dtype,
            value=1,
            name_hint="jnp_nancumprod_target_one",
        )
        masked = ctx.builder.Where(
            mask,
            expanded,
            target_one,
            _outputs=[ctx.fresh_name("jnp_nancumprod_masked")],
        )
        masked.type = ir.TensorType(out_dtype_enum)
        _stamp_type_and_shape(masked, expanded_shape)
        _ensure_value_metadata(ctx, masked)

        reduce_axis = axis + 1
        reduce_axes = _const_i64(
            ctx,
            np.asarray([reduce_axis], dtype=np.int64),
            "jnp_nancumprod_reduce_axis",
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "jnp_nancumprod_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("jnp_nancumprod_out")

        result = ctx.builder.ReduceProd(
            masked,
            reduce_axes,
            keepdims=0,
            _outputs=[desired_name],
        )
        result.type = ir.TensorType(out_dtype_enum)
        out_shape = tuple(int(dim) for dim in getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError(
                    "Original jnp.nancumprod not found for monkey patching"
                )
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                axis: int | None = None,
                dtype: np.dtype[Any] | type | None = None,
                out: Any | None = None,
            ) -> jax.Array:
                if out is not None:
                    raise NotImplementedError(
                        "jnp.nancumprod with 'out' is not supported for ONNX export"
                    )
                axis_arg = None if axis is None else int(axis)
                return cls._PRIM.bind(
                    jnp.asarray(a),
                    axis=axis_arg,
                    dtype=dtype,
                )

            return _patched

        return [
            AssignSpec(
                "jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpNanCumProdPlugin._PRIM.def_impl
def _nancumprod_impl(
    a: ArrayLike,
    *,
    axis: int | None = None,
    dtype: np.dtype[Any] | type | None = None,
) -> jax.Array:
    try:
        orig = get_orig_impl(
            JnpNanCumProdPlugin._PRIM,
            JnpNanCumProdPlugin._FUNC_NAME,
        )
    except RuntimeError:
        orig = _JAX_NANCUMPROD_ORIG
    return orig(jnp.asarray(a), axis=axis, dtype=dtype)


def _nancumprod_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axis: int | None = None,
    dtype: np.dtype[Any] | type | None = None,
) -> tuple[jax.Array, BatchDim]:
    (operand,), (bdim,) = batched_args, batch_dims

    if bdim is None:
        out = JnpNanCumProdPlugin._PRIM.bind(
            operand,
            axis=axis,
            dtype=dtype,
        )
        return out, None

    axis_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, axis_size)

    if axis is None:
        axis_body = None
    else:
        slice_rank = operand.ndim - 1
        axis_int = int(axis)
        axis_body = axis_int % slice_rank if slice_rank and axis_int < 0 else axis_int

    out = jax.vmap(
        lambda x: _JAX_NANCUMPROD_ORIG(x, axis=axis_body, dtype=dtype),
        in_axes=0,
    )(operand)
    return out, 0


batching.primitive_batchers[JnpNanCumProdPlugin._PRIM] = _nancumprod_batch_rule
