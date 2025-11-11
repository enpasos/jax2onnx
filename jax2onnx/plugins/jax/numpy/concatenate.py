# jax2onnx/plugins/jax/numpy/concatenate.py

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, Final, cast

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
import onnx_ir as ir
from jax.interpreters import batching

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_ORIGINAL_JNP_CONCATENATE: Final = jnp.concatenate

ArrayTuple = tuple[ArrayLike, ...]


def _to_tuple(arrays: Iterable[ArrayLike]) -> ArrayTuple:
    if isinstance(arrays, (list, tuple)):
        return tuple(arrays)
    return tuple(arrays)


def _normalize_axis(axis: int, rank: int) -> int:
    if rank == 0:
        return 0
    ax = int(axis)
    return ax % rank if ax < 0 else ax


def _promote_dtype(dtypes: Sequence[np.dtype[Any]]) -> np.dtype[Any]:
    result = dtypes[0]
    for dt in dtypes[1:]:
        result = np.promote_types(result, dt)
    return result


def _concat_dynamic_tile(x: jax.Array) -> jax.Array:
    """Mimic the concat-with-token pattern used in Transformer blocks."""

    # x : (B, N, D)
    d_feature = x.shape[2]
    token = jnp.zeros((1, 1, d_feature), dtype=x.dtype)
    tiled = jnp.broadcast_to(token, (x.shape[0], 1, d_feature))
    return jnp.concatenate([tiled, x], axis=1)


_CONCAT_PRIM: Final = make_jnp_primitive("jax.numpy.concatenate")


@register_primitive(
    jaxpr_primitive=_CONCAT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concatenate.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        }
    ],
    since="v0.8.0",
    context="primitives.jnp",
    component="concatenate",
    testcases=[
        {
            "testcase": "concatenate_basic",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=0),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Concat:6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_mixed_dtypes",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=0),
            "input_shapes": [(3,), (3,)],
            "input_dtypes": [np.float32, np.int32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Concat:6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_with_explicit_dtype",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=0, dtype=jnp.float64),
            "input_shapes": [(3,), (3,)],
            "input_dtypes": [np.float32, np.int32],
            "enable_double_precision": True,
            "post_check_onnx_graph": EG(
                ["Concat:6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_with_explicit_dtype_casts_inputs",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=1, dtype=jnp.float32),
            "input_shapes": [(5, 1), (5, 1)],
            "input_dtypes": [np.int32, np.int32],
            "expected_output_shapes": [(5, 2)],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast:5x1 -> Concat:5x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_abstract_middle_dim",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=1),
            "input_shapes": [("B", 1, 8), ("B", 10, 8)],
            "expected_output_shapes": [("B", 11, 8)],
            "post_check_onnx_graph": EG(
                ["Concat:Bx11x8"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_tile_and_symbolic",
            "callable": _concat_dynamic_tile,
            "input_shapes": [("B", 49, 256)],
            "expected_output_shapes": [("B", 50, 256)],
            "post_check_onnx_graph": EG(
                ["Concat:Bx50x256"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpConcatenatePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _CONCAT_PRIM
    _FUNC_NAME: ClassVar[str] = "concatenate"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def _canonicalize_call(
        *args: object, **kwargs: object
    ) -> tuple[ArrayTuple, int, DTypeLike | None]:
        if not args:
            raise TypeError(
                "concatenate() missing required positional argument 'arrays'"
            )

        arrays_obj = args[0]
        arrays = _to_tuple(cast(Iterable[ArrayLike], arrays_obj))
        if len(args) > 1:
            axis = args[1]
        else:
            axis = kwargs.pop("axis", 0)
        if len(args) > 2:
            dtype = args[2]
        else:
            dtype = kwargs.pop("dtype", None)
        if len(args) > 3 or kwargs:
            raise TypeError("concatenate() received unexpected arguments")

        if not arrays:
            raise ValueError("need at least one array to concatenate")
        return arrays, axis, dtype

    @staticmethod
    def abstract_eval(
        *arrays: core.AbstractValue,
        axis: int = 0,
        dtype: DTypeLike | None = None,
    ) -> core.ShapedArray:
        if not arrays:
            raise ValueError("concatenate requires at least one operand")
        rank = len(arrays[0].shape)
        norm_axis = _normalize_axis(axis, rank)

        axis_sizes = []
        other_dims = arrays[0].shape
        for aval in arrays:
            if len(aval.shape) != rank:
                raise ValueError("all arrays must have the same rank")
            for i, (dim_ref, dim_cur) in enumerate(zip(other_dims, aval.shape)):
                if i == norm_axis:
                    continue
                if dim_ref != dim_cur:
                    raise ValueError("all non-concatenated dimensions must match")
            axis_sizes.append(aval.shape[norm_axis])

        out_shape = list(other_dims)
        if all(isinstance(sz, (int, np.integer)) for sz in axis_sizes):
            out_shape[norm_axis] = int(sum(int(sz) for sz in axis_sizes))
        else:
            out_shape[norm_axis] = axis_sizes[0]
        if dtype is not None:
            out_dtype = np.dtype(dtype)
        else:
            out_dtype = _promote_dtype([np.dtype(a.dtype) for a in arrays])
        return jax.core.ShapedArray(tuple(out_shape), out_dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        out_var = eqn.outvars[0]
        in_vars = list(eqn.invars)
        params = getattr(eqn, "params", {})

        axis = params.get("axis", 0)
        dtype_param = params.get("dtype", None)

        first_shape = tuple(getattr(in_vars[0].aval, "shape", ()))
        rank = len(first_shape)
        norm_axis = _normalize_axis(axis, rank)

        target_dtype = (
            np.dtype(dtype_param)
            if dtype_param is not None
            else _promote_dtype(
                [np.dtype(getattr(v.aval, "dtype", np.float32)) for v in in_vars]
            )
        )
        target_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)

        inputs: list[ir.Value] = []
        for var in in_vars:
            val = ctx.get_value_for_var(var, name_hint=ctx.fresh_name("jnp_concat_in"))
            var_dtype = np.dtype(getattr(var.aval, "dtype", target_dtype))
            if var_dtype != target_dtype:
                cast_val = ctx.builder.Cast(
                    val,
                    _outputs=[ctx.fresh_name("jnp_concat_cast")],
                    to=int(target_enum.value),
                )
                cast_val.type = ir.TensorType(target_enum)
                _stamp_type_and_shape(cast_val, tuple(getattr(var.aval, "shape", ())))
                _ensure_value_metadata(ctx, cast_val)
                inputs.append(cast_val)
            else:
                inputs.append(val)

        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_concat_out")
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Concat")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Concat")

        result = ctx.builder.Concat(
            *inputs,
            axis=int(norm_axis),
            _outputs=[desired_name],
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        result.type = ir.TensorType(target_enum)
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
                raise RuntimeError("Original jnp.concatenate not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(*args: object, **kwargs: object) -> jax.Array:
                arrays, axis, dtype = cls._canonicalize_call(*args, **kwargs)
                axis_int = int(axis)
                return cls._PRIM.bind(*arrays, axis=axis_int, dtype=dtype)

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


@JnpConcatenatePlugin._PRIM.def_impl
def _concatenate_impl(*args: object, **kwargs: object) -> jax.Array:
    try:
        orig = get_orig_impl(
            JnpConcatenatePlugin._PRIM, JnpConcatenatePlugin._FUNC_NAME
        )
    except RuntimeError:
        orig = _ORIGINAL_JNP_CONCATENATE
    return orig(*args, **kwargs)


BatchDim = int | type(batching.not_mapped)


def _concatenate_batch_rule(
    batched_args: tuple[object, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axis: int = 0,
    dtype: DTypeLike | None = None,
) -> tuple[jax.Array, BatchDim]:
    axis_int = int(axis)
    axis_size: int | None = None
    for arg, bdim in zip(batched_args, batch_dims):
        if bdim is batching.not_mapped:
            continue
        shape = getattr(arg, "shape", None)
        if shape is None or bdim >= len(shape):
            continue
        axis_size = shape[bdim]
        break

    if axis_size is None:
        out = JnpConcatenatePlugin._PRIM.bind(*batched_args, axis=axis_int, dtype=dtype)
        return out, batching.not_mapped

    prepared_args = [
        batching.bdim_at_front(arg, bdim, axis_size)
        for arg, bdim in zip(batched_args, batch_dims)
    ]

    try:
        orig = get_orig_impl(
            JnpConcatenatePlugin._PRIM, JnpConcatenatePlugin._FUNC_NAME
        )
    except RuntimeError:
        orig = _ORIGINAL_JNP_CONCATENATE

    def _call_single(*slices: object) -> jax.Array:
        if dtype is None:
            return orig(slices, axis=axis_int)
        return orig(slices, axis=axis_int, dtype=dtype)

    result = jax.vmap(_call_single)(*prepared_args)
    return result, 0


batching.primitive_batchers[JnpConcatenatePlugin._PRIM] = _concatenate_batch_rule
