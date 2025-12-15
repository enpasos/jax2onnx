# jax2onnx/plugins/jax/numpy/take.py

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
import onnx_ir as ir
from flax import nnx
from jax import core
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)


_TAKE_PRIM: Final = make_jnp_primitive("jax.numpy.take")


class _ArangeTakeModule(nnx.Module):
    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        dtype: jnp.dtype | type = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.embedding = nnx.Param(
            jax.random.normal(rngs.params(), (num_embeddings, features), dtype=dtype)
        )

    def __call__(self, x: jax.Array):
        seq_len = x.shape[1]
        indices = jnp.arange(seq_len)
        return jnp.take(self.embedding.value, indices, axis=0)


def _canonical_axis(axis: int, rank: int) -> int:
    axis_norm = axis if axis >= 0 else axis + rank
    if axis_norm < 0 or axis_norm >= rank:
        raise ValueError(f"jnp.take axis {axis} out of bounds for rank {rank}")
    return axis_norm


def _as_int64(
    ctx: LoweringContextProtocol,
    value: ir.Value,
    shape: tuple[int | object, ...],
    name_hint: str,
) -> ir.Value:
    current_type = getattr(value, "type", None)
    current_dtype = getattr(current_type, "dtype", None)
    if current_dtype == ir.DataType.INT64:
        _stamp_type_and_shape(value, shape)
        _ensure_value_metadata(ctx, value)
        return value

    cast_val = ctx.builder.Cast(
        value,
        _outputs=[ctx.fresh_name(name_hint)],
        to=int(ir.DataType.INT64.value),
    )
    cast_val.type = ir.TensorType(ir.DataType.INT64)
    _stamp_type_and_shape(cast_val, shape)
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


@register_primitive(
    jaxpr_primitive=_TAKE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        }
    ],
    since="v0.7.0",
    context="primitives.jnp",
    component="take",
    testcases=[
        {
            "testcase": "take_data_dependent_indices",
            "callable": construct_and_call(
                _ArangeTakeModule,
                num_embeddings=10,
                features=16,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 10)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Range:10 -> Cast:10 -> Gather:Bx16"],
                symbols={"B": None},
            ),
        },
        {
            "testcase": "take_basic_axis1",
            "callable": lambda x, idx: jnp.take(x, idx, axis=1),
            "input_values": [
                np.arange(12, dtype=np.float32).reshape(3, 4),
                np.array([0, 2], dtype=np.int32),
            ],
            "expected_output_shapes": [(3, 2)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Gather:3x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "take_vmap_batching",
            "callable": lambda x: jax.vmap(
                lambda y: jnp.take(y, np.array([0, 2], dtype=np.int32), axis=0)
            )(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpTakePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _TAKE_PRIM
    _FUNC_NAME: ClassVar[str] = "take"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        arr: core.ShapedArray,
        indices: core.ShapedArray,
        *,
        axis: int | None = None,
        mode: str | None = None,
    ) -> core.ShapedArray:
        if mode is not None:
            raise NotImplementedError("jnp.take mode parameter is not supported")
        if axis is None:
            raise NotImplementedError("jnp.take with axis=None is not supported")
        rank = len(arr.shape)
        axis_norm = _canonical_axis(int(axis), rank)
        out_shape = arr.shape[:axis_norm] + indices.shape + arr.shape[axis_norm + 1 :]
        return core.ShapedArray(out_shape, arr.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        axis_param = params.get("axis")
        mode = params.get("mode")
        if mode is not None:
            raise NotImplementedError("jnp.take mode parameter is not supported")
        if axis_param is None:
            raise NotImplementedError("jnp.take with axis=None is not supported")

        arr_var, indices_var = eqn.invars
        out_var = eqn.outvars[0]

        arr_val = ctx.get_value_for_var(
            arr_var, name_hint=ctx.fresh_name("take_data"), prefer_np_dtype=None
        )
        indices_val = ctx.get_value_for_var(
            indices_var,
            name_hint=ctx.fresh_name("take_indices"),
            prefer_np_dtype=np.int64,
        )
        indices_dtype = np.dtype(getattr(indices_var.aval, "dtype", np.int64))
        if not np.issubdtype(indices_dtype, np.integer):
            raise TypeError("jnp.take indices must be integer typed")

        indices_shape = tuple(getattr(indices_var.aval, "shape", ()))
        indices_val = _as_int64(ctx, indices_val, indices_shape, "take_indices_int64")

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        axis = _canonical_axis(int(axis_param), len(arr_shape))

        result = ctx.builder.Gather(
            arr_val,
            indices_val,
            axis=int(axis),
            _outputs=[ctx.fresh_name("Gather")],
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        result.type = ir.TensorType(getattr(arr_val.type, "dtype", ir.DataType.FLOAT))
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
                raise RuntimeError("Original jnp.take not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                arr: ArrayLike,
                indices: ArrayLike,
                *,
                axis: int | None = None,
                mode: str | None = None,
            ) -> jax.Array:
                if axis is None or mode is not None:
                    return orig(arr, indices, axis=axis, mode=mode)
                indices_arr = jnp.asarray(indices)
                return cls._PRIM.bind(
                    jnp.asarray(arr),
                    indices_arr,
                    axis=int(axis),
                    mode=None,
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


@JnpTakePlugin._PRIM.def_impl
def _take_impl(
    arr: ArrayLike,
    indices: ArrayLike,
    *,
    axis: int | None = None,
    mode: str | None = None,
) -> jax.Array:
    orig = get_orig_impl(JnpTakePlugin._PRIM, JnpTakePlugin._FUNC_NAME)
    return orig(arr, indices, axis=axis, mode=mode)


JnpTakePlugin._PRIM.def_abstract_eval(JnpTakePlugin.abstract_eval)


BatchDim = int | type(batching.not_mapped)


def _take_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axis: int | None = None,
    mode: str | None = None,
) -> tuple[jax.Array, BatchDim]:
    arr, indices = batched_args
    arr_bdim, indices_bdim = batch_dims

    mapped = [
        (arg, bd)
        for arg, bd in zip(batched_args, batch_dims)
        if bd is not batching.not_mapped
    ]
    if not mapped:
        out = JnpTakePlugin._PRIM.bind(arr, indices, axis=axis, mode=mode)
        return out, batching.not_mapped

    sample_arg, sample_bd = mapped[0]
    batch_size = sample_arg.shape[sample_bd]

    if arr_bdim is not batching.not_mapped:
        arr = batching.bdim_at_front(arr, arr_bdim, batch_size)
    if indices_bdim is not batching.not_mapped:
        indices = batching.bdim_at_front(indices, indices_bdim, batch_size)

    in_axes = (
        0 if arr_bdim is not batching.not_mapped else None,
        0 if indices_bdim is not batching.not_mapped else None,
    )
    orig = get_orig_impl(JnpTakePlugin._PRIM, JnpTakePlugin._FUNC_NAME)

    def _call_single(a_slice: jax.Array, i_slice: jax.Array) -> jax.Array:
        return orig(a_slice, i_slice, axis=axis, mode=mode)

    result = jax.vmap(_call_single, in_axes=in_axes)(arr, indices)
    return result, 0


batching.primitive_batchers[JnpTakePlugin._PRIM] = _take_batch_rule
