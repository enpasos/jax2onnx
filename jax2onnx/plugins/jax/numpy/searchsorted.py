# jax2onnx/plugins/jax/numpy/searchsorted.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, TypeAlias, cast

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SEARCHSORTED_PRIM: Final = make_jnp_primitive("jax.numpy.searchsorted")

BatchDim: TypeAlias = int | None


def _validate_side(side: str) -> str:
    side_str = str(side)
    if side_str not in {"left", "right"}:
        raise ValueError("searchsorted side must be 'left' or 'right'")
    return side_str


def _result_dtype() -> np.dtype[Any]:
    return np.dtype(np.int32)


def _cast_to_dtype(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    from_dtype: np.dtype[Any],
    to_dtype: np.dtype[Any],
    name_hint: str,
    shape: tuple[object, ...],
) -> ir.Value:
    if from_dtype == to_dtype:
        return val
    target_enum = _dtype_to_ir(to_dtype, ctx.builder.enable_double_precision)
    cast_val = ctx.builder.Cast(
        val,
        to=int(target_enum.value),
        _outputs=[ctx.fresh_name(name_hint)],
    )
    cast_val.type = ir.TensorType(target_enum)
    _stamp_type_and_shape(cast_val, shape)
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


def _unsqueeze(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    axis: int,
    shape: tuple[object, ...],
    name_hint: str,
) -> ir.Value:
    axes = _const_i64(ctx, np.asarray([axis], dtype=np.int64), f"{name_hint}_axes")
    result = ctx.builder.Unsqueeze(
        val,
        axes,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    if getattr(val, "type", None) is not None:
        result.type = val.type
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


@register_primitive(
    jaxpr_primitive=_SEARCHSORTED_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.searchsorted.html",
    onnx=[
        {"component": "Less", "doc": "https://onnx.ai/onnx/operators/onnx__Less.html"},
        {
            "component": "LessOrEqual",
            "doc": "https://onnx.ai/onnx/operators/onnx__LessOrEqual.html",
        },
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
    ],
    since="0.13.0",
    context="primitives.jnp",
    component="searchsorted",
    testcases=[
        {
            "testcase": "jnp_searchsorted_left_vector",
            "callable": lambda a, v: jnp.searchsorted(a, v, side="left"),
            "input_values": [
                np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 2.0, 7.0, 8.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Less:5x4 -> Cast:5x4 -> ReduceSum:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_searchsorted_right_vector",
            "callable": lambda a, v: jnp.searchsorted(a, v, side="right"),
            "input_values": [
                np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 2.0, 7.0, 8.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["LessOrEqual:5x4 -> Cast:5x4 -> ReduceSum:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_searchsorted_scalar_value",
            "callable": lambda a, v: jnp.searchsorted(a, v, side="left"),
            "input_values": [
                np.asarray([1, 3, 5, 7], dtype=np.int32),
                np.asarray(4, dtype=np.int32),
            ],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Less:4 -> Cast:4 -> ReduceSum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "searchsorted_vmap_values",
            "callable": lambda a, v: jax.vmap(
                lambda scalar: jnp.searchsorted(a, scalar, side="left")
            )(v),
            "input_values": [
                np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float32),
                np.asarray([0.0, 2.0, 8.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.int32],
        },
    ],
)
class JnpSearchSortedPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SEARCHSORTED_PRIM
    _FUNC_NAME: ClassVar[str] = "searchsorted"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        a: core.AbstractValue,
        v: core.AbstractValue,
        *,
        side: str = "left",
        method: str = "scan",
    ) -> core.ShapedArray:
        del method
        _validate_side(side)
        if len(tuple(getattr(a, "shape", ()))) != 1:
            raise TypeError("jnp.searchsorted lowering requires 1-D sorted input 'a'")
        return core.ShapedArray(tuple(getattr(v, "shape", ())), _result_dtype())

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        a_var, v_var = eqn.invars
        (out_var,) = eqn.outvars
        params = getattr(eqn, "params", {})
        side = _validate_side(params.get("side", "left"))

        a_shape = tuple(getattr(a_var.aval, "shape", ()))
        v_shape = tuple(getattr(v_var.aval, "shape", ()))
        if len(a_shape) != 1:
            raise TypeError("jnp.searchsorted lowering requires 1-D sorted input 'a'")
        a_len = a_shape[0]
        if not isinstance(a_len, (int, np.integer)):
            raise TypeError("jnp.searchsorted lowering requires static length for 'a'")

        a_dtype: np.dtype[Any] = np.dtype(getattr(a_var.aval, "dtype", np.float32))
        v_dtype: np.dtype[Any] = np.dtype(getattr(v_var.aval, "dtype", a_dtype))
        compare_dtype: np.dtype[Any] = np.promote_types(a_dtype, v_dtype)

        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("searchsorted_a"))
        v_val = ctx.get_value_for_var(v_var, name_hint=ctx.fresh_name("searchsorted_v"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("searchsorted_out")
        )

        a_ready = _cast_to_dtype(
            ctx,
            a_val,
            from_dtype=a_dtype,
            to_dtype=compare_dtype,
            name_hint="searchsorted_a_cast",
            shape=a_shape,
        )
        v_ready = _cast_to_dtype(
            ctx,
            v_val,
            from_dtype=v_dtype,
            to_dtype=compare_dtype,
            name_hint="searchsorted_v_cast",
            shape=v_shape,
        )

        a_broadcast = a_ready
        a_broadcast_shape: tuple[object, ...] = (int(a_len),)
        for _ in range(len(v_shape)):
            a_broadcast_shape = (1, *a_broadcast_shape)
            a_broadcast = _unsqueeze(
                ctx,
                a_broadcast,
                axis=0,
                shape=a_broadcast_shape,
                name_hint="searchsorted_a_unsqueeze",
            )

        v_broadcast = _unsqueeze(
            ctx,
            v_ready,
            axis=len(v_shape),
            shape=(*v_shape, 1),
            name_hint="searchsorted_v_unsqueeze",
        )

        compare_shape = (*v_shape, int(a_len))
        if side == "left":
            comparison = ctx.builder.Less(
                a_broadcast,
                v_broadcast,
                _outputs=[ctx.fresh_name("searchsorted_compare")],
            )
        else:
            comparison = ctx.builder.LessOrEqual(
                a_broadcast,
                v_broadcast,
                _outputs=[ctx.fresh_name("searchsorted_compare")],
            )
        comparison.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(comparison, compare_shape)
        _ensure_value_metadata(ctx, comparison)

        out_dtype = _result_dtype()
        out_enum = numpy_dtype_to_ir(out_dtype)
        counts_input = ctx.builder.Cast(
            comparison,
            to=int(out_enum.value),
            _outputs=[ctx.fresh_name("searchsorted_counts_input")],
        )
        counts_input.type = ir.TensorType(out_enum)
        _stamp_type_and_shape(counts_input, compare_shape)
        _ensure_value_metadata(ctx, counts_input)

        axes = _const_i64(
            ctx,
            np.asarray([len(v_shape)], dtype=np.int64),
            "searchsorted_reduce_axes",
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "searchsorted_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("searchsorted_out")

        result = ctx.builder.ReduceSum(
            counts_input,
            axes,
            keepdims=0,
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = ir.TensorType(out_enum)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
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
                    "Original jnp.searchsorted not found for monkey patching"
                )
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                v: ArrayLike,
                side: str = "left",
                sorter: ArrayLike | None = None,
                *,
                method: str = "scan",
            ) -> jax.Array:
                if sorter is not None:
                    raise NotImplementedError(
                        "jnp.searchsorted with 'sorter' is not supported for ONNX export"
                    )
                return cls._PRIM.bind(
                    jnp.asarray(a),
                    jnp.asarray(v),
                    side=_validate_side(side),
                    method=str(method),
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


@JnpSearchSortedPlugin._PRIM.def_impl
def _searchsorted_impl(
    a: object,
    v: object,
    *,
    side: str = "left",
    method: str = "scan",
) -> jax.Array:
    orig = get_orig_impl(JnpSearchSortedPlugin._PRIM, JnpSearchSortedPlugin._FUNC_NAME)
    return cast(jax.Array, orig(a, v, side=side, sorter=None, method=method))


def _searchsorted_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    side: str = "left",
    method: str = "scan",
) -> tuple[jax.Array, BatchDim]:
    a, v = batched_args
    a_bdim, v_bdim = batch_dims
    if a_bdim is not None:
        raise NotImplementedError(
            "vmap over the sorted input of jnp.searchsorted is not supported"
        )
    result = JnpSearchSortedPlugin._PRIM.bind(
        a,
        v,
        side=_validate_side(side),
        method=method,
    )
    return result, v_bdim


batching.primitive_batchers[JnpSearchSortedPlugin._PRIM] = _searchsorted_batch_rule
