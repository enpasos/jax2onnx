# jax2onnx/plugins/jax/numpy/size.py

from __future__ import annotations

from collections.abc import Sequence as _Seq
from typing import Any, Callable, ClassVar, Final

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


_SIZE_PRIM: Final = make_jnp_primitive("jax.numpy.size")


def _normalize_axes(axes: tuple[int, ...] | None, rank: int) -> tuple[int, ...] | None:
    if axes is None:
        return None
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_axis in axes:
        axis = int(raw_axis)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError(
                f"axis {raw_axis} is out of bounds for array of dimension {rank}"
            )
        if axis in seen:
            raise ValueError(f"repeated axis: {axes}")
        seen.add(axis)
        normalized.append(axis)
    return tuple(normalized)


def _canon_axis_param(
    axis: int | _Seq[int] | None,
) -> tuple[tuple[int, ...] | None, bool]:
    if axis is None:
        return None, False
    if isinstance(axis, _Seq) and not isinstance(axis, (str, bytes)):
        return tuple(int(ax) for ax in axis), True
    return (int(axis),), False


@register_primitive(
    jaxpr_primitive=_SIZE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.size.html",
    onnx=[
        {
            "component": "Size",
            "doc": "https://onnx.ai/onnx/operators/onnx__Size.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="size",
    testcases=[
        {
            "testcase": "jnp_size_all",
            "callable": lambda x: jnp.size(x),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["Size"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_size_axis",
            "callable": lambda x: jnp.size(x, axis=1),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["Shape -> Gather:1 -> Squeeze"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_size_axis_tuple",
            "callable": lambda x: jnp.size(x, axis=(0, 2)),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["Shape -> Gather:2 -> ReduceProd"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_size_dynamic",
            "callable": lambda x: jnp.size(x),
            "input_shapes": [("B", 12, "T")],
            "post_check_onnx_graph": EG(
                ["Size"],
                symbols={"B": None, "T": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpSizePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SIZE_PRIM
    _FUNC_NAME: ClassVar[str] = "size"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        axes: tuple[int, ...] | None = None,
        axes_is_tuple: bool = False,  # kept for bind symmetry
    ) -> core.ShapedArray:
        del x, axes, axes_is_tuple
        use_x64 = bool(jax.config.read("jax_enable_x64"))
        out_dtype = np.int64 if use_x64 else np.int32
        return core.ShapedArray((), out_dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars
        params = dict(getattr(eqn, "params", {}) or {})
        axes_param = params.get("axes")

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("size_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("size_out"))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        axes = _normalize_axes(
            tuple(int(ax) for ax in axes_param) if axes_param is not None else None,
            rank,
        )

        if axes is None:
            size_i64 = ctx.builder.Size(
                x_val,
                _outputs=[ctx.fresh_name("size_i64")],
            )
            size_i64.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(size_i64, ())
            _ensure_value_metadata(ctx, size_i64)
        else:
            if len(axes) == 0:
                size_i64 = _const_i64(ctx, np.asarray(1, dtype=np.int64), "size_one")
            else:
                shape_val = ctx.builder.Shape(
                    x_val,
                    _outputs=[ctx.fresh_name("size_shape")],
                )
                shape_val.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(shape_val, (rank,))
                _ensure_value_metadata(ctx, shape_val)

                axes_idx = _const_i64(
                    ctx, np.asarray(list(axes), dtype=np.int64), "size_axes_idx"
                )
                selected = ctx.builder.Gather(
                    shape_val,
                    axes_idx,
                    axis=0,
                    _outputs=[ctx.fresh_name("size_selected_dims")],
                )
                selected.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(selected, (len(axes),))
                _ensure_value_metadata(ctx, selected)

                if len(axes) == 1:
                    sq_axes = _const_i64(
                        ctx, np.asarray([0], dtype=np.int64), "size_squeeze_axes"
                    )
                    size_i64 = ctx.builder.Squeeze(
                        selected,
                        sq_axes,
                        _outputs=[ctx.fresh_name("size_i64")],
                    )
                    size_i64.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(size_i64, ())
                    _ensure_value_metadata(ctx, size_i64)
                else:
                    red_axes = _const_i64(
                        ctx, np.asarray([0], dtype=np.int64), "size_reduce_axes"
                    )
                    size_i64 = ctx.builder.ReduceProd(
                        selected,
                        red_axes,
                        keepdims=0,
                        _outputs=[ctx.fresh_name("size_i64")],
                    )
                    size_i64.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(size_i64, ())
                    _ensure_value_metadata(ctx, size_i64)

        out_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(out_var, "aval", None), "dtype", np.int32)
        )
        out_enum = _dtype_to_ir(out_dtype, bool(ctx.builder.enable_double_precision))

        if out_enum == ir.DataType.INT64:
            result = size_i64
        else:
            result = ctx.builder.Cast(
                size_i64,
                _outputs=[ctx.fresh_name("size_cast")],
                to=int(out_enum.value),
            )
            result.type = ir.TensorType(out_enum)
            _stamp_type_and_shape(result, ())
            _ensure_value_metadata(ctx, result)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("size_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("size_out")
        result.name = desired_name
        _stamp_type_and_shape(result, ())
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., Any] | None,
        ) -> Callable[..., Any]:
            if orig is None:
                raise RuntimeError("Original jnp.size not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                axis: int | _Seq[int] | None = None,
            ) -> Any:
                axes_param, axes_is_tuple = _canon_axis_param(axis)
                return cls._PRIM.bind(
                    jnp.asarray(a),
                    axes=axes_param,
                    axes_is_tuple=axes_is_tuple,
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


@JnpSizePlugin._PRIM.def_impl
def _size_impl(
    a: ArrayLike,
    *,
    axes: tuple[int, ...] | None = None,
    axes_is_tuple: bool = False,
) -> Any:
    orig = get_orig_impl(JnpSizePlugin._PRIM, JnpSizePlugin._FUNC_NAME)
    axis_arg: int | tuple[int, ...] | None = None
    if axes is not None:
        axis_vals = tuple(int(ax) for ax in axes)
        axis_arg = axis_vals if axes_is_tuple else axis_vals[0]
    return orig(a, axis=axis_arg)


JnpSizePlugin._PRIM.def_abstract_eval(JnpSizePlugin.abstract_eval)


def _size_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[Any, ...],
    *,
    axes: tuple[int, ...] | None = None,
    axes_is_tuple: bool = False,  # kept for bind symmetry
) -> tuple[jax.Array, Any]:
    del axes_is_tuple
    (operand,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = JnpSizePlugin._PRIM.bind(operand, axes=axes, axes_is_tuple=True)
        return out, batching.not_mapped

    batch_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, batch_size)
    slice_rank = operand.ndim - 1

    if axes is None:
        slice_axes = tuple(range(slice_rank))
    else:
        slice_axes_opt = _normalize_axes(tuple(int(ax) for ax in axes), slice_rank)
        if slice_axes_opt is None:
            raise AssertionError("axes normalization unexpectedly returned None")
        slice_axes = slice_axes_opt
    axes_with_batch = tuple(ax + 1 for ax in slice_axes)
    per_slice_scalar = JnpSizePlugin._PRIM.bind(
        operand,
        axes=axes_with_batch,
        axes_is_tuple=True,
    )
    broadcasted = jnp.broadcast_to(per_slice_scalar, (batch_size,))
    return broadcasted, 0


batching.primitive_batchers[JnpSizePlugin._PRIM] = _size_batch_rule
