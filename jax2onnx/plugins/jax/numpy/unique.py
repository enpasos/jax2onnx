# jax2onnx/plugins/jax/numpy/unique.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Final

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
from jax2onnx.plugins.jax.numpy._common import make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_UNIQUE_PRIM: Final = make_jnp_primitive("jax.numpy.unique")
_JNP_UNIQUE_ORIG: Final = jnp.unique


def _normalize_fill_value(fill_value: ArrayLike) -> bool | int | float:
    arr = np.asarray(fill_value)
    if arr.ndim != 0:
        raise NotImplementedError(
            "jnp.unique lowering currently requires scalar fill_value"
        )
    item = arr.item()
    if isinstance(item, (bool, int, float, np.bool_, np.integer, np.floating)):
        if isinstance(item, (np.bool_,)):
            return bool(item)
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return float(item)
        return item
    raise NotImplementedError("Unsupported fill_value type for jnp.unique lowering")


def _unique_bind(
    ar: ArrayLike,
    *,
    size: int,
    fill_value: ArrayLike,
    sorted: bool = True,
) -> Any:
    return _UNIQUE_PRIM.bind(
        ar,
        size=int(size),
        fill_value=_normalize_fill_value(fill_value),
        sorted=bool(sorted),
    )


@register_primitive(
    jaxpr_primitive=_UNIQUE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unique.html",
    onnx=[
        {
            "component": "Unique",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unique.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="unique",
    testcases=[
        {
            "testcase": "jnp_unique_f32_size_fill",
            "callable": lambda x: _unique_bind(x, size=5, fill_value=-1.0),
            "input_values": [np.array([3.0, 1.0, 2.0, 3.0], dtype=np.float32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Unique:? -> Concat:? -> Gather:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_unique_i32_size_fill",
            "callable": lambda x: _unique_bind(x, size=4, fill_value=-7),
            "input_values": [np.array([2, 2, 1, 3], dtype=np.int32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast:? -> Unique -> Cast:? -> Concat:? -> Gather:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_unique_symbolic_size_fill",
            "callable": lambda x: _unique_bind(x, size=6, fill_value=0.0),
            "input_shapes": [("B",)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Unique:? -> Concat:? -> Gather:6"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpUniquePlugin(PrimitiveLeafPlugin):
    """Lower ``jax.numpy.unique`` (value output) to ONNX ``Unique``."""

    _PRIM: ClassVar = _UNIQUE_PRIM
    _FUNC_NAME: ClassVar[str] = "unique"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        ar: core.AbstractValue,
        *,
        size: int,
        fill_value: bool | int | float,
        sorted: bool = True,
    ) -> core.ShapedArray:
        del fill_value, sorted
        size_int = int(size)
        if size_int < 0:
            raise ValueError("jnp.unique size must be non-negative")
        return core.ShapedArray((size_int,), ar.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = dict(getattr(eqn, "params", {}) or {})
        size_param = params.get("size")
        if size_param is None:
            raise NotImplementedError("jnp.unique ONNX lowering requires static `size`")
        size = int(size_param)
        sorted_flag = bool(params.get("sorted", True))
        fill_value = _normalize_fill_value(params.get("fill_value"))

        if size < 0:
            raise ValueError("jnp.unique size must be non-negative")

        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        x_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )
        if x_dtype == np.uint64:
            raise NotImplementedError("jnp.unique uint64 is not supported in ONNX path")

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("unique_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("unique_out")
        )

        enable_double = bool(getattr(ctx.builder, "enable_double_precision", False))
        out_enum = _dtype_to_ir(x_dtype, enable_double)

        needs_unique_cast = (x_dtype == np.bool_) or (
            np.issubdtype(x_dtype, np.integer) and x_dtype != np.int64
        )
        unique_in = x_val
        if needs_unique_cast:
            unique_in = ctx.builder.Cast(
                x_val,
                _outputs=[ctx.fresh_name("unique_input_i64")],
                to=int(ir.DataType.INT64.value),
            )
            unique_in.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(unique_in, x_shape)
            _ensure_value_metadata(ctx, unique_in)

        unique_vals = ctx.builder.Unique(
            unique_in,
            _outputs=[ctx.fresh_name("unique_vals")],
            sorted=int(sorted_flag),
        )
        if needs_unique_cast:
            unique_vals = ctx.builder.Cast(
                unique_vals,
                _outputs=[ctx.fresh_name("unique_vals_cast_back")],
                to=int(out_enum.value),
            )
        unique_vals.type = ir.TensorType(out_enum)
        _stamp_type_and_shape(unique_vals, (None,))
        _ensure_value_metadata(ctx, unique_vals)

        zero = _const_i64(ctx, np.asarray(0, dtype=np.int64), "unique_zero")
        one = _const_i64(ctx, np.asarray(1, dtype=np.int64), "unique_one")
        size_scalar = _const_i64(ctx, np.asarray(size, dtype=np.int64), "unique_size")
        size_vec = _const_i64(
            ctx, np.asarray([size], dtype=np.int64), "unique_size_vec"
        )

        idx = ctx.builder.Range(
            zero,
            size_scalar,
            one,
            _outputs=[ctx.fresh_name("unique_idx")],
        )
        idx.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(idx, (size,))
        _ensure_value_metadata(ctx, idx)

        uniq_shape = ctx.builder.Shape(
            unique_vals,
            _outputs=[ctx.fresh_name("unique_shape")],
        )
        uniq_shape.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(uniq_shape, (1,))
        _ensure_value_metadata(ctx, uniq_shape)

        m_expand = ctx.builder.Expand(
            uniq_shape,
            size_vec,
            _outputs=[ctx.fresh_name("unique_count_expand")],
        )
        m_expand.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(m_expand, (size,))
        _ensure_value_metadata(ctx, m_expand)

        valid_mask = ctx.builder.Less(
            idx,
            m_expand,
            _outputs=[ctx.fresh_name("unique_valid_mask")],
        )
        valid_mask.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(valid_mask, (size,))
        _ensure_value_metadata(ctx, valid_mask)

        select_idx = ctx.builder.Where(
            valid_mask,
            idx,
            m_expand,
            _outputs=[ctx.fresh_name("unique_select_idx")],
        )
        select_idx.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(select_idx, (size,))
        _ensure_value_metadata(ctx, select_idx)

        fill_vec = ctx.bind_const_for_var(
            object(),
            np.asarray([fill_value], dtype=x_dtype),
        )
        fill_vec.type = ir.TensorType(out_enum)
        _stamp_type_and_shape(fill_vec, (1,))
        _ensure_value_metadata(ctx, fill_vec)

        extended = ctx.builder.Concat(
            unique_vals,
            fill_vec,
            _outputs=[ctx.fresh_name("unique_extended")],
            axis=0,
        )
        extended.type = ir.TensorType(out_enum)
        _stamp_type_and_shape(extended, (None,))
        _ensure_value_metadata(ctx, extended)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("unique_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("unique_out")
        result = ctx.builder.Gather(
            extended,
            select_idx,
            _outputs=[desired_name],
            axis=0,
        )
        result.type = ir.TensorType(out_enum)
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", (size,)))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., Any] | None,
        ) -> Callable[..., Any]:
            if orig is None:
                raise RuntimeError("Original jnp.unique not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                ar: ArrayLike,
                return_index: bool = False,
                return_inverse: bool = False,
                return_counts: bool = False,
                axis: int | None = None,
                *,
                equal_nan: bool = True,
                size: int | None = None,
                fill_value: ArrayLike | None = None,
                sorted: bool = True,
            ) -> Any:
                if return_index or return_inverse or return_counts:
                    return orig(
                        ar,
                        return_index=return_index,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                        axis=axis,
                        equal_nan=equal_nan,
                        size=size,
                        fill_value=fill_value,
                        sorted=sorted,
                    )
                if axis is not None:
                    return orig(
                        ar,
                        return_index=False,
                        return_inverse=False,
                        return_counts=False,
                        axis=axis,
                        equal_nan=equal_nan,
                        size=size,
                        fill_value=fill_value,
                        sorted=sorted,
                    )
                if not sorted:
                    raise NotImplementedError(
                        "jnp.unique(sorted=False) is not supported in ONNX lowering"
                    )
                if not equal_nan:
                    raise NotImplementedError(
                        "jnp.unique(equal_nan=False) is not supported in ONNX lowering"
                    )
                if size is None:
                    raise NotImplementedError(
                        "jnp.unique ONNX lowering currently requires a static `size`"
                    )
                if fill_value is None:
                    raise NotImplementedError(
                        "jnp.unique ONNX lowering currently requires `fill_value`"
                    )
                return cls._PRIM.bind(
                    ar,
                    size=int(size),
                    fill_value=_normalize_fill_value(fill_value),
                    sorted=True,
                )

            return _patched

        return [
            AssignSpec("jax.numpy", "unique_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpUniquePlugin._PRIM.def_impl
def _unique_impl(
    ar: ArrayLike,
    *,
    size: int,
    fill_value: bool | int | float,
    sorted: bool = True,
) -> Any:
    return _JNP_UNIQUE_ORIG(
        ar,
        return_index=False,
        return_inverse=False,
        return_counts=False,
        axis=None,
        equal_nan=True,
        size=int(size),
        fill_value=fill_value,
        sorted=sorted,
    )


def _unique_batch_rule(
    batched_args: tuple[Any, ...],
    batch_dims: tuple[object, ...],
    *,
    size: int,
    fill_value: bool | int | float,
    sorted: bool = True,
) -> tuple[Any, object]:
    (x,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = JnpUniquePlugin._PRIM.bind(
            x,
            size=size,
            fill_value=fill_value,
            sorted=sorted,
        )
        return out, batching.not_mapped

    raise NotImplementedError(
        "vmap over jnp.unique is not supported by ONNX lowering yet"
    )


batching.primitive_batchers[JnpUniquePlugin._PRIM] = _unique_batch_rule
