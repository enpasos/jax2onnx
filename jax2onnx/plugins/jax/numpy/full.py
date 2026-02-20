# jax2onnx/plugins/jax/numpy/full.py

from __future__ import annotations

from collections.abc import Sequence as _Seq
from typing import Any, Callable, ClassVar, Final, cast

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


_FULL_PRIM: Final = make_jnp_primitive("jax.numpy.full")


def _normalize_shape(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, _Seq) and not isinstance(shape, (str, bytes)):
        dims = tuple(int(d) for d in shape)
    else:
        dims = (int(shape),)
    if any(d < 0 for d in dims):
        raise ValueError(f"negative dimensions are not allowed: {dims}")
    return dims


def _const_numpy(value: ir.Value) -> np.ndarray[Any, np.dtype[Any]] | None:
    const = getattr(value, "const_value", None)
    if const is None:
        return None
    try:
        arr = np.asarray(const)
    except Exception:
        return None
    return cast(np.ndarray[Any, np.dtype[Any]], arr)


def _np_dtype_from_ir(dtype: ir.DataType) -> np.dtype[Any]:
    mapping: dict[ir.DataType, np.dtype[Any]] = {
        ir.DataType.BOOL: np.dtype(np.bool_),
        ir.DataType.INT8: np.dtype(np.int8),
        ir.DataType.INT16: np.dtype(np.int16),
        ir.DataType.INT32: np.dtype(np.int32),
        ir.DataType.INT64: np.dtype(np.int64),
        ir.DataType.UINT8: np.dtype(np.uint8),
        ir.DataType.FLOAT16: np.dtype(np.float16),
        ir.DataType.FLOAT: np.dtype(np.float32),
        ir.DataType.DOUBLE: np.dtype(np.float64),
    }
    return mapping.get(dtype, np.dtype(np.float32))


@register_primitive(
    jaxpr_primitive=_FULL_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.full.html",
    onnx=[
        {
            "component": "ConstantOfShape",
            "doc": "https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="full",
    testcases=[
        {
            "testcase": "jnp_full_const_scalar",
            "callable": lambda: jnp.full((2, 3), 1.5, dtype=jnp.float32),
            "input_values": [],
            "post_check_onnx_graph": EG(
                ["ConstantOfShape:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_full_expand_scalar_input",
            "callable": lambda x: jnp.full((2, 3), x),
            "input_shapes": [()],
            "post_check_onnx_graph": EG(
                ["Expand:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_full_expand_row_broadcast",
            "callable": lambda x: jnp.full((2, 3), x),
            "input_shapes": [(1, 3)],
            "post_check_onnx_graph": EG(
                ["Expand:2x3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpFullPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _FULL_PRIM
    _FUNC_NAME: ClassVar[str] = "full"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        fill_value: core.AbstractValue,
        *,
        shape: tuple[int, ...],
    ) -> core.ShapedArray:
        dims = _normalize_shape(shape)
        return core.ShapedArray(dims, fill_value.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (fill_var,) = eqn.invars
        (out_var,) = eqn.outvars

        params = dict(getattr(eqn, "params", {}) or {})
        shape = _normalize_shape(params.get("shape"))

        fill_val = ctx.get_value_for_var(
            fill_var, name_hint=ctx.fresh_name("full_fill")
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("full_out"))

        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", shape))
        out_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(out_var, "aval", None), "dtype", np.float32)
        )
        enable_double = bool(ctx.builder.enable_double_precision)
        out_enum = _dtype_to_ir(out_dtype, enable_double)
        out_np_dtype = _np_dtype_from_ir(out_enum)

        fill_dtype = np.dtype(
            getattr(getattr(fill_var, "aval", None), "dtype", out_dtype)
        )
        fill_enum = _dtype_to_ir(fill_dtype, enable_double)
        fill_cast = fill_val
        if fill_enum != out_enum:
            fill_cast = ctx.builder.Cast(
                fill_val,
                _outputs=[ctx.fresh_name("full_fill_cast")],
                to=int(out_enum.value),
            )
            fill_cast.type = ir.TensorType(out_enum)
            fill_shape = tuple(getattr(getattr(fill_var, "aval", None), "shape", ()))
            _stamp_type_and_shape(fill_cast, fill_shape)
            _ensure_value_metadata(ctx, fill_cast)

        shape_arr = np.asarray(shape, dtype=np.int64)
        shape_tensor = _const_i64(ctx, shape_arr, "full_shape")
        _stamp_type_and_shape(shape_tensor, (len(shape),))
        _ensure_value_metadata(ctx, shape_tensor)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("full_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("full_out")

        fill_const = _const_numpy(fill_cast)
        if fill_const is not None and fill_const.size == 1:
            scalar = np.asarray([fill_const.reshape(())], dtype=out_np_dtype)
            result = ctx.builder.ConstantOfShape(
                shape_tensor,
                value=ir.tensor(scalar),
                _outputs=[desired_name],
            )
        else:
            result = ctx.builder.Expand(
                fill_cast,
                shape_tensor,
                _outputs=[desired_name],
            )

        result.type = ir.TensorType(out_enum)
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
                raise RuntimeError("Original jnp.full not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                shape: Any,
                fill_value: ArrayLike,
                dtype: Any | None = None,
                *,
                device: Any | None = None,
            ) -> Any:
                if device is not None:
                    return orig(shape, fill_value, dtype=dtype, device=device)
                norm_shape = _normalize_shape(shape)
                fill_arr = (
                    jnp.asarray(fill_value)
                    if dtype is None
                    else jnp.asarray(fill_value, dtype=dtype)
                )
                return cls._PRIM.bind(fill_arr, shape=norm_shape)

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


@JnpFullPlugin._PRIM.def_impl
def _full_impl(
    fill_value: ArrayLike,
    *,
    shape: tuple[int, ...],
) -> Any:
    orig = get_orig_impl(JnpFullPlugin._PRIM, JnpFullPlugin._FUNC_NAME)
    return orig(shape, fill_value)


JnpFullPlugin._PRIM.def_abstract_eval(JnpFullPlugin.abstract_eval)


def _full_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[Any, ...],
    *,
    shape: tuple[int, ...],
) -> tuple[jax.Array, Any]:
    (fill,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = JnpFullPlugin._PRIM.bind(fill, shape=shape)
        return out, batching.not_mapped

    raise NotImplementedError(
        "vmap over jnp.full is not supported in ONNX lowering yet"
    )


batching.primitive_batchers[JnpFullPlugin._PRIM] = _full_batch_rule
