# jax2onnx/plugins/jax/numpy/cumsum.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Final, Optional

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins._patching import AssignSpec
from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins.jax.numpy._common import make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_CUMSUM_PRIM: Final = make_jnp_primitive("jax.numpy.cumsum")


@register_primitive(
    jaxpr_primitive=_CUMSUM_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cumsum.html",
    onnx=[
        {
            "component": "CumSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__CumSum.html",
        }
    ],
    since="v0.8.0",
    context="primitives.jnp",
    component="cumsum",
    testcases=[
        {
            "testcase": "jnp_cumsum_axis1",
            "callable": lambda x: jnp.cumsum(x, axis=1),
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "jnp_cumsum_reverse_dtype",
            "callable": lambda x: jnp.cumsum(
                x, axis=-1, reverse=True, dtype=jnp.float64
            ),
            "input_shapes": [(1, 5)],
            "enable_double_precision": True,
        },
        {
            "testcase": "cumsum_axis2_i32",
            "callable": lambda x: jnp.cumsum(x, axis=2),
            "input_shapes": [(2, 3, 4)],
            "input_dtypes": [np.int32],
        },
        {
            "testcase": "cumsum_axis2_reverse_i32",
            "callable": lambda x: jnp.cumsum(x, axis=2, reverse=True),
            "input_shapes": [(2, 3, 4)],
            "input_dtypes": [np.int32],
        },
    ],
)
class JnpCumSumPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _CUMSUM_PRIM
    _FUNC_NAME: ClassVar[str] = "cumsum"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x, *, axis: int | None = None, reverse: bool = False, dtype=None, **_
    ):
        out_shape = x.shape
        out_dtype = np.dtype(dtype) if dtype is not None else x.dtype
        return jax.core.ShapedArray(out_shape, out_dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        (operand_var,) = eqn.invars
        (out_var,) = eqn.outvars

        params = getattr(eqn, "params", {})
        axis_param = params.get("axis", 0)
        reverse = bool(params.get("reverse", False))
        req_dtype = params.get("dtype", None)
        exclusive = bool(params.get("exclusive", False))

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("jnp_cumsum_in")
        )
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_cumsum_out")
        )

        operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
        rank = len(operand_shape)

        if axis_param is None:
            axis = rank - 1 if rank else 0
        else:
            axis = int(axis_param)
            if axis < 0 and rank:
                axis = axis % rank

        input_for_cumsum = operand_val
        target_dtype = (
            np.dtype(req_dtype)
            if req_dtype is not None
            else np.dtype(getattr(operand_var.aval, "dtype", np.float32))
        )
        operand_dtype = np.dtype(getattr(operand_var.aval, "dtype", target_dtype))

        if operand_dtype != target_dtype:
            target_enum = _dtype_to_ir(
                target_dtype, ctx.builder.enable_double_precision
            )
            cast_val = ctx.builder.Cast(
                operand_val,
                _outputs=[ctx.fresh_name("jnp_cumsum_cast")],
                to=int(target_enum.value),
            )
            cast_val.type = ir.TensorType(target_enum)
            _stamp_type_and_shape(cast_val, operand_shape)
            _ensure_value_info(ctx, cast_val)
            input_for_cumsum = cast_val
        else:
            target_enum = _dtype_to_ir(
                operand_dtype, ctx.builder.enable_double_precision
            )

        axis_val = _const_i64(ctx, np.asarray(axis, dtype=np.int64), "cumsum_axis")
        _stamp_type_and_shape(axis_val, ())
        _ensure_value_info(ctx, axis_val)

        desired_name = getattr(out_val, "name", None) or ctx.fresh_name("CumSum")
        producer = getattr(out_val, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("CumSum")

        result = ctx.builder.CumSum(
            input_for_cumsum,
            axis_val,
            _outputs=[desired_name],
            exclusive=int(bool(exclusive)),
            reverse=int(bool(reverse)),
        )
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        result.type = ir.TensorType(target_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_info(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec(
                "jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True
            ),
        ]


@JnpCumSumPlugin._PRIM.def_impl
def _cumsum_impl(
    x,
    *rest,
    axis=None,
    dtype=None,
    reverse=False,
    precision=None,
    exclusive=False,
    out=None,
    method=None,
    **kwargs,
):
    if rest:
        raise TypeError("jnp.cumsum expects a single positional argument")
    if out is not None:
        raise NotImplementedError("jnp.cumsum with 'out' is not supported")
    if method is not None:
        raise NotImplementedError("jnp.cumsum 'method' argument is not supported")
    if kwargs:
        raise TypeError(
            f"Unsupported keyword arguments for jnp.cumsum: {tuple(kwargs.keys())}"
        )

    arr = jnp.asarray(x)
    target_dtype = np.dtype(dtype) if dtype is not None else arr.dtype
    if arr.dtype != target_dtype:
        arr = arr.astype(target_dtype)

    cumsum_kwargs = {}
    if precision is not None:
        cumsum_kwargs["precision"] = precision
    if exclusive:
        cumsum_kwargs["exclusive"] = True

    rank = arr.ndim
    if axis is None:
        orig_shape = arr.shape
        flattened = arr.reshape((-1,)) if arr.size else arr.reshape((0,))
        if flattened.ndim == 0:
            flattened = flattened.reshape((1,))
        result = jax.lax.cumsum(flattened, axis=0, reverse=reverse, **cumsum_kwargs)
        return result.reshape(orig_shape)

    if rank == 0:
        arr_1d = arr.reshape((1,))
        result = jax.lax.cumsum(arr_1d, axis=0, reverse=reverse, **cumsum_kwargs)
        return result.reshape(())

    axis_index = int(axis)
    if axis_index < 0:
        axis_index = axis_index % rank

    result = jax.lax.cumsum(arr, axis=axis_index, reverse=reverse, **cumsum_kwargs)
    return result


_ORIGINAL_JNP_CUMSUM: Final[Optional[Callable[..., Any]]] = getattr(jnp, "cumsum", None)


def _runtime_cumsum(
    x,
    *rest,
    axis=None,
    dtype=None,
    reverse=False,
    precision=None,
    exclusive=False,
    out=None,
    method=None,
    **kwargs,
):
    if rest:
        raise TypeError("jnp.cumsum expects a single positional argument")
    if out is not None:
        raise NotImplementedError("jnp.cumsum with 'out' is not supported")
    if method is not None:
        raise NotImplementedError("jnp.cumsum 'method' argument is not supported")
    if kwargs:
        raise TypeError(
            f"Unsupported keyword arguments for jnp.cumsum: {tuple(kwargs.keys())}"
        )

    if (
        not reverse
        and not exclusive
        and precision is None
        and _ORIGINAL_JNP_CUMSUM is not None
    ):
        return _ORIGINAL_JNP_CUMSUM(x, axis=axis, dtype=dtype, out=out)

    return JnpCumSumPlugin._PRIM.bind(
        x,
        axis=axis,
        dtype=dtype,
        reverse=reverse,
        precision=precision,
        exclusive=exclusive,
    )


if getattr(jnp, "cumsum", None) is not _runtime_cumsum:
    setattr(jnp, "cumsum", _runtime_cumsum)
    setattr(jnp, "cumsum_p", JnpCumSumPlugin._PRIM)


JnpCumSumPlugin.ensure_abstract_eval_bound()
