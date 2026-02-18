# jax2onnx/plugins/jax/numpy/cumprod.py

from __future__ import annotations

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


_CUMPROD_PRIM: Final = make_jnp_primitive("jax.numpy.cumprod")
_JAX_CUMPROD_ORIG: Final = jnp.cumprod


@register_primitive(
    jaxpr_primitive=_CUMPROD_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cumprod.html",
    onnx=[
        {
            "component": "CumProd",
            "doc": "https://onnx.ai/onnx/operators/onnx__CumProd.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="cumprod",
    testcases=[
        {
            "testcase": "jnp_cumprod_axis1",
            "callable": lambda x: jnp.cumprod(x, axis=1),
            "input_shapes": [(2, 3, 4)],
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "CumProd:2x3x4",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_cumprod_axis_none_flatten",
            "callable": lambda x: jnp.cumprod(x, axis=None),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(6,)],
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "Reshape:6 -> CumProd:6",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_cumprod_dtype_cast",
            "callable": lambda x: jnp.cumprod(x, axis=1, dtype=jnp.float32),
            "input_shapes": [(2, 3)],
            "input_dtypes": [np.int32],
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "Cast:2x3 -> CumProd:2x3",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpCumProdPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _CUMPROD_PRIM
    _FUNC_NAME: ClassVar[str] = "cumprod"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        axis: int | None = None,
        dtype: np.dtype[Any] | type | None = None,
    ) -> core.ShapedArray:
        out_dtype = np.dtype(dtype) if dtype is not None else np.dtype(x.dtype)
        out_shape = tuple(x.shape)
        if axis is None:
            if all(isinstance(d, (int, np.integer)) for d in out_shape):
                size = int(np.prod([int(d) for d in out_shape], dtype=np.int64))
                out_shape = (size,)
            else:
                out_shape = (None,)
        return core.ShapedArray(out_shape, out_dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (operand_var,) = eqn.invars
        (out_var,) = eqn.outvars

        params = getattr(eqn, "params", {})
        axis_param = params.get("axis", None)
        req_dtype = params.get("dtype", None)

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("jnp_cumprod_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_cumprod_out")
        )

        operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
        rank = len(operand_shape)
        out_shape = tuple(getattr(out_var.aval, "shape", ()))

        input_for_cumprod = operand_val
        target_dtype = (
            np.dtype(req_dtype)
            if req_dtype is not None
            else np.dtype(getattr(operand_var.aval, "dtype", np.float32))
        )
        operand_dtype = np.dtype(getattr(operand_var.aval, "dtype", target_dtype))
        target_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)

        if operand_dtype != target_dtype:
            cast_val = ctx.builder.Cast(
                operand_val,
                _outputs=[ctx.fresh_name("jnp_cumprod_cast")],
                to=int(target_enum.value),
            )
            cast_val.type = ir.TensorType(target_enum)
            _stamp_type_and_shape(cast_val, operand_shape)
            _ensure_value_metadata(ctx, cast_val)
            input_for_cumprod = cast_val

        if axis_param is None:
            flatten_shape = _const_i64(
                ctx,
                np.asarray([-1], dtype=np.int64),
                "cumprod_flatten_shape",
            )
            flattened = ctx.builder.Reshape(
                input_for_cumprod,
                flatten_shape,
                _outputs=[ctx.fresh_name("jnp_cumprod_flatten")],
            )
            flattened.type = ir.TensorType(target_enum)
            _stamp_type_and_shape(flattened, out_shape)
            _ensure_value_metadata(ctx, flattened)
            input_for_cumprod = flattened
            axis = 0
        else:
            axis = int(axis_param)
            if rank and axis < 0:
                axis = axis % rank

        axis_val = _const_i64(ctx, np.asarray(axis, dtype=np.int64), "cumprod_axis")

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("CumProd")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("CumProd")

        result = ctx.builder.CumProd(
            input_for_cumprod,
            axis_val,
            _outputs=[desired_name],
            exclusive=0,
            reverse=0,
        )
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
                raise RuntimeError("Original jnp.cumprod not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                axis: int | None = None,
                dtype: np.dtype[Any] | type | None = None,
                out: Any | None = None,
            ) -> jax.Array:
                if out is not None:
                    raise NotImplementedError(
                        "jnp.cumprod with 'out' is not supported for ONNX export"
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


@JnpCumProdPlugin._PRIM.def_impl
def _cumprod_impl(
    a: ArrayLike,
    *,
    axis: int | None = None,
    dtype: np.dtype[Any] | type | None = None,
) -> jax.Array:
    try:
        orig = get_orig_impl(JnpCumProdPlugin._PRIM, JnpCumProdPlugin._FUNC_NAME)
    except RuntimeError:
        orig = jnp.cumprod
    return orig(jnp.asarray(a), axis=axis, dtype=dtype)


BatchDim = int | type(batching.not_mapped)


def _cumprod_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    axis: int | None = None,
    dtype: np.dtype[Any] | type | None = None,
) -> tuple[jax.Array, BatchDim]:
    (operand,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = JnpCumProdPlugin._PRIM.bind(
            operand,
            axis=axis,
            dtype=dtype,
        )
        return out, batching.not_mapped

    axis_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, axis_size)

    if axis is None:
        axis_body = None
    else:
        slice_rank = operand.ndim - 1
        axis_int = int(axis)
        axis_norm = axis_int % slice_rank if slice_rank and axis_int < 0 else axis_int
        axis_body = axis_norm

    out = jax.vmap(
        lambda x: _JAX_CUMPROD_ORIG(x, axis=axis_body, dtype=dtype),
        in_axes=0,
    )(operand)
    return out, 0


batching.primitive_batchers[JnpCumProdPlugin._PRIM] = _cumprod_batch_rule
