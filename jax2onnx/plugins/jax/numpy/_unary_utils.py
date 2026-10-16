# jax2onnx/plugins/jax/numpy/_unary_utils.py

from __future__ import annotations

from typing import Any, TypeAlias

import jax
from jax2onnx._compat.jax import (
    AbstractValue,
    JaxprEqn,
    Literal,
    Primitive,
    ShapedArray,
    Var,
    batching,
)
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata
from jax2onnx.plugins.jax.numpy._common import get_orig_impl


def abstract_eval_via_orig_unary(
    prim: Primitive,
    func_name: str,
    x: AbstractValue,
) -> ShapedArray:
    """Mirror JAX dtype/shape semantics by delegating to the original jnp impl."""
    shape = tuple(getattr(x, "shape", ()))
    dtype: np.dtype[Any] = np.dtype(getattr(x, "dtype", np.float32))
    shape_dtype = jax.ShapeDtypeStruct(shape, dtype)
    orig = get_orig_impl(prim, func_name)
    out = jax.eval_shape(lambda v: orig(v), shape_dtype)
    out_shape = tuple(getattr(out, "shape", ()))
    out_dtype = np.dtype(getattr(out, "dtype", dtype))
    return ShapedArray(out_shape, out_dtype)


def lower_unary_elementwise_with_optional_cast(
    ctx: NativeLoweringContext,
    eqn: JaxprEqn,
    *,
    op_name: str,
    input_hint: str,
    output_hint: str,
    cast_input_to_output_dtype: bool = False,
    identity_for_integral_input: bool = False,
) -> None:
    (x_var,) = eqn.invars
    (y_var,) = eqn.outvars

    # Preserve the established naming sequence although input lookup allocates no name.
    ctx.fresh_name(input_hint)
    x_val = ctx.require_value_for_var(x_var)
    out_spec = ctx.allocate_value_for_var(y_var, name_hint=ctx.fresh_name(output_hint))
    in_dtype: np.dtype[Any] = _var_dtype(x_var, np.dtype(np.float32))
    out_dtype: np.dtype[Any] = _var_dtype(y_var, in_dtype)

    op_input = x_val
    if identity_for_integral_input and (
        np.issubdtype(in_dtype, np.integer) or in_dtype == np.bool_
    ):
        op_name = "Identity"
    elif cast_input_to_output_dtype and in_dtype != out_dtype:
        op_input = _cast_value(ctx, x_val, out_dtype, output_hint=output_hint)

    result = ctx.builder.op(op_name, [op_input], output=out_spec)
    ctx.bind_value_for_var(y_var, result)


def _var_dtype(var: Var | Literal, default: np.dtype[Any]) -> np.dtype[Any]:
    aval = var.aval
    return np.dtype(aval.dtype) if isinstance(aval, ShapedArray) else default


def _cast_value(
    ctx: NativeLoweringContext,
    value: ir.Value,
    dtype: np.dtype[Any],
    *,
    output_hint: str,
) -> ir.Value:
    cast_dtype = _dtype_to_ir(dtype, ctx.builder.enable_double_precision)
    output = ir.Value(
        name=ctx.fresh_name(f"{output_hint}_cast"),
        type=ir.TensorType(cast_dtype),
        shape=value.shape,
    )
    cast_value = ctx.builder.op(
        "Cast", [value], {"to": int(cast_dtype.value)}, output=output
    )
    _ensure_value_metadata(ctx, cast_value)
    return cast_value


def cast_input_to_output_dtype(
    ctx: NativeLoweringContext,
    x_var: Var | Literal,
    y_var: Var | Literal,
    x_val: ir.Value,
    *,
    output_hint: str,
) -> ir.Value:
    """Cast unary input to output dtype when JAX promotion changed the dtype."""
    in_dtype: np.dtype[Any] = _var_dtype(x_var, np.dtype(np.float32))
    out_dtype: np.dtype[Any] = _var_dtype(y_var, in_dtype)
    if in_dtype == out_dtype:
        return x_val

    return _cast_value(ctx, x_val, out_dtype, output_hint=output_hint)


def register_unary_elementwise_batch_rule(prim: Primitive) -> None:
    """Attach a default batching rule for single-input elementwise primitives."""

    BatchDim: TypeAlias = int | None

    def _batch_rule(
        batched_args: tuple[jax.Array, ...],
        batch_dims: tuple[BatchDim, ...],
        **params: object,
    ) -> tuple[jax.Array, BatchDim]:
        (x,) = batched_args
        (bdim,) = batch_dims
        out = prim.bind(x, **params)
        return out, bdim

    batching.primitive_batchers[prim] = _batch_rule
