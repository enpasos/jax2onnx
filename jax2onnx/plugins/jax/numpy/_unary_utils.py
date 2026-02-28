# jax2onnx/plugins/jax/numpy/_unary_utils.py

from __future__ import annotations

import jax
from jax import core
from jax.extend.core import Primitive
from jax.interpreters import batching
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata
from jax2onnx.plugins.jax.numpy._common import get_orig_impl


def abstract_eval_via_orig_unary(
    prim: Primitive,
    func_name: str,
    x: core.AbstractValue,
) -> core.ShapedArray:
    """Mirror JAX dtype/shape semantics by delegating to the original jnp impl."""
    shape = tuple(getattr(x, "shape", ()))
    dtype = np.dtype(getattr(x, "dtype", np.float32))
    shape_dtype = jax.ShapeDtypeStruct(shape, dtype)
    orig = get_orig_impl(prim, func_name)
    out = jax.eval_shape(lambda v: orig(v), shape_dtype)
    out_shape = tuple(getattr(out, "shape", ()))
    out_dtype = np.dtype(getattr(out, "dtype", dtype))
    return core.ShapedArray(out_shape, out_dtype)


def lower_unary_elementwise_with_optional_cast(
    ctx: LoweringContextProtocol,
    eqn: core.JaxprEqn,
    *,
    op_name: str,
    input_hint: str,
    output_hint: str,
    cast_input_to_output_dtype: bool = False,
    identity_for_integral_input: bool = False,
) -> None:
    (x_var,) = eqn.invars
    (y_var,) = eqn.outvars

    x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name(input_hint))
    out_spec = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name(output_hint))

    in_dtype = np.dtype(getattr(getattr(x_var, "aval", None), "dtype", np.float32))
    out_dtype = np.dtype(getattr(getattr(y_var, "aval", None), "dtype", in_dtype))

    op_input = x_val
    if identity_for_integral_input and (
        np.issubdtype(in_dtype, np.integer) or in_dtype == np.bool_
    ):
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(output_hint)
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name(output_hint)

        result = ctx.builder.Identity(op_input, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(y_var, result)
        return

    if cast_input_to_output_dtype:
        if in_dtype != out_dtype:
            cast_dtype = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
            cast_val = ctx.builder.Cast(
                x_val,
                _outputs=[ctx.fresh_name(f"{output_hint}_cast")],
                to=int(cast_dtype.value),
            )
            cast_val.type = ir.TensorType(cast_dtype)
            cast_val.shape = getattr(x_val, "shape", None)
            _ensure_value_metadata(ctx, cast_val)
            op_input = cast_val

    desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(output_hint)
    producer = getattr(out_spec, "producer", None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name(output_hint)

    builder_op = getattr(ctx.builder, op_name, None)
    if builder_op is None:
        raise AttributeError(f"IR builder missing op '{op_name}'")

    result = builder_op(op_input, _outputs=[desired_name])
    if getattr(out_spec, "type", None) is not None:
        result.type = out_spec.type
    if getattr(out_spec, "shape", None) is not None:
        result.shape = out_spec.shape
    ctx.bind_value_for_var(y_var, result)


def cast_input_to_output_dtype(
    ctx: LoweringContextProtocol,
    x_var: core.AbstractValue,
    y_var: core.AbstractValue,
    x_val: ir.Value,
    *,
    output_hint: str,
) -> ir.Value:
    """Cast unary input to output dtype when JAX promotion changed the dtype."""
    in_dtype = np.dtype(getattr(getattr(x_var, "aval", None), "dtype", np.float32))
    out_dtype = np.dtype(getattr(getattr(y_var, "aval", None), "dtype", in_dtype))
    if in_dtype == out_dtype:
        return x_val

    cast_dtype = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
    cast_val = ctx.builder.Cast(
        x_val,
        _outputs=[ctx.fresh_name(f"{output_hint}_cast")],
        to=int(cast_dtype.value),
    )
    cast_val.type = ir.TensorType(cast_dtype)
    cast_val.shape = getattr(x_val, "shape", None)
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


def register_unary_elementwise_batch_rule(prim: Primitive) -> None:
    """Attach a default batching rule for single-input elementwise primitives."""

    def _batch_rule(batched_args, batch_dims, **params):
        (x,) = batched_args
        (bdim,) = batch_dims
        out = prim.bind(x, **params)
        return out, bdim

    batching.primitive_batchers[prim] = _batch_rule
