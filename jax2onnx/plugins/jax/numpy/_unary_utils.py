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

from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._utils import cast_value_to_dtype
from jax2onnx.plugins.jax.numpy._common import abstract_eval_via_jax, get_orig_impl


def abstract_eval_via_orig_unary(
    prim: Primitive,
    func_name: str,
    x: AbstractValue,
) -> ShapedArray:
    """Mirror JAX dtype/shape semantics by delegating to the original jnp impl."""
    orig = get_orig_impl(prim, func_name)
    return abstract_eval_via_jax(orig, x)


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
    in_dtype: np.dtype[Any] = np.dtype(x_var.aval.dtype)

    op_input = x_val
    if identity_for_integral_input and (
        np.issubdtype(in_dtype, np.integer) or in_dtype == np.bool_
    ):
        op_name = "Identity"
    elif cast_input_to_output_dtype:
        op_input = cast_value_to_dtype(
            ctx,
            x_val,
            numpy_dtype_to_ir(y_var.aval.dtype),
            name_hint=f"{output_hint}_cast",
        )

    result = ctx.builder.op(op_name, [op_input], output=out_spec)
    ctx.bind_value_for_var(y_var, result)


def cast_input_to_output_dtype(
    ctx: NativeLoweringContext,
    x_var: Var | Literal,
    y_var: Var | Literal,
    x_val: ir.Value,
    *,
    output_hint: str,
) -> ir.Value:
    """Cast unary input to output dtype when JAX promotion changed the dtype."""
    return cast_value_to_dtype(
        ctx,
        x_val,
        numpy_dtype_to_ir(y_var.aval.dtype),
        name_hint=f"{output_hint}_cast",
    )


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
