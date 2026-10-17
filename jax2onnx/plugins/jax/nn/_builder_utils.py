# jax2onnx/plugins/jax/nn/_builder_utils.py

from __future__ import annotations

from typing import Any, Mapping, cast

import onnx_ir as ir
from jax2onnx._compat.jax import (
    Primitive,
    batching,
)

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins._precision_utils import promote_float32_operand


def lower_unary_elementwise(
    ctx: LoweringContextProtocol,
    eqn: Any,
    *,
    op_name: str,
    input_hint: str,
    output_hint: str,
    attrs: Mapping[str, Any] | None = None,
) -> None:
    """Materialize a single-output unary op via the builder and bind it."""
    (x_var,) = eqn.invars
    (y_var,) = eqn.outvars

    ctx.fresh_name(input_hint)
    x_val = ctx.require_value_for_var(x_var)
    out_spec = ctx.allocate_value_for_var(y_var, name_hint=ctx.fresh_name(output_hint))
    x_val = promote_float32_operand(
        ctx,
        x_val,
        target_dtype=out_spec.dtype,
        name_hint=f"{output_hint}_input_double",
    )

    desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(output_hint)
    producer = getattr(out_spec, "producer", None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name(output_hint)

    builder_op = getattr(ctx.builder, op_name, None)
    if builder_op is None:
        raise AttributeError(f"IR builder missing op '{op_name}'")

    attrs_dict = dict(attrs or {})
    result = cast(ir.Value, builder_op(x_val, _outputs=[desired_name], **attrs_dict))

    if getattr(out_spec, "type", None) is not None:
        result.type = out_spec.type
    else:
        result.type = getattr(x_val, "type", None)

    if getattr(out_spec, "shape", None) is not None:
        result.shape = out_spec.shape
    else:
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        _stamp_type_and_shape(result, x_shape)

    ctx.bind_value_for_var(y_var, result)


def register_unary_elementwise_batch_rule(prim: Primitive) -> None:
    """Attach a default batching rule for single-input elementwise primitives."""

    def _batch_rule(
        batched_args: tuple[Any, ...], batch_dims: tuple[Any, ...], **params: Any
    ) -> tuple[Any, Any]:
        (x,) = batched_args
        (bd,) = batch_dims
        out = prim.bind(x, **params)
        return out, bd

    batching.primitive_batchers[prim] = _batch_rule
