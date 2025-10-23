# jax2onnx/plugins/jax/nn/_builder_utils.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext  # type: ignore[name-defined]


def lower_unary_elementwise(
    ctx: "IRContext",
    eqn,
    *,
    op_name: str,
    input_hint: str,
    output_hint: str,
    attrs: Mapping[str, Any] | None = None,
) -> None:
    """Materialize a single-output unary op via the builder and bind it."""
    (x_var,) = eqn.invars
    (y_var,) = eqn.outvars

    x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name(input_hint))
    out_spec = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name(output_hint))

    desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(output_hint)
    producer = getattr(out_spec, "producer", None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name(output_hint)

    builder_op = getattr(ctx.builder, op_name, None)
    if builder_op is None:
        raise AttributeError(f"IR builder missing op '{op_name}'")

    attrs_dict = dict(attrs or {})
    result = builder_op(x_val, _outputs=[desired_name], **attrs_dict)

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

    def _batch_rule(batched_args, batch_dims, **params):
        (x,) = batched_args
        (bd,) = batch_dims
        out = prim.bind(x, **params)
        return out, bd

    batching.primitive_batchers[prim] = _batch_rule
