# jax2onnx/plugins/jax/lax/_opset_utils.py

"""Opset-sensitive IR builders shared by primitive lowerings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, cast

import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import const_value_to_numpy
from jax2onnx.plugins.jax.lax._index_utils import _const_i64


_REDUCTION_AXES_INPUT_SINCE: Final[dict[str, int]] = {
    "ReduceL1": 18,
    "ReduceL2": 18,
    "ReduceLogSum": 18,
    "ReduceLogSumExp": 18,
    "ReduceSum": 13,
    "ReduceMean": 18,
    "ReduceMax": 18,
    "ReduceMin": 18,
    "ReduceProd": 18,
    "ReduceSumSquare": 18,
}


def builder_reduce_with_axes(
    ctx: LoweringContextProtocol,
    value: ir.Value,
    *,
    op_type: str,
    axes: Sequence[int] | None,
    axes_input: ir.Value | None = None,
    keepdims: int,
    name_hint: str,
    output_name: str | None = None,
) -> ir.Value:
    """Build a reduction using the axes representation required by the opset."""

    try:
        axes_input_since = _REDUCTION_AXES_INPUT_SINCE[op_type]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported opset-sensitive reduction: {op_type}") from exc

    builder: Any = ctx.builder
    opset = int(getattr(builder, "opset", 21))
    resolved_output_name = (
        output_name if output_name is not None else ctx.fresh_name(name_hint)
    )
    if axes is None:
        if axes_input is not None:
            raise ValueError("axes_input requires explicit axes")
        return cast(
            ir.Value,
            getattr(builder, op_type)(
                value,
                keepdims=int(keepdims),
                _outputs=[resolved_output_name],
            ),
        )

    if opset < axes_input_since:
        if axes_input is not None:
            raise ValueError(
                f"{op_type} uses an axes attribute before opset {axes_input_since}"
            )
        output = ir.Value(name=resolved_output_name)
        builder.add_node(
            op_type=op_type,
            inputs=[value],
            outputs=[output],
            attributes=[
                ir.Attr(
                    "axes",
                    ir.AttributeType.INTS,
                    tuple(int(axis) for axis in axes),
                ),
                ir.Attr("keepdims", ir.AttributeType.INT, int(keepdims)),
            ],
        )
        return output

    axes_value = (
        axes_input
        if axes_input is not None
        else _const_i64(ctx, list(axes), f"{name_hint}_axes")
    )
    return cast(
        ir.Value,
        getattr(builder, op_type)(
            value,
            axes_value,
            keepdims=int(keepdims),
            _outputs=[resolved_output_name],
        ),
    )


def reduction_axes_from_node(node: Any) -> tuple[int, ...] | None:
    """Read static reduction axes from either an input or a legacy attribute."""

    inputs = tuple(getattr(node, "inputs", ()))
    if len(inputs) > 1:
        axes_array = const_value_to_numpy(inputs[1])
        if axes_array is None:
            raise ValueError("Reduction axes input must be a constant")
        return tuple(int(axis) for axis in axes_array.reshape(-1))

    attributes = getattr(node, "attributes", None)
    axes_attr = None
    if attributes is not None:
        get_attr = getattr(attributes, "get", None)
        if callable(get_attr):
            axes_attr = get_attr("axes")
    if axes_attr is None:
        return None

    axes_value = getattr(axes_attr, "value", axes_attr)
    return tuple(int(axis) for axis in axes_value)
