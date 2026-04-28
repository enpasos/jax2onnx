# jax2onnx/converter/output_binding.py

from __future__ import annotations

from typing import Any

import onnx_ir as ir
from jax.extend import core as jcore_ext


def is_drop_var(var: object) -> bool:
    drop_var_type = getattr(jcore_ext, "DropVar", None)
    if isinstance(drop_var_type, type) and isinstance(var, drop_var_type):
        return True
    return type(var).__name__ == "DropVar"


def get_bound_value(ctx: Any, var: object) -> ir.Value | None:
    try:
        value = ctx.builder._var2val.get(var)
    except TypeError:
        return None
    return value if isinstance(value, ir.Value) else None


def _value_is_graph_connected(ctx: Any, value: ir.Value) -> bool:
    if any(existing is value for existing in ctx.builder.inputs):
        return True
    if any(existing is value for existing in ctx.builder.initializers):
        return True
    for node in ctx.builder.nodes:
        if any(output is value for output in node.outputs):
            return True
    return False


def _outvar_needs_binding(ctx: Any, var: object) -> bool:
    value = get_bound_value(ctx, var)
    return value is None or not _value_is_graph_connected(ctx, value)


def _coerce_lowering_result_values(
    result: object, *, primitive_name: str
) -> list[ir.Value] | None:
    if result is None:
        return None
    if isinstance(result, ir.Value):
        return [result]
    if isinstance(result, (list, tuple)):
        values = list(result)
        if all(isinstance(value, ir.Value) for value in values):
            return values
    raise TypeError(
        f"[converter] Primitive '{primitive_name}' returned unsupported lowering "
        f"result {type(result).__name__}; expected ir.Value, a sequence of "
        "ir.Value, or None"
    )


def bind_returned_lowering_values(
    ctx: Any,
    eqn: object,
    result: object,
    *,
    primitive_name: str,
) -> None:
    outvars = list(getattr(eqn, "outvars", ()))
    non_drop_outvars = [
        (index, var) for index, var in enumerate(outvars) if not is_drop_var(var)
    ]
    unbound_outvars = [
        (index, var)
        for index, var in non_drop_outvars
        if _outvar_needs_binding(ctx, var)
    ]
    if not unbound_outvars:
        return

    returned_values = _coerce_lowering_result_values(
        result, primitive_name=primitive_name
    )
    if returned_values is None:
        return

    if len(returned_values) == len(non_drop_outvars):
        indexed_values = zip(non_drop_outvars, returned_values)
        for (_, var), value in indexed_values:
            if _outvar_needs_binding(ctx, var):
                ctx.bind_value_for_var(var, value)
        return

    if len(returned_values) == len(unbound_outvars):
        for (_, var), value in zip(unbound_outvars, returned_values):
            ctx.bind_value_for_var(var, value)
        return

    raise RuntimeError(
        f"[converter] Primitive '{primitive_name}' returned {len(returned_values)} "
        f"value(s), but {len(unbound_outvars)} non-drop outvar(s) remain unbound"
    )


def assert_eqn_outputs_bound(
    ctx: Any,
    eqn: object,
    *,
    primitive_name: str,
    eqn_index: int,
) -> None:
    for out_index, outvar in enumerate(getattr(eqn, "outvars", ())):
        if is_drop_var(outvar):
            continue

        bound_value = get_bound_value(ctx, outvar)
        if bound_value is None:
            raise RuntimeError(
                f"[converter] Primitive '{primitive_name}' at equation "
                f"{eqn_index} did not bind output {out_index}"
            )
        if not _value_is_graph_connected(ctx, bound_value):
            value_name = bound_value.name or "<unnamed>"
            raise RuntimeError(
                f"[converter] Primitive '{primitive_name}' at equation {eqn_index} "
                f"bound output {out_index} to disconnected value '{value_name}'"
            )
