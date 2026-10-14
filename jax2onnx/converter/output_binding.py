# jax2onnx/converter/output_binding.py

from __future__ import annotations

import onnx_ir as ir

from jax2onnx._compat.jax import DropVar, JaxprEqn

from .typing_support import LoweringContextProtocol


def is_drop_var(var: object) -> bool:
    return isinstance(var, DropVar)


def get_bound_value(ctx: LoweringContextProtocol, var: object) -> ir.Value | None:
    """Compatibility name for the context's non-allocating lookup."""
    return ctx.lookup_value_for_var(var)


def assert_eqn_inputs_bound(
    ctx: LoweringContextProtocol,
    eqn: JaxprEqn,
    *,
    primitive_name: str,
    eqn_index: int,
) -> None:
    """Verify that plugin input vars are already bound before lowering starts."""
    for in_index, invar in enumerate(eqn.invars):
        if is_drop_var(invar):
            continue
        try:
            ctx.require_value_for_var(invar)
        except KeyError as exc:
            raise RuntimeError(
                f"[converter] Primitive '{primitive_name}' at equation "
                f"{eqn_index} has unbound input {in_index}"
            ) from exc


def _value_is_graph_connected(ctx: LoweringContextProtocol, value: ir.Value) -> bool:
    """Check native ownership; a name alone does not connect an IR value."""
    graph = ctx.builder.graph
    producer = value.producer()
    if producer is not None:
        return producer.graph is graph
    return value.graph is graph and (value.is_graph_input() or value.is_initializer())


def _legacy_graph_value(
    ctx: LoweringContextProtocol, value: ir.Value
) -> ir.Value | None:
    """Resolve old name-only bindings at the legacy lowering boundary."""
    if _value_is_graph_connected(ctx, value):
        return value
    if not value.name:
        return None
    graph = ctx.builder.graph
    for candidate in graph.inputs:
        if candidate.name == value.name:
            return candidate
    for candidate in graph.initializers.values():
        if candidate.name == value.name:
            return candidate
    for node in graph:
        for candidate in node.outputs:
            if candidate.name == value.name:
                return candidate
    return None


def _outvar_needs_binding(ctx: LoweringContextProtocol, var: object) -> bool:
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
    ctx: LoweringContextProtocol,
    eqn: JaxprEqn,
    result: object,
    *,
    primitive_name: str,
) -> None:
    """Adapt legacy results and name-only aliases to native value bindings."""
    outvars = eqn.outvars
    non_drop_outvars = [
        (index, var) for index, var in enumerate(outvars) if not is_drop_var(var)
    ]
    for _, var in non_drop_outvars:
        bound = ctx.lookup_value_for_var(var)
        if bound is not None:
            connected = _legacy_graph_value(ctx, bound)
            if connected is not None and connected is not bound:
                ctx.bind_value_for_var(var, connected)
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
                ctx.bind_value_for_var(var, _legacy_graph_value(ctx, value) or value)
        return

    if len(returned_values) == len(unbound_outvars):
        for (_, var), value in zip(unbound_outvars, returned_values):
            ctx.bind_value_for_var(var, _legacy_graph_value(ctx, value) or value)
        return

    raise RuntimeError(
        f"[converter] Primitive '{primitive_name}' returned {len(returned_values)} "
        f"value(s), but {len(unbound_outvars)} non-drop outvar(s) remain unbound"
    )


def finalize_eqn_lowering_outputs(
    ctx: LoweringContextProtocol,
    eqn: JaxprEqn,
    lowering_result: object,
    *,
    primitive_name: str,
    eqn_index: int,
) -> None:
    """Finalize a legacy lowering; canonical plugins bind their own outputs."""
    bind_returned_lowering_values(
        ctx,
        eqn,
        lowering_result,
        primitive_name=primitive_name,
    )
    assert_eqn_outputs_bound(
        ctx,
        eqn,
        primitive_name=primitive_name,
        eqn_index=eqn_index,
    )


def assert_eqn_outputs_bound(
    ctx: LoweringContextProtocol,
    eqn: JaxprEqn,
    *,
    primitive_name: str,
    eqn_index: int,
) -> None:
    for out_index, outvar in enumerate(eqn.outvars):
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
