# jax2onnx/converter/lowering_dispatch.py

from __future__ import annotations

import inspect
from contextlib import AbstractContextManager, contextmanager, nullcontext
from collections.abc import Mapping
from typing import Iterator, TypeVar

from jax2onnx._compat.jax import Jaxpr, JaxprEqn, Literal, ShapedArray, Var

from jax2onnx.utils.debug import RecordedPrimitiveCallLog, save_primitive_calls_log

from .output_binding import (
    assert_eqn_inputs_bound,
    assert_eqn_outputs_bound,
)
from .typing_support import (
    FunctionLowering,
    IRBuilderAPI,
    ConverterContext,
    PrimitiveLowering,
)
from .lowering_registry import (
    RegisteredLowering,
    make_converter_facade as make_converter_facade,
    prepare_legacy_call,
    prepare_lowering,
    prepared_lowerings,
)

_PluginT = TypeVar("_PluginT")


def get_registered_lowering_plugin(
    registry: Mapping[str, _PluginT],
    primitive_name: str,
    *,
    source: str,
    detail: str | None = None,
) -> _PluginT:
    """Return the registered plugin for ``primitive_name`` or raise a source label."""
    plugin = registry.get(primitive_name)
    if plugin is not None:
        return plugin

    detail_text = f" {detail}" if detail else ""
    raise NotImplementedError(
        f"[{source}] No plugins registered for primitive "
        f"'{primitive_name}'{detail_text}"
    )


def identify_lowering_plugin(
    plugin: object, primitive_name: str
) -> tuple[str, str | None]:
    """Return a stable plugin identifier and optional source line for metadata."""
    if plugin is None:
        return primitive_name, None

    plugin_class = f"{type(plugin).__module__}.{type(plugin).__name__}"
    if isinstance(plugin, PrimitiveLowering):
        lower = plugin.lower
        func_name = getattr(lower, "__name__", "lower")
        identifier = f"{plugin_class}.{func_name}"
        try:
            _, start_line = inspect.getsourcelines(lower)
        except (OSError, TypeError):
            return identifier, None
        return identifier, str(start_line)

    if isinstance(plugin, FunctionLowering):
        return f"{plugin_class}.get_handler", None
    return plugin_class, None


def _aval_log_entry(var: Var | Literal) -> tuple[tuple[object, ...], str, str]:
    aval = var.aval
    if isinstance(aval, ShapedArray):
        return tuple(aval.shape), str(aval.dtype), type(aval).__name__
    return (), "", type(aval).__name__


def _var_log_name(var: object) -> str:
    try:
        return str(var)
    except Exception:
        return repr(var)


def _bound_value_name(ctx: ConverterContext, var: object) -> str:
    value = ctx.lookup_value_for_var(var)
    if value is None:
        return ""
    return value.name or ""


def _params_repr(params: Mapping[str, object]) -> str:
    if not params:
        return ""
    lines: list[str] = []
    for key in sorted(params):
        try:
            value_repr = repr(params[key])
        except Exception:
            value_repr = f"<unrepresentable:{type(params[key]).__name__}>"
        lines.append(f"  {key}: {value_repr}")
    return "\n".join(lines)


def _plugin_file_hint(plugin_ref: object, prim_name: str) -> str:
    if plugin_ref is None:
        return prim_name
    return f"{type(plugin_ref).__module__}.{type(plugin_ref).__name__}"


def _primitive_call_record(
    ctx: ConverterContext,
    eqn: JaxprEqn,
    *,
    eqn_index: int,
    primitive_name: str,
    plugin_ref: object,
) -> RecordedPrimitiveCallLog:
    invars = eqn.invars
    outvars = eqn.outvars
    params = eqn.params
    return RecordedPrimitiveCallLog(
        sequence_id=eqn_index,
        primitive_name=primitive_name,
        plugin_file_hint=_plugin_file_hint(plugin_ref, primitive_name),
        params=dict(params),
        params_repr=_params_repr(params),
        inputs_aval=[_aval_log_entry(var) for var in invars],
        outputs_aval=[_aval_log_entry(var) for var in outvars],
        inputs_jax_vars=[_var_log_name(var) for var in invars],
        inputs_onnx_names=[_bound_value_name(ctx, var) for var in invars],
        outputs_jax_vars=[_var_log_name(var) for var in outvars],
        outputs_onnx_names=[_bound_value_name(ctx, var) for var in outvars],
    )


def _append_primitive_call_record(
    ctx: ConverterContext,
    eqn: JaxprEqn,
    *,
    eqn_index: int,
    primitive_name: str,
    plugin_ref: object,
) -> None:
    if not ctx.record_primitive_calls_file:
        return
    ctx._lowering_record_owner._primitive_call_records.append(
        _primitive_call_record(
            ctx,
            eqn,
            eqn_index=eqn_index,
            primitive_name=primitive_name,
            plugin_ref=plugin_ref,
        )
    )


@contextmanager
def primitive_recording_scope(ctx: ConverterContext) -> Iterator[None]:
    owner = ctx._lowering_record_owner
    previous_depth = owner._lowering_record_depth
    owner._lowering_record_depth = previous_depth + 1
    try:
        yield
    finally:
        owner._lowering_record_depth = previous_depth
        if previous_depth == 0 and ctx.record_primitive_calls_file:
            save_primitive_calls_log(
                owner._primitive_call_records, ctx.record_primitive_calls_file
            )


def _eqn_jax_traceback(eqn: JaxprEqn) -> str | None:
    traceback = eqn.source_info.traceback
    return str(traceback) if traceback is not None else None


@contextmanager
def staged_lowering_metadata(
    builder: IRBuilderAPI,
    *,
    eqn: JaxprEqn,
    plugin_ref: object,
    primitive_name: str,
) -> Iterator[None]:
    prev_jax_trace = builder.current_jax_traceback
    prev_plugin_id = builder.current_plugin_identifier
    prev_plugin_line = builder.current_plugin_line
    jax_trace: str | None = None
    plugin_identifier: str | None = None
    plugin_line: str | None = None

    if builder.stacktrace_metadata_enabled:
        jax_trace = _eqn_jax_traceback(eqn)
        plugin_identifier, plugin_line = identify_lowering_plugin(
            plugin_ref,
            primitive_name,
        )

    builder.set_current_jax_traceback(jax_trace)
    builder.set_current_plugin_identifier(plugin_identifier, plugin_line)
    try:
        yield
    finally:
        builder.set_current_jax_traceback(prev_jax_trace)
        builder.set_current_plugin_identifier(prev_plugin_id, prev_plugin_line)


@contextmanager
def current_eqn_scope(ctx: ConverterContext, eqn: JaxprEqn) -> Iterator[None]:
    previous_eqn = ctx._current_eqn
    ctx._current_eqn = eqn
    try:
        yield
    finally:
        ctx._current_eqn = previous_eqn


def dispatch_plugin_lowering(
    plugin: object,
    *,
    ctx: ConverterContext,
    eqn: JaxprEqn,
    primitive_name: str,
    source: str,
    converter: object | None = None,
) -> object:
    """Compatibility call for raw plugins; preserve their returned result."""
    call = prepare_legacy_call(plugin, primitive_name=primitive_name, source=source)
    return call(ctx, eqn, converter)


def lower_equation_with_plugin(
    plugin: object,
    *,
    ctx: ConverterContext,
    eqn: JaxprEqn,
    primitive_name: str,
    eqn_index: int,
    source: str,
    converter: object | None = None,
) -> None:
    """Prepare a raw plugin at this compatibility boundary, then lower it."""
    entry = prepare_lowering(plugin, primitive_name=primitive_name, source=source)
    if converter is not None:
        entry = entry.with_converter(converter)
    _lower_registered_equation(
        entry,
        ctx=ctx,
        eqn=eqn,
        primitive_name=primitive_name,
        eqn_index=eqn_index,
    )


def _lower_registered_equation(
    entry: RegisteredLowering,
    *,
    ctx: ConverterContext,
    eqn: JaxprEqn,
    primitive_name: str,
    eqn_index: int,
) -> None:
    """Invoke the canonical contract and validate graph-connected outputs."""
    with (
        current_eqn_scope(ctx, eqn),
        staged_lowering_metadata(
            ctx.builder,
            eqn=eqn,
            plugin_ref=entry.plugin,
            primitive_name=primitive_name,
        ),
    ):
        assert_eqn_inputs_bound(
            ctx,
            eqn,
            primitive_name=primitive_name,
            eqn_index=eqn_index,
        )
        if ctx._source_structure is not None:
            ctx._source_structure.before_equation(ctx, eqn, eqn_index)
        structure_dispatch = (
            ctx._source_structure.dispatch(eqn, eqn_index)
            if ctx._source_structure is not None
            else nullcontext()
        )
        with structure_dispatch:
            entry.lower(ctx, eqn)
        assert_eqn_outputs_bound(
            ctx,
            eqn,
            primitive_name=primitive_name,
            eqn_index=eqn_index,
        )
        if ctx._source_structure is not None:
            ctx._source_structure.equation(ctx, eqn, eqn_index)
        _append_primitive_call_record(
            ctx,
            eqn,
            eqn_index=eqn_index,
            primitive_name=primitive_name,
            plugin_ref=entry.plugin,
        )


def lower_jaxpr_with_plugins(
    *,
    ctx: ConverterContext,
    jaxpr: Jaxpr,
    registry: Mapping[str, object],
    source: str,
    converter: object | None = None,
    missing_plugin_detail: str | None = None,
) -> None:
    """Lower equations through prepared ``(ctx, eqn) -> None`` plugins."""
    if ctx._jaxpr_validator is not None:
        ctx._jaxpr_validator(jaxpr, ctx.opset)
    lowerings = prepared_lowerings(registry, converter)
    structure_scope: AbstractContextManager[None] = nullcontext()
    if ctx._source_structure is not None:
        from jax2onnx.plugins._source_structure import freeze_source_structure_rule

        structure_scope = ctx._source_structure.scope(
            ctx, jaxpr, freeze_source_structure_rule
        )
    with (
        primitive_recording_scope(ctx),
        ctx._const_folder.producer_scope(jaxpr),
        structure_scope,
    ):
        for eqn_index, eqn in enumerate(jaxpr.eqns):
            primitive_name = eqn.primitive.name
            plugin = get_registered_lowering_plugin(
                lowerings,
                primitive_name,
                source=source,
                detail=missing_plugin_detail,
            )
            _lower_registered_equation(
                plugin,
                ctx=ctx,
                eqn=eqn,
                primitive_name=primitive_name,
                eqn_index=eqn_index,
            )
