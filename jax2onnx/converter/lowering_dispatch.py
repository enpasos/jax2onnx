# jax2onnx/converter/lowering_dispatch.py

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Iterator

from jax2onnx.utils.debug import RecordedPrimitiveCallLog, save_primitive_calls_log

from .output_binding import (
    assert_eqn_inputs_bound,
    finalize_eqn_lowering_outputs,
    get_bound_value,
)
from .typing_support import (
    FunctionLowering,
    LoweringContextProtocol,
    PrimitiveLowering,
)

_LOWER_SIGNATURE_CACHE: dict[object, bool] = {}
_MISSING: object = object()


def _lower_accepts_params(lower: Any) -> bool:
    cache_key = getattr(lower, "__func__", lower)
    try:
        cached = _LOWER_SIGNATURE_CACHE.get(cache_key)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    try:
        accepts_params = "params" in inspect.signature(lower).parameters
    except (TypeError, ValueError):
        accepts_params = False
    try:
        _LOWER_SIGNATURE_CACHE[cache_key] = accepts_params
    except TypeError:
        pass
    return accepts_params


def make_converter_facade(ctx: LoweringContextProtocol) -> SimpleNamespace:
    """Build the minimal converter facade expected by function lowerings."""
    return SimpleNamespace(builder=getattr(ctx, "builder", None), ctx=ctx)


def get_registered_lowering_plugin(
    registry: Mapping[str, object],
    primitive_name: str,
    *,
    source: str,
    detail: str | None = None,
) -> object:
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

    try:
        if isinstance(plugin, PrimitiveLowering):
            lower = plugin.lower
            func_name = getattr(lower, "__name__", "lower")
            identifier = (
                f"{type(plugin).__module__}.{type(plugin).__name__}.{func_name}"
            )
            try:
                _, start_line = inspect.getsourcelines(lower)
            except (OSError, TypeError):
                return identifier, None
            return identifier, str(start_line)

        if isinstance(plugin, FunctionLowering):
            return (
                f"{type(plugin).__module__}.{type(plugin).__name__}.get_handler",
                None,
            )

        if hasattr(plugin, "__class__"):
            return f"{type(plugin).__module__}.{type(plugin).__name__}", None
    except Exception:
        pass

    return primitive_name, None


def _aval_log_entry(var: object) -> tuple[tuple[object, ...], str, str]:
    aval = getattr(var, "aval", None)
    if aval is None:
        return ((), "", type(var).__name__)
    raw_shape = getattr(aval, "shape", ())
    try:
        shape = tuple(raw_shape)
    except TypeError:
        shape = ()
    dtype = getattr(aval, "dtype", "")
    return (shape, str(dtype), type(aval).__name__)


def _var_log_name(var: object) -> str:
    try:
        return str(var)
    except Exception:
        return repr(var)


def _bound_value_name(ctx: object, var: object) -> str:
    value = get_bound_value(ctx, var)
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
    ctx: object,
    eqn: object,
    *,
    eqn_index: int,
    primitive_name: str,
    plugin_ref: object,
) -> RecordedPrimitiveCallLog:
    invars = list(getattr(eqn, "invars", ()))
    outvars = list(getattr(eqn, "outvars", ()))
    params = getattr(eqn, "params", {})
    if not isinstance(params, Mapping):
        params = {}
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


def _record_owner(ctx: object) -> object:
    return getattr(ctx, "_lowering_record_owner", ctx)


def _append_primitive_call_record(
    ctx: object,
    eqn: object,
    *,
    eqn_index: int,
    primitive_name: str,
    plugin_ref: object,
) -> None:
    if not getattr(ctx, "record_primitive_calls_file", None):
        return
    owner = _record_owner(ctx)
    records = getattr(owner, "_primitive_call_records", None)
    if not isinstance(records, list):
        records = []
        setattr(owner, "_primitive_call_records", records)
    records.append(
        _primitive_call_record(
            ctx,
            eqn,
            eqn_index=eqn_index,
            primitive_name=primitive_name,
            plugin_ref=plugin_ref,
        )
    )


@contextmanager
def primitive_recording_scope(ctx: object) -> Iterator[None]:
    owner = _record_owner(ctx)
    previous_depth = int(getattr(owner, "_lowering_record_depth", 0) or 0)
    setattr(owner, "_lowering_record_depth", previous_depth + 1)
    try:
        yield
    finally:
        next_depth = max(int(getattr(owner, "_lowering_record_depth", 1)) - 1, 0)
        setattr(owner, "_lowering_record_depth", next_depth)
        record_file = getattr(ctx, "record_primitive_calls_file", None)
        records = getattr(owner, "_primitive_call_records", None)
        if next_depth == 0 and record_file and isinstance(records, list):
            save_primitive_calls_log(records, str(record_file))


def _eqn_jax_traceback(eqn: object) -> str | None:
    source_info = getattr(eqn, "source_info", None)
    if source_info is None:
        return None
    traceback = getattr(source_info, "traceback", None)
    if traceback is None:
        return None
    try:
        return str(traceback)
    except Exception:
        return None


def _stacktrace_metadata_enabled(builder: object) -> bool:
    enabled = getattr(builder, "stacktrace_metadata_enabled", False)
    if callable(enabled):
        try:
            return bool(enabled())
        except Exception:
            return False
    return bool(enabled)


@contextmanager
def staged_lowering_metadata(
    builder: object,
    *,
    eqn: object,
    plugin_ref: object,
    primitive_name: str,
) -> Iterator[None]:
    set_jax_traceback = getattr(builder, "set_current_jax_traceback", None)
    set_plugin_identifier = getattr(builder, "set_current_plugin_identifier", None)
    if not callable(set_jax_traceback) or not callable(set_plugin_identifier):
        yield
        return

    prev_jax_trace = getattr(builder, "current_jax_traceback", None)
    prev_plugin_id = getattr(builder, "current_plugin_identifier", None)
    prev_plugin_line = getattr(builder, "current_plugin_line", None)
    jax_trace: str | None = None
    plugin_identifier: str | None = None
    plugin_line: str | None = None

    if _stacktrace_metadata_enabled(builder):
        jax_trace = _eqn_jax_traceback(eqn)
        plugin_identifier, plugin_line = identify_lowering_plugin(
            plugin_ref,
            primitive_name,
        )

    set_jax_traceback(jax_trace)
    set_plugin_identifier(plugin_identifier, plugin_line)
    try:
        yield
    finally:
        set_jax_traceback(prev_jax_trace)
        set_plugin_identifier(prev_plugin_id, prev_plugin_line)


@contextmanager
def current_eqn_scope(ctx: object, eqn: object) -> Iterator[None]:
    previous_eqn = getattr(ctx, "_current_eqn", _MISSING)
    setattr(ctx, "_current_eqn", eqn)
    try:
        yield
    finally:
        if previous_eqn is _MISSING:
            try:
                delattr(ctx, "_current_eqn")
            except AttributeError:
                pass
        else:
            setattr(ctx, "_current_eqn", previous_eqn)


def dispatch_plugin_lowering(
    plugin: object,
    *,
    ctx: LoweringContextProtocol,
    eqn: object,
    primitive_name: str,
    source: str,
    converter: object | None = None,
) -> object:
    """Invoke a registered lowering plugin for a single JAXPR equation."""
    if isinstance(plugin, PrimitiveLowering):
        lower = plugin.lower
        if _lower_accepts_params(lower):
            return lower(ctx, eqn, getattr(eqn, "params", None))
        return lower(ctx, eqn)

    if isinstance(plugin, FunctionLowering):
        converter_facade = converter or make_converter_facade(ctx)
        handler = plugin.get_handler(converter_facade)
        return handler(
            converter_facade,
            eqn,
            getattr(eqn, "params", {}) or {},
        )

    raise NotImplementedError(
        f"[{source}] Unsupported plugin type for primitive '{primitive_name}'"
    )


def lower_equation_with_plugin(
    plugin: object,
    *,
    ctx: LoweringContextProtocol,
    eqn: object,
    primitive_name: str,
    eqn_index: int,
    source: str,
    converter: object | None = None,
) -> object:
    """Dispatch a plugin and verify the equation output contract."""
    builder = getattr(ctx, "builder", None)
    metadata_scope = (
        staged_lowering_metadata(
            builder,
            eqn=eqn,
            plugin_ref=plugin,
            primitive_name=primitive_name,
        )
        if builder is not None
        else nullcontext()
    )
    with current_eqn_scope(ctx, eqn), metadata_scope:
        assert_eqn_inputs_bound(
            ctx,
            eqn,
            primitive_name=primitive_name,
            eqn_index=eqn_index,
        )
        lowering_result = dispatch_plugin_lowering(
            plugin,
            ctx=ctx,
            eqn=eqn,
            primitive_name=primitive_name,
            source=source,
            converter=converter,
        )
        finalize_eqn_lowering_outputs(
            ctx,
            eqn,
            lowering_result,
            primitive_name=primitive_name,
            eqn_index=eqn_index,
        )
        _append_primitive_call_record(
            ctx,
            eqn,
            eqn_index=eqn_index,
            primitive_name=primitive_name,
            plugin_ref=plugin,
        )
    return lowering_result


def lower_jaxpr_with_plugins(
    *,
    ctx: LoweringContextProtocol,
    jaxpr: object,
    registry: Mapping[str, object],
    source: str,
    converter: object | None = None,
    missing_plugin_detail: str | None = None,
) -> None:
    """Lower every equation in a JAXPR-like object with registered plugins."""
    const_folder = getattr(ctx, "_const_folder", None)
    producer_scope = getattr(const_folder, "producer_scope", None)
    scope = producer_scope(jaxpr) if callable(producer_scope) else nullcontext()
    with primitive_recording_scope(ctx), scope:
        for eqn_index, eqn in enumerate(getattr(jaxpr, "eqns", ())):
            primitive_name = eqn.primitive.name
            plugin = get_registered_lowering_plugin(
                registry,
                primitive_name,
                source=source,
                detail=missing_plugin_detail,
            )
            lower_equation_with_plugin(
                plugin,
                ctx=ctx,
                eqn=eqn,
                primitive_name=primitive_name,
                eqn_index=eqn_index,
                source=source,
                converter=converter,
            )
