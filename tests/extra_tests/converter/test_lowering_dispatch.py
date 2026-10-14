# tests/extra_tests/converter/test_lowering_dispatch.py

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import onnx_ir as ir
import pytest

from jax2onnx import to_onnx
from jax2onnx._compat.jax import Jaxpr, JaxprEqn
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.converter import lowering_registry
from jax2onnx.converter.lowering_dispatch import (
    dispatch_plugin_lowering,
    get_registered_lowering_plugin,
    identify_lowering_plugin,
    lower_equation_with_plugin,
    lower_jaxpr_with_plugins,
    make_converter_facade,
)
from jax2onnx.converter.lowering_registry import PluginRegistry
from jax2onnx.plugins import plugin_system


def _recorded_function(x):
    return x + x


def _call_recorded_function(x):
    return _recorded_function(x)


class _PrimitiveNoParams:
    def lower(self, ctx: Any, eqn: Any) -> tuple[str, Any, Any]:
        return ("primitive", ctx, eqn)


class _PrimitiveWithParams:
    def lower(self, ctx: Any, eqn: Any, params: Any) -> tuple[str, Any, Any, Any]:
        return ("primitive_params", ctx, eqn, params)


class _ReturnGraphInputPlugin:
    def lower(self, ctx: Any, eqn: Any) -> ir.Value:
        return ctx.builder.inputs[0]


class _FunctionStylePlugin:
    def get_handler(self, converter: Any) -> Any:
        def handler(conv: Any, eqn: Any, params: Any) -> tuple[Any, Any, Any, Any]:
            return (converter, conv, eqn, params)

        return handler


class _AssertMetadataPlugin:
    def __init__(self, traceback: str, fail: bool = False) -> None:
        self.traceback = traceback
        self.fail = fail

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        assert ctx._current_eqn is eqn
        assert ctx.builder.current_jax_traceback == self.traceback
        identifier = ctx.builder.current_plugin_identifier
        assert identifier is not None
        assert identifier.endswith("._AssertMetadataPlugin.lower")
        if self.fail:
            raise ValueError("lowering failed")
        ctx.bind_value_for_var(eqn.outvars[0], ctx.builder.inputs[0])


class _NestedRecordingPlugin:
    def __init__(self, jaxpr: Jaxpr) -> None:
        self.jaxpr = jaxpr

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        ctx.bind_value_for_var(self.jaxpr.invars[0], ctx.builder.inputs[0])
        lower_jaxpr_with_plugins(
            ctx=ctx,
            jaxpr=self.jaxpr,
            registry={"add": _ReturnGraphInputPlugin()},
            source="inner",
        )
        ctx.bind_value_for_var(eqn.outvars[0], ctx.builder.inputs[0])


def _jaxpr_and_context(
    *, nested: bool = False, stacktrace_metadata: bool = False
) -> tuple[Jaxpr, IRContext]:
    fn = (lambda x: jax.jit(lambda y: y + y)(x)) if nested else (lambda x: x + x)
    jaxpr = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((2,), jnp.float32)).jaxpr
    ctx = IRContext(
        opset=21,
        enable_double_precision=False,
        input_specs=[],
        stacktrace_metadata=stacktrace_metadata,
    )
    ctx.add_input_for_invar(jaxpr.invars[0], 0)
    return jaxpr, ctx


def test_get_registered_lowering_plugin_returns_registered_plugin() -> None:
    plugin = _PrimitiveNoParams()

    result = get_registered_lowering_plugin(
        {"sample": plugin},
        "sample",
        source="test",
    )

    assert result is plugin


def test_get_registered_lowering_plugin_reports_missing_plugin() -> None:
    with pytest.raises(
        NotImplementedError,
        match=r"\[test\] No plugins registered for primitive 'missing' in body",
    ):
        get_registered_lowering_plugin(
            {},
            "missing",
            source="test",
            detail="in body",
        )


def test_identify_lowering_plugin_reports_primitive_lowering_metadata() -> None:
    identifier, line = identify_lowering_plugin(_PrimitiveNoParams(), "sample")

    assert identifier.endswith("._PrimitiveNoParams.lower")
    assert line is None or line.isdigit()


def test_identify_lowering_plugin_reports_function_lowering_metadata() -> None:
    identifier, line = identify_lowering_plugin(_FunctionStylePlugin(), "sample")

    assert identifier.endswith("._FunctionStylePlugin.get_handler")
    assert line is None


def test_identify_lowering_plugin_falls_back_to_primitive_name() -> None:
    identifier, line = identify_lowering_plugin(None, "sample")

    assert identifier == "sample"
    assert line is None


def test_make_converter_facade_exposes_builder_and_context() -> None:
    ctx = SimpleNamespace(builder=SimpleNamespace())

    converter = make_converter_facade(ctx)

    assert converter.builder is ctx.builder
    assert converter.ctx is ctx


def test_dispatches_primitive_without_params() -> None:
    ctx = SimpleNamespace(builder=SimpleNamespace())
    eqn = SimpleNamespace(params={"ignored": True})

    result = dispatch_plugin_lowering(
        _PrimitiveNoParams(),
        ctx=ctx,
        eqn=eqn,
        primitive_name="sample",
        source="test",
    )

    assert result == ("primitive", ctx, eqn)


def test_dispatches_primitive_with_params() -> None:
    ctx = SimpleNamespace(builder=SimpleNamespace())
    eqn = SimpleNamespace(params={"axis": 1})

    result = dispatch_plugin_lowering(
        _PrimitiveWithParams(),
        ctx=ctx,
        eqn=eqn,
        primitive_name="sample",
        source="test",
    )

    assert result == ("primitive_params", ctx, eqn, {"axis": 1})


def test_registered_lowering_inspects_signature_once(monkeypatch) -> None:
    signature_calls = 0
    original_signature = lowering_registry._inspect_signature

    def _counted_signature(obj: Any) -> Any:
        nonlocal signature_calls
        signature_calls += 1
        return original_signature(obj)

    monkeypatch.setattr(lowering_registry, "_inspect_signature", _counted_signature)
    jaxpr, ctx = _jaxpr_and_context()
    registry = PluginRegistry({"add": _ReturnGraphInputPlugin()})
    lower = registry.prepared_lowerings()["add"].lower
    lower(ctx, jaxpr.eqns[0])
    lower(ctx, jaxpr.eqns[0])

    assert signature_calls == 1


def test_dispatches_function_plugin_with_supplied_converter() -> None:
    ctx = SimpleNamespace(builder=SimpleNamespace())
    converter = SimpleNamespace(builder=ctx.builder, ctx=ctx)
    eqn = SimpleNamespace(params={"marker": "function"})

    result = dispatch_plugin_lowering(
        _FunctionStylePlugin(),
        ctx=ctx,
        eqn=eqn,
        primitive_name="sample_function",
        source="test",
        converter=converter,
    )

    assert result == (converter, converter, eqn, {"marker": "function"})


def test_dispatches_function_plugin_with_default_converter_facade() -> None:
    ctx = SimpleNamespace(builder=SimpleNamespace())
    eqn = SimpleNamespace()

    converter, conv, handled_eqn, params = dispatch_plugin_lowering(
        _FunctionStylePlugin(),
        ctx=ctx,
        eqn=eqn,
        primitive_name="sample_function",
        source="test",
    )

    assert converter is conv
    assert converter.builder is ctx.builder
    assert converter.ctx is ctx
    assert handled_eqn is eqn
    assert params == {}


def test_lower_equation_with_plugin_dispatches_and_finalizes_outputs() -> None:
    jaxpr, ctx = _jaxpr_and_context()
    eqn = jaxpr.eqns[0]

    result = lower_equation_with_plugin(
        _ReturnGraphInputPlugin(),
        ctx=ctx,
        eqn=eqn,
        primitive_name="add",
        eqn_index=0,
        source="test",
    )

    assert result is None
    assert ctx.lookup_value_for_var(eqn.outvars[0]) is ctx.builder.inputs[0]


def test_lower_equation_with_plugin_rejects_unbound_input() -> None:
    jaxpr, _ = _jaxpr_and_context()
    ctx = IRContext(opset=21, enable_double_precision=False, input_specs=[])

    with pytest.raises(RuntimeError, match="Primitive 'add'.*has unbound input 0"):
        lower_equation_with_plugin(
            _ReturnGraphInputPlugin(),
            ctx=ctx,
            eqn=jaxpr.eqns[0],
            primitive_name="add",
            eqn_index=0,
            source="test",
        )


@pytest.mark.parametrize("fail", [False, True])
def test_lower_equation_with_plugin_stages_metadata_and_current_eqn(fail: bool) -> None:
    jaxpr, ctx = _jaxpr_and_context(stacktrace_metadata=True)
    eqn = jaxpr.eqns[0]
    traceback = str(eqn.source_info.traceback)
    builder = ctx.builder
    builder.set_current_jax_traceback("previous-trace")
    builder.set_current_plugin_identifier("previous-plugin", "7")
    plugin = _AssertMetadataPlugin(traceback, fail)

    if fail:
        with pytest.raises(ValueError, match="lowering failed"):
            lower_equation_with_plugin(
                plugin,
                ctx=ctx,
                eqn=eqn,
                primitive_name="add",
                eqn_index=0,
                source="test",
            )
    else:
        lower_equation_with_plugin(
            plugin, ctx=ctx, eqn=eqn, primitive_name="add", eqn_index=0, source="test"
        )

    assert ctx._current_eqn is None
    assert builder.current_jax_traceback == "previous-trace"
    assert builder.current_plugin_identifier == "previous-plugin"
    assert builder.current_plugin_line == "7"


def test_lower_jaxpr_with_plugins_lowers_all_equations() -> None:
    jaxpr = jax.make_jaxpr(lambda x: (x + x) + x)(
        jax.ShapeDtypeStruct((2,), jnp.float32)
    ).jaxpr
    ctx = IRContext(opset=21, enable_double_precision=False, input_specs=[])
    ctx.add_input_for_invar(jaxpr.invars[0], 0)

    lower_jaxpr_with_plugins(
        ctx=ctx, jaxpr=jaxpr, registry={"add": _ReturnGraphInputPlugin()}, source="test"
    )

    assert len(jaxpr.eqns) == 2
    for eqn in jaxpr.eqns:
        assert ctx.lookup_value_for_var(eqn.outvars[0]) is ctx.builder.inputs[0]


def test_lower_jaxpr_with_plugins_records_nested_calls_once(tmp_path) -> None:
    jaxpr, ctx = _jaxpr_and_context(nested=True)
    inner_jaxpr = jaxpr.eqns[0].params["jaxpr"].jaxpr
    log_path = tmp_path / "primitive_calls.json"
    ctx.record_primitive_calls_file = str(log_path)

    lower_jaxpr_with_plugins(
        ctx=ctx,
        jaxpr=jaxpr,
        registry={"jit": _NestedRecordingPlugin(inner_jaxpr)},
        source="outer",
    )

    records = json.loads(log_path.read_text(encoding="utf-8"))
    assert [record["primitive_name"] for record in records] == ["add", "jit"]
    assert ctx._lowering_record_depth == 0
    assert len(ctx._primitive_call_records) == 2


def test_function_body_recording_shares_the_parent_log(tmp_path) -> None:
    plugin_system.onnx_function(_recorded_function)
    qualified = f"{_recorded_function.__module__}.{_recorded_function.__name__}"
    log_path = tmp_path / "function_calls.json"
    try:
        to_onnx(
            _call_recorded_function,
            inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
            record_primitive_calls_file=str(log_path),
            return_mode="ir",
        )
        records = json.loads(log_path.read_text(encoding="utf-8"))
        assert [record["primitive_name"] for record in records] == [
            "add",
            f"onnx_fn::{qualified}",
        ]
        assert all(record["outputs_onnx_names"] for record in records)
    finally:
        plugin_system.PLUGIN_REGISTRY.pop(f"onnx_fn::{qualified}", None)
        plugin_system.ONNX_FUNCTION_PLUGIN_REGISTRY.pop(qualified, None)


def test_lower_jaxpr_with_plugins_restores_existing_constant_producers() -> None:
    outer_jaxpr, ctx = _jaxpr_and_context()
    inner_jaxpr, _ = _jaxpr_and_context()
    ctx.bind_value_for_var(inner_jaxpr.invars[0], ctx.builder.inputs[0])
    ctx._const_folder.install_producers(outer_jaxpr)

    lower_jaxpr_with_plugins(
        ctx=ctx,
        jaxpr=inner_jaxpr,
        registry={"add": _ReturnGraphInputPlugin()},
        source="test",
    )

    assert ctx.lookup_value_for_var(inner_jaxpr.outvars[0]) is ctx.builder.inputs[0]
    assert id(outer_jaxpr.outvars[0]) in ctx._const_folder._producer
    assert id(inner_jaxpr.outvars[0]) not in ctx._const_folder._producer


def test_lower_jaxpr_with_plugins_reports_missing_plugin_detail() -> None:
    jaxpr, ctx = _jaxpr_and_context()

    with pytest.raises(
        NotImplementedError,
        match=r"\[onnx_function\] No plugins registered for primitive 'add' in function body",
    ):
        lower_jaxpr_with_plugins(
            ctx=ctx,
            jaxpr=jaxpr,
            registry={},
            source="onnx_function",
            missing_plugin_detail="in function body",
        )
    assert ctx._lowering_record_depth == 0


def test_reports_unsupported_plugin_type() -> None:
    with pytest.raises(
        NotImplementedError,
        match=r"\[test\] Unsupported plugin type for primitive 'missing'",
    ):
        dispatch_plugin_lowering(
            object(),
            ctx=SimpleNamespace(builder=SimpleNamespace()),
            eqn=SimpleNamespace(),
            primitive_name="missing",
            source="test",
        )
