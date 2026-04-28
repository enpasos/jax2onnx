# tests/extra_tests/converter/test_lowering_dispatch.py

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jax2onnx.converter.lowering_dispatch import (
    dispatch_plugin_lowering,
    get_registered_lowering_plugin,
    identify_lowering_plugin,
    make_converter_facade,
)


class _PrimitiveNoParams:
    def lower(self, ctx: Any, eqn: Any) -> tuple[str, Any, Any]:
        return ("primitive", ctx, eqn)


class _PrimitiveWithParams:
    def lower(self, ctx: Any, eqn: Any, params: Any) -> tuple[str, Any, Any, Any]:
        return ("primitive_params", ctx, eqn, params)


class _FunctionStylePlugin:
    def get_handler(self, converter: Any) -> Any:
        def handler(conv: Any, eqn: Any, params: Any) -> tuple[Any, Any, Any, Any]:
            return (converter, conv, eqn, params)

        return handler


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
