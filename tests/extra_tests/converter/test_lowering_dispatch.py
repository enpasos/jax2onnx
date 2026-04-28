# tests/extra_tests/converter/test_lowering_dispatch.py

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import onnx_ir as ir
import pytest

from jax2onnx.converter.lowering_dispatch import (
    dispatch_plugin_lowering,
    get_registered_lowering_plugin,
    identify_lowering_plugin,
    lower_equation_with_plugin,
    lower_jaxpr_with_plugins,
    make_converter_facade,
)


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


def test_lower_equation_with_plugin_dispatches_and_finalizes_outputs() -> None:
    outvar = object()
    input_value = ir.Value(name="input")
    ctx = SimpleNamespace(
        builder=SimpleNamespace(
            inputs=[input_value],
            initializers=[],
            nodes=[],
            _var2val={},
        )
    )
    ctx.bind_value_for_var = lambda var, value: ctx.builder._var2val.__setitem__(
        var, value
    )
    eqn = SimpleNamespace(outvars=[outvar])

    result = lower_equation_with_plugin(
        _ReturnGraphInputPlugin(),
        ctx=ctx,
        eqn=eqn,
        primitive_name="return_input",
        eqn_index=0,
        source="test",
    )

    assert result is input_value
    assert ctx.builder._var2val[outvar] is input_value


def test_lower_jaxpr_with_plugins_lowers_all_equations() -> None:
    outvars = [object(), object()]
    input_value = ir.Value(name="input")
    ctx = SimpleNamespace(
        builder=SimpleNamespace(
            inputs=[input_value],
            initializers=[],
            nodes=[],
            _var2val={},
        )
    )
    ctx.bind_value_for_var = lambda var, value: ctx.builder._var2val.__setitem__(
        var, value
    )
    jaxpr = SimpleNamespace(
        eqns=[
            SimpleNamespace(
                primitive=SimpleNamespace(name="return_input"),
                outvars=[outvars[0]],
            ),
            SimpleNamespace(
                primitive=SimpleNamespace(name="return_input"),
                outvars=[outvars[1]],
            ),
        ]
    )

    lower_jaxpr_with_plugins(
        ctx=ctx,
        jaxpr=jaxpr,
        registry={"return_input": _ReturnGraphInputPlugin()},
        source="test",
    )

    assert ctx.builder._var2val[outvars[0]] is input_value
    assert ctx.builder._var2val[outvars[1]] is input_value


def test_lower_jaxpr_with_plugins_reports_missing_plugin_detail() -> None:
    jaxpr = SimpleNamespace(
        eqns=[
            SimpleNamespace(
                primitive=SimpleNamespace(name="missing_inner"),
                outvars=[object()],
            )
        ]
    )

    with pytest.raises(
        NotImplementedError,
        match=(
            r"\[onnx_function\] No plugins registered for primitive "
            r"'missing_inner' in function body"
        ),
    ):
        lower_jaxpr_with_plugins(
            ctx=SimpleNamespace(builder=SimpleNamespace()),
            jaxpr=jaxpr,
            registry={},
            source="onnx_function",
            missing_plugin_detail="in function body",
        )


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
