# tests/extra_tests/converter/test_conversion_lowering_helpers.py

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx.converter.conversion_api import (
    _bind_closed_jaxpr_constants,
    _bind_jaxpr_inputs,
    _create_ir_context,
    _current_eqn_scope,
    _lower_jaxpr_equations,
    _staged_lowering_metadata,
    _trace_to_jaxpr,
)
from jax2onnx.plugins.plugin_system import PLUGIN_REGISTRY


def _add_one(x):
    return x + 1


class _ReturnOnlyAddPlugin:
    def lower(self, ctx: Any, eqn: Any) -> object:
        lhs = ctx.get_value_for_var(eqn.invars[0])
        rhs = ctx.get_value_for_var(eqn.invars[1])
        out = ctx.builder.Add(lhs, rhs, _outputs=[ctx.fresh_name("add_return")])
        out.type = lhs.type
        out.shape = lhs.shape
        return out


class _FakeMetadataBuilder:
    stacktrace_metadata_enabled = True

    def __init__(self) -> None:
        self.current_jax_traceback = "previous-trace"
        self.current_plugin_identifier = "previous-plugin"
        self.current_plugin_line = "7"

    def set_current_jax_traceback(self, trace: str | None) -> None:
        self.current_jax_traceback = trace

    def set_current_plugin_identifier(
        self, identifier: str | None, line: str | None = None
    ) -> None:
        self.current_plugin_identifier = identifier
        self.current_plugin_line = line


def _prepare_add_context():
    trace = _trace_to_jaxpr(
        fn=_add_one,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_params=None,
        enable_double_precision=False,
        inputs_as_nchw=None,
        outputs_as_nchw=None,
        input_names=["x"],
        output_names=["y"],
    )
    ctx = _create_ir_context(
        opset=21,
        enable_double_precision=False,
        input_specs=trace.sds_list,
        frozen_params=trace.frozen_params,
        record_primitive_calls_file=None,
    )
    _bind_closed_jaxpr_constants(
        ctx,
        trace.jaxpr,
        trace.closed_jaxpr.consts,
        default_float=np.dtype(np.float32),
        enable_double_precision=False,
    )
    _bind_jaxpr_inputs(ctx, trace.jaxpr, inputs_as_nchw=())
    return trace, ctx


def test_lower_jaxpr_equations_dispatches_plugin_and_binds_result(monkeypatch) -> None:
    trace, ctx = _prepare_add_context()
    monkeypatch.setitem(PLUGIN_REGISTRY, "add", _ReturnOnlyAddPlugin())

    _lower_jaxpr_equations(ctx, trace.jaxpr)

    out_value = ctx.get_value_for_var(trace.jaxpr.outvars[0])
    producer = out_value.producer()
    assert producer is not None
    assert producer.op_type == "Add"
    assert ctx._current_eqn is None


def test_lower_jaxpr_equations_reports_missing_plugin(monkeypatch) -> None:
    trace, ctx = _prepare_add_context()
    primitive_name = trace.jaxpr.eqns[0].primitive.name
    monkeypatch.delitem(PLUGIN_REGISTRY, primitive_name, raising=False)

    with pytest.raises(
        NotImplementedError,
        match=f"No plugins registered for primitive '{primitive_name}'",
    ):
        _lower_jaxpr_equations(ctx, trace.jaxpr)


def test_current_eqn_scope_restores_previous_value() -> None:
    _, ctx = _prepare_add_context()
    previous_eqn = object()
    active_eqn = object()
    ctx._current_eqn = previous_eqn

    with _current_eqn_scope(ctx, active_eqn):
        assert ctx._current_eqn is active_eqn

    assert ctx._current_eqn is previous_eqn


def test_staged_lowering_metadata_sets_and_restores_builder_state() -> None:
    builder = _FakeMetadataBuilder()
    eqn = SimpleNamespace(source_info=SimpleNamespace(traceback="jax traceback text"))

    with _staged_lowering_metadata(
        builder,
        eqn=eqn,
        plugin_ref=_ReturnOnlyAddPlugin(),
        primitive_name="add",
    ):
        assert builder.current_jax_traceback == "jax traceback text"
        assert builder.current_plugin_identifier.endswith("._ReturnOnlyAddPlugin.lower")
        assert (
            builder.current_plugin_line is None or builder.current_plugin_line.isdigit()
        )

    assert builder.current_jax_traceback == "previous-trace"
    assert builder.current_plugin_identifier == "previous-plugin"
    assert builder.current_plugin_line == "7"
