# tests/extra_tests/converter/test_conversion_guardrails.py

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from jax2onnx.converter import conversion_api
from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY,
    import_all_plugins,
    onnx_function,
)
from jax2onnx.plugins.jax.lax._control_flow_utils import lower_jaxpr_eqns
from jax2onnx.user_interface import to_onnx


def _add_one(x):
    return x + 1


@onnx_function
def _function_body_add_one(x):
    return x + 1


def _call_function_body_add_one(x):
    return _function_body_add_one(x)


def _convert_add_one() -> object:
    return conversion_api.to_onnx(
        fn=_add_one,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_params=None,
        model_name="conversion_guardrail",
        opset=21,
        enable_double_precision=False,
        record_primitive_calls_file=None,
    )


class _ReturnOnlyAddPlugin:
    def lower(self, ctx: Any, eqn: Any) -> object:
        lhs = ctx.get_value_for_var(eqn.invars[0])
        rhs = ctx.get_value_for_var(eqn.invars[1])
        ctx.get_value_for_var(eqn.outvars[0], name_hint="preallocated_add")
        out = ctx.builder.Add(lhs, rhs, _outputs=[ctx.fresh_name("add_return")])
        out.type = lhs.type
        out.shape = lhs.shape
        return out


class _NoBindAddPlugin:
    def lower(self, ctx: Any, eqn: Any) -> None:
        return None


class _DisconnectedOutputAddPlugin:
    def lower(self, ctx: Any, eqn: Any) -> None:
        ctx.get_value_for_var(eqn.outvars[0], name_hint="dangling_add")


def test_returned_lowering_value_binds_unbound_outvar(monkeypatch) -> None:
    import_all_plugins()
    monkeypatch.setitem(PLUGIN_REGISTRY, "add", _ReturnOnlyAddPlugin())

    model = _convert_add_one()

    assert model.graph.outputs
    producer = model.graph.outputs[0].producer()
    assert producer is not None
    assert producer.op_type == "Add"


def test_missing_outvar_binding_fails_at_primitive(monkeypatch) -> None:
    import_all_plugins()
    monkeypatch.setitem(PLUGIN_REGISTRY, "add", _NoBindAddPlugin())

    with pytest.raises(RuntimeError, match="Primitive 'add'.*did not bind output 0"):
        _convert_add_one()


def test_missing_outvar_binding_fails_in_function_body(monkeypatch) -> None:
    import_all_plugins()
    monkeypatch.setitem(PLUGIN_REGISTRY, "add", _NoBindAddPlugin())

    with pytest.raises(RuntimeError, match="Primitive 'add'.*did not bind output 0"):
        to_onnx(
            _call_function_body_add_one,
            inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
            return_mode="ir",
        )


def test_disconnected_outvar_binding_fails_at_primitive(monkeypatch) -> None:
    import_all_plugins()
    monkeypatch.setitem(PLUGIN_REGISTRY, "add", _DisconnectedOutputAddPlugin())

    with pytest.raises(
        RuntimeError,
        match="Primitive 'add'.*bound output 0 to disconnected value 'dangling_add'",
    ):
        _convert_add_one()


def test_nested_lowering_source_labels_missing_plugin_errors() -> None:
    jaxpr = SimpleNamespace(
        eqns=[
            SimpleNamespace(primitive=SimpleNamespace(name="missing_nested_primitive"))
        ]
    )

    with pytest.raises(
        NotImplementedError,
        match=r"\[jit\] No plugins registered for primitive 'missing_nested_primitive'",
    ):
        lower_jaxpr_eqns(SimpleNamespace(), jaxpr, source="jit")
