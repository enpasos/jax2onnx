# tests/extra_tests/converter/test_lowering_registry.py

"""Registration compatibility and the canonical equation-loop contract."""

from __future__ import annotations

import gc
from inspect import Signature
from typing import Any
import weakref

import jax
import jax.numpy as jnp
import onnx_ir as ir
import pytest

from jax2onnx import to_onnx
from jax2onnx._compat.jax import Jaxpr, JaxprEqn
from jax2onnx.converter import lowering_registry
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.lowering_dispatch import lower_jaxpr_with_plugins
from jax2onnx.converter.lowering_registry import PluginRegistry
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins import plugin_system
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, PrimitivePlugin


def _typed_registry_double(x):
    return jax.jit(lambda y: y + y)(x)


def _call_typed_registry_double(x):
    return _typed_registry_double(x)


def _jaxpr_and_context() -> tuple[Jaxpr, IRContext]:
    jaxpr = jax.make_jaxpr(lambda x: (x + x) + x)(
        jax.ShapeDtypeStruct((2,), jnp.float32)
    ).jaxpr
    ctx = IRContext(opset=21, enable_double_precision=False, input_specs=[])
    ctx.add_input_for_invar(jaxpr.invars[0], 0)
    return jaxpr, ctx


class _CanonicalPlugin:
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        ctx.bind_value_for_var(eqn.outvars[0], ctx.builder.inputs[0])


class _SlotsCanonicalPlugin:
    __slots__ = ()

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        ctx.bind_value_for_var(eqn.outvars[0], ctx.builder.inputs[0])


class _ReturnOnlyLeaf(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> ir.Value:
        return ctx.builder.inputs[0]


class _UnannotatedReturnLeaf(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        return ctx.builder.inputs[0]


class _AnnotatedReturnLeaf(PrimitiveLeafPlugin):
    # An unchecked extension annotation must not select the native contract.
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        return ctx.builder.inputs[0]


class _ParamsLeaf(PrimitiveLeafPlugin):
    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []

    def lower(
        self,
        ctx: LoweringContextProtocol,
        eqn: JaxprEqn,
        params: dict[str, Any],
    ) -> None:
        self.params.append(params)
        ctx.bind_value_for_var(eqn.outvars[0], ctx.builder.inputs[0])


class _FunctionHandler:
    def __init__(self) -> None:
        self.converters: list[object] = []

    def get_handler(self, converter: Any) -> Any:
        self.converters.append(converter)

        def handler(conv: Any, eqn: JaxprEqn, params: dict[str, Any]) -> ir.Value:
            assert conv is converter
            assert params == eqn.params
            return conv.ctx.builder.inputs[0]

        return handler


class _MetadataOnlyPlugin(PrimitivePlugin):
    def get_patch_params(self) -> list[tuple[Any, str, Any]]:
        return []


def test_registry_keeps_plugin_identity_through_all_mapping_operations() -> None:
    first = _CanonicalPlugin()
    second = _CanonicalPlugin()
    registry = PluginRegistry()
    registry.register_canonical("add", first)

    assert registry["add"] is first
    assert list(registry.values()) == [first]
    assert dict(registry.items()) == {"add": first}
    assert registry.copy() == {"add": first}
    assert registry.setdefault("add", second) is first
    assert registry.setdefault("alias", first) is first
    assert registry.prepared_lowerings()["alias"].lower == first.lower

    registry.register_canonical("add", second)
    registry.update({"add": second})
    assert registry["add"] is second
    assert registry.prepared_lowerings()["add"].lower == second.lower
    registry["add"] = first
    assert registry.prepared_lowerings()["add"].lower == first.lower
    assert registry.popitem() == ("alias", first)
    assert "alias" not in registry.prepared_lowerings()
    assert registry.pop("add") is first
    assert "add" not in registry.prepared_lowerings()
    registry["add"] = second
    registry.clear()
    assert not registry
    assert not registry.prepared_lowerings()


def test_canonical_registration_and_restoration_never_inspect_signatures(
    monkeypatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError(
            "Explicit native registration must not inspect annotations"
        )

    monkeypatch.setattr(lowering_registry, "_inspect_signature", forbidden)
    monkeypatch.setattr(lowering_registry, "bind_returned_lowering_values", forbidden)
    plugin = _CanonicalPlugin()
    registry = PluginRegistry()
    registry.register_canonical("add", plugin)
    registry.setdefault("alias", plugin)
    original = registry.pop("add")
    registry["add"] = original
    registry.clear()
    registry["add"] = original
    jaxpr, ctx = _jaxpr_and_context()

    lower_jaxpr_with_plugins(ctx=ctx, jaxpr=jaxpr, registry=registry, source="test")

    assert registry["add"] is plugin
    assert registry.prepared_lowerings()["add"].lower == plugin.lower


def test_native_identity_catalog_does_not_retain_removed_plugins() -> None:
    registry = PluginRegistry()
    plugin = _CanonicalPlugin()
    plugin_ref = weakref.ref(plugin)
    registry.register_canonical("add", plugin)
    registry.pop("add")
    del plugin
    gc.collect()

    assert plugin_ref() is None


def test_slots_only_native_plugin_requires_no_weak_reference_or_inspection(
    monkeypatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError("Explicit native registration must not inspect plugins")

    monkeypatch.setattr(lowering_registry, "_inspect_signature", forbidden)
    monkeypatch.setattr(lowering_registry, "bind_returned_lowering_values", forbidden)
    plugin = _SlotsCanonicalPlugin()
    registry = PluginRegistry()
    registry.register_canonical("add", plugin)
    jaxpr, ctx = _jaxpr_and_context()

    lower_jaxpr_with_plugins(ctx=ctx, jaxpr=jaxpr, registry=registry, source="test")

    assert registry["add"] is plugin
    assert ctx.require_value_for_var(jaxpr.outvars[0]) is ctx.builder.inputs[0]


def test_native_primitive_decorator_registers_a_typed_method_without_inspection(
    monkeypatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError("Native decorators must not infer the lowering contract")

    monkeypatch.setattr(lowering_registry, "_inspect_signature", forbidden)
    monkeypatch.setattr(lowering_registry, "bind_returned_lowering_values", forbidden)
    key = "test_native_extension"

    @plugin_system.register_native_primitive(jaxpr_primitive=key, component="typed")
    class NativeLeaf(PrimitiveLeafPlugin):
        def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
            ctx.bind_value_for_var(eqn.outvars[0], ctx.builder.inputs[0])

    try:
        jaxpr, ctx = _jaxpr_and_context()
        plugin = plugin_system.PLUGIN_REGISTRY[key]
        entry = plugin_system.PLUGIN_REGISTRY.prepared_lowerings()[key]
        entry.lower(ctx, jaxpr.eqns[0])

        assert type(plugin) is NativeLeaf
        assert plugin.metadata == {"jaxpr_primitive": key, "component": "typed"}
        assert entry.lower == plugin.lower
        assert (
            ctx.require_value_for_var(jaxpr.eqns[0].outvars[0]) is ctx.builder.inputs[0]
        )
    finally:
        plugin_system.PLUGIN_REGISTRY.pop(key, None)


@pytest.mark.parametrize(
    "plugin_cls",
    [_ReturnOnlyLeaf, _UnannotatedReturnLeaf, _AnnotatedReturnLeaf, _ParamsLeaf],
)
def test_external_leaf_subclasses_keep_legacy_call_conventions(plugin_cls) -> None:
    jaxpr, ctx = _jaxpr_and_context()
    plugin = plugin_cls()
    registry = PluginRegistry({"add": plugin})

    lower_jaxpr_with_plugins(ctx=ctx, jaxpr=jaxpr, registry=registry, source="test")

    assert all(
        ctx.require_value_for_var(var) is ctx.builder.inputs[0] for var in jaxpr.outvars
    )
    assert registry["add"] is plugin
    if isinstance(plugin, _ParamsLeaf):
        assert plugin.params == [eqn.params for eqn in jaxpr.eqns]


@pytest.mark.parametrize(
    "plugin_cls",
    [_ReturnOnlyLeaf, _UnannotatedReturnLeaf, _AnnotatedReturnLeaf, _ParamsLeaf],
)
def test_public_legacy_decorator_keeps_extension_call_conventions(plugin_cls) -> None:
    jaxpr, ctx = _jaxpr_and_context()
    key = "test_legacy_extension"
    decorated = plugin_system.register_primitive(jaxpr_primitive=key)(plugin_cls)
    try:
        plugin_system.PLUGIN_REGISTRY.prepared_lowerings()[key].lower(
            ctx, jaxpr.eqns[0]
        )

        assert decorated is plugin_cls
        assert (
            ctx.require_value_for_var(jaxpr.eqns[0].outvars[0]) is ctx.builder.inputs[0]
        )
        assert type(plugin_system.PLUGIN_REGISTRY[key]) is plugin_cls
    finally:
        plugin_system.PLUGIN_REGISTRY.pop(key, None)


def test_ordinary_mapping_is_prepared_once_for_the_whole_jaxpr(monkeypatch) -> None:
    jaxpr, ctx = _jaxpr_and_context()
    calls = 0
    original_signature = lowering_registry._inspect_signature

    def inspect_once(obj: object) -> Signature:
        nonlocal calls
        calls += 1
        return original_signature(obj)

    monkeypatch.setattr(lowering_registry, "_inspect_signature", inspect_once)

    lower_jaxpr_with_plugins(
        ctx=ctx, jaxpr=jaxpr, registry={"add": _ReturnOnlyLeaf()}, source="test"
    )

    assert calls == 1
    assert ctx.require_value_for_var(jaxpr.outvars[0]) is ctx.builder.inputs[0]


@pytest.mark.parametrize("registered", [False, True])
def test_explicit_converter_is_bound_once_without_leaking_to_default_calls(
    monkeypatch, registered: bool
) -> None:
    jaxpr, ctx = _jaxpr_and_context()
    plugin = _FunctionHandler()
    registry = PluginRegistry({"add": plugin}) if registered else {"add": plugin}
    converter = lowering_registry.make_converter_facade(ctx)
    converter.marker = "explicit"

    def no_signature(obj: object) -> Signature:
        raise AssertionError("Rebinding function handlers must not inspect signatures")

    monkeypatch.setattr(lowering_registry, "_inspect_signature", no_signature)
    lower_jaxpr_with_plugins(
        ctx=ctx, jaxpr=jaxpr, registry=registry, source="test", converter=converter
    )
    assert plugin.converters == [converter, converter]

    lower_jaxpr_with_plugins(ctx=ctx, jaxpr=jaxpr, registry=registry, source="test")
    assert all(facade is not converter for facade in plugin.converters[2:])
    assert all(facade.ctx is ctx for facade in plugin.converters[2:])
    assert ctx.require_value_for_var(jaxpr.outvars[0]) is ctx.builder.inputs[0]


def test_native_entries_are_identical_in_an_explicit_converter_view() -> None:
    registry = PluginRegistry()
    registry.register_canonical("add", _CanonicalPlugin())
    entry = registry.prepared_lowerings()["add"]

    assert registry.prepared_lowerings(object())["add"] is entry


def test_metadata_only_raw_entries_fail_only_when_used() -> None:
    jaxpr, ctx = _jaxpr_and_context()
    plugin = _MetadataOnlyPlugin()
    registry = PluginRegistry({"add": plugin})

    assert registry["add"] is plugin
    with pytest.raises(NotImplementedError, match="Unsupported plugin type.*'add'"):
        lower_jaxpr_with_plugins(ctx=ctx, jaxpr=jaxpr, registry=registry, source="test")


def test_native_exports_need_neither_signature_inspection_nor_return_binding(
    monkeypatch,
) -> None:
    plugin_system.import_all_plugins()

    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError(
            "Native lowerings must use their prepared binding contract"
        )

    monkeypatch.setattr(lowering_registry, "_inspect_signature", forbidden)
    monkeypatch.setattr(lowering_registry, "bind_returned_lowering_values", forbidden)
    monkeypatch.setattr(plugin_system, "_extract_ir_ctx", forbidden)
    plugin_system.onnx_function(_typed_registry_double)
    qualified = f"{_typed_registry_double.__module__}.{_typed_registry_double.__name__}"
    try:
        model = to_onnx(
            _call_typed_registry_double,
            inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
            return_mode="ir",
        )
        assert model.functions
        assert model.graph.outputs
    finally:
        plugin_system.PLUGIN_REGISTRY.pop(f"onnx_fn::{qualified}", None)
        plugin_system.ONNX_FUNCTION_PLUGIN_REGISTRY.pop(qualified, None)
