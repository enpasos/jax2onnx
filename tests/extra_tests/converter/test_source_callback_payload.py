# tests/extra_tests/converter/test_source_callback_payload.py

"""Function-owned memory is isolated without changing executable source code."""

from __future__ import annotations

import functools
import gc
import sys
import weakref

import pytest

from jax2onnx._compat.python_annotations import isolated_annotation_parts
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins.flax.nnx._source_callback import CallbackCapture


_ANNOTATION_CALLS: list[int] = []


def _annotation_probe() -> type[int]:
    _ANNOTATION_CALLS.append(1)
    return int


def _capture(*functions):
    capture = CallbackCapture()
    tokens = tuple(capture.capture(function, "test callback") for function in functions)
    graph = capture.finish()
    rebuilt = graph.rebuild()
    return graph, tuple(rebuilt.resolve(token) for token in tokens)


def test_same_code_and_numerics_with_fresh_cells_and_shared_function_aliases():
    offset = 3

    def original(x, scale=2, *, bias=1):
        return x * scale + bias + offset

    original.label = ("safe", frozenset({1, 2}))
    graph, (first, alias) = _capture(original, original)
    assert first is alias
    assert first is not original
    assert first.__code__ is original.__code__
    assert first.__globals__ is original.__globals__
    assert first.__builtins__ is original.__builtins__
    assert first.__closure__[0] is not original.__closure__[0]
    assert first.__dict__ is not original.__dict__
    assert first.__kwdefaults__ is not original.__kwdefaults__
    assert first(4) == original(4) == 12
    renormalized = CallbackCapture()
    for function in (first, alias):
        renormalized.capture(function, "rebuilt callback")
    assert renormalized.finish() == graph
    graph.assert_current()


def test_shared_closure_cells_are_preserved_across_distinct_callbacks():
    offset = 2

    def left(x):
        return x + offset

    def right(x):
        return x * offset

    _, (fresh_left, fresh_right) = _capture(left, right)
    assert fresh_left.__closure__[0] is fresh_right.__closure__[0]
    assert fresh_left.__closure__[0] is not left.__closure__[0]
    fresh_left.__closure__[0].cell_contents = 7
    assert fresh_left(3) == 10
    assert fresh_right(3) == 21
    assert left(3) == 5
    assert right(3) == 6


def test_metadata_mapping_aliases_and_wrapped_edges_are_detached():
    def original(x, *, flag="kept"):
        return x

    shared = {"flag": "kept"}
    original.__dict__ = shared
    original.__kwdefaults__ = shared
    original.__annotations__ = shared

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return original(*args, **kwargs)

    _, (fresh, fresh_wrapped) = _capture(original, wrapped)
    assert fresh.__dict__ is fresh.__kwdefaults__
    assert fresh.__dict__ is fresh.__annotations__
    assert fresh.__dict__ is not shared
    assert fresh_wrapped.__wrapped__ is fresh
    assert fresh_wrapped.__closure__[0].cell_contents is fresh
    assert fresh_wrapped(4) == wrapped(4)
    fresh.__dict__["flag"] = "isolated"
    assert shared == {"flag": "kept"}


@pytest.mark.parametrize("slot", ["closure", "defaults", "attributes"])
def test_unknown_mutable_payloads_reject_without_running_body_or_user_hooks(slot):
    executions = []

    class Opaque:
        def __hash__(self):
            raise AssertionError("hash hook executed")

        def __deepcopy__(self, memo):
            raise AssertionError("copy hook executed")

    value = Opaque()
    if slot == "closure":

        def original(x):
            executions.append(value)
            return x

    else:

        def original(x):
            _ANNOTATION_CALLS.append(x)
            return x

        if slot == "defaults":
            original.__defaults__ = (value,)
        else:
            original.payload = value
    before = len(_ANNOTATION_CALLS)
    with pytest.raises(SourcePrecisionError, match="mutable or opaque"):
        _capture(original)
    assert not executions
    assert len(_ANNOTATION_CALLS) == before


@pytest.mark.parametrize("slot", ["closure", "attributes", "defaults", "kwdefaults"])
def test_delayed_original_mutation_rejects_without_touching_clone(slot):
    offset = 2

    def original(x=1, *, scale=3):
        return x * scale + offset

    original.label = "before"
    graph, (fresh,) = _capture(original)
    if slot == "closure":
        original.__closure__[0].cell_contents = 9
    elif slot == "attributes":
        original.label = "after"
    elif slot == "defaults":
        original.__defaults__ = (8,)
    else:
        original.__kwdefaults__["scale"] = 7
    with pytest.raises(SourcePrecisionError, match="changed"):
        graph.assert_current()
    assert fresh() == 5
    assert fresh.label == "before"


def test_clone_memory_mutation_changes_normalized_schema_only_for_clone():
    offset = 2

    def original(x):
        return x + offset

    graph, (fresh,) = _capture(original)
    fresh.__closure__[0].cell_contents = 7
    changed = CallbackCapture()
    changed.capture(fresh)
    assert changed.finish() != graph
    graph.assert_current()
    assert original(1) == 3


def test_lazy_annotation_snapshot_and_clone_materialization_keep_source_cold():
    namespace = {}
    exec(
        compile(
            "def callback(x: _annotation_probe()) -> _annotation_probe():\n    return x\n",
            __file__,
            "exec",
            dont_inherit=True,
        ),
        globals(),
        namespace,
    )
    original = namespace["callback"]
    before = len(_ANNOTATION_CALLS)
    original_parts = isolated_annotation_parts(original)
    graph, (fresh,) = _capture(original)
    assert len(_ANNOTATION_CALLS) == before
    assert fresh.__annotations__ == {"x": int, "return": int}
    assert len(_ANNOTATION_CALLS) == before + (2 if sys.version_info >= (3, 14) else 0)
    assert isolated_annotation_parts(original) == original_parts
    graph.assert_current()
    normalized = CallbackCapture()
    normalized.capture(fresh)
    assert normalized.finish() == graph
    fresh.__annotations__["return"] = str
    if sys.version_info >= (3, 14):
        with pytest.raises(SourcePrecisionError, match="cache bindings changed"):
            normalized.finish().assert_current()
    assert original.__annotations__ == {"x": int, "return": int}


def test_original_annotation_warm_transition_stales_and_warm_cache_is_isolated():
    def original(x: int) -> str:
        return str(x)

    graph, _ = _capture(original)
    original.__annotations__
    if sys.version_info >= (3, 14):
        with pytest.raises(SourcePrecisionError, match="changed"):
            graph.assert_current()
    _, (fresh,) = _capture(original)
    assert fresh.__annotations__ == original.__annotations__
    assert fresh.__annotations__ is not original.__annotations__


def test_snapshot_and_rebuild_ownership_releases_after_descriptor_lifetime():
    def original(x):
        return x + 1

    graph, (fresh,) = _capture(original)
    original_ref = weakref.ref(original)
    fresh_ref = weakref.ref(fresh)
    del original, fresh
    gc.collect()
    assert original_ref() is not None
    del graph
    gc.collect()
    assert original_ref() is None
    assert fresh_ref() is None


def test_bound_methods_and_partial_callables_reject_before_execution():
    class Owner:
        def method(self, x):
            raise AssertionError("body executed")

    for function in (Owner().method, functools.partial(int, base=10)):
        with pytest.raises(SourcePrecisionError, match="bound methods"):
            _capture(function)


def test_managed_converter_wrapper_retains_bounded_role_after_patch_scope(monkeypatch):
    from jax2onnx.converter.precision_policy import (
        PrecisionPolicy,
        precision_policy_scope,
    )
    from jax2onnx.plugins import plugin_system

    def original(x):
        return x + 1

    qualified = f"{__name__}.callback_test_role"
    plugin = plugin_system.FunctionPlugin(f"onnx_fn::{qualified}", original)
    monkeypatch.setitem(
        plugin_system.ONNX_FUNCTION_PLUGIN_REGISTRY, plugin._qualified_target, plugin
    )
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        wrapper = plugin._make_patch_fn(plugin.primitive, is_class=False)(original)
        with monkeypatch.context() as patch_scope:
            patch_scope.setitem(
                plugin_system._PATCH_STATE,
                (sys.modules[__name__], "callback_test_role"),
                {
                    "orig": original,
                    "patched": wrapper,
                    "owner": plugin,
                    "policy": PrecisionPolicy.SOURCE,
                },
            )
            capture = CallbackCapture()
            callback_token = capture.capture(wrapper)
            graph = capture.finish()
    graph.assert_current()
    fresh = graph.rebuild().resolve(callback_token)
    assert plugin._unbound_orig_fn is original
    assert fresh.__code__ is wrapper.__code__
    assert fresh.__wrapped__ is not original
    old_cells = dict(
        zip(wrapper.__code__.co_freevars, wrapper.__closure__, strict=True)
    )
    new_cells = dict(zip(fresh.__code__.co_freevars, fresh.__closure__, strict=True))
    assert new_cells["self"].cell_contents is plugin
    assert new_cells["primitive"].cell_contents is plugin.primitive
    assert new_cells["params"].cell_contents == old_cells["params"].cell_contents
    assert new_cells["params"].cell_contents is not old_cells["params"].cell_contents
    normalized = CallbackCapture(expected=graph)
    normalized.capture(fresh)
    assert normalized.finish() == graph
    with pytest.raises(SourcePrecisionError, match="mutable or opaque"):
        CallbackCapture().capture(wrapper)
    token = plugin_system._IN_FUNCTION_BUILD.set({plugin.name})
    try:
        assert fresh(3) == 4
    finally:
        plugin_system._IN_FUNCTION_BUILD.reset(token)


@pytest.mark.parametrize("marker", [None, 1, object()])
def test_nonfunction_wrapped_marker_rejects_without_running_callback(marker):
    from jax2onnx.plugins.plugin_system import _source_parameter_names

    def original(x):
        raise AssertionError("body executed")

    original.__wrapped__ = marker
    with pytest.raises(SourcePrecisionError, match="ordinary function edge"):
        _capture(original)
    with pytest.raises(SourcePrecisionError, match="ordinary function header"):
        _source_parameter_names(original)


def test_passive_parameter_names_keep_native_positional_varargs_keyword_order():
    from jax2onnx.plugins.plugin_system import _source_parameter_names

    def full(first: _annotation_probe(), /, second=2, *args, self=1, other=3, **kwargs):
        raise AssertionError("body executed")

    def keyword_after_varargs(*args, self=1):
        raise AssertionError("body executed")

    def keyword_only(*, self=1):
        raise AssertionError("body executed")

    def only_kwargs(**kwargs):
        raise AssertionError("body executed")

    before = len(_ANNOTATION_CALLS)
    before_parts = isolated_annotation_parts(full)
    assert _source_parameter_names(full) == [
        "first",
        "second",
        "args",
        "self",
        "other",
        "kwargs",
    ]
    assert _source_parameter_names(keyword_after_varargs) == ["args", "self"]
    assert _source_parameter_names(keyword_only) == ["self"]
    assert _source_parameter_names(only_kwargs) == ["kwargs"]
    assert len(_ANNOTATION_CALLS) == before
    assert isolated_annotation_parts(full) == before_parts


@pytest.mark.parametrize("nested", [False, True])
def test_function_descendants_in_frozenset_reject_without_callback_execution(nested):
    def left(x):
        raise AssertionError("body executed")

    def right(x):
        raise AssertionError("body executed")

    def original(x):
        raise AssertionError("body executed")

    original.payload = (
        frozenset({(left,), (right,)}) if nested else frozenset({left, right})
    )
    with pytest.raises(SourcePrecisionError, match="function-valued frozensets"):
        _capture(original)


def test_native_array_annotation_keeps_exact_type_with_fresh_annotation_cache():
    from jax2onnx._compat.jax import Array
    from jax2onnx.plugins.flax.nnx._source_callback import _canonical_annotation

    def original(x):
        return x

    original.__annotations__ = {"return": Array}
    assert _canonical_annotation(Array)
    assert not _canonical_annotation(1)
    graph, (fresh,) = _capture(original)
    assert fresh.__annotations__ is not original.__annotations__
    assert fresh.__annotations__["return"] is Array
    graph.assert_current()


@pytest.mark.parametrize("payload", [[], {}, object()])
def test_model_owned_annotation_objects_reject_before_callback_body(payload):
    def original(x):
        raise AssertionError("body executed")

    original.__annotations__ = {"return": payload}
    with pytest.raises(SourcePrecisionError, match="mutable or opaque"):
        _capture(original)
