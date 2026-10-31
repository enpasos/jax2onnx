# tests/extra_tests/converter/test_source_state_preparation.py

"""Production whole-call state lifting, with independent native state oracles."""

from contextlib import nullcontext
from functools import partial
from typing import Any

from flax import linen, nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx._linen_bridge import _LinenToNNXCallable
from jax2onnx._source_state_preparation import (
    NativeClosure,
    _prepare_nnx_closure_call,
    prepare_linen_call,
    prepare_native_control_call,
    prepare_nnx_call,
    prepare_pure_control_call,
)
from tests.extra_tests.converter.source_precision_support import raw_source_case
from tests.extra_tests.converter.test_source_fixed_parameter_models import _instantiate
from jax2onnx.plugins.flax import test_utils
from jax2onnx.plugins.flax.nnx import _source_function_state as nnx_state


def _clone(owner: Any) -> Any:
    graph, state = nnx.split(owner)
    return nnx.merge(graph, state, copy=True)


def _native_words(owner: Any, mutable: Any = nnx.RngState) -> tuple[Any, ...]:
    return tuple(
        (
            jax.random.key_data(value)
            if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
            else value
        )
        for value in jax.tree_util.tree_leaves(nnx.state(owner, mutable))
    )


def _exact(actual: Any, expected: Any) -> None:
    lhs, lhs_tree = jax.tree_util.tree_flatten(actual)
    rhs, rhs_tree = jax.tree_util.tree_flatten(expected)
    assert lhs_tree == rhs_tree
    for value, reference in zip(lhs, rhs, strict=True):
        np.testing.assert_array_equal(value, reference, strict=True)


class _Streams(nnx.Module):
    def __init__(self) -> None:
        self.rngs = nnx.Rngs(
            dropout=jax.random.key(17, impl="threefry2x32"),
            noise=jax.random.key(23, impl="threefry2x32"),
        )
        self.alias = self.rngs
        self.drop = nnx.Dropout(rate=0.25, rngs=self.rngs)
        self.fixed = nnx.Param(jnp.asarray(2, dtype=jnp.float32))

    def __call__(self, x: Any, *, deterministic: bool = True, offset: Any = 3) -> Any:
        y = self.drop(x, deterministic=deterministic)
        if not deterministic:
            y = y + jax.random.uniform(self.alias.noise(), x.shape, dtype=x.dtype)
        return {"primary": y * self.fixed.value, "nested": (y + offset,)}


def test_all_rng_streams_aliases_and_result_tree_survive_branch_specific_effects() -> (
    None
):
    original = _Streams()
    native = _clone(original)
    initial = _native_words(original)
    prepared = prepare_nnx_call(original)
    _exact(prepared.initial_words, initial)
    assert len(set(prepared.state_names)) == len(prepared.initial_words)
    assert all(spec.paths for spec in prepared.state_specs)
    assert any("noise" in name for name in prepared.state_names)
    fresh = prepared.fresh_original()
    assert fresh.alias is fresh.rngs
    assert fresh is not original
    assert fresh.fixed.value is original.fixed.value
    x = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)
    state = prepared.initial_words
    compiled = jax.jit(lambda x, flag, *words: prepared(x, flag, *words, offset=5))
    for flag in (True, False, False, True):
        reference = native(x, deterministic=flag, offset=5)
        result, *state = compiled(x, jnp.asarray(flag), *state)
        _exact(result, reference)
        _exact(tuple(state), _native_words(native))
    _exact(_native_words(original), initial)
    reset, *reset_state = prepared(
        x, jnp.asarray(True), *prepared.initial_words, offset=5
    )
    _exact(reset, _clone(original)(x, deterministic=True, offset=5))
    _exact(tuple(reset_state), initial)


class _OtherEffect(nnx.Module):
    def __init__(self) -> None:
        self.counter = nnx.BatchStat(jnp.asarray(0, dtype=jnp.int32))

    def __call__(self, x: Any, *, deterministic: bool = True) -> Any:
        if not deterministic:
            self.counter.value += 1
        return x + self.counter.value


def test_effects_outside_declared_mutable_state_fail_without_changing_caller() -> None:
    original = _OtherEffect()
    prepared = prepare_nnx_call(original)
    with pytest.raises(ValueError, match="outside the declared mutable selection"):
        prepared.concrete(False, jnp.asarray(2, dtype=jnp.int32))
    assert int(original.counter.value) == 0
    # Explicitly selecting this real Variable allows both branch effect layouts.
    declared = prepare_nnx_call(original, mutable=nnx.BatchStat)
    state = declared.initial_words
    for flag, expected in ((True, 0), (False, 1), (False, 2), (True, 2)):
        _, *state = declared(jnp.asarray(2, dtype=jnp.int32), jnp.asarray(flag), *state)
        assert int(state[0]) == expected
    assert int(original.counter.value) == 0


def test_actual_key_implementation_is_retained_and_unselected_key_fails() -> None:
    original = _Streams()
    prepared = prepare_nnx_call(original)
    assert any(
        spec.key_implementation == "threefry2x32" for spec in prepared.state_specs
    )
    with pytest.raises(ValueError, match="Every declared mutable state word"):
        prepared.concrete(True, jnp.ones((2, 5)), *prepared.initial_words[:-1])
    changed = list(prepared.initial_words)
    changed[0] = jnp.asarray(changed[0], dtype=jnp.float32)
    with pytest.raises(ValueError, match="shape, dtype or weakness"):
        prepared.concrete(True, jnp.ones((2, 5)), *changed)
    original.rngs.noise.key.value = jax.random.key(23, impl="rbg")
    with pytest.raises(ValueError, match="selected scalar native key"):
        prepare_nnx_call(original)


def test_declared_closure_owner_preserves_body_defaults_and_other_cells() -> None:
    owner, bias = _Streams(), {"offset": 7}

    def original(x: Any, *, deterministic: bool = True, gain: int = 2) -> Any:
        return owner(x * gain, deterministic=deterministic, offset=bias["offset"])

    envelope = NativeClosure.capture(original, "owner")
    prepared = _prepare_nnx_closure_call(original, owner_name="owner")
    fresh = prepared.fresh_original()
    assert fresh.__code__ is original.__code__
    assert fresh.__globals__ is original.__globals__
    assert fresh.__kwdefaults__ is original.__kwdefaults__
    assert fresh.__defaults__ is original.__defaults__
    native = envelope.rebind_owner(_clone(owner))
    x = jnp.ones((2, 5), dtype=jnp.float32)
    result, *_ = prepared.concrete(False, x, *prepared.initial_words, gain=3)
    _exact(result, native(x, deterministic=False, gain=3))
    original.__kwdefaults__ = {"deterministic": False, "gain": 2}
    with pytest.raises(ValueError, match="attributes changed"):
        prepared.concrete(True, x, *prepared.initial_words)


def test_pure_original_callable_keeps_caller_key_defaults_and_output_tree() -> None:
    key = jax.random.key(37)
    token = object()

    def original(x: Any, inference: Any, key: Any = None, model: Any = token) -> Any:
        if model is not token:
            raise ValueError("default binding changed")
        if inference:
            return {"y": x, "extra": (x + 1,)}
        if key is None:
            raise ValueError("the caller must supply a key")
        return {"y": x * jax.random.bernoulli(key, 0.5, x.shape), "extra": (x + 1,)}

    missing = prepare_pure_control_call(original, control_parameter="inference")
    with pytest.raises(ValueError, match="caller must supply"):
        missing.concrete(False, jnp.ones((8,)))
    bound = partial(original, key=key)
    prepared = prepare_pure_control_call(bound, control_parameter="inference")
    assert prepared.original is bound
    assert prepared.initial_words == prepared.state_names == ()
    x = jnp.arange(8, dtype=jnp.float32)
    for flag in (True, False, False, True):
        (actual,) = prepared(x, jnp.asarray(flag))
        _exact(actual, original(x, flag, key=key))
    with pytest.raises(TypeError, match="Python bool"):
        prepared.concrete(jnp.asarray(True), x)
    with pytest.raises(TypeError, match="supplied twice"):
        prepared.concrete(True, x, inference=False)


def test_original_linen_bridge_identity_and_fixed_key_semantics() -> None:
    assert test_utils._LinenToNNXCallable is _LinenToNNXCallable
    # Retain the actual registered bridge owner: its default is handled by
    # _DropoutWithDefault, so lazy initialization and both call controls remain
    # valid without supplying Linen Dropout's parameter twice.
    original = _instantiate(raw_source_case("plugin::linen.dropout", 1)["callable"])
    prepared = prepare_linen_call(original)
    native = _LinenToNNXCallable(_clone(original._model), original._rngs)
    before = _native_words(original._model)
    state = prepared.initial_words
    x = jnp.arange(20, dtype=jnp.float32).reshape(2, 10)
    assert all("__bridge_rngs__" not in name for name in prepared.state_names)
    for flag in (True, False, False, True):
        (actual, *state) = prepared(x, jnp.asarray(flag), *state)
        _exact(actual, native(x, deterministic=flag))
        _exact(tuple(state), _native_words(native._model))
    assert prepared.fresh_original()._rngs is original._rngs
    _exact(_native_words(original._model), before)


def test_native_control_is_explicit_and_never_inferred_from_one_zero_rate_child() -> (
    None
):
    class Source(nnx.Module):
        def __init__(self) -> None:
            self.drop = nnx.Dropout(rate=0.0, rngs=nnx.Rngs(43))

        def __call__(self, x: Any, *, deterministic: bool = True) -> Any:
            y = self.drop(x, deterministic=deterministic)
            return y if deterministic else -y

    source = Source()
    prepared = prepare_nnx_call(source)
    x = jnp.ones((3,), dtype=jnp.float32)
    _exact(prepared(x, jnp.asarray(False), *prepared.initial_words)[0], -x)
    explicit = prepare_native_control_call(source)
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.make_jaxpr(explicit)(x, jnp.asarray(False), *explicit.initial_words)


@pytest.mark.parametrize("kind", ("nnx", "pure-function"))
def test_prepared_callable_supports_direct_jit_first_and_warmed_calls(
    kind: str,
) -> None:
    if kind == "nnx":
        prepared = prepare_nnx_call(_Streams())
        assert prepared.initial_words
    else:

        def source(x: Any, *, deterministic: bool = True) -> Any:
            return {"result": x if deterministic else -x}

        prepared = prepare_pure_control_call(source, control_parameter="deterministic")
        assert prepared.initial_words == ()
    assert hash(prepared) == object.__hash__(prepared)
    assert prepared == prepared and prepared != object()
    compiled = jax.jit(prepared)
    x = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)
    words = prepared.initial_words
    for flag in (True, False, False, True):
        expected = prepared.concrete(flag, x, *words)
        for _ in range(2):
            actual = compiled(x, jnp.asarray(flag), *words)
            _exact(actual, expected)
        words = tuple(expected[1:])


def test_numpy_word_feeds_remain_unchanged_across_concrete_training_and_reset() -> None:
    original = _Streams()
    native = _clone(original)
    prepared = prepare_nnx_call(original)
    words = tuple(np.array(value, copy=True) for value in prepared.initial_words)
    before = tuple(value.tobytes() for value in words)
    owner_before = _native_words(original)
    x = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)
    expected = native(x, deterministic=False)
    first = prepared.concrete(False, x, *words)
    assert tuple(value.tobytes() for value in words) == before
    _exact(first[0], expected)
    _exact(first[1:], _native_words(native))
    reset = prepared.concrete(False, x, *words)
    assert tuple(value.tobytes() for value in words) == before
    _exact(reset, first)
    _exact(_native_words(original), owner_before)


@pytest.mark.parametrize("failure", ("body-error", "undeclared-effect"))
def test_numpy_counter_feeds_survive_source_and_effect_failures(failure: str) -> None:
    class Source(_Streams):
        def __init__(self) -> None:
            super().__init__()
            self.counter = nnx.BatchStat(jnp.asarray(0, dtype=jnp.int32))

        def __call__(self, x: Any, *, deterministic: bool = True) -> Any:
            result = super().__call__(x, deterministic=deterministic)
            if not deterministic:
                if failure == "body-error":
                    raise RuntimeError("original body failed after advancing RNG")
                self.counter.value += 1
            return result

    source = Source()
    prepared = prepare_nnx_call(source)
    words = tuple(np.array(value, copy=True) for value in prepared.initial_words)
    before = tuple(value.tobytes() for value in words)
    owner_before = _native_words(source)
    expected_type = RuntimeError if failure == "body-error" else ValueError
    match = (
        "original body failed"
        if failure == "body-error"
        else "outside the declared mutable"
    )
    with pytest.raises(expected_type, match=match):
        prepared.concrete(False, jnp.ones((2, 5), jnp.float32), *words)
    assert tuple(value.tobytes() for value in words) == before
    _exact(_native_words(source), owner_before)
    assert int(source.counter.value) == 0


@pytest.mark.parametrize("storage", ("fixed-parameter", "selected-counter"))
def test_native_owner_numpy_storage_is_rejected_before_source_execution(
    storage: str,
) -> None:
    calls: list[bool] = []

    class Source(_Streams):
        def __call__(self, x: Any, *, deterministic: bool = True) -> Any:
            calls.append(deterministic)
            return super().__call__(x, deterministic=deterministic)

    source = Source()
    payload = (
        np.asarray(2, dtype=np.float32)
        if storage == "fixed-parameter"
        else np.asarray(0, dtype=np.uint32)
    )
    variable = (
        source.fixed if storage == "fixed-parameter" else source.rngs.dropout.count
    )
    variable.value = payload
    assert variable.value is payload
    before = payload.tobytes()
    with pytest.raises(
        ValueError, match="mutable NumPy storage needs an isolation/effect adapter"
    ):
        prepare_nnx_call(source)
    assert calls == [] and variable.value is payload and payload.tobytes() == before


@onnx_function(name="SourcePreparedWholeCallBoundary")
class _NamedWholeCall(_Streams):
    def __call__(self, x: Any, *, deterministic: bool = True) -> Any:
        return super().__call__(x, deterministic=deterministic)


def _equations(graph: Any) -> Any:
    if isinstance(graph, ClosedJaxpr):
        inner = graph.jaxpr
        if inner is not graph:
            graph = inner
    for equation in graph.eqns:
        yield equation
        for parameter in equation.params.values():
            values = parameter if isinstance(parameter, (tuple, list)) else (parameter,)
            for value in values:
                if isinstance(value, (ClosedJaxpr, Jaxpr)):
                    yield from _equations(value)


def test_prepared_root_nnx_call_retains_its_actual_named_function_boundary() -> None:
    prepared = prepare_nnx_call(_NamedWholeCall())
    specs = [
        jax.ShapeDtypeStruct((2, 5), np.float32),
        jax.ShapeDtypeStruct((), np.bool_),
        *(
            jax.ShapeDtypeStruct(word.shape, word.dtype)
            for word in prepared.initial_words
        ),
    ]
    native = jax.make_jaxpr(lambda *args: prepared(*args))(*specs)
    assert not any(e.primitive.name.startswith("onnx_fn::") for e in _equations(native))
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(lambda *args: prepared(*args))(*specs)
    assert any(e.primitive.name.startswith("onnx_fn::") for e in _equations(activated))
    model = ui._to_onnx(
        prepared, specs, opset=23, precision_policy=PrecisionPolicy.SOURCE
    )
    assert isinstance(model, onnx.ModelProto)
    onnx.checker.check_model(model, full_check=True)
    assert model.functions
    assert any(
        function.name == "SourcePreparedWholeCallBoundary"
        for function in model.functions
    )


@pytest.mark.parametrize("ambient_graph", (True, False))
def test_mutable_selection_uses_canonical_graph_state_under_both_ambient_modes(
    ambient_graph: bool,
) -> None:
    original = _Streams()
    original.count_alias = original.rngs.noise.count
    native = _clone(original)
    x = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)
    expected = native(x, deterministic=False)
    expected_words = _native_words(native)
    explicit_graph = nnx_state._nnx_graph_protocol()
    before = nnx.set_graph_mode.current_value() if explicit_graph else None
    # Older supported Flax always uses graph semantics. Exercise its ordinary
    # native path fully for both stable IDs instead of skipping the test.
    mode = nnx.set_graph_mode(ambient_graph) if explicit_graph else nullcontext()
    with mode:
        if explicit_graph:
            assert nnx.set_graph_mode.current_value() is ambient_graph
        prepared = prepare_nnx_call(original)
        fresh = prepared.fresh_original()
        assert fresh.alias is fresh.rngs
        assert fresh.count_alias is fresh.rngs.noise.count
        actual = prepared.concrete(False, x, *prepared.initial_words)
        if explicit_graph:
            assert nnx.set_graph_mode.current_value() is ambient_graph
    if explicit_graph:
        assert nnx.set_graph_mode.current_value() is before
    _exact(actual[0], expected)
    _exact(actual[1:], expected_words)
    _exact(_native_words(original), prepared.initial_words)


@pytest.mark.parametrize("nested", (False, True))
def test_native_linen_numeric_configuration_is_not_ignored_as_static(
    nested: bool,
) -> None:
    class Declaration(linen.Module):
        payload: Any

        def __call__(self, x: Any) -> Any:
            pytest.fail("declaration validation must not execute the source body")

    payload = jnp.asarray(3, dtype=jnp.float32)
    module = Declaration({"nested": (payload,)} if nested else payload)
    with pytest.raises(ValueError, match="Linen numeric or nonscalar configuration"):
        nnx_state._carrier_fields(module)
    raw = object.__getattribute__(module, "__dict__")
    assert raw["payload"] is module.payload


@pytest.mark.parametrize(
    "hook", ("field-descriptor", "parent-descriptor", "custom-hash")
)
def test_native_linen_unknown_descriptors_and_hash_are_not_executed(hook: str) -> None:
    touched: list[str] = []

    class Declaration(linen.Module):
        rate: float = 0.5

        def __call__(self, x: Any) -> Any:
            pytest.fail("declaration validation must not execute the source body")

    class Descriptor:
        def __get__(self, instance: Any, owner: Any = None) -> Any:
            touched.append("descriptor")
            raise AssertionError("unknown descriptor executed")

    if hook == "custom-hash":

        class CustomHash(linen.Module):
            rate: float = 0.5

            def __hash__(self) -> int:
                touched.append("hash")
                raise AssertionError("unknown hash executed")

        module = CustomHash()
    else:
        module = Declaration()
        setattr(
            Declaration,
            "rate" if hook == "field-descriptor" else "parent",
            Descriptor(),
        )
    with pytest.raises(ValueError, match="descriptor|custom hash"):
        nnx_state._carrier_fields(module)
    assert touched == []


def test_bound_native_linen_declaration_requires_an_explicit_adapter() -> None:
    class Declaration(linen.Module):
        rate: float = 0.5

    module = Declaration()
    object.__setattr__(module, "_parent_ref", object())
    with pytest.raises(ValueError, match="Bound or scoped Linen"):
        nnx_state._carrier_fields(module)
