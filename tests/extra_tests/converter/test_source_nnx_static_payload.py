# tests/extra_tests/converter/test_source_nnx_static_payload.py

"""Pre-invocation containment is distinct from mutable-static support."""

from dataclasses import dataclass, replace
import inspect
from typing import Any

from flax import linen, nnx
from flax.core import FrozenDict
from flax.nnx import graph as nnx_graph, pytreelib
from flax.nnx.bridge import wrappers as nnx_wrappers
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins.flax.nnx._source_function_state import (
    prepare_nnx_function_state,
)


@dataclass(eq=False)
class _HashableCounter:
    count: int = 0


@dataclass(frozen=True)
class _FrozenEnvelope:
    counter: _HashableCounter


class _Owner(nnx.Module):
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        shared = jnp.asarray(1, np.float32)
        self.value = nnx.Variable({"nested": [shared, (shared,)]})
        self.alias = self.value

    def __call__(self, x: Any) -> Any:
        return x


@pytest.mark.parametrize("envelope", ["direct", "tuple", "frozen"])
def test_hashable_mutable_static_counter_rejects_before_body(envelope: str) -> None:
    counter = _HashableCounter()
    payload = (
        counter
        if envelope == "direct"
        else ("config", counter) if envelope == "tuple" else _FrozenEnvelope(counter)
    )
    owner = _Owner(payload)
    variable = owner.value
    raw = variable.get_raw_value()
    nested, pair = raw["nested"], raw["nested"][1]
    shared = nested[0]
    metadata = variable.get_metadata()
    calls = []

    def mutate(receiver: Any, x: Any) -> Any:
        calls.append(receiver)
        target = (
            receiver.payload
            if envelope == "direct"
            else (
                receiver.payload[1] if envelope == "tuple" else receiver.payload.counter
            )
        )
        target.count += 1
        receiver.value[...] = x
        return x

    with pytest.raises(
        SourcePrecisionError, match="mutable static payload support remains unfinished"
    ) as failure:
        prepared = prepare_nnx_function_state(owner, mutate, 27)
        assert prepared is not None
        adapter, initial = prepared
        adapter.invoke((jnp.asarray(2, np.float32),), {}, initial)
    assert failure.value.category == "lowering-defect"
    assert calls == [] and counter.count == 0
    assert owner.payload is payload and owner.value is owner.alias is variable
    assert variable.get_raw_value() is raw and raw["nested"] is nested
    assert nested[1] is pair and nested[0] is pair[0] is shared
    current = variable.get_metadata()
    assert current.keys() == metadata.keys()
    assert all(current[key] is value for key, value in metadata.items())
    np.testing.assert_array_equal(np.asarray(shared), np.asarray(1, np.float32))


def test_rejected_static_payload_is_not_hashed() -> None:
    hashes = []

    class Opaque:
        def __hash__(self) -> int:
            hashes.append(self)
            return 0

    owner = _Owner(Opaque())
    with pytest.raises(SourcePrecisionError, match="GraphDef.Static payload requires"):
        prepare_nnx_function_state(owner, type(owner).__call__, 28)
    assert hashes == []


@jax.tree_util.register_pytree_node_class
class _AuxiliaryPayload:
    def __init__(self, value: Any, metadata: Any) -> None:
        self.value = value
        self.metadata = metadata

    def tree_flatten(self) -> Any:
        # A rejected flatten hook must not get a chance to modify its owner.
        self.metadata["tags"].append("flatten")
        return (self.value,), self.metadata

    @classmethod
    def tree_unflatten(cls, metadata: Any, values: Any) -> Any:
        metadata["tags"].append("unflatten")
        return cls(values[0], metadata)


def test_mutable_pytree_auxiliary_rejects_before_body() -> None:
    metadata = {"tags": []}
    payload = _AuxiliaryPayload(jnp.asarray(1, np.float32), metadata)
    owner = _Owner("safe")
    owner.value.set_raw_value(payload)
    calls = []

    def mutate(receiver: Any) -> Any:
        calls.append(receiver)
        receiver.value.get_raw_value().metadata["tags"].append("changed")
        return receiver.value.get_raw_value().value

    with pytest.raises(
        SourcePrecisionError, match="custom PyTree flatten/reconstruction hooks"
    ):
        prepared = prepare_nnx_function_state(owner, mutate, 29)
        assert prepared is not None
        adapter, initial = prepared
        adapter.invoke((), {}, initial)
    assert calls == [] and metadata == {"tags": []}
    assert owner.value is owner.alias
    assert owner.value.get_raw_value() is payload and payload.metadata is metadata


@pytest.mark.parametrize("cached_only", [False, True], ids=["live-hook", "cached-hook"])
def test_custom_graph_flatten_rejects_without_running_hook(
    cached_only: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class Unsupported(nnx.Module):
        def __init__(self) -> None:
            self.count = 0

        def _graph_node_flatten(self) -> Any:
            calls.append(self)
            self.count += 1
            return super()._graph_node_flatten()

    owner = Unsupported()
    if cached_only:
        # NNX registered the callback when the class was defined. Restoring its
        # current attribute cannot retroactively make that callback safe.
        monkeypatch.setattr(
            Unsupported, "_graph_node_flatten", pytreelib.Pytree._graph_node_flatten
        )
    with pytest.raises(SourcePrecisionError, match="custom NNX graph callbacks"):
        prepare_nnx_function_state(owner, lambda self: self.count, 33)
    assert calls == [] and owner.count == 0


@pytest.mark.parametrize("metadata", ["list", "callback"])
def test_mutable_or_callable_variable_metadata_rejects_before_body(
    metadata: str,
) -> None:
    owner = _Owner("safe")
    calls = []
    payload = [] if metadata == "list" else lambda value: calls.append(value)
    owner.value.set_metadata(witness=payload)
    raw = owner.value.get_raw_value()

    def operation(receiver: Any) -> Any:
        calls.append(receiver)
        return receiver.value.get_raw_value()["nested"][0]

    with pytest.raises(SourcePrecisionError, match="Variable metadata requires"):
        prepared = prepare_nnx_function_state(owner, operation, 34)
        assert prepared is not None
        adapter, state = prepared
        adapter.invoke((), {}, state)
    assert calls == [] and owner.value.get_raw_value() is raw
    assert owner.value.get_metadata()["witness"] is payload


@pytest.mark.parametrize("ingress", ["original", "reconstruct", "invoke"])
def test_native_mutable_numeric_buffer_rejects_before_source_can_modify_it(
    ingress: str,
) -> None:
    owner = _Owner("safe")
    buffer = np.asarray([1, 2], np.float32)
    owner.value.set_raw_value(buffer if ingress == "original" else jnp.asarray(buffer))
    original = owner.value.get_raw_value()
    calls = []

    def operation(receiver: Any) -> Any:
        calls.append(receiver)
        receiver.value.get_raw_value()[...] = 0
        return receiver.value.get_raw_value()

    with pytest.raises(SourcePrecisionError, match="mutable native numeric buffers"):
        prepared = prepare_nnx_function_state(owner, operation, 35)
        assert prepared is not None
        adapter, state = prepared
        changed = (buffer,)
        assert len(state) == 1
        if ingress == "reconstruct":
            adapter.reconstruct(changed)
        else:
            adapter.invoke((), {}, changed)
    assert calls == [] and owner.value.get_raw_value() is original
    np.testing.assert_array_equal(buffer, np.asarray([1, 2], np.float32))


def test_safe_immutable_configuration_and_metadata_remain_supported() -> None:
    payload = ("scale", (1, 2.0, None), frozenset({"a", "b"}))
    owner = _Owner(payload)
    witness = (("tags", ("first", "second")),)
    owner.value.set_metadata(witness=witness)
    calls = []

    def operation(receiver: Any, x: Any) -> Any:
        calls.append(receiver)
        assert receiver.payload is payload
        assert receiver.value.get_metadata()["witness"] is witness
        return x + receiver.value.get_raw_value()["nested"][0]

    prepared = prepare_nnx_function_state(owner, operation, 30)
    assert prepared is not None
    adapter, state = prepared
    result, updated, effects = adapter.invoke((jnp.asarray(3, np.float32),), {}, state)
    assert len(calls) == 1 and calls[0] is not owner
    assert effects == () and all(
        after is before for before, after in zip(state, updated, strict=True)
    )
    assert np.asarray(result).dtype == np.float32 and np.asarray(result).shape == ()
    np.testing.assert_array_equal(np.asarray(result), np.asarray(4, np.float32))


@pytest.mark.parametrize("payload_location", ["static", "metadata"])
def test_delayed_descriptor_rechecks_new_mutable_payload_before_body(
    payload_location: str,
) -> None:
    owner = _Owner("safe")
    calls = []

    def operation(receiver: Any) -> Any:
        calls.append(receiver)
        return receiver.value.get_raw_value()["nested"][0]

    prepared = prepare_nnx_function_state(owner, operation, 31)
    assert prepared is not None
    adapter, state = prepared
    witness = {"tags": []}
    if payload_location == "static":
        # Native GraphDef has frozen fields around an exposed live outer list.
        adapter.graphdef.attributes.append(("late_payload", nnx_graph.Static(witness)))
    else:
        node = next(
            item
            for item in adapter.graphdef.nodes
            if type(item) is nnx_graph.VariableDef
        )
        node.metadata._mapping["late_payload"] = witness
    with pytest.raises(
        SourcePrecisionError, match="mutable static payload support remains unfinished"
    ):
        adapter.invoke((), {}, state)
    assert calls == [] and witness == {"tags": []}


def test_native_mutable_classification_mapping_is_reconstructed_without_aliasing() -> (
    None
):
    owner = _Owner("safe")
    original = owner._pytree__nodes
    original_items = tuple(original.items())

    def operation(receiver: Any) -> Any:
        # NNX itself treats this mapping as replaceable immutable metadata, but
        # the actual envelope contains an ordinary mutable dict.
        receiver._pytree__nodes._mapping["payload"] = True
        return receiver.value.get_raw_value()["nested"][0]

    prepared = prepare_nnx_function_state(owner, operation, 32)
    assert prepared is not None
    adapter, state = prepared
    first, second = adapter.reconstruct(state), adapter.reconstruct(state)
    assert first._pytree__nodes is not original
    assert second._pytree__nodes is not original
    assert first._pytree__nodes is not second._pytree__nodes
    # Make the mutation directly on the reconstructed owner: topology checks
    # deliberately reject it after the method, but the caller must stay intact.
    operation(first)
    assert owner._pytree__nodes is original
    assert tuple(original.items()) == original_items
    assert tuple(second._pytree__nodes.items()) == original_items


@pytest.mark.parametrize(
    "configuration",
    [
        np.float32,
        jnp.float32,
        np.dtype("float32"),
        jnp.bfloat16,
        np.dtype(jnp.bfloat16),
        jax.lax.Precision.HIGH,
        (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST),
    ],
)
def test_finite_native_scalar_configuration_preserves_actual_anchor(
    configuration: Any,
) -> None:
    owner = _Owner(configuration)
    prepared = prepare_nnx_function_state(owner, type(owner).__call__, 36)
    assert prepared is not None
    adapter, state = prepared
    clone = adapter.reconstruct(state)
    if type(configuration) is tuple:
        assert all(a is b for a, b in zip(clone.payload, configuration, strict=True))
    else:
        assert clone.payload is configuration


def test_numpy_and_jax_dtype_constructors_keep_distinct_descriptor_signatures() -> None:
    def operation(receiver: Any) -> Any:
        return receiver.payload

    signatures = []
    for constructor in (np.float32, jnp.float32):
        prepared = prepare_nnx_function_state(_Owner(constructor), operation, 37)
        assert prepared is not None
        signatures.append(prepared[0].signature)
    assert signatures[0] != signatures[1]


@pytest.mark.parametrize("kind", ["linear", "linear-general", "batch-norm"])
def test_reviewed_native_callable_roles_preserve_pure_module_behavior(
    kind: str,
) -> None:
    if kind == "linear":
        owner = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    elif kind == "linear-general":
        owner = nnx.LinearGeneral(2, 2, rngs=nnx.Rngs(0))
    else:
        owner = nnx.BatchNorm(2, use_running_average=True, rngs=nnx.Rngs(0))
    inputs = jnp.asarray([[1, 2], [3, 4]], np.float32)
    expected = owner(inputs)
    prepared = prepare_nnx_function_state(owner, lambda self, x: self(x), 38)
    assert prepared is not None
    adapter, state = prepared
    actual, _, effects = adapter.invoke((inputs,), {}, state)
    assert effects == ()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_native_callable_identity_does_not_authorize_a_foreign_field() -> None:
    with pytest.raises(SourcePrecisionError, match="GraphDef.Static payload requires"):
        owner = _Owner(jax.lax.dot_general)
        prepare_nnx_function_state(owner, type(owner).__call__, 39)


@pytest.mark.parametrize("field", ["dot_general", "promote_dtype"])
def test_native_owner_custom_callback_rejects_before_running_it(field: str) -> None:
    owner = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    calls = []

    def custom(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return args[0]

    setattr(owner, field, custom)
    with pytest.raises(SourcePrecisionError, match="GraphDef.Static payload requires"):
        prepare_nnx_function_state(owner, lambda self, x: self(x), 40)
    assert calls == [] and getattr(owner, field) is custom


@pytest.mark.parametrize("anchor", ["dtype", "precision", "callable"])
def test_delayed_native_anchor_revalidation_rejects_added_mutable_payload(
    anchor: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    owner.precision = jax.lax.Precision.HIGH
    calls = []

    def operation(receiver: Any) -> Any:
        calls.append(receiver)
        return receiver.kernel[...]

    prepared = prepare_nnx_function_state(owner, operation, 41)
    assert prepared is not None
    adapter, state = prepared
    target = (
        jnp.float32
        if anchor == "dtype"
        else jax.lax.Precision.HIGH if anchor == "precision" else owner.dot_general
    )
    with monkeypatch.context() as context:
        context.setattr(target, "_unsupported_payload", [], raising=False)
        with pytest.raises(SourcePrecisionError, match="native .* changed"):
            adapter.invoke((), {}, state)
    assert calls == []


def test_audited_linen_declaration_rebuilds_mutable_native_envelope_and_aliases() -> (
    None
):
    class Declaration(linen.Module):
        scale: float = 0.5

        def __call__(self, x: Any) -> Any:
            return x * self.scale

    declaration = Declaration()
    owner = _Owner(declaration)
    owner.declaration_alias = declaration
    raw = object.__getattribute__(declaration, "__dict__")
    native_state, identifier = raw["_state"], raw["_id"]
    original_id = identifier.id
    children, cursor = native_state.children, native_state.autoname_cursor

    def mutate(receiver: Any, x: Any) -> Any:
        assert receiver.payload is receiver.declaration_alias
        receiver.payload._state.children["changed"] = object()
        receiver.payload._id.id += 1
        return x

    prepared = prepare_nnx_function_state(owner, mutate, 42)
    assert prepared is not None
    adapter, state = prepared
    clone = adapter.reconstruct(state)
    assert clone.payload is clone.declaration_alias and clone.payload is not declaration
    cloned_raw = object.__getattribute__(clone.payload, "__dict__")
    assert cloned_raw["_state"] is not native_state
    assert cloned_raw["_id"] is not identifier and cloned_raw["_id"].id == original_id
    with pytest.raises(SourcePrecisionError, match="Only pristine unbound Linen"):
        adapter.invoke((jnp.asarray(1, np.float32),), {}, state)
    assert owner.payload is owner.declaration_alias is declaration
    assert raw["_state"] is native_state and raw["_id"] is identifier
    assert native_state.children is children and native_state.autoname_cursor is cursor
    assert children == {} and cursor == {} and identifier.id == original_id


@pytest.mark.parametrize("hook", ["graph-registration", "variable-constructor"])
def test_delayed_native_reconstruction_rejects_replaced_hook_before_execution(
    hook: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _Owner("safe")

    class PlainVariable(nnx.Variable):
        pass

    owner.value = PlainVariable(owner.value.get_raw_value())
    owner.alias = owner.value
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise AssertionError("unsupported native reconstruction hook executed")

    prepared = prepare_nnx_function_state(owner, lambda self: self.value[...], 43)
    assert prepared is not None
    adapter, state = prepared
    with monkeypatch.context() as context:
        if hook == "graph-registration":
            implementation = nnx_graph.GRAPH_REGISTRY[type(owner)]
            context.setitem(
                nnx_graph.GRAPH_REGISTRY,
                type(owner),
                replace(implementation, create_empty=forbidden),
            )
        else:
            context.setattr(PlainVariable, "_new", classmethod(forbidden))
        with pytest.raises(
            SourcePrecisionError, match="custom .* (callbacks|storage/hooks)"
        ):
            adapter.invoke((), {}, state)
    assert calls == [] and owner.value is owner.alias


@pytest.mark.parametrize("batched", [False, True], ids=["empty", "nonempty"])
def test_native_linear_general_batch_axes_preserve_behavior(batched: bool) -> None:
    axes = {0: 2} if batched else {}
    owner = nnx.LinearGeneral(2, 3, batch_axis=axes, rngs=nnx.Rngs(0))
    shape = (2, 3, 2) if batched else (3, 2)
    inputs = jnp.asarray(np.arange(np.prod(shape)).reshape(shape) + 1, np.float32)
    expected = owner(inputs)
    original = owner.batch_axis
    raw = object.__getattribute__(original, "_dict")
    hash(original)
    prepared = prepare_nnx_function_state(owner, lambda self, x: self(x), 44)
    assert prepared is not None
    adapter, state = prepared
    clone = adapter.reconstruct(state)
    assert clone.batch_axis is not original
    assert object.__getattribute__(clone.batch_axis, "_dict") is not raw
    assert object.__getattribute__(clone.batch_axis, "_hash") is None
    actual, _, effects = adapter.invoke((inputs,), {}, state)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert effects == () and owner.batch_axis is original
    assert object.__getattribute__(original, "_dict") is raw and raw == axes


def test_native_batch_axes_preserve_shared_envelope_aliases_and_original_storage() -> (
    None
):
    class Pair(nnx.Module):
        def __init__(self) -> None:
            shared = FrozenDict({0: 2})
            self.left = nnx.LinearGeneral(2, 2, batch_axis={0: 2}, rngs=nnx.Rngs(0))
            self.right = nnx.LinearGeneral(2, 2, batch_axis={0: 2}, rngs=nnx.Rngs(1))
            self.left.batch_axis = self.right.batch_axis = shared

    owner = Pair()
    original = owner.left.batch_axis
    raw = object.__getattribute__(original, "_dict")
    before = tuple(raw.items())
    calls = []

    def mutate(receiver: Any) -> Any:
        calls.append(receiver)
        assert receiver.left.batch_axis is receiver.right.batch_axis
        object.__getattribute__(receiver.left.batch_axis, "_dict")[1] = 3
        return receiver.left.kernel[...]

    prepared = prepare_nnx_function_state(owner, mutate, 45)
    assert prepared is not None
    adapter, state = prepared
    with pytest.raises(SourcePrecisionError, match="changed graph topology"):
        adapter.invoke((), {}, state)
    assert len(calls) == 1 and calls[0] is not owner
    assert owner.left.batch_axis is owner.right.batch_axis is original
    assert object.__getattribute__(original, "_dict") is raw
    assert tuple(raw.items()) == before


@pytest.mark.parametrize("payload", [[1], object(), True])
def test_native_batch_axes_reject_noninteger_payload_before_body(payload: Any) -> None:
    owner = nnx.LinearGeneral(2, 2, rngs=nnx.Rngs(0))
    original = FrozenDict({0: payload})
    owner.batch_axis = original
    raw = object.__getattribute__(original, "_dict")
    calls = []

    def operation(receiver: Any) -> Any:
        calls.append(receiver)
        return receiver.kernel[...]

    with pytest.raises(SourcePrecisionError, match="batch_axis needs exact integer"):
        prepared = prepare_nnx_function_state(owner, operation, 46)
        assert prepared is not None
        adapter, state = prepared
        adapter.invoke((), {}, state)
    assert calls == [] and owner.batch_axis is original
    assert object.__getattribute__(original, "_dict") is raw and raw[0] is payload


def test_frozen_dict_is_not_a_general_static_payload_exception() -> None:
    owner = _Owner(FrozenDict({}))
    with pytest.raises(SourcePrecisionError, match="GraphDef.Static payload requires"):
        prepare_nnx_function_state(owner, type(owner).__call__, 47)


def test_native_tonnx_fallback_retains_pure_linen_method_dispatch() -> None:
    class Declaration(linen.Module):
        scale: float = 0.5

        def __call__(self, x: Any) -> Any:
            return x * self.scale

        def twice(self, x: Any) -> Any:
            return self(x) + self(x)

    declaration = Declaration()
    owner = nnx_wrappers.ToNNX(declaration, rngs=None)
    raw = object.__getattribute__(declaration, "__dict__")
    native_state = raw["_state"]
    inputs = jnp.asarray([1, 2], np.float32)
    expected = owner.twice(inputs)
    prepared = prepare_nnx_function_state(owner, lambda self, x: self.twice(x), 48)
    assert prepared is not None
    adapter, state = prepared
    actual, _, effects = adapter.invoke((inputs,), {}, state)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert effects == () and owner.to_nnx__module is declaration
    assert raw["_state"] is native_state and native_state.children == {}


@pytest.mark.parametrize("native", [nnx.Rngs, nnx_wrappers.ToNNX])
def test_native_fallback_alias_does_not_authorize_another_owner(native: Any) -> None:
    class Foreign(nnx.Module):
        __getattr__ = inspect.getattr_static(native, "__getattr__")

    owner = object.__new__(Foreign)
    object.__getattribute__(owner, "__dict__").update(vars(_Owner("safe")))
    with pytest.raises(SourcePrecisionError, match="custom NNX attribute fallback"):
        prepare_nnx_function_state(owner, lambda self: (), 49)


@pytest.mark.parametrize("field", ["to_nnx__module", "to_nnx__rngs"])
def test_native_tonnx_rejects_unknown_wrapped_storage_before_body(field: str) -> None:
    class Declaration(linen.Module):
        pass

    owner = nnx_wrappers.ToNNX(Declaration(), rngs=None)
    unsupported = object()
    object.__setattr__(owner, field, unsupported)
    calls = []

    def operation(receiver: Any) -> Any:
        calls.append(receiver)
        return ()

    with pytest.raises(SourcePrecisionError, match="ToNNX needs"):
        prepared = prepare_nnx_function_state(owner, operation, 50)
        assert prepared is not None
        adapter, state = prepared
        adapter.invoke((), {}, state)
    assert calls == [] and object.__getattribute__(owner, field) is unsupported


@pytest.mark.parametrize("ingress", ["initial", "delayed"])
@pytest.mark.parametrize("hook", ["__get__", "__set__", "__delete__"])
def test_carrier_descriptor_hooks_reject_before_reconstruction_or_body(
    ingress: str, hook: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    @dataclass
    class Carrier:
        value: Any
        label: int = 1

    carrier = Carrier(jnp.asarray(1, np.float32))
    owner = _Owner(carrier)
    calls = []

    def forbidden(*args: Any) -> Any:
        calls.append(args)
        raise AssertionError("unsupported carrier descriptor executed")

    def operation(receiver: Any) -> Any:
        calls.append(receiver)
        return receiver.payload.value

    prepared = (
        prepare_nnx_function_state(owner, operation, 51)
        if ingress == "delayed"
        else None
    )
    descriptor = type("UnsupportedDescriptor", (), {hook: forbidden})()
    with monkeypatch.context() as context:
        context.setattr(Carrier, "value", descriptor, raising=False)
        with pytest.raises(SourcePrecisionError, match="carrier field descriptor"):
            if ingress == "initial":
                prepared = prepare_nnx_function_state(owner, operation, 51)
            assert prepared is not None
            adapter, state = prepared
            adapter.invoke((), {}, state)
    assert calls == [] and owner.payload is carrier
    np.testing.assert_array_equal(np.asarray(carrier.value), np.asarray(1, np.float32))


def test_delayed_carrier_static_metadata_rejects_nonliteral_before_body() -> None:
    @dataclass
    class Carrier:
        value: Any
        label: int = 1

    carrier = Carrier(jnp.asarray(1, np.float32))
    owner = _Owner(carrier)
    calls = []

    def operation(receiver: Any) -> Any:
        calls.append(receiver)
        return receiver.payload.value

    prepared = prepare_nnx_function_state(owner, operation, 52)
    assert prepared is not None
    adapter, state = prepared
    spec = adapter.carriers[0]
    label = next(item for item in spec.fields if item.name == "label")
    assert label.static is not None
    unsafe = replace(label, static=replace(label.static, value=[]))
    changed = replace(
        spec, fields=tuple(unsafe if item is label else item for item in spec.fields)
    )
    adapter = replace(adapter, carriers=(changed,))
    with pytest.raises(
        SourcePrecisionError, match="nonliteral or nested carrier metadata"
    ):
        adapter.invoke((), {}, state)
    assert calls == [] and owner.payload is carrier and carrier.label == 1


def test_plain_callback_snapshot_preserves_aliases_and_detaches_function_memory() -> (
    None
):
    increment = 2

    def callback(x: Any) -> Any:
        return x + increment

    callback.tag = "original"
    owner = _Owner(callback)
    owner.callback_alias = callback
    prepared = prepare_nnx_function_state(owner, lambda self, x: self.payload(x), 53)
    assert prepared is not None
    adapter, state = prepared
    assert adapter.callbacks is not None
    hash(adapter.signature)
    clone = adapter.reconstruct(state)
    assert clone.payload is clone.callback_alias and clone.payload is not callback
    assert clone.payload.__globals__ is callback.__globals__
    assert clone.payload.__dict__ is not callback.__dict__
    assert clone.payload.__closure__[0] is not callback.__closure__[0]
    actual, updated, effects = adapter.invoke((jnp.asarray(3, np.float32),), {}, state)
    assert actual.dtype == np.float32 and actual.shape == ()
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(5, np.float32))
    assert effects == () and all(a is b for a, b in zip(updated, state, strict=True))
    assert (
        owner.payload is owner.callback_alias is callback and callback.tag == "original"
    )


def test_callback_snapshot_preserves_shared_metadata_between_distinct_roots() -> None:
    def first(x: Any) -> Any:
        return x

    def second(x: Any) -> Any:
        return x + 1

    shared = {"tag": "original"}
    first.__dict__ = second.__dict__ = shared
    owner = _Owner(first)
    owner.second_callback = second
    prepared = prepare_nnx_function_state(owner, lambda self, x: self.payload(x), 54)
    assert prepared is not None
    adapter, state = prepared
    clone = adapter.reconstruct(state)
    assert clone.payload is not first and clone.second_callback is not second
    assert clone.payload.__dict__ is clone.second_callback.__dict__
    assert clone.payload.__dict__ is not shared
    clone.payload.__dict__["tag"] = "changed"
    assert shared == {"tag": "original"}


@pytest.mark.parametrize("effect", ["attribute", "cell"])
def test_callback_owned_effects_reject_without_mutating_original_memory(
    effect: str,
) -> None:
    count = 0

    def callback(x: Any) -> Any:
        nonlocal count
        count += 1
        return x

    callback.calls = 0
    owner = _Owner(callback)
    variable, raw = owner.value, owner.value.get_raw_value()

    def operation(receiver: Any, x: Any) -> Any:
        if effect == "attribute":
            receiver.payload.calls += 1
            return x
        return receiver.payload(x)

    prepared = prepare_nnx_function_state(owner, operation, 55)
    assert prepared is not None
    adapter, state = prepared
    with pytest.raises(
        SourcePrecisionError, match="callback effects need explicit state/update"
    ):
        adapter.invoke((jnp.asarray(1, np.float32),), {}, state)
    assert count == 0 and callback.calls == 0 and owner.payload is callback
    assert owner.value is owner.alias is variable and variable.get_raw_value() is raw


@pytest.mark.parametrize("binding", ["attribute", "cell"])
def test_delayed_original_callback_mutation_rejects_before_source_body(
    binding: str,
) -> None:
    offset = 1

    def callback(x: Any) -> Any:
        return x + offset

    callback.tag = "original"
    owner = _Owner(callback)
    calls = []

    def operation(receiver: Any, x: Any) -> Any:
        calls.append(receiver)
        return receiver.payload(x)

    prepared = prepare_nnx_function_state(owner, operation, 56)
    assert prepared is not None
    adapter, state = prepared
    if binding == "attribute":
        callback.tag = "changed"
    else:
        callback.__closure__[0].cell_contents = 2
    with pytest.raises(SourcePrecisionError, match="prepared callback .* changed"):
        adapter.invoke((jnp.asarray(1, np.float32),), {}, state)
    assert calls == [] and owner.payload is callback


@pytest.mark.parametrize("placement", ["defaults", "kwdefaults", "attributes"])
def test_callback_nested_mutable_payload_rejects_before_source_body(
    placement: str,
) -> None:
    def callback(x: Any, amount: Any = 1, *, scale: Any = 1) -> Any:
        return x * scale + amount

    payload = []
    if placement == "defaults":
        callback.__defaults__ = (payload,)
    elif placement == "kwdefaults":
        callback.__kwdefaults__ = {"scale": payload}
    else:
        callback.mutable_payload = payload
    owner = _Owner(callback)
    calls = []

    def operation(receiver: Any, x: Any) -> Any:
        calls.append(receiver)
        return receiver.payload(x)

    with pytest.raises(
        SourcePrecisionError,
        match="GraphDef.Static payload requires a supported callback",
    ):
        prepared = prepare_nnx_function_state(owner, operation, 57)
        assert prepared is not None
        adapter, state = prepared
        adapter.invoke((jnp.asarray(1, np.float32),), {}, state)
    assert calls == [] and payload == [] and owner.payload is callback


def test_prepared_managed_callback_retains_role_after_patch_scope_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the descriptor directly; this does not qualify a live Jaxpr."""
    from jax2onnx.converter.precision_policy import (
        PrecisionPolicy,
        precision_policy_scope,
    )
    from jax2onnx.plugins import plugin_system

    def original(x: Any) -> Any:
        return x + 1

    @dataclass
    class Carrier:
        value: Any

    qualified = f"{__name__}.retained_nnx_callback_role"
    plugin = plugin_system.FunctionPlugin(f"onnx_fn::{qualified}", original)
    monkeypatch.setitem(
        plugin_system.ONNX_FUNCTION_PLUGIN_REGISTRY, plugin._qualified_target, plugin
    )
    patch_key = (__name__, "retained_nnx_callback_role")
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        wrapper = plugin._make_patch_fn(plugin.primitive, is_class=False)(original)
        owner = _Owner(wrapper)
        owner.carrier = Carrier(jnp.asarray(2, np.float32))
        with monkeypatch.context() as active:
            active.setitem(
                plugin_system._PATCH_STATE,
                patch_key,
                {
                    "orig": original,
                    "patched": wrapper,
                    "owner": plugin,
                    "policy": PrecisionPolicy.SOURCE,
                },
            )
            prepared = prepare_nnx_function_state(
                owner, lambda self, x: self.payload(x), 58
            )
            assert prepared is not None
            adapter, state = prepared
    assert patch_key not in plugin_system._PATCH_STATE
    raw = owner.value.get_raw_value()
    clone = adapter.reconstruct(state)
    assert clone.payload is not wrapper
    assert all(a is b for a, b in zip(adapter.current_state(owner), state, strict=True))
    assert adapter.ports_for_filter(owner, nnx.Variable)
    native_effects = tuple(
        index
        for index in range(adapter.state_count)
        if index not in adapter._foreign_ports()
    )
    adapter.with_effects(native_effects).validate_owner(owner)
    assert owner.value.get_raw_value() is raw
    token = plugin_system._IN_FUNCTION_BUILD.set({plugin.name})
    try:
        actual, updated, effects = adapter.invoke(
            (jnp.asarray(3, np.float32),), {}, state
        )
    finally:
        plugin_system._IN_FUNCTION_BUILD.reset(token)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(4, np.float32))
    assert effects == () and all(a is b for a, b in zip(updated, state, strict=True))
    assert owner.payload is wrapper and owner.value.get_raw_value() is raw
