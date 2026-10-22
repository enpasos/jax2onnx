# jax2onnx/plugins/flax/nnx/_source_function_state.py

"""Public NNX state lifting for the internal SOURCE Function binder."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
import inspect
import marshal
from types import GetSetDescriptorType, MemberDescriptorType
from typing import Any, Callable, Mapping, cast

from flax import errors, nnx
from flax.nnx import graph as nnx_graph
import jax
import jax.tree_util as jtu
import numpy as np

from jax2onnx._compat.jax import ShapedArray, Tracer
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins.flax.nnx._source_linen_declaration import (
    is_static_linen_declaration,
)


def _failure(message: str, *, defect: bool = False) -> SourcePrecisionError:
    return SourcePrecisionError(
        message,
        phase="profile-validation",
        category="lowering-defect" if defect else "profile-exclusion",
        location="NNX Function state",
        requirement="fixed NNX topology and explicitly carried numeric state owned by the current trace",
        action="prepare a fresh NNX owner with split/merge inside the enclosing traced call",
    )


_NNX_GRAPH_SIGNATURES: tuple[
    tuple[tuple[str, inspect._ParameterKind, object], ...], ...
] = tuple(
    tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    )
    for function in (nnx.split, nnx.iter_graph)
)


def _nnx_graph_protocol() -> bool:
    """Select graph-only or explicit graph mode from exact native API shapes."""
    parameter = inspect.Parameter
    if not (
        type(_NNX_GRAPH_SIGNATURES) is tuple
        and len(_NNX_GRAPH_SIGNATURES) == 2
        and all(type(signature) is tuple for signature in _NNX_GRAPH_SIGNATURES)
        and all(
            type(item) is tuple
            and len(item) == 3
            and type(item[0]) is str
            and any(
                item[1] is kind
                for kind in (
                    parameter.POSITIONAL_ONLY,
                    parameter.POSITIONAL_OR_KEYWORD,
                    parameter.VAR_POSITIONAL,
                    parameter.KEYWORD_ONLY,
                )
            )
            and (item[2] is parameter.empty or item[2] is None)
            for signature in _NNX_GRAPH_SIGNATURES
            for item in signature
        )
    ):
        raise _failure("NNX graph API schema needs an exact adapter", defect=True)
    split = (
        ("node", parameter.POSITIONAL_OR_KEYWORD, parameter.empty),
        ("filters", parameter.VAR_POSITIONAL, parameter.empty),
    )
    iteration = (("node", parameter.POSITIONAL_ONLY, parameter.empty),)
    graph = ("graph", parameter.KEYWORD_ONLY, None)
    if _NNX_GRAPH_SIGNATURES == (split, iteration):
        return False
    if _NNX_GRAPH_SIGNATURES == ((*split, graph), (*iteration, graph)):
        return True
    raise _failure("NNX graph API schema needs an exact adapter", defect=True)


def _split_graph(owner: Any) -> Any:
    if _nnx_graph_protocol():
        return nnx.split(owner, graph=True)
    return nnx.split(owner)


def _iter_graph(owner: Any) -> Any:
    if _nnx_graph_protocol():
        return nnx.iter_graph(owner, graph=True)
    return nnx.iter_graph(owner)


def _nnx_variable_roles() -> tuple[str, str]:
    fields = nnx.Variable.required_metadata
    if type(fields) is frozenset and all(type(name) is str for name in fields):
        if fields == frozenset({"has_ref", "is_hijax", "is_mutable", "eager_sharding"}):
            return "has_ref", "is_hijax"
        if fields == frozenset({"ref", "hijax", "eager_sharding"}):
            return "ref", "hijax"
    raise _failure("NNX Variable metadata schema needs an exact adapter", defect=True)


def _require_ordinary_variable_metadata(metadata: Any) -> None:
    roles = _nnx_variable_roles()
    if not (
        type(metadata) is dict
        and all(type(name) is str for name in metadata)
        and nnx.Variable.required_metadata <= metadata.keys()
        and all(type(metadata[role]) is bool for role in roles)
    ):
        raise _failure("NNX Variable metadata is malformed", defect=True)
    if any(metadata[role] for role in roles):
        raise _failure("Mutable NNX Function state requires ordinary non-Ref Variables")


@dataclass(frozen=True)
class _StateSpec:
    shape: tuple[Any, ...]
    dtype: Any
    weak_type: bool

    @classmethod
    def of(cls, value: Any) -> _StateSpec:
        aval = jax.typeof(value)
        if not isinstance(aval, ShapedArray):
            raise _failure(
                "NNX Function state must contain ordinary numeric JAX values"
            )
        return cls(tuple(aval.shape), aval.dtype, bool(aval.weak_type))


@dataclass(frozen=True)
class _StateEntry:
    path: tuple[Any, ...]
    tree: jtu.PyTreeDef
    start: int
    stop: int


def _foreign_failure(message: str) -> SourcePrecisionError:
    return _failure(
        f"NNX foreign numeric carrier support remains unfinished: {message}",
        defect=True,
    )


@dataclass(frozen=True)
class _CarrierToken:
    index: int


@dataclass(frozen=True)
class _StaticField:
    value_type: type[Any]
    fingerprint: bytes
    value: Any = field(compare=False, hash=False)


@dataclass(frozen=True)
class _CarrierField:
    name: str
    port: int | None
    static: _StaticField | None


@dataclass(frozen=True)
class _CarrierSpec:
    carrier_type: type[Any]
    fields: tuple[_CarrierField, ...]


def _numeric_carrier_field(value: Any) -> bool:
    return isinstance(
        value, (jax.Array, Tracer, jax.ShapeDtypeStruct, np.ndarray, np.generic)
    )


def _static_carrier_field(value: Any) -> _StaticField:
    if type(value) not in {type(None), bool, int, float, complex, str, bytes}:
        raise _foreign_failure("nonliteral or nested carrier metadata needs an adapter")
    return _StaticField(type(value), marshal.dumps(value), value)


def _carrier_fields(value: Any) -> tuple[tuple[str, Any], ...] | None:
    if is_static_linen_declaration(value, failure=_foreign_failure):
        return None
    if isinstance(value, (type, nnx.Module)) or not is_dataclass(value):
        return None
    declared = fields(value)
    names = tuple(field.name for field in declared)
    # Only ordinary actual dataclass fields are read. Descriptors/custom field
    # accessors require their own adapter; no header or constructor is evaluated.
    for name in names:
        descriptor = inspect.getattr_static(type(value), name, None)
        if (
            hasattr(type(descriptor), "__get__")
            and type(descriptor) is not MemberDescriptorType
        ):
            raise _foreign_failure("carrier field descriptor needs an adapter")
    values = tuple((name, object.__getattribute__(value, name)) for name in names)
    if not any(_numeric_carrier_field(item) for _, item in values):
        return None
    if type(type(value)) is not type or any(
        base is not object and not is_dataclass(base)
        for base in cast(tuple[type[Any], ...], type(value).__mro__)
    ):
        raise _foreign_failure("nonordinary dataclass inheritance needs an adapter")
    if any(
        name in vars(base)
        for base in cast(tuple[type[Any], ...], type(value).__mro__)
        if base is not object
        for name in ("__getattribute__", "__getattr__")
    ):
        raise _foreign_failure("custom dataclass field access needs an adapter")
    dictionary_descriptor = inspect.getattr_static(type(value), "__dict__", None)
    if type(dictionary_descriptor) not in {
        type(None),
        GetSetDescriptorType,
        MemberDescriptorType,
    }:
        raise _foreign_failure("custom dataclass storage descriptor needs an adapter")
    try:
        dictionary = object.__getattribute__(value, "__dict__")
    except AttributeError:
        dictionary = None
    if dictionary is not None and (
        type(dictionary) is not dict or set(dictionary) != set(names)
    ):
        raise _foreign_failure("carrier has state outside its declared fields")
    for _, item in values:
        if _numeric_carrier_field(item):
            if isinstance(item, np.ndarray):
                raise _foreign_failure(
                    "mutable foreign-linked buffers need isolation/effect support"
                )
            _StateSpec.of(item)
        else:
            _static_carrier_field(item)
    return values


def _normalize_carriers(
    graphdef: Any, native: tuple[Any, ...]
) -> tuple[Any, tuple[_CarrierSpec, ...], tuple[int, ...], tuple[Any, ...]]:
    if type(graphdef) is not nnx.GraphDef or tuple(
        field.name for field in fields(graphdef)
    ) != ("nodes", "attributes", "num_leaves"):
        raise _foreign_failure("native GraphDef schema needs an exact adapter")
    raw_carriers: list[tuple[type[Any], tuple[tuple[str, Any], ...]]] = []
    identities: dict[int, int] = {}
    replacements: dict[int, int] = {}
    connected: set[int] = set()
    for index, (_, attribute) in enumerate(graphdef.attributes):
        if type(attribute) is not nnx_graph.Static:
            continue
        value = attribute.value
        carrier_values = _carrier_fields(value)
        if carrier_values is None:
            continue
        identity = id(value)
        if identity not in identities:
            identities[identity] = len(raw_carriers)
            raw_carriers.append((type(value), carrier_values))
            connected.update(
                id(item) for _, item in carrier_values if _numeric_carrier_field(item)
            )
        replacements[index] = identities[identity]
    if not raw_carriers:
        return graphdef, (), (), native
    # Only foreign-connected identity groups are canonicalized. Every unrelated
    # existing Variable/container port retains its own original position.
    state_values: list[Any] = []
    ports: dict[int, int] = {}
    aliases = []
    for item in native:
        identity = id(item)
        if identity not in connected or identity not in ports:
            port = len(state_values)
            state_values.append(item)
            if identity in connected:
                ports[identity] = port
        else:
            port = ports[identity]
        aliases.append(port)
    carriers = []
    for carrier_type, raw_fields in raw_carriers:
        carrier_fields = []
        for name, item in raw_fields:
            if _numeric_carrier_field(item):
                identity = id(item)
                if identity not in ports:
                    ports[identity] = len(state_values)
                    state_values.append(item)
                carrier_fields.append(_CarrierField(name, ports[identity], None))
            else:
                carrier_fields.append(
                    _CarrierField(name, None, _static_carrier_field(item))
                )
        carriers.append(_CarrierSpec(carrier_type, tuple(carrier_fields)))
    attributes = [
        (
            (key, replace(attribute, value=_CarrierToken(replacements[index])))
            if index in replacements
            else (key, attribute)
        )
        for index, (key, attribute) in enumerate(graphdef.attributes)
    ]
    return (
        replace(graphdef, attributes=attributes),
        tuple(carriers),
        tuple(aliases),
        tuple(state_values),
    )


def _split(
    owner: Any,
) -> tuple[
    Any,
    tuple[_StateEntry, ...],
    tuple[_CarrierSpec, ...],
    tuple[int, ...],
    tuple[Any, ...],
]:
    graphdef, state = _split_graph(owner)
    entries: list[_StateEntry] = []
    leaves: list[Any] = []
    for path, value in nnx.to_flat_state(state):
        values, tree = jtu.tree_flatten(value)
        start = len(leaves)
        leaves.extend(values)
        entries.append(_StateEntry(tuple(path), tree, start, len(leaves)))
    # Reconstruction must not retain a surrounding NNX update-context index.
    graphdef, carriers, aliases, state_values = _normalize_carriers(
        graphdef.with_no_outer_index(), tuple(leaves)
    )
    return graphdef, tuple(entries), carriers, aliases, state_values


@dataclass(frozen=True)
class NNXFunctionState:
    """Topology and a pure unbound call, never a live owner or state payload."""

    graphdef: Any
    entries: tuple[_StateEntry, ...]
    specs: tuple[_StateSpec, ...]
    method: Callable[..., Any]
    instance_key: int
    effects: tuple[int, ...] | None = None
    carriers: tuple[_CarrierSpec, ...] = ()
    native_aliases: tuple[int, ...] = ()

    @property
    def state_count(self) -> int:
        return len(self.specs)

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            "nnx-state",
            self.method,
            self.instance_key,
            self.graphdef,
            self.entries,
            self.specs,
            self.effects,
            self.carriers,
            self.native_aliases,
        )

    def _foreign_ports(self) -> frozenset[int]:
        return frozenset(
            field.port
            for carrier in self.carriers
            for field in carrier.fields
            if field.port is not None
        )

    def with_effects(self, effects: tuple[int, ...]) -> NNXFunctionState:
        if tuple(sorted(set(effects))) != effects or any(
            index < 0 or index >= self.state_count for index in effects
        ):
            raise _failure("Malformed NNX Function state effect layout", defect=True)
        if self._foreign_ports().intersection(effects):
            raise _foreign_failure("foreign-linked effects need update support")
        return replace(self, effects=effects)

    def _check_values(self, state: tuple[Any, ...]) -> None:
        if len(state) != self.state_count:
            raise _failure(
                "NNX Function state changed shape, dtype, weakness or port count"
            )
        foreign = self._foreign_ports()
        if any(isinstance(state[index], np.ndarray) for index in foreign):
            raise _foreign_failure(
                "mutable foreign-linked buffers need isolation/effect support"
            )
        actual = tuple(_StateSpec.of(value) for value in state)
        if actual != self.specs:
            if any(actual[index] != self.specs[index] for index in foreign):
                raise _foreign_failure(
                    "foreign-linked shape, dtype or weakness changed"
                )
            raise _failure(
                "NNX Function state changed shape, dtype, weakness or port count"
            )

    def _entry_ports(self, entry: _StateEntry) -> tuple[int, ...]:
        if self.native_aliases:
            return self.native_aliases[entry.start : entry.stop]
        return tuple(range(entry.start, entry.stop))

    def current_state(self, owner: object) -> tuple[Any, ...]:
        """Read numeric ports while checking exact topology, metadata and aliases."""
        return self._owner_state(owner)

    def paths_for_port(self, port: int) -> tuple[tuple[Any, ...], ...]:
        """Return every native entry path sharing this canonical numeric port."""
        if type(port) is not int or not 0 <= port < self.state_count:
            raise _failure("Malformed NNX state port selection", defect=True)
        return tuple(
            entry.path for entry in self.entries if port in self._entry_ports(entry)
        )

    def ports_for_filter(self, owner: object, mutable: Any) -> tuple[int, ...]:
        """Select from the same native graph protocol as this descriptor.

        Ambient tree mode can duplicate shared-owner/Variable paths. Filtering
        the exact graph-mode split state keeps selection on canonical entries
        without changing the caller's ambient graph configuration.
        """
        self._owner_state(owner)
        _, state = _split_graph(owner)
        selected = nnx.filter_state(state, mutable)
        return self.ports_for_paths(
            tuple(tuple(path) for path, _ in nnx.to_flat_state(selected))
        )

    def ports_for_paths(self, paths: tuple[tuple[Any, ...], ...]) -> tuple[int, ...]:
        """Select canonical ports, closing the existing native alias relation."""
        selected = set(paths)
        known = {entry.path for entry in self.entries}
        if not selected <= known:
            raise _failure("Selected NNX state path is absent from the native schema")
        return tuple(
            sorted(
                {
                    port
                    for entry in self.entries
                    if entry.path in selected
                    for port in self._entry_ports(entry)
                }
            )
        )

    def _state(self, state: tuple[Any, ...], *, changed_only: bool = False) -> Any:
        self._check_values(state)
        changed = set(self.effects or ())
        return nnx.from_flat_state(
            (
                entry.path,
                jtu.tree_unflatten(
                    entry.tree, [state[index] for index in self._entry_ports(entry)]
                ),
            )
            for entry in self.entries
            if not changed_only or changed.intersection(self._entry_ports(entry))
        )

    def reconstruct(self, state: tuple[Any, ...]) -> Any:
        """Merge current carried values into fresh actual native owner/carriers."""
        self._check_values(state)
        graphdef = self.graphdef
        if self.carriers:
            carriers = []
            for spec in self.carriers:
                # Object allocation bypasses __new__/__init__/__post_init__.
                carrier = object.__new__(spec.carrier_type)
                for field in spec.fields:
                    if field.port is None:
                        assert field.static is not None
                        value = field.static.value
                    else:
                        value = state[field.port]
                    object.__setattr__(carrier, field.name, value)
                carriers.append(carrier)
            attributes = [
                (
                    (key, replace(attribute, value=carriers[attribute.value.index]))
                    if type(attribute) is nnx_graph.Static
                    and type(attribute.value) is _CarrierToken
                    else (key, attribute)
                )
                for key, attribute in graphdef.attributes
            ]
            # This graph exists only for reconstruction, never descriptor/cache.
            graphdef = replace(graphdef, attributes=attributes)
        return nnx.merge(graphdef, self._state(state), copy=True)

    def _owner_state(self, owner: object) -> tuple[Any, ...]:
        graphdef, entries, carriers, aliases, state = _split(owner)
        if (
            graphdef != self.graphdef
            or entries != self.entries
            or carriers != self.carriers
            or aliases != self.native_aliases
        ):
            if self.carriers or carriers:
                raise _foreign_failure(
                    "carrier schema, metadata or alias topology changed"
                )
            raise _failure("NNX Function changed graph topology or state metadata")
        self._check_values(state)
        return state

    def _foreign_instances(self, owner: object) -> tuple[object, ...]:
        if not self.carriers:
            return ()
        graphdef, _ = _split_graph(owner)
        instances = []
        for index, (_, attribute) in enumerate(self.graphdef.attributes):
            if (
                type(attribute) is nnx_graph.Static
                and type(attribute.value) is _CarrierToken
            ):
                if (
                    index >= len(graphdef.attributes)
                    or type(graphdef.attributes[index][1]) is not nnx_graph.Static
                ):
                    raise _foreign_failure("carrier holder layout changed")
                instances.append(graphdef.attributes[index][1].value)
        return tuple(instances)

    def invoke(
        self,
        arguments: tuple[Any, ...],
        keywords: Mapping[str, object],
        state: tuple[Any, ...],
    ) -> tuple[Any, tuple[Any, ...], tuple[int, ...]]:
        owner = self.reconstruct(state)
        foreign_before = self._foreign_instances(owner)
        result = self.method(owner, *arguments, **keywords)
        if any(
            before is not after
            for before, after in zip(
                foreign_before, self._foreign_instances(owner), strict=True
            )
        ):
            raise _foreign_failure(
                "carrier holder replacement needs effect/update support"
            )
        updated = self._owner_state(owner)
        if any(updated[index] is not state[index] for index in self._foreign_ports()):
            raise _foreign_failure(
                "foreign-linked numeric mutation needs effect/update support"
            )
        # This is an exact def-use alias test, not numerical tracer equality.
        effects = tuple(
            index
            for index, (before, after) in enumerate(zip(state, updated, strict=True))
            if after is not before
        )
        if self.effects is not None and effects != self.effects:
            raise _failure("NNX Function state alias effects changed while retracing")
        return result, updated, effects

    def validate_owner(self, owner: object) -> None:
        if self.effects is not None and self._foreign_ports().intersection(
            self.effects
        ):
            raise _foreign_failure("foreign-linked effects need update support")
        self._owner_state(owner)
        if self.effects is None:
            raise _failure(
                "NNX Function state has no traced effect layout", defect=True
            )
        if not self.effects:
            return
        try:
            # Public update checks every changed Variable/attribute at this trace
            # level. It writes the exact original objects, advancing no state.
            nnx.update(owner, self._original_state(owner))
        except errors.TraceContextError as exc:
            raise _failure(
                "Mutable NNX Function state requires a trace-local prepared owner"
            ) from exc

    def _original_state(self, owner: object) -> Any:
        """Transient envelopes retain original raw containers and numeric aliases.

        Flatten/unflatten is appropriate for the explicit numeric ports, but
        would rebuild a container-valued Variable's raw payload. A no-advance
        ownership check and rollback must instead retain that exact object.
        Nothing from this snapshot is stored in the descriptor or cache.
        """
        values = dict(_iter_graph(owner))
        changed = set(self.effects or ())
        originals = []
        for entry in self.entries:
            if not changed.intersection(self._entry_ports(entry)):
                continue
            value = values[entry.path]
            if isinstance(value, nnx.Variable):
                metadata = value.get_metadata()
                _require_ordinary_variable_metadata(metadata)
                value = type(value).from_metadata(value.get_raw_value(), metadata)
            originals.append((entry.path, value))
        return nnx.from_flat_state(originals)

    def update_owner(self, owner: object, state: tuple[Any, ...]) -> None:
        self.validate_owner(owner)
        self._check_values(state)
        if not self.effects:
            return
        original = self._original_state(owner)
        try:
            nnx.update(owner, self._state(state, changed_only=True))
        except BaseException:
            # Restore existing Variables and aliases, including a partial update.
            nnx.update(owner, original)
            raise


def prepare_nnx_function_state(
    owner: object, method: Callable[..., Any], instance_key: int
) -> tuple[NNXFunctionState, tuple[Any, ...]] | None:
    """Describe an actual NNX owner using only public split/merge state APIs."""
    if not is_nnx_function_owner(owner):
        return None
    graphdef, entries, carriers, aliases, state = _split(owner)
    adapter = NNXFunctionState(
        graphdef,
        entries,
        tuple(_StateSpec.of(value) for value in state),
        method,
        instance_key,
        carriers=carriers,
        native_aliases=aliases,
    )
    return adapter, state


def is_nnx_function_owner(owner: object) -> bool:
    return isinstance(owner, nnx.Module)
