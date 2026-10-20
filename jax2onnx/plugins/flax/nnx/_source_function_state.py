# jax2onnx/plugins/flax/nnx/_source_function_state.py

"""Public NNX state lifting for the internal SOURCE Function binder."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from typing import Any, Callable, Mapping

from flax import errors, nnx
import jax
import jax.tree_util as jtu

from jax2onnx._compat.jax import ShapedArray
from jax2onnx.converter.source_profile import SourcePrecisionError


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


def _split(owner: Any) -> tuple[Any, tuple[_StateEntry, ...], tuple[Any, ...]]:
    graphdef, state = _split_graph(owner)
    entries: list[_StateEntry] = []
    leaves: list[Any] = []
    for path, value in nnx.to_flat_state(state):
        values, tree = jtu.tree_flatten(value)
        start = len(leaves)
        leaves.extend(values)
        entries.append(_StateEntry(tuple(path), tree, start, len(leaves)))
    # Reconstruction must not retain a surrounding NNX update-context index.
    return graphdef.with_no_outer_index(), tuple(entries), tuple(leaves)


@dataclass(frozen=True)
class NNXFunctionState:
    """Topology and a pure unbound call, never a live owner or state payload."""

    graphdef: Any
    entries: tuple[_StateEntry, ...]
    specs: tuple[_StateSpec, ...]
    method: Callable[..., Any]
    instance_key: int
    effects: tuple[int, ...] | None = None

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
        )

    def with_effects(self, effects: tuple[int, ...]) -> NNXFunctionState:
        if tuple(sorted(set(effects))) != effects or any(
            index < 0 or index >= self.state_count for index in effects
        ):
            raise _failure("Malformed NNX Function state effect layout", defect=True)
        return replace(self, effects=effects)

    def _check_values(self, state: tuple[Any, ...]) -> None:
        if (
            len(state) != self.state_count
            or tuple(_StateSpec.of(value) for value in state) != self.specs
        ):
            raise _failure(
                "NNX Function state changed shape, dtype, weakness or port count"
            )

    def _state(self, state: tuple[Any, ...], *, changed_only: bool = False) -> Any:
        self._check_values(state)
        changed = set(self.effects or ())
        return nnx.from_flat_state(
            (
                entry.path,
                jtu.tree_unflatten(entry.tree, state[entry.start : entry.stop]),
            )
            for entry in self.entries
            if not changed_only or changed.intersection(range(entry.start, entry.stop))
        )

    def _owner_state(self, owner: object) -> tuple[Any, ...]:
        graphdef, entries, state = _split(owner)
        if graphdef != self.graphdef or entries != self.entries:
            raise _failure("NNX Function changed graph topology or state metadata")
        self._check_values(state)
        return state

    def invoke(
        self,
        arguments: tuple[Any, ...],
        keywords: Mapping[str, object],
        state: tuple[Any, ...],
    ) -> tuple[Any, tuple[Any, ...], tuple[int, ...]]:
        owner = nnx.merge(self.graphdef, self._state(state), copy=True)
        result = self.method(owner, *arguments, **keywords)
        updated = self._owner_state(owner)
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
            if not changed.intersection(range(entry.start, entry.stop)):
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
    graphdef, entries, state = _split(owner)
    adapter = NNXFunctionState(
        graphdef,
        entries,
        tuple(_StateSpec.of(value) for value in state),
        method,
        instance_key,
    )
    return adapter, state


def is_nnx_function_owner(owner: object) -> bool:
    return isinstance(owner, nnx.Module)
