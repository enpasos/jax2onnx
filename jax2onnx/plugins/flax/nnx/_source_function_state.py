# jax2onnx/plugins/flax/nnx/_source_function_state.py

"""Public NNX state lifting for the internal SOURCE Function binder."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    graphdef, state = nnx.split(owner, graph=True)
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
        values = dict(nnx.iter_graph(owner, graph=True))
        changed = set(self.effects or ())
        originals = []
        for entry in self.entries:
            if not changed.intersection(range(entry.start, entry.stop)):
                continue
            value = values[entry.path]
            if isinstance(value, nnx.Variable):
                metadata = value.get_metadata()
                if metadata.get("ref") or metadata.get("hijax"):
                    raise _failure(
                        "Mutable NNX Function state requires ordinary non-Ref Variables"
                    )
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
