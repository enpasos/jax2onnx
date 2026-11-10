# jax2onnx/converter/state_inputs.py

"""Export selected model arrays as named graph inputs with saved defaults.

A state export is the ordinary export of ``g(*data, *state)``, where ``g`` is
the exported callable with every declared array replaced by an argument. The
arrays are located among the pytree leaves of the callable and traced as
inputs, so the trace cannot read their values. After graph rewriting has
finished, each array becomes the default initializer of its input.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Any, Final

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import Tracer

RESERVED_METADATA_PREFIX: Final[str] = "pkg.jax2onnx."
_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class StateCaptureError(ValueError):
    """A declared state array cannot be exported as a state input."""


@dataclass(frozen=True, eq=False)
class StateInput:
    """Export one array of the model as a named ONNX input with a saved default.

    Attributes:
        array: A concrete ``jax.Array`` that is a pytree leaf of the exported
            callable. The object identifies the state, not its value.
        name: Name of the graph input and of its default tensor; a C90
            identifier.
        metadata: String properties stored on the default tensor. jax2onnx
            does not interpret them.
        overridable: Keep the array in ``graph.input`` so callers can replace
            it. When false it is an ordinary initializer, which runtimes may
            fold.
    """

    array: Any
    name: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    overridable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME_PATTERN.fullmatch(self.name) is None:
            raise ValueError(
                f"StateInput name must be a C90 identifier, got {self.name!r}."
            )
        if isinstance(self.array, Tracer) or not isinstance(self.array, jax.Array):
            raise TypeError(
                f"StateInput {self.name!r} needs a concrete jax.Array, got "
                f"{type(self.array).__name__}. Convert NumPy leaves of the model "
                "with jnp.asarray before the export."
            )
        if np.issubdtype(self.array.dtype, np.complexfloating):
            raise TypeError(f"StateInput {self.name!r} cannot be complex.")
        if self.array.size == 0:
            raise ValueError(f"StateInput {self.name!r} cannot be an empty array.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError(f"StateInput {self.name!r} metadata must be a mapping.")
        for key, value in self.metadata.items():
            if not isinstance(key, str) or not key or not isinstance(value, str):
                raise TypeError(
                    f"StateInput {self.name!r} metadata must map nonempty "
                    "strings to strings."
                )
            try:
                key.encode("utf-8")
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(
                    f"StateInput {self.name!r} metadata must be encodable as UTF-8."
                ) from error
            if key.startswith(RESERVED_METADATA_PREFIX):
                raise ValueError(
                    f"StateInput {self.name!r} metadata key {key!r} uses the "
                    f"reserved prefix {RESERVED_METADATA_PREFIX!r}."
                )
        if not isinstance(self.overridable, bool):
            raise TypeError(f"StateInput {self.name!r} overridable must be a bool.")


class StateInputs:
    """The declared state of one export.

    It locates the declared arrays in the exported callable, provides the
    callable and input specifications used for tracing, rejects declared
    arrays that reach the trace as constants, and attaches the defaults.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        entries: Sequence[StateInput],
        *,
        data_input_count: int,
        reserved_names: Collection[str],
        allow_unused: bool,
    ) -> None:
        self._entries: tuple[StateInput, ...] = tuple(entries)
        self._data_input_count = data_input_count
        self._allow_unused = allow_unused

        names = [entry.name for entry in self._entries]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"State names must be unique; duplicate {duplicates}.")
        collisions = sorted(set(names).intersection(reserved_names))
        if collisions:
            raise ValueError(
                f"State names {collisions} collide with input, output or "
                "input_params names."
            )
        self._index_by_id: dict[int, int] = {}
        for index, entry in enumerate(self._entries):
            if self._index_by_id.setdefault(id(entry.array), index) != index:
                raise ValueError(
                    f"State {entry.name!r} declares an array object that "
                    f"{self._entries[self._index_by_id[id(entry.array)]].name!r} "
                    "already declares."
                )

        leaves, self._treedef = jax.tree_util.tree_flatten(fn)
        self._leaves: tuple[Any, ...] = tuple(leaves)
        self._positions: tuple[tuple[int, ...], ...] = tuple(
            tuple(
                position
                for position, leaf in enumerate(self._leaves)
                if leaf is entry.array
            )
            for entry in self._entries
        )
        missing = [
            entry.name
            for entry, positions in zip(self._entries, self._positions)
            if not positions
        ]
        if missing:
            raise StateCaptureError(
                f"State arrays {missing} are not pytree leaves of the exported "
                "callable. Pass the model itself, or a jax.tree_util.Partial "
                "that holds the arrays, instead of a function closing over them."
            )

    @property
    def names(self) -> list[str]:
        return [entry.name for entry in self._entries]

    @property
    def specs(self) -> list[jax.ShapeDtypeStruct]:
        """Tracing specifications; a state input keeps its array's weak type."""
        return [
            jax.ShapeDtypeStruct(
                entry.array.shape, entry.array.dtype, weak_type=entry.array.weak_type
            )
            for entry in self._entries
        ]

    def traced(self, *args: Any, **kwargs: Any) -> Any:
        """Call the exported callable with its declared leaves replaced."""
        leaves = list(self._leaves)
        for index, positions in enumerate(self._positions):
            for position in positions:
                leaves[position] = args[self._data_input_count + index]
        fn = jax.tree_util.tree_unflatten(self._treedef, leaves)
        return fn(*args[: self._data_input_count], **kwargs)

    def reject_captured(self, array: object) -> None:
        """Fail when state reaches a traced scope as a captured constant."""
        index = self._index_by_id.get(id(array))
        if index is not None:
            raise StateCaptureError(
                f"State array {self._entries[index].name!r} is also captured as a "
                "constant: the computation reaches it through a reference outside "
                "the pytree of the exported callable, such as a global or a closure."
            )
        if isinstance(array, Tracer):
            raise StateCaptureError(
                "A traced value is captured inside an ONNX function body. An "
                "@onnx_function module cannot read declared state from itself; "
                "leave its arrays undeclared or pass state to a free "
                "@onnx_function as an argument."
            )

    def attach_defaults(self, model: ir.Model, values: Sequence[ir.Value]) -> None:
        """Store each array on its input once graph rewriting has finished.

        Attaching last keeps the values invisible to constant evaluation, so
        the optimizer cannot fold or copy state.
        """
        graph = model.graph
        unused = {
            entry.name
            for entry, value in zip(self._entries, values)
            if not value.uses() and not value.is_graph_output()
        }
        if unused and not self._allow_unused:
            raise StateCaptureError(
                f"State arrays {sorted(unused)} are not read by the exported "
                "computation. Remove them from `state` or pass "
                "allow_unused_state=True to leave them out of the model."
            )
        for entry, value in zip(self._entries, values):
            if entry.name in unused:
                graph.inputs.remove(value)
                continue
            tensor = ir.tensor(np.asarray(entry.array), name=entry.name)
            tensor.metadata_props.update(entry.metadata)
            value.const_value = tensor
            graph.register_initializer(value)
            if not entry.overridable:
                graph.inputs.remove(value)


def prepare_state_inputs(
    fn: Callable[..., Any],
    state: Sequence[StateInput] | None,
    *,
    data_input_count: int,
    reserved_names: Collection[str],
    allow_unused: bool,
) -> StateInputs | None:
    """Validate ``state`` and locate its arrays; ``None`` when nothing is declared."""
    if not isinstance(allow_unused, bool):
        raise TypeError("allow_unused_state must be a bool.")
    if state is None:
        return None
    if isinstance(state, StateInput) or isinstance(state, (str, bytes, Mapping)):
        raise TypeError("state must be a sequence of StateInput entries.")
    entries = list(state)
    for entry in entries:
        if not isinstance(entry, StateInput):
            raise TypeError(
                f"state entries must be StateInput, got {type(entry).__name__}."
            )
    if not entries:
        return None
    return StateInputs(
        fn,
        entries,
        data_input_count=data_input_count,
        reserved_names=reserved_names,
        allow_unused=allow_unused,
    )
