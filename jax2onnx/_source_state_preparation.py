# jax2onnx/_source_state_preparation.py

"""Explicit whole-call control and state preparation of native computations.

Preparation never executes the source, changes its precision, advances a caller's
owner, or chooses an RNG splitting policy. Every returned state word must be fed
to the next invocation. These internal constructors do not change export defaults.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import CellType, FunctionType
from typing import Any, cast

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx._compat.jax import (
    is_selected_prng_key_dtype,
    selected_prng_key_from_words,
    selected_prng_key_words,
)
from jax2onnx._linen_bridge import _LinenToNNXCallable
from jax2onnx.plugins.flax.nnx._source_function_state import (
    NNXFunctionState,
    prepare_nnx_function_state,
)


def closure_contents(target: FunctionType) -> dict[str, Any]:
    return {
        name: cell.cell_contents
        for name, cell in zip(
            target.__code__.co_freevars, target.__closure__ or (), strict=True
        )
    }


def _cell(value: Any) -> Any:
    return CellType(value)


def _attributes(target: FunctionType) -> tuple[Any, ...]:
    return (
        target.__code__,
        target.__globals__,
        target.__name__,
        target.__qualname__,
        target.__module__,
        target.__doc__,
        target.__defaults__,
        target.__kwdefaults__,
        target.__annotations__,
        target.__dict__,
    )


@dataclass(frozen=True)
class NativeClosure:
    """Preserve the original body/cells; rebind only its declared owner cell."""

    original: FunctionType
    owner_name: str | None
    attributes: tuple[Any, ...]
    captures: tuple[Any, ...]
    mapping_entries: tuple[tuple[str, tuple[tuple[Any, Any], ...]], ...]

    @classmethod
    def capture(cls, original: Any, owner_name: str | None) -> NativeClosure:
        if type(original) is not FunctionType:
            raise TypeError("Preparation requires the original factory function")
        captures = tuple(cell.cell_contents for cell in original.__closure__ or ())
        if owner_name is not None and owner_name not in original.__code__.co_freevars:
            raise ValueError("The declared mutable owner is not an actual capture")
        mappings = tuple(
            (name, tuple(value.items()))
            for name in ("__dict__", "__annotations__", "__kwdefaults__")
            if (value := getattr(original, name)) is not None
        )
        return cls(original, owner_name, _attributes(original), captures, mappings)

    def assert_original(self) -> None:
        """Validate identity, including binding changes hidden by equal values."""
        if not all(
            actual is expected
            for actual, expected in zip(
                _attributes(self.original), self.attributes, strict=True
            )
        ):
            raise ValueError("The original callable's attributes changed")
        for name, entries in self.mapping_entries:
            mapping = getattr(self.original, name)
            if list(mapping) != [key for key, _ in entries] or not all(
                mapping[key] is value for key, value in entries
            ):
                raise ValueError("The original callable's metadata bindings changed")
        if not all(
            cell.cell_contents is expected
            for cell, expected in zip(
                self.original.__closure__ or (), self.captures, strict=True
            )
        ):
            raise ValueError("The original callable's captured bindings changed")

    def rebind_owner(self, owner: Any) -> FunctionType:
        self.assert_original()
        if self.owner_name is None:
            raise ValueError("This callable has no declared owner capture")
        cells = tuple(
            _cell(owner) if name == self.owner_name else cell
            for name, cell in zip(
                self.original.__code__.co_freevars,
                self.original.__closure__ or (),
                strict=True,
            )
        )
        result = FunctionType(
            self.original.__code__,
            self.original.__globals__,
            self.original.__name__,
            self.original.__defaults__,
            cells,
        )
        result.__qualname__ = self.original.__qualname__
        result.__module__ = self.original.__module__
        result.__doc__ = self.original.__doc__
        result.__kwdefaults__ = self.original.__kwdefaults__
        result.__annotations__ = self.original.__annotations__
        result.__dict__.update(self.original.__dict__)
        return result


@dataclass(frozen=True)
class StateWordSpec:
    """One physical state port and its actual native logical key encoding."""

    name: str
    shape: tuple[Any, ...]
    dtype: Any
    weak_type: bool
    key_implementation: str | None = None
    paths: tuple[tuple[Any, ...], ...] = ()

    @classmethod
    def capture(
        cls, name: str, value: Any, *, paths: tuple[tuple[Any, ...], ...] = ()
    ) -> tuple[StateWordSpec, Any]:
        implementation = None
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            if not is_selected_prng_key_dtype(value.dtype) or tuple(value.shape) != ():
                raise ValueError("State preparation needs a selected scalar native key")
            implementation = str(jax.random.key_impl(value))
            word = selected_prng_key_words(value)
        else:
            word = value
        aval = jax.typeof(word)
        return (
            cls(
                name,
                tuple(aval.shape),
                aval.dtype,
                bool(aval.weak_type),
                implementation,
                paths,
            ),
            word,
        )

    def decode(self, word: Any) -> Any:
        aval = jax.typeof(word)
        if (
            tuple(aval.shape) != self.shape
            or aval.dtype != self.dtype
            or bool(aval.weak_type) != self.weak_type
        ):
            raise ValueError("Prepared state word changed shape, dtype or weakness")
        if isinstance(word, (np.ndarray, np.generic)):
            # Physical host buffers are storage for a native logical state
            # value. A fresh immutable JAX value prevents an NNX count update
            # from mutating the caller's feed/reset buffer in place.
            word = jnp.array(word, dtype=self.dtype, copy=True)
            native = jax.typeof(word)
            if (
                tuple(native.shape) != self.shape
                or native.dtype != self.dtype
                or bool(native.weak_type) != self.weak_type
            ):
                raise ValueError(
                    "Native state ingress changed shape, dtype or weakness"
                )
        if self.key_implementation is None:
            return word
        key = selected_prng_key_from_words(word)
        if str(jax.random.key_impl(key)) != self.key_implementation:
            raise ValueError("Prepared native key implementation changed")
        return key

    def encode(self, value: Any) -> Any:
        if self.key_implementation is None:
            word = value
        else:
            if (
                not is_selected_prng_key_dtype(value.dtype)
                or str(jax.random.key_impl(value)) != self.key_implementation
            ):
                raise ValueError("Prepared native key implementation changed")
            word = selected_prng_key_words(value)
        self.decode(word)
        return word


def _control_parameter(name: str) -> str:
    if type(name) is not str or not name.isidentifier():
        raise ValueError("The control parameter must be an explicit keyword name")
    return name


@dataclass(frozen=True)
class _NNXCallState:
    descriptor: NNXFunctionState
    fixed_state: tuple[Any, ...] = field(repr=False)
    selected_ports: tuple[int, ...]
    word_specs: tuple[StateWordSpec, ...]
    owner_from_target: Callable[[Any], Any] = field(repr=False)
    target_from_owner: Callable[[Any], Any] = field(repr=False)

    def state_from_words(self, words: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(words) != len(self.selected_ports):
            raise ValueError("Every declared mutable state word must be supplied")
        state = list(self.fixed_state)
        for index, spec, word in zip(
            self.selected_ports, self.word_specs, words, strict=True
        ):
            state[index] = spec.decode(word)
        return tuple(state)

    def words_from_state(self, state: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            spec.encode(state[index])
            for index, spec in zip(self.selected_ports, self.word_specs, strict=True)
        )

    def target_from_words(self, words: tuple[Any, ...]) -> Any:
        return self.target_from_owner(
            self.descriptor.reconstruct(self.state_from_words(words))
        )

    def words_from_target(self, target: Any) -> tuple[Any, ...]:
        return self.words_from_state(
            self.descriptor.current_state(self.owner_from_target(target))
        )

    def invoke(
        self, x: Any, keywords: Mapping[str, Any], words: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        state = self.state_from_words(words)
        result, updated, effects = self.descriptor.invoke((x,), keywords, state)
        if not set(effects).issubset(self.selected_ports):
            raise ValueError(
                "The native call changed state outside the declared mutable selection"
            )
        return (result, *self.words_from_state(updated))


@dataclass(frozen=True, eq=False)
class PreparedStatefulCall:
    """A native result tree followed by the complete next explicit state tuple."""

    original: Callable[..., Any] = field(repr=False)
    control_parameter: str
    initial_words: tuple[Any, ...] = field(repr=False)
    state_specs: tuple[StateWordSpec, ...]
    _state: _NNXCallState | None = field(default=None, repr=False)
    _native_control: bool = False
    _closure: NativeClosure | None = field(default=None, repr=False)

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.state_specs)

    def _invoke(
        self, x: Any, flag: Any, words: tuple[Any, ...], keywords: Mapping[str, Any]
    ) -> tuple[Any, ...]:
        if self.control_parameter in keywords:
            raise TypeError("The declared control operand was supplied twice")
        if self._closure is not None:
            self._closure.assert_original()
        named = {**keywords, self.control_parameter: flag}
        if self._state is None:
            if words:
                raise ValueError("A pure prepared call has no mutable state words")
            return (self.original(x, **named),)
        return self._state.invoke(x, named, words)

    def concrete(
        self, flag: bool, x: Any, *words: Any, **keywords: Any
    ) -> tuple[Any, ...]:
        if type(flag) is not bool:
            raise TypeError("A concrete branch requires an actual Python bool")
        return self._invoke(x, flag, tuple(words), keywords)

    def __call__(
        self, x: Any, flag: Any, *words: Any, **keywords: Any
    ) -> tuple[Any, ...]:
        if self._native_control:
            # Explicitly chosen for a source whose own native short circuit
            # accepts this operand. No inference from a child's Dropout rate.
            return self._invoke(x, flag, tuple(words), keywords)
        return cast(
            tuple[Any, ...],
            jax.lax.cond(
                flag,
                lambda operands: self.concrete(True, *operands, **keywords),
                lambda operands: self.concrete(False, *operands, **keywords),
                (x, *words),
            ),
        )

    def target_from_words(self, words: tuple[Any, ...]) -> Any:
        if self._state is None:
            if words:
                raise ValueError("A pure prepared call has no mutable state words")
            if self._closure is not None:
                self._closure.assert_original()
            return self.original
        return self._state.target_from_words(words)

    def fresh_original(self) -> Any:
        return self.target_from_words(self.initial_words)

    def words_from_target(self, target: Any) -> tuple[Any, ...]:
        if self._state is None:
            if target is not self.original:
                raise ValueError("The supplied pure callable is not the original")
            if self._closure is not None:
                self._closure.assert_original()
            return ()
        return self._state.words_from_target(target)


def _prepare_owner(
    original: Callable[..., Any],
    owner: Any,
    method: Callable[..., Any],
    *,
    control_parameter: str,
    mutable: Any,
    owner_from_target: Callable[[Any], Any],
    target_from_owner: Callable[[Any], Any],
    native_control: bool = False,
    closure: NativeClosure | None = None,
) -> PreparedStatefulCall:
    described = prepare_nnx_function_state(owner, method, 0)
    if described is None:
        raise TypeError("Preparation requires an actual NNX owner")
    descriptor, state = described
    if any(isinstance(value, np.ndarray) for value in state):
        # Coercing the original owner's ndarray storage would change its source
        # type/mutation semantics. Keep that explicit implementation gap until
        # a storage/effect adapter can preserve those semantics in isolation.
        raise ValueError(
            "Native owner state with mutable NumPy storage needs an isolation/effect adapter"
        )
    ports = descriptor.ports_for_filter(owner, mutable)
    specs: list[StateWordSpec] = []
    words: list[Any] = []
    for port in ports:
        paths_for_port = descriptor.paths_for_port(port)
        name = (
            "state_"
            + str(port)
            + "_"
            + "__".join(".".join(str(part) for part in path) for path in paths_for_port)
        )
        spec, word = StateWordSpec.capture(name, state[port], paths=paths_for_port)
        specs.append(spec)
        words.append(word)
    adapter = _NNXCallState(
        descriptor, state, ports, tuple(specs), owner_from_target, target_from_owner
    )
    return PreparedStatefulCall(
        original,
        _control_parameter(control_parameter),
        tuple(words),
        tuple(specs),
        adapter,
        native_control,
        closure,
    )


def _identity(value: Any) -> Any:
    return value


def _call_owner(receiver: Any, *arguments: Any, **keywords: Any) -> Any:
    # Dispatch on the fresh actual receiver so activation can retain its
    # original named Function boundary. Capturing an unbound __call__ before
    # activation would bypass that boundary's installed wrapper.
    return receiver(*arguments, **keywords)


def prepare_nnx_call(
    original: Any,
    *,
    control_parameter: str = "deterministic",
    mutable: Any = nnx.RngState,
) -> PreparedStatefulCall:
    """Lift the actual selected NNX state, preserving all aliases and streams."""
    if not isinstance(original, nnx.Module):
        raise TypeError("prepare_nnx_call requires an actual NNX module")
    return _prepare_owner(
        original,
        original,
        _call_owner,
        control_parameter=control_parameter,
        mutable=mutable,
        owner_from_target=_identity,
        target_from_owner=_identity,
    )


def prepare_linen_call(
    original: Any, *, control_parameter: str = "deterministic"
) -> PreparedStatefulCall:
    """Prepare the initialized original ToNNX wrapper, retaining its fixed key."""
    from flax.nnx import bridge

    if (
        type(original) is not _LinenToNNXCallable
        or type(original._model) is not bridge.ToNNX
    ):
        raise TypeError(
            "prepare_linen_call requires the initialized native Linen bridge"
        )
    key = original._rngs

    def target_from_owner(owner: Any) -> Any:
        return _LinenToNNXCallable(owner, key)

    def invoke(owner: Any, *arguments: Any, **keywords: Any) -> Any:
        return target_from_owner(owner)(*arguments, **keywords)

    def owner_from_target(target: Any) -> Any:
        if type(target) is not _LinenToNNXCallable or target._rngs is not key:
            raise ValueError("The original Linen wrapper or fixed bridge key changed")
        return target._model

    return _prepare_owner(
        original,
        original._model,
        invoke,
        control_parameter=control_parameter,
        mutable=nnx.RngState,
        owner_from_target=owner_from_target,
        target_from_owner=target_from_owner,
    )


def prepare_pure_control_call(
    original: Callable[..., Any], *, control_parameter: str
) -> PreparedStatefulCall:
    """Call the complete original computation with concrete True/False controls.

    Bind keys and other caller arguments explicitly, for example with partial.
    This constructor imports no optional Equinox package and invents no RNG state.
    """
    if not callable(original):
        raise TypeError("Preparation requires a callable")
    closure = (
        NativeClosure.capture(original, None)
        if type(original) is FunctionType
        else None
    )
    return PreparedStatefulCall(
        original, _control_parameter(control_parameter), (), (), _closure=closure
    )


def prepare_native_control_call(
    original: Any,
    *,
    control_parameter: str = "deterministic",
    mutable: Any = nnx.RngState,
) -> PreparedStatefulCall:
    """Explicitly preserve a source's native flag handling and short circuits."""
    prepared = prepare_nnx_call(
        original, control_parameter=control_parameter, mutable=mutable
    )
    return PreparedStatefulCall(
        prepared.original,
        prepared.control_parameter,
        prepared.initial_words,
        prepared.state_specs,
        prepared._state,
        True,
    )


def _prepare_nnx_closure_call(
    original: FunctionType,
    *,
    owner_name: str,
    control_parameter: str = "deterministic",
    native_control: bool = False,
    call_body: Callable[..., Any] | None = None,
) -> PreparedStatefulCall:
    """Compatibility boundary for callers declaring one actual mutable cell."""
    closure = NativeClosure.capture(original, owner_name)
    owner = closure_contents(original)[owner_name]

    def invoke(receiver: Any, *arguments: Any, **keywords: Any) -> Any:
        target = closure.rebind_owner(receiver)
        if call_body is None:
            return target(*arguments, **keywords)
        return call_body(*arguments, target=target, **keywords)

    def owner_from_target(target: Any) -> Any:
        return closure_contents(target)[owner_name]

    return _prepare_owner(
        original,
        owner,
        invoke,
        control_parameter=control_parameter,
        mutable=nnx.RngState,
        owner_from_target=owner_from_target,
        target_from_owner=closure.rebind_owner,
        native_control=native_control,
        closure=closure,
    )
