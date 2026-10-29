# jax2onnx/_compat/python_annotations.py

"""Passive annotation snapshots for private Python callable preparation.

CPython 3.14 exposes its cached annotation storage through ``func_traverse``.
Match the complete ordered traversal before using that storage: public annotation
getters may evaluate a lazy annotation callback. Other interpreters and changed
traversal layouts require a separate adapter rather than an evaluating fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import gc
import importlib
import sys
from types import CodeType, FunctionType
from typing import Any, cast
import weakref


class AnnotationStateError(ValueError):
    """Annotation metadata cannot be retained or no longer matches its snapshot."""


_ABSENT = object()
_EMPTY_CELL = object()
_STATES: weakref.WeakKeyDictionary[
    FunctionType, weakref.ReferenceType[AnnotationState]
] = weakref.WeakKeyDictionary()


def _same(actual: tuple[object, ...], expected: tuple[object, ...]) -> bool:
    return len(actual) == len(expected) and all(
        left is right for left, right in zip(actual, expected, strict=True)
    )


def _is_dict(value: object) -> bool:
    return issubclass(type(value), dict)


def _is_exact_dict(value: object) -> bool:
    # A dict annotation also permits subclasses in static typing. Runtime
    # isolation deliberately requires the exact native storage type.
    return type(value) is dict


def _uses_legacy_annotations() -> bool:
    # Keep both runtime paths checked under the project's Python 3.12 mypy target.
    return sys.version_info < (3, 14)


def _supports_passive_annotations() -> bool:
    return sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 14)


def _cached_annotations(fn: FunctionType) -> object:
    if _uses_legacy_annotations():
        return fn.__annotations__
    if not _supports_passive_annotations():
        raise AnnotationStateError(
            "passive lazy annotation inspection needs a supported interpreter adapter"
        )

    # Objects/funcobject.c:func_traverse, CPython 3.14. NULL slots are omitted;
    # an explicitly stored None is still visited. Preserve ordered occurrences
    # because annotations may alias globals, builtins, kwdefaults or __dict__.
    prefixes: list[tuple[object, ...]] = [
        (fn.__code__, fn.__globals__, fn.__builtins__)
    ]
    for value, nullable in (
        (fn.__module__, True),
        (fn.__defaults__, False),
        (fn.__kwdefaults__, False),
        (fn.__doc__, True),
        (fn.__name__, False),
        (fn.__dict__, False),
        (fn.__closure__, False),
    ):
        if value is not None:
            prefixes = [(*prefix, value) for prefix in prefixes]
        elif nullable:
            prefixes = [
                candidate
                for prefix in prefixes
                for candidate in (prefix, (*prefix, None))
            ]

    annotate = getattr(fn, "__annotate__", None)
    suffixes: list[tuple[object, ...]] = (
        [(annotate,)] if annotate is not None else [(), (None,)]
    )
    type_params = getattr(fn, "__type_params__", ())
    if not issubclass(type(type_params), tuple) or tuple.__len__(type_params):
        raise AnnotationStateError("generic function headers need separate support")
    suffixes = [
        candidate
        for suffix in suffixes
        for candidate in (
            (*suffix, fn.__qualname__),
            (*suffix, type_params, fn.__qualname__),
        )
    ]
    referents = tuple(gc.get_referents(fn))
    candidates: list[object] = []
    for prefix in prefixes:
        for suffix in suffixes:
            if len(referents) < len(prefix) + len(suffix):
                continue
            if not _same(referents[: len(prefix)], prefix) or not _same(
                referents[-len(suffix) :], suffix
            ):
                continue
            middle = referents[len(prefix) : -len(suffix)]
            if not middle:
                candidates.append(_ABSENT)
            elif len(middle) == 1 and _is_dict(middle[0]):
                candidates.append(middle[0])
    if not candidates or not all(value is candidates[0] for value in candidates):
        raise AnnotationStateError(
            "function annotation storage does not match the supported CPython traversal"
        )
    return candidates[0]


def _entries(mapping: dict[str, Any]) -> tuple[tuple[str, object], ...]:
    # A dict subclass may override items/iteration. Snapshot native bindings
    # without invoking its user-defined methods.
    return tuple(dict.items(mapping))


def _cell_value(cell: Any) -> object:
    try:
        return cell.cell_contents
    except ValueError:
        return _EMPTY_CELL


def _annotator_metadata(annotate: object) -> tuple[object, ...]:
    if annotate is None:
        return ()
    if type(annotate) is not FunctionType:
        raise AnnotationStateError(
            "non-function annotation callbacks need separate preparation support"
        )
    kwdefaults = annotate.__kwdefaults__
    return (
        annotate.__code__,
        annotate.__globals__,
        annotate.__builtins__,
        annotate.__defaults__,
        annotate.__kwdefaults__,
        *(
            item
            for entry in (dict.items(kwdefaults) if kwdefaults is not None else ())
            for item in entry
        ),
        annotate.__closure__,
        *(_cell_value(cell) for cell in annotate.__closure__ or ()),
    )


@dataclass
class AnnotationState:
    """Watch cached identity and entries without evaluating annotation callbacks.

    Only an explicit VALUE read through our prepared function may establish a
    previously absent cache. An original-side read instead invalidates the cold
    record, because mutations before observation cannot be distinguished safely.
    """

    function: FunctionType
    annotate: object
    cached: object
    entries: tuple[tuple[str, object], ...]
    annotator_metadata: tuple[object, ...]
    allow_absent: bool = False
    peers: list[weakref.ReferenceType[AnnotationState]] = field(default_factory=list)
    active: bool = False

    @classmethod
    def capture(cls, fn: FunctionType) -> AnnotationState:
        previous_ref = _STATES.get(fn)
        previous = previous_ref() if previous_ref is not None else None
        if previous is not None:
            try:
                previous.assert_current()
            except AnnotationStateError:
                pass
            else:
                return previous
        cached = _cached_annotations(fn)
        state = cls(
            fn,
            getattr(fn, "__annotate__", None),
            cached,
            _entries(cast(dict[str, Any], cached)) if _is_dict(cached) else (),
            _annotator_metadata(getattr(fn, "__annotate__", None)),
        )
        # Weak values matter too: each live tracker owns its function. The
        # registry itself must not retain either object after callers drop it.
        _STATES[fn] = weakref.ref(state)
        return state

    def _assert_annotator_current(self) -> None:
        if getattr(self.function, "__annotate__", None) is not self.annotate:
            raise AnnotationStateError(
                "prepared dependency annotation callback changed"
            )
        if not _same(_annotator_metadata(self.annotate), self.annotator_metadata):
            raise AnnotationStateError(
                "prepared dependency annotation callback code or bindings changed"
            )

    def assert_current(self) -> None:
        self._assert_annotator_current()
        actual = _cached_annotations(self.function)
        if actual is not self.cached and not (self.allow_absent and actual is _ABSENT):
            raise AnnotationStateError(
                "prepared dependency annotation metadata changed; an original-side "
                "first annotation read requires fresh preparation"
            )
        if actual is self.cached:
            # A known authorized cache has now been observed in the function.
            # Never permit a later removal to reopen its prior cold state.
            self.allow_absent = False
        if _is_dict(self.cached):
            entries = _entries(cast(dict[str, Any], self.cached))
            if len(entries) != len(self.entries) or any(
                key is not expected_key or value is not expected_value
                for (key, value), (expected_key, expected_value) in zip(
                    entries, self.entries, strict=True
                )
            ):
                raise AnnotationStateError(
                    "prepared dependency annotation metadata bindings changed"
                )

    def attach(
        self, prepared: FunctionType, *, check_metadata: Callable[[], None]
    ) -> AnnotationState:
        """Retain a cached dict or defer its shared first read to the caller."""
        self.assert_current()
        if _uses_legacy_annotations():
            prepared.__annotations__ = cast(dict[str, Any], self.cached)
            return self.capture(prepared)

        requester: list[AnnotationState] = []

        def current_peers() -> list[AnnotationState]:
            current = []
            for peer_ref in self.peers:
                peer = peer_ref()
                if peer is not None:
                    try:
                        peer.assert_current()
                    except AnnotationStateError:
                        # A changed clone stays stale; it neither acquires a
                        # repaired baseline nor poisons another valid clone.
                        continue
                    current.append(peer)
            return current

        def forward(format: int) -> dict[str, Any]:
            if self.active:
                raise AnnotationStateError(
                    "reentrant annotation materialization needs separate support"
                )
            self.active = True
            try:
                check_metadata()
                self.assert_current()
                requesting_state = requester[0]
                requesting_state.assert_current()
                if format != 1:
                    # 2 is annotationlib's internal fake-globals protocol. Its
                    # fallback must clone the original compiler thunk, never
                    # this forwarding closure. Public FORWARDREF/STRING requests
                    # are completed by annotationlib on that original thunk.
                    if format == 2:
                        raise NotImplementedError(format)
                    try:
                        annotationlib = importlib.import_module("annotationlib")
                        if not callable(self.annotate):
                            known = dict(self.entries) if _is_dict(self.cached) else {}
                            if format == 3:
                                return known
                            if format == 4:
                                return cast(
                                    dict[str, Any],
                                    annotationlib.annotations_to_string(known),
                                )
                            raise ValueError(f"Invalid format: {format!r}")
                        return cast(
                            dict[str, Any],
                            annotationlib.call_annotate_function(
                                self.annotate, format, owner=self.function
                            ),
                        )
                    finally:
                        check_metadata()
                        self.assert_current()
                        requesting_state.assert_current()
                try:
                    value = self.function.__annotations__
                except BaseException:
                    check_metadata()
                    self.assert_current()
                    requesting_state.assert_current()
                    raise
                check_metadata()
                requesting_state.assert_current()
                self._assert_annotator_current()
                if _cached_annotations(self.function) is not value:
                    raise AnnotationStateError(
                        "annotation callback changed its function metadata"
                    )
                # Snapshot before the dict can escape. Repeated direct VALUE
                # calls must never refresh the baseline after a mutation.
                if not _is_dict(self.cached):
                    valid_peers = current_peers()
                    self.cached = value
                    self.entries = _entries(value)
                    for peer in valid_peers:
                        peer.cached = value
                        peer.entries = self.entries
                        # A direct __annotate__(VALUE) call leaves the
                        # prepared native cache absent until its getter runs.
                        peer.allow_absent = True
                return value
            finally:
                self.active = False

        setattr(prepared, "__annotate__", forward)
        prepared_state = self.capture(prepared)
        requester.append(prepared_state)
        if _is_dict(self.cached):
            prepared_state.cached = self.cached
            prepared_state.entries = self.entries
            prepared_state.allow_absent = True
        self.peers.append(weakref.ref(prepared_state))
        return prepared_state


@dataclass(slots=True)
class _IsolatedAnnotations:
    """One prepared function's annotation cache, never shared with its source."""

    function: FunctionType
    annotate: FunctionType | None
    initial: dict[str, Any] | None
    cached: dict[str, Any] | None
    entries: tuple[tuple[str, object], ...]
    validate: Callable[[dict[str, Any]], None]
    forward: object = None
    active: bool = False
    origin: tuple[object, ...] = ()

    def assert_current(self) -> None:
        if (
            len(self.origin) != 4
            or self.function is not self.origin[0]
            or self.annotate is not self.origin[1]
            or self.initial is not self.origin[2]
            or self.validate is not self.origin[3]
            or type(self.active) is not bool
            or (self.cached is not None and not _is_exact_dict(self.cached))
            or type(self.entries) is not tuple
            or any(
                type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not str
                for entry in self.entries
            )
            or type(self.forward) is not FunctionType
            or self.forward.__defaults__ is not None
            or self.forward.__kwdefaults__ is not None
            or not _is_exact_dict(self.forward.__dict__)
            or self.forward.__dict__
        ):
            raise AnnotationStateError("isolated annotation ownership schema changed")
        if getattr(self.function, "__annotate__", None) is not self.forward:
            raise AnnotationStateError("isolated annotation callback changed")
        actual = _cached_annotations(self.function)
        if actual is not _ABSENT and actual is not self.cached:
            raise AnnotationStateError("isolated annotation cache changed")
        if self.cached is not None and (
            len(self.cached) != len(self.entries)
            or any(
                key is not old_key or value is not old_value
                for (key, value), (old_key, old_value) in zip(
                    dict.items(self.cached), self.entries, strict=True
                )
            )
        ):
            raise AnnotationStateError("isolated annotation cache bindings changed")


def _isolated_forwarder(state: _IsolatedAnnotations) -> FunctionType:
    def forward(format: int) -> dict[str, Any]:
        state.assert_current()
        if state.active:
            raise AnnotationStateError("reentrant isolated annotation materialization")
        state.active = True
        try:
            if format == 1 and state.cached is not None:
                return state.cached
            if state.annotate is None:
                known = dict(state.entries)
                if format == 1:
                    state.validate(known)
                    state.cached = known
                    state.entries = _entries(known)
                    return known
                if format == 3:
                    return known
                if format == 4:
                    annotationlib = importlib.import_module("annotationlib")
                    return cast(
                        dict[str, Any], annotationlib.annotations_to_string(known)
                    )
                raise ValueError(f"Invalid format: {format!r}")
            if format == 2:
                raise NotImplementedError(format)
            if format == 1:
                result = state.annotate(format)
                state.assert_current()
                if type(result) is not dict:
                    raise AnnotationStateError("isolated annotator needs an exact dict")
                state.validate(result)
                state.cached = result
                state.entries = _entries(result)
                return result
            annotationlib = importlib.import_module("annotationlib")
            result = annotationlib.call_annotate_function(
                state.annotate, format, owner=state.function
            )
            state.assert_current()
            return cast(dict[str, Any], result)
        finally:
            state.active = False

    if type(forward) is not FunctionType:
        raise AnnotationStateError("isolated forwarder needs a native function")
    return forward


_ISOLATED_FORWARD_CODE: CodeType = next(
    value
    for value in _isolated_forwarder.__code__.co_consts
    if type(value) is CodeType and value.co_name == "forward"
)


def isolated_annotation_parts(
    function: FunctionType,
) -> tuple[FunctionType | None, dict[str, Any] | None]:
    """Read an inert snapshot, including our isolated cache's logical baseline.

    Lazy materialization by a clone may warm its private cache. It must neither
    warm the source nor change the clone's original structural declaration.
    No getter or annotation callback is evaluated here.
    """
    annotate: object = getattr(function, "__annotate__", None)
    if (
        type(annotate) is FunctionType
        and annotate.__code__ is _ISOLATED_FORWARD_CODE
        and annotate.__globals__ is globals()
        and annotate.__closure__ is not None
        and len(annotate.__closure__) == 1
    ):
        try:
            state = annotate.__closure__[0].cell_contents
        except ValueError as exc:
            raise AnnotationStateError("isolated annotation ownership changed") from exc
        if (
            type(state) is not _IsolatedAnnotations
            or state.function is not function
            or state.forward is not annotate
        ):
            raise AnnotationStateError("isolated annotation ownership changed")
        state.assert_current()
        return state.annotate, state.initial
    if annotate is not None and type(annotate) is not FunctionType:
        raise AnnotationStateError("non-function isolated annotator needs an adapter")
    cached = _cached_annotations(function)
    if cached is not _ABSENT and type(cached) is not dict:
        raise AnnotationStateError("isolated annotations need an exact dict")
    return (
        annotate,
        None if cached is _ABSENT else cast(dict[str, Any], cached),
    )


def install_isolated_annotations(
    function: FunctionType,
    *,
    annotate: FunctionType | None,
    cached: dict[str, Any] | None,
    validate: Callable[[dict[str, Any]], None],
) -> None:
    """Install already detached annotation objects without running a thunk."""
    if _uses_legacy_annotations():
        function.__annotations__ = {} if cached is None else cached
        return
    state = _IsolatedAnnotations(
        function,
        annotate,
        cached,
        cached,
        () if cached is None else _entries(cached),
        validate,
    )
    state.origin = (function, annotate, cached, validate)
    forward = _isolated_forwarder(state)
    state.forward = forward
    setattr(function, "__annotate__", forward)
