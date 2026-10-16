# jax2onnx/plugins/_function_keys.py

"""Exact, immutable cache keys for static ONNX-function configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import jax
import numpy as np


@dataclass(frozen=True, eq=False)
class _OpaqueValue:
    """Retain an unsupported value without sharing its specialized function."""

    value: object = field(repr=False)


@dataclass(frozen=True, eq=False)
class _ArrayContents:
    """Hash cheaply while verifying exact contents of matching candidates.

    JAX arrays are immutable, so retaining them avoids copying model weights.
    NumPy arrays have an immutable backing buffer shared with their snapshot.
    """

    digest: bytes
    value: np.ndarray | jax.Array = field(repr=False)

    def __hash__(self) -> int:
        return hash(self.digest)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ArrayContents):
            return NotImplemented
        return self.digest == other.digest and _array_bytes(self.value) == _array_bytes(
            other.value
        )


def _array_bytes(value: np.ndarray | jax.Array) -> memoryview:
    array = np.ascontiguousarray(np.asarray(value))
    return memoryview(array.reshape(-1).view(np.uint8))


def _array_digest(value: np.ndarray | jax.Array) -> bytes:
    return hashlib.sha256(_array_bytes(value)).digest()


def _immutable_numpy_array(value: np.ndarray) -> np.ndarray:
    backing: object = value
    while isinstance(backing, np.ndarray):
        backing = backing.base
    if isinstance(backing, bytes) and value.flags.c_contiguous:
        return value
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _dtype_key(dtype: np.dtype) -> object:
    # NumPy dtype equality ignores metadata, whose objects may also be mutable.
    if dtype.metadata is not None:
        return _OpaqueValue(dtype)
    if dtype.fields:
        return (dtype, tuple(_dtype_key(field[0]) for field in dtype.fields.values()))
    if dtype.subdtype:
        return (dtype, _dtype_key(dtype.subdtype[0]))
    return dtype


def snapshot_static_value(value: object) -> object:
    """Freeze supported mutable configuration at the primitive-bind boundary.

    Opaque objects retain their identity and never participate in function reuse.
    Cyclic containers cannot be snapshotted as static Python configuration.
    """
    return _snapshot(value, set())


def _snapshot(value: object, active: set[int]) -> object:
    value_type = type(value)
    if value_type not in (dict, list, tuple, set, np.ndarray):
        return value
    identity = id(value)
    if identity in active:
        raise ValueError("Cyclic static ONNX function configuration is unsupported")
    active.add(identity)
    try:
        if value_type is np.ndarray:
            assert isinstance(value, np.ndarray)
            if not value.dtype.hasobject:
                return _immutable_numpy_array(value)
            copied = value.copy()
            if value.dtype.names:
                for name in value.dtype.names:
                    copied[name] = _snapshot(value[name], active)
            elif value.dtype.hasobject:
                for index in np.ndindex(value.shape):
                    copied[index] = _snapshot(value[index], active)
            return copied
        if isinstance(value, dict):
            return {
                _snapshot(key, active): _snapshot(item, active)
                for key, item in value.items()
            }
        assert isinstance(value, (list, tuple, set))
        entries = [_snapshot(item, active) for item in value]
        if isinstance(value, tuple):
            return tuple(entries)
        if isinstance(value, set):
            return set(entries)
        return entries
    finally:
        active.remove(identity)


def static_value_key(value: object) -> tuple[object, ...]:
    """Snapshot supported values; conservatively avoid reuse for opaque objects.

    Keys compare exact contents rather than hashes or representations alone.
    Container order is retained because Python configuration can affect tracing through
    iteration order. Opaque objects and cycles receive a fresh identity token,
    which also keeps the original object alive for the lifetime of the key.
    """
    return _value_key(value, set())


def _value_key(value: object, active: set[int]) -> tuple[object, ...]:
    value_type = type(value)
    if value is None or value_type in (bool, int, str, bytes):
        return (value_type, value)
    if value_type in (float, complex):
        # Bytes preserve signed zero and NaN payloads, unlike scalar equality.
        return (value_type, np.asarray(value).tobytes())
    if isinstance(value, np.dtype):
        return (np.dtype, _dtype_key(value))

    is_array = (
        value_type is np.ndarray
        or isinstance(value, jax.Array)
        or (isinstance(value, np.generic) and value_type.__module__ == "numpy")
    )
    is_container = value_type in (tuple, list, dict, set, frozenset)
    if not is_array and not is_container:
        return ("opaque", _OpaqueValue(value))
    identity = id(value)
    if identity in active:
        return ("opaque", _OpaqueValue(value))
    active.add(identity)
    try:
        if is_array:
            array = np.asarray(value)
            if array.dtype.hasobject:
                contents: object = _value_key(array.tolist(), active)
            else:
                storage = (
                    value
                    if isinstance(value, jax.Array)
                    else _immutable_numpy_array(array)
                )
                contents = _ArrayContents(_array_digest(storage), storage)
            return (value_type, _dtype_key(array.dtype), tuple(array.shape), contents)
        if isinstance(value, dict):
            entries = tuple(
                (_value_key(key, active), _value_key(item, active))
                for key, item in value.items()
            )
            return (value_type, entries)
        assert isinstance(value, (tuple, list, set, frozenset))
        return (value_type, tuple(_value_key(item, active) for item in value))
    finally:
        active.remove(identity)
