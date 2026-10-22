# jax2onnx/plugins/_patching.py

from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Final, Iterator, Union, Tuple

_MISSING: Final[object] = object()


@dataclass(frozen=True)
class _AttributeSnapshot:
    value: Any

    def restore(
        self, target: Any, attr: str, *, ignore_missing_delete_errors: bool = False
    ) -> None:
        if self.value is not _MISSING:
            setattr(target, attr, self.value)
        elif ignore_missing_delete_errors:
            try:
                delattr(target, attr)
            except Exception:
                # Preserve the existing missing-attribute leaf behavior.
                pass
        else:
            delattr(target, attr)


def _snapshot_attribute(target: Any, attr: str, resolved: Any) -> _AttributeSnapshot:
    # Class/module getattr can bind descriptors or find an inherited attribute.
    # Instances and slots retain the previous resolved-value/setter semantics.
    original = (
        vars(target).get(attr, _MISSING)
        if isinstance(target, (type, ModuleType))
        else resolved
    )
    return _AttributeSnapshot(original)


def _resolve(target: Union[str, Any]) -> Any:
    if not isinstance(target, str):
        return target
    # "flax.nnx.Linear" -> (module "flax.nnx", attr "Linear")
    mod_path, _, attr = target.rpartition(".")
    obj = import_module(mod_path) if mod_path else globals()
    return getattr(obj, attr) if attr else obj


@dataclass
class AssignSpec:
    target: Union[str, Any]
    attr: str
    value: Any
    delete_if_missing: bool = True  # if original missing, restore by deleting


@dataclass
class MonkeyPatchSpec:
    target: Union[str, Any]
    attr: str
    make_value: Callable[[Any], Any]  # receives original value, returns new value
    delete_if_missing: bool = False


PatchSpec = Union[AssignSpec, MonkeyPatchSpec]


@dataclass(frozen=True, eq=False)
class _OwnedPatch:
    target: Any
    attr: str
    snapshot: _AttributeSnapshot
    installed: Any
    owner: object
    leaf: bool


_OWNED_PATCHES: list[_OwnedPatch] = []


def _record_owned_patch(
    target: Any,
    attr: str,
    snapshot: _AttributeSnapshot,
    installed: Any,
    *,
    owner: object,
    leaf: bool,
) -> _OwnedPatch:
    record = _OwnedPatch(target, attr, snapshot, installed, owner, leaf)
    _OWNED_PATCHES.append(record)
    return record


def _forget_owned_patch(record: _OwnedPatch) -> None:
    _OWNED_PATCHES.remove(record)


def _has_owned_leaf_patches() -> bool:
    return any(record.leaf for record in _OWNED_PATCHES)


def _installed_attribute(target: Any, attr: str) -> Any:
    return (
        vars(target).get(attr, _MISSING)
        if isinstance(target, (type, ModuleType))
        else getattr(target, attr, _MISSING)
    )


@contextmanager
def _suspend_owned_patches() -> Iterator[frozenset[tuple[int, str]]]:
    """Suspend only exact converter-owned assignments, then replay their values.

    A caller replacement above an owned stack is never unwrapped or replaced.
    Return those conflicting keys so genuine Function instrumentation can fail
    explicitly rather than quietly losing its requested boundary.
    """
    original = tuple(_OWNED_PATCHES)
    top = {(id(record.target), record.attr): record for record in original}
    blocked = frozenset(
        key
        for key, record in top.items()
        if _installed_attribute(record.target, record.attr) is not record.installed
    )
    suspended = [
        record for record in original if (id(record.target), record.attr) not in blocked
    ]
    removed: list[_OwnedPatch] = []
    try:
        for record in reversed(suspended):
            record.snapshot.restore(record.target, record.attr)
            _forget_owned_patch(record)
            removed.append(record)
        yield blocked
    finally:
        # Replay the exact wrapper/descriptor objects; never evaluate a patch
        # factory or follow an arbitrary caller-owned original marker chain.
        for record in reversed(removed):
            setattr(record.target, record.attr, record.installed)
        remaining = [record for record in _OWNED_PATCHES if record not in original]
        _OWNED_PATCHES[:] = [*original, *remaining]


@contextmanager
def apply_patches(
    specs: list[PatchSpec], *, owner: object | None = None
) -> Iterator[None]:
    applied: list[Tuple[Any, str, _AttributeSnapshot, bool, _OwnedPatch | None]] = []
    try:
        for s in specs:
            tgt = _resolve(s.target)
            orig = getattr(tgt, s.attr, _MISSING)
            snapshot = _snapshot_attribute(tgt, s.attr, orig)
            if isinstance(s, AssignSpec):
                setattr(tgt, s.attr, s.value)
            else:  # MonkeyPatchSpec
                new_val = s.make_value(None if orig is _MISSING else orig)
                setattr(tgt, s.attr, new_val)
            record = (
                _record_owned_patch(
                    tgt,
                    s.attr,
                    snapshot,
                    _installed_attribute(tgt, s.attr),
                    owner=owner,
                    leaf=True,
                )
                if owner is not None
                else None
            )
            applied.append((tgt, s.attr, snapshot, orig is _MISSING, record))
        yield
    finally:
        # unwind in reverse order
        for tgt, attr, snapshot, originally_missing, record in reversed(applied):
            try:
                snapshot.restore(
                    tgt, attr, ignore_missing_delete_errors=originally_missing
                )
            finally:
                if record is not None:
                    _forget_owned_patch(record)
