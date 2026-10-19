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


@contextmanager
def apply_patches(specs: list[PatchSpec]) -> Iterator[None]:
    applied: list[Tuple[Any, str, _AttributeSnapshot, bool]] = []
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
            applied.append((tgt, s.attr, snapshot, orig is _MISSING))
        yield
    finally:
        # unwind in reverse order
        for tgt, attr, snapshot, originally_missing in reversed(applied):
            snapshot.restore(tgt, attr, ignore_missing_delete_errors=originally_missing)
