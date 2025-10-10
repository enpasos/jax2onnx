# jax2onnx/plugins/_patching.py

from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Callable, Final, Iterator, Union, Tuple

_MISSING: Final[object] = object()


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
    applied: list[Tuple[Any, str, Any]] = []
    try:
        for s in specs:
            tgt = _resolve(s.target)
            orig = getattr(tgt, s.attr, _MISSING)
            if isinstance(s, AssignSpec):
                setattr(tgt, s.attr, s.value)
            else:  # MonkeyPatchSpec
                new_val = s.make_value(None if orig is _MISSING else orig)
                setattr(tgt, s.attr, new_val)
            applied.append((tgt, s.attr, orig))
        yield
    finally:
        # unwind in reverse order
        for tgt, attr, orig in reversed(applied):
            if orig is _MISSING:
                try:
                    delattr(tgt, attr)
                except Exception:
                    # if delete_if_missing False, leave as-is
                    pass
            else:
                setattr(tgt, attr, orig)
