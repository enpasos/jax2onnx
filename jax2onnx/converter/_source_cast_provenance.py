# jax2onnx/converter/_source_cast_provenance.py

"""Read-only provenance lease for the first ordinary native cast syntax.

This never prepares, replaces, evaluates or invokes the supplied callable.
The immutable proof is syntax/binding evidence, not compilation authority.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from types import FunctionType, ModuleType
from typing import Any, Final

import jax
import numpy as np

from jax2onnx._compat.jax import pin_native_cast_callable

from jax2onnx._source_callable_preparation import (
    _FunctionState,
    _code_contract,
    _compile_body,
    _definition,
    _global_reads,
    callable_preimage,
)
from jax2onnx._source_preparation import SourcePreparationError

_CONVERT: Final[Any] = jax.lax.convert_element_type
_DTYPE: Final[Any] = np.dtype
_HALF: Final[Any] = np.float16
_HALF_DTYPE: Final[np.dtype[Any]] = np.dtype("float16")


@dataclass(frozen=True, slots=True)
class CallableCastProof:
    source_sha256: str
    code_sha256: str
    grammar: str
    bindings: tuple[tuple[str, str], ...]


class CallableCastLease:
    """Strong native Python references exist only until ``close``."""

    def __init__(self, function: object) -> None:
        self._state: _FunctionState | None = None
        self._native: Any = None
        self._members: list[tuple[dict[str, Any], str, object]] = []
        self.proof: CallableCastProof | None = None
        self.reason: str | None = None
        self._closed = False
        try:
            self._capture(function)
        except (SourcePreparationError, ValueError, TypeError, AttributeError, OSError):
            self.reason = "ordinary-cast-callable-provenance-unavailable"
            self.proof = None
            self._state = None
            self._release_native()
            self._members.clear()

    def _capture(self, function: object) -> None:
        if type(function) is not FunctionType:
            raise ValueError("ordinary function required")
        if function.__defaults__ is not None or function.__kwdefaults__ is not None:
            raise ValueError("default arguments are outside the grammar")
        self._state = _FunctionState.capture(function, watch_globals=True)
        preimage = callable_preimage(function)
        definition = _definition(function, preimage.source)
        args = definition.args
        formals = (*args.posonlyargs, *args.args)
        if (
            len(formals) != 1
            or args.kwonlyargs
            or args.vararg is not None
            or args.kwarg is not None
            or args.defaults
            or args.kw_defaults
            or definition.decorator_list
        ):
            raise ValueError("ordinary unary argument grammar differs")
        body = definition.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and type(body[0].value.value) is str
        ):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            raise ValueError("whole callable is not a single return")
        call = body[0].value
        if (
            not isinstance(call, ast.Call)
            or len(call.args) != 2
            or call.keywords
            or not isinstance(call.args[0], ast.Name)
            or call.args[0].id != formals[0].arg
            or self._resolve(function, call.func) is not _CONVERT
        ):
            raise ValueError("native conversion expression differs")
        target = call.args[1]
        if isinstance(target, ast.Call):
            if (
                len(target.args) != 1
                or target.keywords
                or self._resolve(function, target.func) is not _DTYPE
            ):
                raise ValueError("dtype constructor differs")
            target = target.args[0]
        dtype = self._resolve(function, target)
        if not (
            (type(dtype) is str and dtype == "float16")
            or dtype is _HALF
            or (type(dtype) is type(_HALF_DTYPE) and dtype == _HALF_DTYPE)
        ):
            raise ValueError("immutable half dtype required")
        imported: tuple[str, ...] = ()
        compiled = _compile_body(function, definition, function.__code__.co_filename)
        if _code_contract(compiled) != _code_contract(function.__code__):
            imported = tuple(
                name
                for name in _global_reads(function.__code__)
                if type(function.__globals__.get(name)) is ModuleType
            )
            compiled = _compile_body(
                function,
                definition,
                function.__code__.co_filename,
                imported_globals=imported,
            )
        if _code_contract(compiled) != _code_contract(function.__code__):
            raise ValueError("original inspected source/live code differs")
        self._native = pin_native_cast_callable()
        # These exact selected module members are outside _FunctionState's
        # shallow global identity fence. No getattr or unknown resolver runs.
        self._members.extend(
            (
                (vars(jax.lax), "convert_element_type", _CONVERT),
                (vars(np), "dtype", _DTYPE),
                (vars(np), "float16", _HALF),
            )
        )
        names = tuple(
            (name, type(value).__module__ + "." + type(value).__qualname__)
            for _, name, value in self._members
        )
        self.proof = CallableCastProof(
            preimage.source_sha256,
            preimage.code_sha256,
            "ordinary-native-unary-f64-f16-v1",
            names,
        )
        self.assert_current()

    def _resolve(self, function: FunctionType, expression: ast.expr) -> object:
        if isinstance(expression, ast.Constant) and type(expression.value) is str:
            return expression.value
        if isinstance(expression, ast.Name):
            cells = dict(
                zip(
                    function.__code__.co_freevars,
                    function.__closure__ or (),
                    strict=True,
                )
            )
            if expression.id in cells:
                return cells[expression.id].cell_contents
            namespace = function.__globals__
            if expression.id not in namespace:
                raise ValueError("unbound source expression")
            value = dict.__getitem__(namespace, expression.id)
            self._members.append((namespace, expression.id, value))
            return value
        if isinstance(expression, ast.Attribute):
            module = self._resolve(function, expression.value)
            if type(module) is not ModuleType:
                raise ValueError("only exact module member access is supported")
            namespace = vars(module)
            if expression.attr not in namespace:
                raise ValueError("module member unavailable")
            value = dict.__getitem__(namespace, expression.attr)
            self._members.append((namespace, expression.attr, value))
            return value
        raise ValueError("expression requires an unsupported runtime operation")

    def assert_current(self) -> None:
        if self._closed or self._state is None:
            raise ValueError("callable provenance lease unavailable or closed")
        self._state.assert_current()
        if self._native is None:
            raise ValueError("native conversion provenance unavailable")
        self._native.assert_current()
        for namespace, name, expected in self._members:
            if dict.get(namespace, name) is not expected:
                raise ValueError("selected source module member changed")

    def _release_native(self) -> None:
        if self._native is not None:
            self._native.close()
        self._native = None

    def close(self) -> None:
        self._state = None
        self._release_native()
        self._members.clear()
        self._closed = True


def proof_digest(value: CallableCastProof) -> str:
    payload = (value.source_sha256, value.code_sha256, value.grammar, value.bindings)
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()
