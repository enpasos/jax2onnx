# tests/extra_tests/converter/test_source_dropout_parameters.py

"""Native Dropout preparation; retain separate diagnostic sample properties."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import equinox as eqx
from flax.errors import ModifyScopeVariableError, TraceContextError
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function
from jax2onnx import user_interface as ui
from jax2onnx._compat.jax import (
    DropVar,
    Literal,
    ShapedArray,
    is_selected_prng_key_dtype,
    is_unpartitioned_prng_placement,
    selected_prng_key_words,
    scan_arity,
)
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins.flax.nnx.dropout import DropoutPlugin
from jax2onnx.plugins.flax.linen.spectral_norm import _SpectralNormDense
from jax2onnx.plugins.plugin_system import (
    _DynamicParamRef,
    _StaticParam,
    import_all_plugins,
)
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter import source_precision_diagnostics as diagnostics
from tests.extra_tests.converter.source_state_preparation import (
    EqxDropoutState,
    ExplicitStatePreparation,
    LinenDropoutState,
    NNXDropoutState,
    NativeClosure,
    closure_contents,
)


class DropoutDiagnosticValidator(diagnostics.DiagnosticValidator):
    """The existing finite diagnostic families plus three exact Dropout schemas."""

    _rates = {
        "nnx.dropout": "rate",
        "linen.dropout": "rate",
        "eqx.nn.dropout": "p",
    }

    def __call__(self, jaxpr: Any, opset: int) -> None:
        rules = {
            **diagnostics._RULES,
            **{
                name: (2, 1, frozenset({rate, "call_time"}))
                for name, rate in self._rates.items()
            },
        }
        with patch.object(diagnostics, "_RULES", rules):
            super().__call__(jaxpr, opset)

    def _ordinary(self, eqn: Any, site: str, opset: int) -> None:
        name = eqn.primitive.name
        if name in self._rates:
            if len(eqn.invars) != 2 or len(eqn.outvars) != 1:
                self._fail(
                    "Dropout requires two operands and one output", site, opset, name
                )
            value, flag = (self._tensor(var, site, opset) for var in eqn.invars)
            output = self._tensor(eqn.outvars[0], site, opset)
            if (
                value[0] not in {"float16", "bfloat16", "float32", "float64"}
                or output != value
            ):
                self._fail(
                    "Dropout must preserve its floating input signature",
                    site,
                    opset,
                    name,
                )
            if flag[:2] != ("bool", ()):
                self._fail("Dropout flag must be BOOL[]", site, opset, name)
            rate_name = self._rates[name]
            if set(eqn.params) != {rate_name, "call_time"}:
                self._fail(
                    "Unreviewed Dropout parameter schema",
                    site,
                    opset,
                    name,
                    pending=True,
                )
            rate = eqn.params[rate_name]
            if (
                isinstance(rate, (bool, np.bool_))
                or not isinstance(rate, Real)
                or not np.isfinite(rate)
                or not 0 <= rate < 1
            ):
                self._fail(
                    "Dropout rate must be finite and in [0, 1)", site, opset, name
                )
            if type(eqn.params["call_time"]) is not bool:
                self._fail("Dropout call_time must be a Python BOOL", site, opset, name)
        super()._ordinary(eqn, site, opset)

    def _function(self, eqn: Any, site: str, opset: int) -> None:
        reserved = {"__jax2onnx_source_trace", "instance_key"}
        dynamic = []
        for name, value in eqn.params.items():
            if name in reserved:
                continue
            if isinstance(value, _DynamicParamRef):
                if (
                    type(value.index) is not int
                    or not 0 <= value.index < len(eqn.invars)
                    or self._tensor(eqn.invars[value.index], site, opset)[:2]
                    != ("bool", ())
                ):
                    self._fail(
                        "Function keyword is not an actual BOOL[] operand",
                        site,
                        opset,
                        eqn.primitive.name,
                    )
                dynamic.append(value.index)
            elif isinstance(value, _StaticParam) and type(value.value) is bool:
                continue
            elif (
                isinstance(value, _StaticParam)
                and eqn.primitive.name == f"onnx_fn::{__name__}._dropout_named"
                and name == "target"
                and callable(value.value)
                and value.value.__module__ == __name__
                and value.value.__qualname__.startswith("_target.<locals>.")
            ):
                # Only this test factory's three explicit framework adapters.
                continue
            elif (
                isinstance(value, _StaticParam)
                and eqn.primitive.name == f"onnx_fn::{__name__}._dropout_state_named"
                and name == "preparation"
                and type(value.value) is ExplicitStatePreparation
                and type(value.value.adapter)
                in {NNXDropoutState, LinenDropoutState, EqxDropoutState}
                and value.value.original_body.__module__ == __name__
                and value.value.original_body.__name__ == "_dropout_named"
            ):
                value.value.adapter.original.assert_original()
                continue
            else:
                self._fail(
                    "Unreviewed Function keyword parameter",
                    site,
                    opset,
                    eqn.primitive.name,
                    pending=True,
                )
        if sorted(dynamic) != list(
            range(len(eqn.invars) - len(dynamic), len(eqn.invars))
        ):
            self._fail(
                "Function keyword operands are not the exact trailing bindings",
                site,
                opset,
                eqn.primitive.name,
            )
        checked = SimpleNamespace(
            primitive=eqn.primitive,
            invars=eqn.invars,
            outvars=eqn.outvars,
            params={
                name: value for name, value in eqn.params.items() if name in reserved
            },
        )
        super()._function(checked, site, opset)


class NativeDropoutDiagnosticValidator(DropoutDiagnosticValidator):
    """Compose selected native PRNG proofs with the existing finite loop checks.

    No product loop admission is implied. Only exact selected PRNG equations
    and proven Bernoulli/uniform carriers bypass the old diagnostic families.
    Every other operation still takes the original structural validator route.
    """

    _random = frozenset(
        {"random_seed", "random_wrap", "random_unwrap", "random_fold_in", "random_bits"}
    )

    def _tensor(self, var: Any, location: str, opset: int) -> Any:
        dtype = var.aval.dtype
        if jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            if not is_selected_prng_key_dtype(dtype):
                self._fail(
                    "Unselected native key implementation",
                    location,
                    opset,
                    pending=True,
                )
            if (
                not isinstance(var.aval, ShapedArray)
                or tuple(var.aval.shape) != ()
                or var.aval.weak_type is not False
                or not is_unpartitioned_prng_placement(var.aval.sharding, 0)
            ):
                self._fail("Malformed scalar selected-key aval", location, opset)
            return ("key<fry>", (), False)
        return super()._tensor(var, location, opset)

    def _signature(self, graph: Any) -> Any:
        body = diagnostics._body(graph)
        return (
            [self._tensor(var, "nested input", 23)[:2] for var in body.invars],
            [self._tensor(var, "nested output", 23)[:2] for var in body.outvars],
        )

    def _nested(
        self, eqn: Any, site: str, opset: int, ancestors: frozenset[int]
    ) -> None:
        # Same complete finite nested checks, using validated logical key types.
        name = eqn.primitive.name
        if set(eqn.params) - diagnostics._NESTED[name]:
            self._fail(
                "Unreviewed nested operation parameters",
                site,
                opset,
                name,
                pending=True,
            )
        ins = [self._tensor(var, site, opset)[:2] for var in eqn.invars]
        outs = [self._tensor(var, site, opset)[:2] for var in eqn.outvars]
        children: list[tuple[str, Any]] = []
        valid = False
        try:
            if name in {"jit", "pjit"}:
                child = eqn.params.get("jaxpr", eqn.params.get("call_jaxpr"))
                valid = self._signature(child) == (ins, outs)
                children = [("body", child)]
                for key, value in eqn.params.items():
                    if key not in {"jaxpr", "call_jaxpr"} and not self._parameter(
                        value, key
                    ):
                        self._fail(
                            "Unreviewed JIT parameter form",
                            site,
                            opset,
                            name,
                            pending=True,
                        )
            elif name == "cond":
                branches = eqn.params["branches"]
                if len(branches) != 2:
                    self._fail(
                        "Only two-branch conditionals are inspected",
                        site,
                        opset,
                        name,
                        pending=True,
                    )
                valid = (
                    bool(ins)
                    and ins[0][0] in {"int32", "int64"}
                    and ins[0][1] == ()
                    and all(
                        self._signature(branch) == (ins[1:], outs)
                        for branch in branches
                    )
                )
                children = [
                    (f"branch {index}", branch) for index, branch in enumerate(branches)
                ]
            elif name == "while":
                nc, nb = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
                if not isinstance(nc, Integral) or not isinstance(nb, Integral):
                    self._fail("Invalid while constant counts", site, opset, name)
                nc, nb = int(nc), int(nb)
                carry = ins[nc + nb :]
                cond, body = eqn.params["cond_jaxpr"], eqn.params["body_jaxpr"]
                condition_signature = self._signature(cond)
                condition_outputs = condition_signature[1]
                if (
                    len(condition_outputs) == 1
                    and condition_outputs[0][0] == "bool"
                    and condition_outputs[0][1]
                ):
                    self._fail(
                        "Unreviewed batched while predicate",
                        site,
                        opset,
                        name,
                        pending=True,
                    )
                valid = (
                    nc >= 0
                    and nb >= 0
                    and condition_signature == (ins[:nc] + carry, [("bool", ())])
                    and self._signature(body) == (ins[nc : nc + nb] + carry, carry)
                    and outs == carry
                )
                children = [("condition", cond), ("body", body)]
            elif name == "scan":
                child = eqn.params["jaxpr"]
                nc, ny, _ = scan_arity(eqn.params, len(ins))
                length = eqn.params["length"]
                if not isinstance(length, Integral):
                    self._fail("Invalid scan length", site, opset, name)
                length = int(length)
                carry = ins[nc : nc + ny]
                xs = ins[nc + ny :]
                ys = outs[ny:]
                valid = (
                    length >= 0
                    and all(shape and shape[0] == length for _, shape in (*xs, *ys))
                    and self._signature(child)
                    == (
                        ins[:nc] + carry + [(dtype, shape[1:]) for dtype, shape in xs],
                        carry + [(dtype, shape[1:]) for dtype, shape in ys],
                    )
                    and outs[:ny] == carry
                )
                if "linear" in eqn.params:
                    valid = (
                        valid
                        and len(eqn.params["linear"]) == len(ins)
                        and all(
                            isinstance(value, bool) for value in eqn.params["linear"]
                        )
                    )
                if "ft_out" in eqn.params:
                    groups = getattr(eqn.params["ft_out"], "elts", ())
                    valid = (
                        valid
                        and len(groups) == 2
                        and [len(group) for group in groups] == [ny, len(ys)]
                    )
                if (
                    not isinstance(eqn.params["reverse"], bool)
                    or not isinstance(eqn.params["unroll"], Integral)
                    or eqn.params["unroll"] < 1
                    or eqn.params.get("_split_transpose", False)
                ):
                    self._fail(
                        "Unreviewed scan execution parameter form",
                        site,
                        opset,
                        name,
                        pending=True,
                    )
                children = [("body", child)]
        except SourcePrecisionError:
            raise
        except (TypeError, ValueError, KeyError, AttributeError, IndexError) as exc:
            self._fail(f"Malformed nested signature: {exc}", site, opset, name)
        if not valid:
            self._fail("Nested dtype/shape/arity signature mismatch", site, opset, name)
        for label, child in children:
            self._walk(child, opset, f"{site}/{label}", ancestors)

    def _selected(self, eqn: Any, site: str, opset: int) -> bool:
        from jax2onnx.plugins.jax.random._source_prng import (
            NativePRNGExclusion,
            native_bernoulli_uniform,
            prove_native_uniform,
            validate_selected_random_equation,
        )

        try:
            if eqn.primitive.name in self._random:
                validate_selected_random_equation(eqn)
                return True
            if eqn.primitive is jax.lax.stop_gradient_p or (
                eqn.primitive.name == "jit"
                and (
                    prove_native_uniform(eqn) is not None
                    or native_bernoulli_uniform(eqn) is not None
                )
            ):
                # Keep all actual carrier context, placement, parameter, literal,
                # conversion and definition/use checks from the audited packet.
                # Native Eqx also stops the probability's gradient. Its exact
                # pure copy schema must pass the mandatory product validator.
                formals = []
                for var in eqn.invars:
                    if not isinstance(var, Literal) and all(
                        var is not old for old in formals
                    ):
                        formals.append(var)
                validate_source_jaxpr(
                    SimpleNamespace(
                        invars=formals,
                        constvars=(),
                        eqns=[eqn],
                        outvars=eqn.outvars,
                        effects=(),
                        is_high=False,
                    ),
                    opset,
                )
                return True
        except NativePRNGExclusion as exc:
            self._fail(str(exc), site, opset, eqn.primitive.name, pending=True)
        except SourcePrecisionError:
            raise
        except (TypeError, ValueError, AttributeError) as exc:
            self._fail(str(exc), site, opset, eqn.primitive.name)
        return False

    def _walk(
        self, closed: Any, opset: int, location: str, ancestors: frozenset[int]
    ) -> None:
        # The old walk's binding/effect/constant checks are retained explicitly;
        # only typed key payloads and the selected PRNG dispatch differ.
        graph = diagnostics._body(closed)
        if id(graph) in ancestors or len(ancestors) > 20:
            self._fail(
                "Cyclic or excessively nested Jaxpr", location, opset, pending=True
            )
        ancestors = ancestors | {id(graph)}
        if opset not in (23, 26) or getattr(graph, "effects", True):
            self._fail(
                "Unreviewed opset or effectful scope", location, opset, pending=True
            )
        bound = set()
        for role, values in (("constant", graph.constvars), ("input", graph.invars)):
            for index, var in enumerate(values):
                self._tensor(var, f"{location}/{role} {index}", opset)
                if id(var) in bound:
                    self._fail("Duplicate scope binding", location, opset)
                bound.add(id(var))
        consts = getattr(closed, "consts", None)
        if consts is not None:
            if len(consts) != len(graph.constvars):
                self._fail("Constant payload arity mismatch", location, opset)
            for var, value in zip(graph.constvars, consts, strict=True):
                selected_key = is_selected_prng_key_dtype(var.aval.dtype)
                array = np.asarray(
                    selected_prng_key_words(value) if selected_key else value
                )
                expected = (
                    ("uint32", (2,))
                    if selected_key
                    else self._tensor(var, location, opset)[:2]
                )
                if (array.dtype.name, array.shape) != expected:
                    self._fail("Constant payload dtype/shape mismatch", location, opset)
        operations = []
        for index, eqn in enumerate(graph.eqns):
            name, site = eqn.primitive.name, f"{location}/equation {index}"
            if eqn.effects:
                self._fail("Effectful equation", site, opset, name, pending=True)
            for var in eqn.invars:
                self._tensor(var, site, opset)
                if not isinstance(var, Literal) and id(var) not in bound:
                    self._fail("Equation uses an unbound variable", site, opset, name)
            for var in eqn.outvars:
                self._tensor(var, site, opset)
                if not isinstance(var, DropVar):
                    if id(var) in bound:
                        self._fail(
                            "Equation redefines a bound variable", site, opset, name
                        )
                    bound.add(id(var))
            plugin = self.lowerers.get(name)
            if plugin is None:
                self._fail(
                    "No registered lowerer for this diagnostic operation",
                    site,
                    opset,
                    name,
                    pending=True,
                )
            if self._selected(eqn, site, opset):
                pass
            elif name.startswith("onnx_fn::"):
                self._function(eqn, site, opset)
            elif name in diagnostics._NESTED:
                self._nested(eqn, site, opset, ancestors)
            elif name in diagnostics._RULES:
                self._ordinary(eqn, site, opset)
            else:
                self._fail(
                    "Operation has no explicit diagnostic structural family",
                    site,
                    opset,
                    name,
                    pending=True,
                )
            operations.append(
                (
                    name,
                    f"{type(plugin).__module__}.{type(plugin).__qualname__}",
                    tuple(sorted(eqn.params)),
                )
            )
        for var in graph.outvars:
            self._tensor(var, f"{location}/output", opset)
            if not isinstance(var, Literal) and id(var) not in bound:
                self._fail("Scope returns an unbound variable", location, opset)
        self.scopes.append(
            diagnostics.StructuralScope(
                location,
                tuple(self._tensor(var, location, opset) for var in graph.invars),
                tuple(self._tensor(var, location, opset) for var in graph.outvars),
                tuple(operations),
            )
        )


@pytest.fixture(autouse=True)
def _configuration() -> Iterator[None]:
    policy = active_precision_policy()
    callables = (
        nnx.Dropout.__call__,
        nn.Dropout.__call__,
        eqx.nn.Dropout.__call__,
        jax.random.key,
        jax.random.fold_in,
        jax.random.bernoulli,
        jax.random.uniform,
    )
    previous = {
        name: getattr(jax.config, name)
        for name in (
            "jax_enable_x64",
            "jax_platforms",
            "jax_default_matmul_precision",
            "jax_numpy_dtype_promotion",
            "jax_numpy_rank_promotion",
        )
    }
    import_all_plugins()
    try:
        yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)
        assert active_precision_policy() is policy
        assert callables == (
            nnx.Dropout.__call__,
            nn.Dropout.__call__,
            eqx.nn.Dropout.__call__,
            jax.random.key,
            jax.random.fold_in,
            jax.random.bernoulli,
            jax.random.uniform,
        )


@onnx_function
def _dropout_named(
    x: Any, *, target: Callable[..., Any], deterministic: Any = True
) -> Any:
    return target(x, deterministic)


def _target(framework: str, rate: float) -> Callable[..., Any]:
    if framework == "nnx":
        block = nnx.Dropout(rate, deterministic=True, rngs=nnx.Rngs(31))
        return lambda x, flag: block(x, deterministic=flag)
    if framework == "linen":
        # This unchanged Linen module receives a fixed external bridge key.
        block = nn.Dropout(rate)
        key = jax.random.key(31)
        return lambda x, flag: block.apply(
            {}, x, deterministic=flag, rngs={"dropout": key}
        )
    block = eqx.nn.Dropout(p=rate, inference=True)
    key = jax.random.key(31)
    return lambda x, flag: block(x, inference=flag, key=key)


def _model(target: Callable[..., Any], scope: str) -> Callable[..., Any]:
    def model(x: Any, *, runtime: Any = True) -> Any:
        if scope == "root":
            return target(x, runtime)
        if scope == "function":
            return _dropout_named(x, target=target, deterministic=runtime)
        if scope == "if":
            return jax.lax.cond(
                jax.lax.reduce_sum(x, axes=(0,)) > np.float32(0),
                lambda value: _dropout_named(
                    value, target=target, deterministic=runtime
                ),
                lambda value: -_dropout_named(
                    -value, target=target, deterministic=runtime
                ),
                x,
            )
        return jax.lax.fori_loop(
            0,
            2,
            lambda index, value: _dropout_named(
                value, target=target, deterministic=runtime
            ),
            x,
        )

    return model


@onnx_function
def _dropout_state_named(
    x: Any,
    *words: Any,
    preparation: ExplicitStatePreparation,
    deterministic: Any = True,
) -> Any:
    # The owner is reconstructed by preparation inside this Function's actual
    # concrete-Boolean branch, never captured from a parent trace.
    return preparation(x, deterministic, *words)


def _lifted_model(
    preparation: ExplicitStatePreparation, scope: str
) -> Callable[..., Any]:
    """External state lifting of the four original scope skeletons.

    The original _model remains the whole-model reference wherever valid. Only
    this prepared workflow adds state carry and two loop-iteration audit tapes.
    """

    def named(x: Any, runtime: Any, words: tuple[Any, ...]) -> Any:
        return _dropout_state_named(
            x, *words, preparation=preparation, deterministic=runtime
        )

    def model(x: Any, *words: Any, runtime: Any = True) -> Any:
        if scope == "root":
            return preparation(x, runtime, *words)
        if scope == "function":
            return named(x, runtime, words)
        if scope == "if":

            def positive(operands: Any) -> Any:
                value, *state = operands
                return named(value, runtime, tuple(state))

            def negative(operands: Any) -> Any:
                value, *state = operands
                y, *next_state = named(-value, runtime, tuple(state))
                return (-y, *next_state)

            return jax.lax.cond(
                jax.lax.reduce_sum(x, axes=(0,)) > np.float32(0),
                positive,
                negative,
                (x, *words),
            )
        width = 1 + len(words)

        def body(index: Any, carry: Any) -> Any:
            value, *state = carry[:width]
            result = named(value, runtime, tuple(state))
            first, second = carry[width : 2 * width], carry[2 * width :]
            return (
                *result,
                *(
                    jax.lax.select(index == 0, new, old)
                    for new, old in zip(result, first, strict=True)
                ),
                *(
                    jax.lax.select(index == 1, new, old)
                    for new, old in zip(result, second, strict=True)
                ),
            )

        return jax.lax.fori_loop(0, 2, body, (x, *words, x, *words, x, *words))

    return model


@contextmanager
def _native_settings(x64: bool) -> Iterator[None]:
    previous = {
        name: getattr(jax.config, name)
        for name in ("jax_default_prng_impl", "jax_threefry_partitionable")
    }
    try:
        jax.config.update("jax_default_prng_impl", "threefry2x32")
        jax.config.update("jax_threefry_partitionable", True)
        with (
            jax.enable_x64(x64),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)


def _prepare(framework: str, rate: float) -> ExplicitStatePreparation:
    target = _target(framework, rate)
    adapter = (
        NNXDropoutState.from_target(target)
        if framework == "nnx"
        else (
            LinenDropoutState.from_target(target)
            if framework == "linen"
            else EqxDropoutState.from_target(target)
        )
    )
    return ExplicitStatePreparation(adapter, _dropout_named)


def _state(target: Any, framework: str) -> tuple[np.ndarray, ...]:
    if framework != "nnx":
        return ()
    # Independent reference extraction from the original owner. It does not
    # call preparation.words_from_target or reconstruct an owner from words.
    owner = closure_contents(target)["block"]
    return tuple(
        np.asarray(
            jax.random.key_data(value)
            if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
            else value
        ).copy()
        for value in jax.tree_util.tree_leaves(nnx.state(owner, nnx.RngState))
    )


def _exact(actual: Any, expected: Any) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@dataclass(frozen=True)
class _NativeStep:
    output: tuple[np.ndarray, ...]
    before: tuple[np.ndarray, ...]
    after: tuple[np.ndarray, ...]


_FLAGS = (True, False, False, True)


def _sequential_native(
    framework: str, rate: float, scope: str, x: np.ndarray, flags: tuple[bool, ...]
) -> tuple[_NativeStep, ...]:
    """Actual original calls, selected in host order without captured mutation."""
    target = _target(framework, rate)
    steps = []
    for flag in flags:
        before = _state(target, framework)
        value = jnp.asarray(x)
        tape = []
        for _ in range(2 if scope == "loop" else 1):
            if scope == "if" and not bool(np.sum(x) > np.float32(0)):
                value = -_dropout_named(-value, target=target, deterministic=flag)
            elif scope == "root":
                value = target(value, flag)
            else:
                value = _dropout_named(value, target=target, deterministic=flag)
            tape.append((np.asarray(value).copy(), *_state(target, framework)))
        after = _state(target, framework)
        output = (np.asarray(value).copy(), *after)
        if scope == "loop":
            output += tuple(item for iteration in tape for item in iteration)
        steps.append(_NativeStep(output, before, after))
    return tuple(steps)


def _whole_native(
    framework: str,
    rate: float,
    scope: str,
    x: np.ndarray,
    flags: tuple[bool, ...],
    reference: tuple[_NativeStep, ...],
    local: Any = None,
) -> dict[str, Any]:
    """Use the unchanged whole model where the original owner contract permits."""
    prerequisite = framework == "nnx" and rate > 0 and scope in {"if", "loop"}
    receipt: dict[str, Any] = {
        "raw_captured_mutation_requires_state_lifting": prerequisite,
        "whole_original_eager_checked": [],
        "whole_original_jit_checked": [],
        "_saved_native_arrays": {},
    }
    for role in ("eager", "jit"):
        target = _target(framework, rate)
        model = (
            _model(target, scope)
            if local is None
            else _original_local_model(target, local)
        )
        parameter = "runtime" if local is None else "deterministic"
        if framework == "nnx":
            owner = closure_contents(target)["block"]
            original_closure = NativeClosure.capture(target, "block")

            def original_call(block: Any, values: Any, *, flag: bool) -> Any:
                rebound = original_closure.rebind_owner(block)
                original = (
                    _model(rebound, scope)
                    if local is None
                    else _original_local_model(rebound, local)
                )
                return original(values, **{parameter: flag})

            compiled = nnx.jit(original_call, static_argnames=("flag",))
        for index, (flag, step) in enumerate(zip(flags, reference, strict=True)):
            if prerequisite and not flag:
                continue
            before = _state(target, framework)
            if role == "eager":
                y = model(jnp.asarray(x), **{parameter: flag})
            elif framework == "nnx":
                y = compiled(owner, jnp.asarray(x), flag=flag)
            else:
                # A fresh native compilation for this concrete original flag.
                y = jax.jit(lambda values: model(values, **{parameter: flag}))(
                    jnp.asarray(x)
                )
            _exact(y, step.output[0])
            after = _state(target, framework)
            if not prerequisite:
                for actual, expected in zip(after, step.after, strict=True):
                    _exact(actual, expected)
            else:
                for actual, expected in zip(after, before, strict=True):
                    _exact(actual, expected)
            saved = receipt["_saved_native_arrays"]
            saved[f"whole_{role}_step{index}_y"] = np.asarray(y).copy()
            saved.update(
                {
                    f"whole_{role}_step{index}_before{j}": value
                    for j, value in enumerate(before)
                }
            )
            saved.update(
                {
                    f"whole_{role}_step{index}_after{j}": value
                    for j, value in enumerate(after)
                }
            )
            receipt[f"whole_original_{role}_checked"].append(flag)
    return receipt


def _prepared_native(
    model: Callable[..., Any],
    preparation: ExplicitStatePreparation,
    x: np.ndarray,
    flags: tuple[bool, ...],
    reference: tuple[_NativeStep, ...],
    parameter: str = "runtime",
) -> tuple[tuple[np.ndarray, ...], ...]:
    outputs = []
    for compiled in (False, True):
        runner = jax.jit(model) if compiled else model
        words = preparation.adapter.initial_words
        for flag, step in zip(flags, reference, strict=True):
            result = tuple(
                np.asarray(value)
                for value in runner(
                    jnp.asarray(x), *words, **{parameter: jnp.asarray(flag)}
                )
            )
            assert len(result) == len(step.output)
            for actual, expected in zip(result, step.output, strict=True):
                _exact(actual, expected)
            words = tuple(jnp.asarray(value) for value in result[1 : 1 + len(words)])
            outputs.append(result)
    return tuple(outputs)


def _original_local_model(target: Any, local: Any) -> Any:
    def model(x: Any, *, deterministic: Any = False) -> Any:
        if local == "default":
            return _dropout_named(x, target=target)
        return _dropout_named(x, target=target, deterministic=local)

    return model


def _lifted_local_model(preparation: ExplicitStatePreparation, local: Any) -> Any:
    def model(x: Any, *words: Any, deterministic: Any = False) -> Any:
        if local == "default":
            return _dropout_state_named(x, *words, preparation=preparation)
        return _dropout_state_named(
            x, *words, preparation=preparation, deterministic=local
        )

    return model


def _state_contract(
    preparation: ExplicitStatePreparation,
    rate: float,
    scope: str,
    flag: bool,
    step: _NativeStep,
) -> None:
    width = len(preparation.adapter.initial_words)
    assert len(step.before) == len(step.after) == width
    for index, (before, after) in enumerate(zip(step.before, step.after, strict=True)):
        assert before.dtype == after.dtype == np.dtype(np.uint32)
        assert (
            before.shape
            == after.shape
            == preparation.adapter.initial_words[index].shape
        )
        if preparation.adapter.state_names[index].startswith("rng_count"):
            expected = before + np.asarray(
                int(not flag and rate > 0) * (2 if scope == "loop" else 1), np.uint32
            )
            _exact(after, expected)
        else:
            _exact(after, before)


def _physical_contract(
    actual: Any,
    step: _NativeStep,
    x: np.ndarray,
    flag: bool,
    rate: float,
    scope: str,
    state_width: int,
) -> None:
    assert len(actual) == len(step.output)
    for value, expected in zip(actual, step.output, strict=True):
        _exact(value, expected)
    iterations = [np.asarray(actual[0])]
    if scope == "loop":
        width = 1 + state_width
        iterations = [np.asarray(actual[width]), np.asarray(actual[2 * width])]
        _exact(iterations[-1], actual[0])
        for value, final in zip(actual[2 * width + 1 :], actual[1:width], strict=True):
            _exact(value, final)
        previous_state = step.before
        for index in range(2):
            state = actual[(index + 1) * width + 1 : (index + 2) * width]
            for before, after in zip(previous_state, state, strict=True):
                increment = (
                    np.asarray(int(not flag and rate > 0), np.uint32)
                    if before.shape == ()
                    else np.zeros_like(before)
                )
                _exact(after, before + increment)
            previous_state = tuple(state)
    previous = x
    for value in iterations:
        assert value.dtype == np.dtype(np.float32) and value.shape == (32,)
        assert np.isfinite(value).all()
        if flag or rate == 0:
            _exact(value, previous)
        else:
            # Exact native masks on nonzero inputs, exact native zero signs,
            # and zeros/scaling everywhere; no tolerance or mask reconstruction.
            assert np.logical_or(value == 0, value == previous * np.float32(2)).all()
            assert np.all(value[previous == 0] == 0)
        previous = value


def _save_native(path: Path, reference: Any, prepared: Any, receipt: Any) -> None:
    arrays = dict(receipt["_saved_native_arrays"])
    for index, step in enumerate(reference):
        arrays.update(
            {f"step{index}_output{j}": value for j, value in enumerate(step.output)}
        )
        arrays.update(
            {f"step{index}_before{j}": value for j, value in enumerate(step.before)}
        )
        arrays.update(
            {f"step{index}_after{j}": value for j, value in enumerate(step.after)}
        )
    arrays.update(
        {
            f"prepared{index}_output{j}": value
            for index, result in enumerate(prepared)
            for j, value in enumerate(result)
        }
    )
    np.savez(path.with_suffix(".npz"), **arrays)
    receipt = {
        key: value for key, value in receipt.items() if key != "_saved_native_arrays"
    }
    receipt["saved_array_sha256"] = {
        name: hashlib.sha256(value.tobytes()).hexdigest()
        for name, value in arrays.items()
    }
    path.with_suffix(".json").write_text(json.dumps(receipt, sort_keys=True, indent=2))


def _native_matrix(
    framework: str,
    rate: float,
    scope: str,
    preparation: ExplicitStatePreparation,
    model: Any,
    tmp_path: Path,
    *,
    local: Any = None,
) -> dict[str, tuple[np.ndarray, tuple[_NativeStep, ...]]]:
    families = _families(scope)
    effective = (
        _FLAGS
        if local is None
        else (True if local == "default" else local,) * len(_FLAGS)
    )
    result = {}
    for family, x in families.items():
        reference = _sequential_native(framework, rate, scope, x, effective)
        if framework in {"linen", "eqx"}:
            training = [
                step
                for flag, step in zip(effective, reference, strict=True)
                if not flag
            ]
            for step in training[1:]:
                for actual, expected in zip(
                    step.output, training[0].output, strict=True
                ):
                    _exact(actual, expected)
        original = _whole_native(framework, rate, scope, x, _FLAGS, reference, local)
        prepared = _prepared_native(
            model,
            preparation,
            x,
            _FLAGS,
            reference,
            "runtime" if local is None else "deterministic",
        )
        for flag, step in zip(effective, reference, strict=True):
            _state_contract(preparation, rate, scope, flag, step)
            _physical_contract(
                step.output,
                step,
                x,
                flag,
                rate,
                scope,
                len(preparation.adapter.initial_words),
            )
        _save_native(
            tmp_path / family,
            reference,
            prepared,
            {
                **original,
                "external_state_lifting": True,
                "native_flags": effective,
                "root_flags": _FLAGS,
                "state_names": preparation.adapter.state_names,
                "state_shapes": [
                    value.shape for value in preparation.adapter.initial_words
                ],
                "whole_model_prerequisite": (
                    "captured NNX mutation in nested raw JAX scopes is not a native state-carry workflow"
                    if original["raw_captured_mutation_requires_state_lifting"]
                    else None
                ),
                "original_body_code_identity": id(preparation.original_body.__code__),
                "original_target_code_identity": id(
                    preparation.adapter.original.original.__code__
                ),
                "other_original_capture_identities": {
                    name: id(value)
                    for name, value in closure_contents(
                        preparation.adapter.original.original
                    ).items()
                    if name != "block" or framework != "nnx"
                },
            },
        )
        result[family] = (x, reference)
    preparation.adapter.original.assert_original()
    captures = closure_contents(preparation.adapter.original.original)
    if framework in {"linen", "eqx"}:
        np.savez(
            tmp_path / "original-captured-key.npz",
            words=np.asarray(selected_prng_key_words(captures["key"])),
        )
    return result


def _families(scope: str) -> dict[str, np.ndarray]:
    families = {"ones": np.ones(32, np.float32), "zero": np.zeros(32, np.float32)}
    if scope == "if":
        families["negative"] = -np.ones(32, np.float32)
    return families


def _eqx_native(rate: float, scope: str, tmp_path: Path, *, local: Any = None) -> Any:
    matrix = {}
    effective = (
        _FLAGS
        if local is None
        else (True if local == "default" else local,) * len(_FLAGS)
    )
    for family, x in _families(scope).items():
        reference = _sequential_native("eqx", rate, scope, x, effective)
        original = _whole_native("eqx", rate, scope, x, _FLAGS, reference, local)
        _save_native(
            tmp_path / family,
            reference,
            (),
            {
                **original,
                "root_flags": _FLAGS,
                "native_flags": effective,
                "target_criterion": "unchanged diagnostic Dropout sample properties; no native stochastic mask equivalence claim",
                "prepared_native_equality": "separate bitwise evidence from the actual concrete-flag native branches",
                "raw_dynamic_flag_native_contract": "Original Equinox uses a Python inference condition; a traced flag requires the explicitly prepared caller",
            },
        )
        matrix[family] = (x, reference)
    return matrix


def _export_prepared(
    model: Any, preparation: ExplicitStatePreparation, scope: str, params: Any
) -> Any:
    observer = NativeDropoutDiagnosticValidator()
    specs = [jax.ShapeDtypeStruct((32,), np.float32)] + [
        jax.ShapeDtypeStruct(value.shape, value.dtype)
        for value in preparation.adapter.initial_words
    ]
    names = ["result", *(f"next_{name}" for name in preparation.adapter.state_names)]
    if scope == "loop":
        names += [
            f"iteration_{index}_{name}"
            for index in range(2)
            for name in ("result", *preparation.adapter.state_names)
        ]
    proto = ui._to_onnx(
        model,
        specs,
        input_params=params,
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=observer,
        input_names=["features", *preparation.adapter.state_names],
        output_names=names,
    )
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-diagnostic-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.UNVERIFIED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    assert observer.scopes
    assert len(proto.graph.output) == len(names)
    return proto, observer


class DropoutFlow:
    """Expand actual Function/If/Loop bindings before following control dependencies."""

    def __init__(self, model: onnx.ModelProto):
        self.functions = {(fn.domain, fn.name): fn for fn in model.functions}
        self.edges: dict[str, set[str]] = {}
        self.constants: dict[str, np.ndarray] = {}
        self.controls: list[dict[str, Any]] = []
        self.conditionals: list[dict[str, Any]] = []
        self.loops: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.outputs = self._graph(model.graph, "root", {})

    def _edge(self, target: str, *sources: str) -> None:
        self.edges.setdefault(target, set()).update(
            source for source in sources if source
        )

    def _graph(
        self,
        graph: Any,
        scope: str,
        outer: dict[str, str],
        arguments: list[str] | None = None,
    ) -> list[str]:
        function = isinstance(graph, onnx.FunctionProto)
        inputs = (
            list(graph.input) if function else [value.name for value in graph.input]
        )
        outputs = (
            list(graph.output) if function else [value.name for value in graph.output]
        )
        initializers = [] if function else list(graph.initializer)
        names = (
            set(inputs)
            | {value.name for value in initializers}
            | {name for node in graph.node for name in node.output if name}
        )
        env = {**outer, **{name: f"{scope}/{name}" for name in names}}

        def resolve(name: str) -> str:
            assert not name or name in env, f"Unbound serialized operand {scope}/{name}"
            return env[name] if name else ""

        if arguments is not None:
            assert len(inputs) == len(arguments)
            for name, argument in zip(inputs, arguments, strict=True):
                self._edge(resolve(name), argument)
        for value in initializers:
            self.constants[resolve(value.name)] = onnx.numpy_helper.to_array(value)
        for index, node in enumerate(graph.node):
            ins = [resolve(name) for name in node.input]
            outs = [resolve(name) for name in node.output]
            path = f"{scope}/{index}:{node.op_type}"
            key = node.domain, node.op_type
            if key in self.functions:
                self.calls.append(dict(path=path, key=key, inputs=ins, outputs=outs))
                returned = self._graph(self.functions[key], path, {}, ins)
                for output, returned_value in zip(outs, returned, strict=True):
                    self._edge(output, returned_value)
            elif node.op_type == "If":
                self.conditionals.append(
                    dict(path=path, predicate=ins[0], outputs=outs)
                )
                branches = [
                    self._graph(attr.g, path + "/" + attr.name, env)
                    for attr in node.attribute
                    if attr.type == onnx.AttributeProto.GRAPH
                ]
                assert len(branches) == 2
                for position, output in enumerate(outs):
                    self._edge(
                        output, ins[0], *(branch[position] for branch in branches)
                    )
            elif node.op_type == "Loop":
                self.loops.append(dict(path=path, inputs=ins, outputs=outs))
                body = next(attr.g for attr in node.attribute if attr.name == "body")
                returned = self._graph(
                    body, path + "/body", env, [path + "/iteration", *ins[1:]]
                )
                assert len(body.input) == len(ins)
                assert len(returned) == len(outs) + 1
                for formal, returned_value in zip(
                    body.input[1:], returned, strict=True
                ):
                    self._edge(path + "/body/" + formal.name, returned_value)
                for output, returned_value in zip(outs, returned[1:], strict=True):
                    self._edge(output, returned_value, *ins[:2])
            else:
                for output in outs:
                    self._edge(output, *ins)
                if node.op_type == "Constant":
                    attribute = next(
                        attr for attr in node.attribute if attr.name == "value"
                    )
                    self.constants[outs[0]] = onnx.numpy_helper.to_array(attribute.t)
                elif (
                    node.op_type in {"Identity", "Cast", "Not"}
                    and ins[0] in self.constants
                ):
                    value = self.constants[ins[0]]
                    self.constants[outs[0]] = (
                        np.logical_not(value) if node.op_type == "Not" else value
                    )
                if node.op_type == "Dropout":
                    assert len(ins) == 3
                    producer = next(
                        (item for item in graph.node if node.input[2] in item.output),
                        None,
                    )
                    self.controls.append(
                        dict(
                            path=path,
                            training=ins[2],
                            training_operator=(
                                producer.op_type if producer is not None else None
                            ),
                            outputs=outs,
                            rate=float(self.constants[ins[1]]),
                        )
                    )
        return [resolve(name) for name in outputs]

    def reach(self, roots: list[str]) -> set[str]:
        seen: set[str] = set()
        pending = list(roots)
        while pending:
            value = pending.pop()
            if value not in seen:
                seen.add(value)
                pending.extend(self.edges.get(value, ()))
        return seen


def _export(
    model: Callable[..., Any], params: dict[str, Any]
) -> tuple[onnx.ModelProto, DropoutDiagnosticValidator]:
    observer = DropoutDiagnosticValidator()
    artifact = ui._to_onnx(
        model,
        [jax.ShapeDtypeStruct((32,), np.float32)],
        input_params=params,
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=observer,
        input_names=["features"],
        output_names=["result"],
    )
    onnx.checker.check_model(artifact, full_check=True)
    report = get_precision_report(artifact)
    assert report is not None and report.profile == "source-diagnostic-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.UNVERIFIED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    return artifact, observer


def _session(model: onnx.ModelProto, optimize: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _prepared_flow(
    proto: Any,
    preparation: ExplicitStatePreparation,
    framework: str,
    rate: float,
    scope: str,
    *,
    local: Any = None,
) -> None:
    parameter = "runtime" if local is None else "deterministic"
    assert [value.name for value in proto.graph.input] == [
        "features",
        *preparation.adapter.state_names,
        parameter,
    ]
    flag = proto.graph.input[-1].type.tensor_type
    assert flag.elem_type == onnx.TensorProto.BOOL and not flag.shape.dim
    for value, expected in zip(
        proto.graph.input[1:-1], preparation.adapter.initial_words, strict=True
    ):
        tensor = value.type.tensor_type
        assert tensor.elem_type == onnx.TensorProto.UINT32
        assert tuple(dim.dim_value for dim in tensor.shape.dim) == expected.shape
    flow = DropoutFlow(proto)
    live = flow.reach(flow.outputs)
    assert "root/features" in live
    active_training = rate > 0 and local not in {"default", True}
    if local is None and rate > 0:
        assert f"root/{parameter}" in flow.reach([flow.outputs[0]])
        for index, name in enumerate(preparation.adapter.state_names):
            if name.startswith("rng_count"):
                assert f"root/{parameter}" in flow.reach([flow.outputs[1 + index]])
    elif local is not None or rate == 0:
        # Identity/default/literal cases need no artificial control or sink.
        assert f"root/{parameter}" not in live
    if scope != "root":
        assert flow.calls
        assert any(
            any(output in live for output in call["outputs"]) for call in flow.calls
        )
    if scope == "if":
        assert any(
            "root/features" in flow.reach([item["predicate"]])
            for item in flow.conditionals
        )
    if scope == "loop":
        assert flow.loops
        assert any(
            any(output in live for output in loop["outputs"]) for loop in flow.loops
        )
    if framework in {"linen", "eqx"} and active_training and scope != "root":
        key = closure_contents(preparation.adapter.original.original)["key"]
        expected = np.asarray(selected_prng_key_words(key))
        paths = [
            path
            for path, value in flow.constants.items()
            if value.dtype == np.uint32
            and value.shape == (2,)
            and value.tobytes() == expected.tobytes()
        ]
        assert (
            paths
        ), "The original fixed typed key must remain a physical Function constant"
        if scope == "loop":
            assert any("/body/" in path for path in paths)


def _runtime_prepared(
    proto: Any,
    preparation: ExplicitStatePreparation,
    matrix: Any,
    rate: float,
    scope: str,
    tmp_path: Path,
    *,
    local: Any = None,
    sample_properties: Callable[..., None] | None = None,
) -> None:
    payload = proto.SerializeToString()
    parameter = "runtime" if local is None else "deterministic"
    effective = (
        _FLAGS
        if local is None
        else (True if local == "default" else local,) * len(_FLAGS)
    )
    state_width = len(preparation.adapter.initial_words)
    (tmp_path / "model.onnx").write_bytes(payload)
    (tmp_path / "native-prefreeze.json").write_text(
        json.dumps(
            {
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                "criterion": "bitwise float32 outputs and uint32 state, including both loop iteration tapes",
                "families": list(matrix),
                "flags": _FLAGS,
                "effective_flags": effective,
                "reset_replays": 2,
                "optimizations": ["disabled", "all"],
                "external_state_lifting": True,
            },
            sort_keys=True,
            indent=2,
        )
    )
    for optimize in (False, True):
        session = _session(proto, optimize)
        saved = {}
        for family, (x, reference) in matrix.items():
            for replay in range(2):
                words = tuple(
                    np.asarray(value).copy()
                    for value in preparation.adapter.initial_words
                )
                for index, (root_flag, flag, step) in enumerate(
                    zip(_FLAGS, effective, reference, strict=True)
                ):
                    for actual, expected in zip(words, step.before, strict=True):
                        _exact(actual, expected)
                    feed = {"features": x, parameter: np.asarray(root_flag)}
                    feed.update(
                        zip(preparation.adapter.state_names, words, strict=True)
                    )
                    actual = session.run(None, feed)
                    _physical_contract(actual, step, x, flag, rate, scope, state_width)
                    if sample_properties is not None:
                        sample_properties(actual[0], step, x, flag, rate, scope)
                    words = tuple(actual[1 : 1 + state_width])
                    for state, expected in zip(words, step.after, strict=True):
                        _exact(state, expected)
                    saved.update(
                        {
                            f"{family}_replay{replay}_step{index}_output{j}": value
                            for j, value in enumerate(actual)
                        }
                    )
        np.savez(tmp_path / f"actual-{'all' if optimize else 'disabled'}.npz", **saved)
    assert proto.SerializeToString() == payload
    preparation.adapter.original.assert_original()
    for actual, expected in zip(
        preparation.adapter.words_from_target(preparation.adapter.original.original),
        preparation.adapter.initial_words,
        strict=True,
    ):
        _exact(actual, expected)
    assert get_precision_report(proto).checks[-1].status is CheckStatus.UNVERIFIED


def _eqx_sample_properties(
    actual: Any, step: _NativeStep, x: Any, inference: bool, rate: float, scope: str
) -> None:
    # Retain the pre-existing diagnostic criterion separately from the new
    # native bitwise comparisons in _physical_contract and saved evidence.
    assert actual.dtype == np.dtype(np.float32) and actual.shape == (32,)
    assert np.isfinite(actual).all()
    if inference or rate == 0:
        _exact(actual, step.output[0])
    else:
        factor = np.float32(4 if scope == "loop" else 2)
        assert np.logical_or(actual == 0, actual == x * factor).all()


def _eqx_runtime(scope: str, rate: float, tmp_path: Path, *, local: Any = None) -> None:
    # _target, _model and _eqx_native remain unchanged native reference owners.
    # Only this explicit caller adapts native Python flag selection to lax.cond.
    matrix = _eqx_native(rate, scope, tmp_path, local=local)
    preparation = _prepare("eqx", rate)
    model = (
        _lifted_model(preparation, scope)
        if local is None
        else _lifted_local_model(preparation, local)
    )
    parameter = "runtime" if local is None else "deterministic"
    prepared_arrays = {}
    for family, (x, reference) in matrix.items():
        prepared = _prepared_native(model, preparation, x, _FLAGS, reference, parameter)
        prepared_arrays.update(
            {
                f"{family}_call{index}_output{j}": value
                for index, result in enumerate(prepared)
                for j, value in enumerate(result)
            }
        )
    np.savez(tmp_path / "prepared-native-equality.npz", **prepared_arrays)
    original_key = closure_contents(preparation.adapter.original.original)["key"]
    np.savez(
        tmp_path / "original-captured-key.npz",
        words=np.asarray(selected_prng_key_words(original_key)),
    )
    if local not in {"default", True}:
        for _, reference in matrix.values():
            training = [
                step for flag, step in zip(_FLAGS, reference, strict=True) if not flag
            ]
            for step in training[1:]:
                for actual, expected in zip(
                    step.output, training[0].output, strict=True
                ):
                    _exact(actual, expected)
    params = {parameter: True if local is None else False}
    proto, observer = _export_prepared(model, preparation, scope, params)
    assert observer.scopes
    _prepared_flow(proto, preparation, "eqx", rate, scope, local=local)
    # Native Bernoulli/Threefry is visible here, never ONNX Dropout randomness.
    assert not DropoutFlow(proto).controls
    _runtime_prepared(
        proto,
        preparation,
        matrix,
        rate,
        scope,
        tmp_path,
        local=local,
        sample_properties=_eqx_sample_properties,
    )


@pytest.mark.parametrize("framework", ["nnx", "linen", "eqx"])
@pytest.mark.parametrize("scope", ["root", "function", "if", "loop"])
@pytest.mark.parametrize("rate", [0.5, 0.0], ids=["positive-rate", "zero-rate"])
def test_dropout_uses_real_named_operand_through_nested_scope(
    framework: str, scope: str, rate: float, tmp_path: Path
) -> None:
    # Keep the original 24 identities; both caller-x64 modes run inside each.
    for x64 in (False, True):
        with _native_settings(x64):
            if framework == "eqx":
                evidence = tmp_path / f"x64-{x64}"
                evidence.mkdir()
                _eqx_runtime(scope, rate, evidence)
                continue
            evidence = tmp_path / f"x64-{x64}"
            evidence.mkdir()
            preparation = _prepare(framework, rate)
            model = _lifted_model(preparation, scope)
            # Independent original calls and fresh native eager/JIT preparation
            # outputs are saved before the first plugin activation/export.
            matrix = _native_matrix(
                framework, rate, scope, preparation, model, evidence
            )
            proto, _ = _export_prepared(model, preparation, scope, {"runtime": True})
            _prepared_flow(proto, preparation, framework, rate, scope)
            _runtime_prepared(proto, preparation, matrix, rate, scope, evidence)


@pytest.mark.parametrize("framework", ["nnx", "linen", "eqx"])
@pytest.mark.parametrize("local", ["default", True, False])
def test_dropout_nested_default_and_literal_override_ignore_same_named_root_sample(
    framework: str, local: Any, tmp_path: Path
) -> None:
    for x64 in (False, True):
        with _native_settings(x64):
            if framework == "eqx":
                evidence = tmp_path / f"x64-{x64}"
                evidence.mkdir()
                _eqx_runtime("function", 0.5, evidence, local=local)
                continue
            evidence = tmp_path / f"x64-{x64}"
            evidence.mkdir()
            preparation = _prepare(framework, 0.5)
            model = _lifted_local_model(preparation, local)
            matrix = _native_matrix(
                framework, 0.5, "function", preparation, model, evidence, local=local
            )
            proto, _ = _export_prepared(
                model, preparation, "function", {"deterministic": False}
            )
            _prepared_flow(proto, preparation, framework, 0.5, "function", local=local)
            _runtime_prepared(
                proto, preparation, matrix, 0.5, "function", evidence, local=local
            )


@pytest.mark.parametrize("scope", ["if", "loop"])
def test_original_captured_nnx_training_requires_state_lifting(scope: str) -> None:
    """The raw prerequisite is separate from every positive prepared workflow."""
    for x64 in (False, True):
        with _native_settings(x64):
            target = _target("nnx", 0.5)
            before = _state(target, "nnx")
            with pytest.raises(TraceContextError, match="Cannot mutate RngCount"):
                _model(target, scope)(jnp.ones(32, np.float32), runtime=False)
            for actual, expected in zip(_state(target, "nnx"), before, strict=True):
                _exact(actual, expected)


def test_state_preparation_preserves_actual_callable_and_nonowner_captures() -> None:
    for x64 in (False, True):
        with _native_settings(x64):
            for framework in ("nnx", "linen", "eqx"):
                preparation = _prepare(framework, 0.5)
                original = preparation.adapter.original.original
                restored = preparation.adapter.target_from_words(
                    preparation.adapter.initial_words
                )
                assert restored.__code__ is original.__code__
                assert restored.__defaults__ is original.__defaults__
                assert restored.__kwdefaults__ is original.__kwdefaults__
                captures, restored_captures = closure_contents(
                    original
                ), closure_contents(restored)
                assert captures.keys() == restored_captures.keys()
                for name in captures:
                    if framework == "nnx" and name == "block":
                        assert restored_captures[name] is not captures[name]
                    else:
                        assert restored_captures[name] is captures[name]
                preparation.adapter.original.assert_original()
                for actual, expected in zip(
                    preparation.adapter.words_from_target(restored),
                    preparation.adapter.initial_words,
                    strict=True,
                ):
                    _exact(actual, expected)


def test_native_dropout_composition_keeps_unknown_operations_and_keys_pending() -> None:
    graph = jax.make_jaxpr(jax.lax.exp)(jax.ShapeDtypeStruct((32,), np.float32))
    with pytest.raises(
        SourcePrecisionError, match="no explicit diagnostic structural family"
    ) as error:
        NativeDropoutDiagnosticValidator()(graph, 23)
    assert error.value.category == "profile-exclusion"

    # A real selected key captured/passed through a conditional must reach the
    # same finite nested checks without NumPy dtype coercion.
    def conditional_key(flag: Any, key: Any) -> Any:
        return jax.lax.cond(
            flag,
            lambda value: jax.random.fold_in(value, np.uint32(1)),
            lambda value: jax.random.fold_in(value, np.uint32(2)),
            key,
        )

    with _native_settings(False):
        selected = jax.ShapeDtypeStruct((), jax.random.key(31).dtype)
        graph = jax.make_jaxpr(conditional_key)(
            jax.ShapeDtypeStruct((), np.bool_), selected
        )
        validator = NativeDropoutDiagnosticValidator()
        validator(graph, 23)
        nested = next(eqn for eqn in graph.jaxpr.eqns if eqn.primitive.name == "cond")
        mismatched = nested.replace(outvars=[nested.invars[0]])
        with pytest.raises(
            SourcePrecisionError, match="Nested dtype/shape/arity signature mismatch"
        ) as error:
            validator._nested(mismatched, "selected-key conditional", 23, frozenset())
        assert error.value.category == "lowering-defect"
        unselected = jax.ShapeDtypeStruct((), jax.random.key(31, impl="rbg").dtype)
        graph = jax.make_jaxpr(conditional_key)(
            jax.ShapeDtypeStruct((), np.bool_), unselected
        )
        nested = next(eqn for eqn in graph.jaxpr.eqns if eqn.primitive.name == "cond")
        with pytest.raises(
            SourcePrecisionError, match="Unselected native key implementation"
        ) as error:
            validator._nested(nested, "unselected-key conditional", 23, frozenset())
        assert error.value.category == "profile-exclusion"
    graph = jax.make_jaxpr(lambda words: jax.random.wrap_key_data(words, impl="rbg"))(
        jax.ShapeDtypeStruct((4,), np.uint32)
    )
    with pytest.raises(
        SourcePrecisionError, match="Unselected native key implementation"
    ) as error:
        NativeDropoutDiagnosticValidator()(graph, 23)
    assert error.value.category == "profile-exclusion"


@pytest.mark.parametrize("rate", [0.0, 0.5])
def test_original_nnx_dropout_rng_counter_prerequisite(rate: float) -> None:
    assert nnx.Dropout.__call__.__module__.startswith("flax.nnx.")
    block = nnx.Dropout(rate, rngs=nnx.Rngs(17))

    def count() -> int:
        return int(np.asarray(block.rngs.count.value))

    before = count()
    inference = np.asarray(block(jnp.ones(32, np.float32), deterministic=True))
    assert count() == before
    np.testing.assert_array_equal(inference, np.ones(32, np.float32), strict=True)
    training = np.asarray(block(jnp.ones(32, np.float32), deterministic=False))
    assert count() == before + int(rate > 0)
    assert np.isin(training, [0, 2] if rate else [1]).all()


def test_original_nnx_training_requires_rng_state() -> None:
    with pytest.raises(ValueError, match="rngs"):
        nnx.Dropout(0.5)(jnp.ones(32, np.float32), deterministic=False)


def test_original_linen_fixed_external_key_is_repeatable() -> None:
    assert nn.Dropout.__call__.__module__.startswith("flax.linen.")
    target = _target("linen", 0.5)
    x = jnp.ones(32, np.float32)
    np.testing.assert_array_equal(target(x, True), x, strict=True)
    first, repeated = np.asarray(target(x, False)), np.asarray(target(x, False))
    np.testing.assert_array_equal(first, repeated, strict=True)
    assert np.isin(first, [0, 2]).all()


def test_original_equinox_missing_key_and_external_key_preparation() -> None:
    assert eqx.nn.Dropout.__call__.__module__.startswith("equinox.")
    block = eqx.nn.Dropout(p=0.5)
    x = jnp.ones(32, np.float32)
    np.testing.assert_array_equal(block(x, inference=np.asarray(True)), x, strict=True)
    with pytest.raises(RuntimeError, match="key"):
        block(x, inference=np.asarray(False))
    training = np.asarray(block(x, inference=False, key=jax.random.key(9)))
    assert training.dtype == np.dtype(np.float32) and training.shape == (32,)
    assert np.isin(training, [0, 2]).all()


def test_original_spectral_norm_statistics_require_mutable_state() -> None:
    block = _SpectralNormDense(features=4, n_steps=1)
    x = jnp.arange(8, dtype=np.float32).reshape(1, 8)
    variables = block.init(jax.random.key(17), x, update_stats=False)
    fixed = block.apply(variables, x, update_stats=False)
    assert fixed.shape == (1, 4) and np.isfinite(np.asarray(fixed)).all()
    with pytest.raises(ModifyScopeVariableError):
        block.apply(variables, x, update_stats=True)
    _, updated = block.apply(variables, x, update_stats=True, mutable=["batch_stats"])
    before = jax.tree_util.tree_leaves(variables["batch_stats"])
    after = jax.tree_util.tree_leaves(updated["batch_stats"])
    assert any(
        not np.array_equal(np.asarray(old), np.asarray(new))
        for old, new in zip(before, after, strict=True)
    )


def test_dropout_validator_keeps_unknown_registered_operations_pending() -> None:
    original_rules = diagnostics._RULES
    graph = jax.make_jaxpr(jax.lax.exp)(jax.ShapeDtypeStruct((2,), np.float32))
    with pytest.raises(
        SourcePrecisionError, match="no explicit diagnostic structural family"
    ) as caught:
        DropoutDiagnosticValidator()(graph, 23)
    assert caught.value.category == "profile-exclusion"
    assert diagnostics._RULES is original_rules
    assert "nnx.dropout" not in diagnostics._RULES


@pytest.mark.parametrize(
    "defect",
    [
        "arity",
        "flag-dtype",
        "flag-shape",
        "output",
        "params",
        "rate-nan",
        "rate-one",
        "call-time",
    ],
)
def test_dropout_validator_rejects_malformed_explicit_schema(defect: str) -> None:
    def var(shape: tuple[int, ...], dtype: Any) -> Any:
        return SimpleNamespace(aval=ShapedArray(shape, dtype))

    eqn = SimpleNamespace(
        primitive=DropoutPlugin._PRIM,
        invars=[var((32,), np.float32), var((), np.bool_)],
        outvars=[var((32,), np.float32)],
        params={"rate": 0.5, "call_time": True},
    )
    if defect == "arity":
        eqn.invars.pop()
    elif defect == "flag-dtype":
        eqn.invars[1] = var((), np.int32)
    elif defect == "flag-shape":
        eqn.invars[1] = var((1,), np.bool_)
    elif defect == "output":
        eqn.outvars[0] = var((32,), np.float16)
    elif defect == "params":
        eqn.params["extra"] = 1
    elif defect == "call-time":
        eqn.params["call_time"] = 1
    else:
        eqn.params["rate"] = np.nan if defect == "rate-nan" else 1.0
    with pytest.raises(SourcePrecisionError):
        DropoutDiagnosticValidator()._ordinary(eqn, "malformed-dropout", 23)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("fault", ["other-schema", "unknown", "dtype", "boolean"])
def test_native_uniform_multiply_schema_cannot_ignore_or_reinterpret_parameters(
    x64: bool, fault: str
) -> None:
    from jax2onnx._compat.jax import ClosedJaxpr
    from jax2onnx.plugins.jax.random._source_prng import prove_native_uniform

    with _native_settings(x64):
        key = jax.random.key(17, impl="threefry2x32")
        equation = jax.make_jaxpr(
            lambda value: jax.random.uniform(value, (2, 3), dtype=jnp.float32)
        )(key).jaxpr.eqns[0]
        # Calibrate the actual installed native carrier before one corruption.
        assert prove_native_uniform(equation) is not None
        closed = equation.params["jaxpr"]
        body = closed.jaxpr
        equations = list(body.eqns)
        multiply = equations[10]
        assert multiply.primitive is jax.lax.mul_p
        original = dict(multiply.params)
        if fault == "other-schema":
            damaged = {} if original else {"out_dtype": None}
        elif fault == "unknown":
            damaged = {**original, "unowned": None}
        else:
            damaged = {
                **original,
                "out_dtype": np.float32 if fault == "dtype" else False,
            }
        equations[10] = multiply.replace(params=damaged)
        changed_body = body.replace(eqns=equations)
        changed = equation.replace(
            params={
                **equation.params,
                "jaxpr": ClosedJaxpr(changed_body, closed.consts),
            }
        )
        with pytest.raises(ValueError, match="primitive parameter"):
            prove_native_uniform(changed)
        assert dict(multiply.params) == original
        assert prove_native_uniform(equation) is not None


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("form", ["literal-probability", "weak-probability", "tensor"])
def test_native_dropout_stop_gradient_reuses_mandatory_product_validation(
    x64: bool, form: str
) -> None:
    with _native_settings(x64):
        if form == "literal-probability":
            graph = jax.make_jaxpr(lambda: jax.lax.stop_gradient(0.5))()
        else:
            dtype = np.float64 if x64 else np.float32
            graph = jax.make_jaxpr(jax.lax.stop_gradient)(
                jax.ShapeDtypeStruct(
                    () if form == "weak-probability" else (32,),
                    dtype,
                    weak_type=form == "weak-probability",
                )
            )
        body = diagnostics._body(graph)
        (equation,) = body.eqns
        assert equation.primitive is jax.lax.stop_gradient_p
        product_validator = validate_source_jaxpr
        calls = []
        original_rules = diagnostics._RULES

        def mandatory(scope: Any, opset: int) -> None:
            assert scope.eqns == [equation]
            assert scope.outvars is equation.outvars
            assert scope.constvars == () and scope.effects == ()
            product_validator(scope, opset)
            calls.append((scope, opset))

        with patch(f"{__name__}.validate_source_jaxpr", side_effect=mandatory):
            NativeDropoutDiagnosticValidator()(graph, 23)
        assert len(calls) == 1 and calls[0][1] == 23
        assert diagnostics._RULES is original_rules
        assert "stop_gradient" not in original_rules


@pytest.mark.parametrize(
    "fault",
    [
        "parameter",
        "empty-placement-parameter",
        "input-arity",
        "output-arity",
        "dtype",
        "shape",
        "weakness",
        "binding",
        "effects",
    ],
)
def test_native_dropout_stop_gradient_cannot_bypass_product_schema(fault: str) -> None:
    from jax2onnx._compat.jax import Var

    with _native_settings(False):
        graph = jax.make_jaxpr(jax.lax.stop_gradient)(
            jax.ShapeDtypeStruct((32,), np.float32)
        )
        (equation,) = diagnostics._body(graph).eqns
        validator = NativeDropoutDiagnosticValidator()
        assert validator._selected(equation, "native-stop-gradient-control", 23)
        if fault == "parameter":
            damaged = equation.replace(params={"unowned": None})
        elif fault == "empty-placement-parameter":
            damaged = equation.replace(params={"out_sharding": None})
        elif fault == "input-arity":
            damaged = equation.replace(invars=[])
        elif fault == "output-arity":
            damaged = equation.replace(outvars=[])
        elif fault in {"dtype", "shape", "weakness"}:
            updates = {
                "dtype": {"dtype": np.dtype(np.float16)},
                "shape": {"shape": (16, 2)},
                "weakness": {"weak_type": True},
            }[fault]
            damaged = equation.replace(
                outvars=[Var(equation.outvars[0].aval.update(**updates))]
            )
        elif fault == "binding":
            damaged = equation.replace(outvars=[equation.invars[0]])
        else:
            damaged = equation.replace(effects=frozenset({"unowned-effect"}))
        with pytest.raises(SourcePrecisionError):
            validator._selected(damaged, "damaged-stop-gradient", 23)
        assert validator._selected(equation, "native-stop-gradient-control", 23)


def test_native_dropout_stop_gradient_rejects_a_name_only_impostor() -> None:
    with _native_settings(False):
        graph = jax.make_jaxpr(jax.lax.stop_gradient)(
            jax.ShapeDtypeStruct((32,), np.float32)
        )
        body = diagnostics._body(graph)
        (equation,) = body.eqns
        impostor = SimpleNamespace(
            primitive=SimpleNamespace(name="stop_gradient"),
            invars=equation.invars,
            outvars=equation.outvars,
            params=equation.params,
            effects=equation.effects,
        )
        validator = NativeDropoutDiagnosticValidator()
        assert not validator._selected(impostor, "name-only-impostor", 23)
        with pytest.raises(
            SourcePrecisionError, match="no explicit diagnostic structural family"
        ):
            validator._walk(
                SimpleNamespace(
                    invars=body.invars,
                    constvars=body.constvars,
                    outvars=body.outvars,
                    eqns=[impostor],
                    effects=body.effects,
                ),
                23,
                "name-only-impostor",
                frozenset(),
            )
