# tests/extra_tests/converter/test_source_nnx_function_state.py

"""NNX owners reconstructed outside the actual named Function boundary."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
from jax.experimental.key_reuse import _core as key_reuse_core
from jax.extend.core import jaxpr_as_fun
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnxruntime.capi import _pybind_state
import pytest

from jax2onnx import onnx_function, to_onnx, user_interface as ui
from jax2onnx._compat.jax import (
    FunctionKeyReuse,
    Literal,
    Tracer,
    function_key_reuse_signature,
)
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.plugins import plugin_system
from jax2onnx.plugins._source_function_state import (
    FunctionOperandLayout,
    FunctionStateCall,
)
from jax2onnx.plugins.plugin_system import FunctionPlugin
from jax2onnx.plugins.flax.nnx._source_function_state import prepare_nnx_function_state
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter.test_source_mlp_runtime_acceptance import (
    _FLAGS,
    _ROLES,
    _configuration,
    _native_references,
    _port,
    _runtime_configuration,
    _same_state,
    _state,
)
from tests.extra_tests.converter.test_source_mlp_state_preparation import (
    PreparedMLP,
    array_snapshot,
)
from tests.extra_tests.converter.test_source_square_variance import _payload

_OUTPUT_CRITERIA = (2e-6, 1e-7)
_FIXED_TYPED_KEY: Any = None


@onnx_function
class _TwoDropoutOwner(nnx.Module):
    def __init__(self) -> None:
        rngs = nnx.Rngs(17)
        self.dropout1 = nnx.Dropout(rate=0.1, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.1, rngs=rngs)
        self.scale = nnx.Param(jnp.asarray(np.arange(10) / 16 + 0.5, np.float32))

    def __call__(self, x: Any, deterministic: bool = True) -> Any:
        value = self.dropout1(x, deterministic=deterministic)
        return self.dropout2(value * self.scale[...], deterministic=deterministic)


@onnx_function
class _CountingChild(nnx.Module):
    def __init__(self, scale: float, seed: int) -> None:
        rngs = nnx.Rngs(seed)
        self.dropout1 = nnx.Dropout(rate=0.1, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.1, rngs=rngs)
        self.scale = nnx.Param(jnp.asarray(scale, np.float32))
        self.calls = nnx.Variable(jnp.asarray(0, np.float32))

    def __call__(
        self, x: Any, deterministic: bool = True, *, increment: Any = np.float32(1)
    ) -> Any:
        self.calls[...] = self.calls[...] + increment
        value = self.dropout1(x, deterministic=deterministic)
        value = self.dropout2(value * self.scale[...], deterministic=deterministic)
        return value, x, x, self.scale[...]


class _TwoCalls(nnx.Module):
    def __init__(self, shared: bool) -> None:
        self.left = _CountingChild(0.75, 23)
        self.right = self.left if shared else _CountingChild(1.25, 29)

    def __call__(self, x: Any, deterministic: bool = True) -> Any:
        # This array constant checks keyword/state packing. It does not vary a
        # runtime keyword value across calls.
        increment = jnp.asarray(np.float32(1))
        first, *_ = self.left(x, deterministic=deterministic, increment=increment)
        second, *_ = self.right(first, deterministic=deterministic, increment=increment)
        # Retain original argument aliases and each original instance parameter.
        return second, x, x, self.left.scale[...], self.right.scale[...]


@onnx_function
class _StatelessOwner(nnx.Module):
    def __init__(self) -> None:
        self.scale = nnx.Param(jnp.asarray(0.75, np.float32))

    def __call__(self, x: Any) -> Any:
        return x * self.scale[...], x, x, self.scale[...]


class _UpdatePair(nnx.Module):
    def __init__(self) -> None:
        shared_leaf = jnp.asarray(1, np.float32)
        raw = {"nested": [shared_leaf, (shared_leaf, jnp.asarray(2, np.float32))]}
        self.first = nnx.Variable(raw, witness_metadata={"tags": [object(), object()]})
        self.second = nnx.Variable(jnp.asarray(3, np.float32))
        self.alias = self.first

    def __call__(self, x: Any) -> Any:
        return x


@onnx_function
class _ConsumedKeyOwner(nnx.Module):
    def __init__(self, key: Any, behavior: str, *, repeat_inside: bool = False) -> None:
        self.base = nnx.Variable(key)
        self.behavior = behavior
        self.repeat_inside = repeat_inside

    def __call__(self, x: Any) -> Any:
        if self.behavior == "split":
            first = jax.random.split(self.base[...])
            result = (
                (first, jax.random.split(self.base[...]))
                if self.repeat_inside
                else first
            )
        else:
            first = jax.random.uniform(self.base[...], x.shape)
            result = (
                (first, jax.random.uniform(self.base[...], x.shape))
                if self.repeat_inside
                else first
            )
        return result, x


@onnx_function
class _CapturedKeyOwner(nnx.Module):
    def __init__(self) -> None:
        self.scale = nnx.Param(jnp.asarray(0.75, np.float32))

    def __call__(self, x: Any) -> Any:
        return x * self.scale[...] + jax.random.uniform(_FIXED_TYPED_KEY, x.shape)


@onnx_function
class _KeyAliasOwner(nnx.Module):
    def __init__(self, key: Any) -> None:
        self.key = nnx.RngKey(key, tag="alias")

    def __call__(self) -> Any:
        return self.key[...], self.key[...]


@onnx_function
class _DebugFlagOwner(nnx.Module):
    def __call__(self, x: Any) -> Any:
        return x + np.float32(0.125 if jax.config.jax_debug_key_reuse else 0.25)


@onnx_function
class _LoopKeyReuseOwner(nnx.Module):
    def __init__(self, key: Any) -> None:
        self.base = nnx.Variable(key)

    def __call__(self, x: Any, *, offset: Any) -> Any:
        def step(carry: Any, unused: Any) -> Any:
            return carry + offset + jax.random.uniform(self.base[...], x.shape), None

        return jax.lax.scan(step, x, None, length=2)[0]


@dataclass(frozen=True)
class _AllStateCall:
    """This finite fixture exposes all native state at the outer API boundary."""

    original: Any
    graphdef: Any
    tree: Any
    implementations: tuple[str | None, ...]
    initial: tuple[Any, ...]

    @classmethod
    def from_owner(cls, owner: Any) -> _AllStateCall:
        graphdef, state = nnx.split(owner)
        leaves, tree = jax.tree.flatten(state)
        implementations = tuple(
            (
                str(jax.random.key_impl(value))
                if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
                else None
            )
            for value in leaves
        )
        initial = tuple(
            jax.random.key_data(value) if implementation is not None else value
            for value, implementation in zip(leaves, implementations, strict=True)
        )
        return cls(owner, graphdef, tree, implementations, initial)

    def state(self, values: Any) -> Any:
        return jax.tree.unflatten(
            self.tree,
            [
                (
                    jax.random.wrap_key_data(value, impl=implementation)
                    if implementation is not None
                    else value
                )
                for value, implementation in zip(
                    values, self.implementations, strict=True
                )
            ],
        )

    def fresh_original(self) -> Any:
        return nnx.merge(self.graphdef, self.state(self.initial), copy=True)

    def values(self, owner: Any) -> tuple[Any, ...]:
        graphdef, state = nnx.split(owner)
        leaves, tree = jax.tree.flatten(state)
        assert graphdef == self.graphdef and tree == self.tree
        return tuple(
            jax.random.key_data(value) if implementation is not None else value
            for value, implementation in zip(leaves, self.implementations, strict=True)
        )

    def _branch(self, deterministic: bool, x: Any, *values: Any) -> Any:
        # This merge must stay OUTSIDE the decorated child call and INSIDE cond.
        owner = nnx.merge(self.graphdef, self.state(values), copy=True)
        result = owner(x, deterministic=deterministic)
        return (*result, *self.values(owner))

    def __call__(self, x: Any, deterministic: Any, *values: Any) -> Any:
        return jax.lax.cond(
            deterministic,
            lambda args: self._branch(True, *args),
            lambda args: self._branch(False, *args),
            (x, *values),
        )


@contextmanager
def _deny_runtime() -> Any:
    attempts = []

    def denied(*args: Any, **kwargs: Any) -> Any:
        attempts.append(True)
        raise AssertionError("NNX native and restoration roles deny ORT constructors")

    with (
        patch.object(ort, "InferenceSession", denied),
        patch.object(_pybind_state, "InferenceSession", denied),
    ):
        yield
    assert not attempts


def _specs(initial: Any) -> list[Any]:
    return [
        jax.ShapeDtypeStruct((2, 10), np.float32),
        jax.ShapeDtypeStruct((), np.bool_),
        *(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in initial),
    ]


def _same_values(actual: Any, expected: Any) -> None:
    assert tuple(_payload(value) for value in actual) == tuple(
        _payload(value) for value in expected
    )


def _same_results(actual: Any, expected: Any) -> None:
    assert len(actual) == len(expected)
    assert _payload(actual[0])[:2] == _payload(expected[0])[:2]
    assert np.all(np.isfinite(actual[0]))
    assert np.all(np.isfinite(expected[0]))
    np.testing.assert_allclose(
        actual[0], expected[0], rtol=_OUTPUT_CRITERIA[0], atol=_OUTPUT_CRITERIA[1]
    )
    _same_values(actual[1:], expected[1:])


def _no_tracers(owner: Any) -> None:
    assert all(
        not isinstance(value, Tracer) for value in jax.tree.leaves(nnx.state(owner))
    )


def _binding_context() -> Any:
    return (
        plugin_system._IN_FUNCTION_BUILD.get(),
        _TwoDropoutOwner.__call__,
        _CountingChild.__call__,
        _StatelessOwner.__call__,
        _ConsumedKeyOwner.__call__,
        _CapturedKeyOwner.__call__,
        _KeyAliasOwner.__call__,
        _DebugFlagOwner.__call__,
        _LoopKeyReuseOwner.__call__,
        tuple(
            (id(plugin), id(plugin._orig_fn))
            for plugin in plugin_system.ONNX_FUNCTION_PLUGIN_REGISTRY.values()
            if plugin.target
            in (
                _TwoDropoutOwner,
                _CountingChild,
                _StatelessOwner,
                _ConsumedKeyOwner,
                _CapturedKeyOwner,
                _KeyAliasOwner,
                _DebugFlagOwner,
                _LoopKeyReuseOwner,
            )
        ),
    )


def _assert_function_ports(model: Any, built: list[Any]) -> None:
    """Compare actual serialized definitions with actual returned body avals."""
    assert model.functions and len(model.functions) == len(built)
    serialized = {
        (function.domain, function.name): function for function in model.functions
    }
    for definition, body in built:
        function = serialized[definition.domain, definition.name]
        actual = {value.name: value for value in function.value_info}
        assert len(function.input) == len(body.jaxpr.invars)
        assert len(function.output) == len(body.jaxpr.outvars)
        assert len(set(function.output)) == len(function.output)
        assert not set(function.input) & set(function.output)
        key_ports = count_ports = 0
        for names, variables in (
            (function.input, body.jaxpr.invars),
            (function.output, body.jaxpr.outvars),
        ):
            for name, variable in zip(names, variables, strict=True):
                representation = representation_for_aval(variable.aval)
                _port(
                    actual[name],
                    representation.physical_dtype,
                    representation.physical_shape,
                )
                if representation.encoding == "threefry-key-words":
                    key_ports += 1
                elif (
                    variable.aval.dtype == np.dtype(np.uint32)
                    and variable.aval.shape == ()
                ):
                    count_ports += 1
        assert key_ports >= 2 and count_ports >= 2


@contextmanager
def _observe_functions() -> Any:
    bodies = []
    built = []
    original_trace = FunctionPlugin._trace_function_body
    original_build = FunctionPlugin._build_function

    def trace(plugin: Any, *args: Any, **kwargs: Any) -> Any:
        body = original_trace(plugin, *args, **kwargs)
        bodies.append(body)
        return body

    def build(plugin: Any, *args: Any, **kwargs: Any) -> Any:
        start = len(bodies)
        definition = original_build(plugin, *args, **kwargs)
        assert len(bodies) == start + 1
        built.append((definition, bodies[-1]))
        return definition

    with (
        patch.object(FunctionPlugin, "_trace_function_body", trace),
        patch.object(FunctionPlugin, "_build_function", build),
    ):
        yield built
    assert FunctionPlugin._trace_function_body is original_trace
    assert FunctionPlugin._build_function is original_build


def test_external_merge_two_dropout_owner_threads_tfft_and_reset(
    tmp_path: Path, record_property: Any
) -> None:
    with _runtime_configuration(False):
        prepared = PreparedMLP.from_original(_TwoDropoutOwner(), "nnx")
        canonical = array_snapshot(prepared.original)
        initial = _state(prepared.initial_words)
        x = ((np.arange(20, dtype=np.float32) - 9) / 16).reshape(2, 10)
        with _deny_runtime():
            references, held, identities = _native_references(
                prepared, x, "nonzero", tmp_path
            )
            assert len(held) == 4
            before_export = _configuration()
            binding_context = _binding_context()
            with _observe_functions() as built, jax.checking_leaks():
                model = ui._to_onnx(
                    prepared,
                    _specs(initial),
                    precision_policy=PrecisionPolicy.SOURCE,
                    return_mode="proto",
                    opset=23,
                )
            assert _configuration() == before_export
            assert _binding_context() == binding_context
            _assert_function_ports(model, built)
            assert array_snapshot(prepared.original) == canonical
            _no_tracers(prepared.original)
        path = tmp_path / "external-owner.onnx"
        onnx.save_model(model, path)
        onnx.checker.check_model(onnx.load_model(path), full_check=True)
        report = get_precision_report(model)
        assert report is not None and report.profile == "source-static-v1"
        assert all(
            check.status is CheckStatus.UNVERIFIED
            for check in report.checks
            if check.category in {"runtime-execution", "numerical-acceptance"}
        )
        for index, word in enumerate(initial):
            _port(model.graph.input[index + 2], np.uint32, word.shape)
            _port(model.graph.output[index + 1], np.uint32, word.shape)
        sessions = runs = 0
        for optimize in (False, True):
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                if optimize
                else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            session = ort.InferenceSession(
                str(path), options, providers=["CPUExecutionProvider"]
            )
            sessions += 1
            assert session.get_providers() == ["CPUExecutionProvider"]
            names = [value.name for value in session.get_inputs()]
            for replay in range(2):
                words = initial
                for index, flag in enumerate(_FLAGS):
                    outputs = session.run(
                        None,
                        dict(zip(names, (x, np.asarray(flag), *words), strict=True)),
                    )
                    np.savez(
                        tmp_path / f"target-{optimize}-{replay}-{index}.npz",
                        **{f"output{i}": value for i, value in enumerate(outputs)},
                        input=x,
                        flag=np.asarray(flag),
                        **{f"state-before-{i}": value for i, value in enumerate(words)},
                    )
                    runs += 1
                    for role in _ROLES:
                        step = references[role][index]
                        _same_state(words, step.before)
                        _same_state(outputs[1:], step.after)
                        assert outputs[0].dtype == step.y.dtype
                        assert outputs[0].shape == step.y.shape
                        assert np.all(np.isfinite(outputs[0]))
                        assert np.all(np.isfinite(step.y))
                        np.testing.assert_allclose(
                            outputs[0],
                            step.y,
                            rtol=_OUTPUT_CRITERIA[0],
                            atol=_OUTPUT_CRITERIA[1],
                        )
                    if flag:
                        _same_state(outputs[1:], words)
                    words = tuple(outputs[1:])
        assert sessions == 2 and runs == 16
        assert array_snapshot(prepared.original) == canonical
        record_property(
            "external-owner-state",
            json.dumps(
                {
                    "flags": list(_FLAGS),
                    "replays": 2,
                    "sessions": sessions,
                    "runs": runs,
                    "native_roles": identities,
                    "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
                sort_keys=True,
            ),
        )


@pytest.mark.parametrize(
    "shared", [True, False], ids=["shared-child-twice", "two-instances"]
)
def test_external_merge_preserves_non_rng_updates_aliases_and_instance_state(
    shared: bool, tmp_path: Path
) -> None:
    with _runtime_configuration(False), _deny_runtime():
        prepared = _AllStateCall.from_owner(_TwoCalls(shared))
        canonical = array_snapshot(prepared.original)
        x = ((np.arange(20, dtype=np.float32) - 9) / 16).reshape(2, 10)
        native = prepared.fresh_original()
        compiled_owner = prepared.fresh_original()
        replay_owner = prepared.fresh_original()
        replay_children = (replay_owner.left, replay_owner.right)
        compiled = nnx.jit(
            lambda owner, values, *, deterministic: owner(
                values, deterministic=deterministic
            ),
            static_argnames=("deterministic",),
        )
        pure_jit = jax.jit(lambda *args: prepared(*args))
        states = [prepared.initial, prepared.initial]
        references = []
        binding_context = _binding_context()
        for index, flag in enumerate(_FLAGS):
            eager = native(jnp.asarray(x), deterministic=flag)
            compiled_output = compiled(
                compiled_owner, jnp.asarray(x), deterministic=flag
            )
            expected_state = prepared.values(native)
            _same_results(compiled_output, eager)
            _same_values(prepared.values(compiled_owner), expected_state)
            for role, runner in enumerate((prepared, pure_jit)):
                actual = runner(jnp.asarray(x), np.asarray(flag), *states[role])
                _same_results(actual[:5], eager)
                _same_values(actual[5:], expected_state)
                states[role] = actual[5:]
            nnx.update(replay_owner, prepared.state(states[0]))
            assert replay_owner.left is replay_children[0]
            assert replay_owner.right is replay_children[1]
            assert (replay_owner.left is replay_owner.right) is shared
            _same_values(prepared.values(replay_owner), expected_state)
            _same_values(eager[1:3], (x, x))
            _same_values(
                eager[3:],
                (
                    np.asarray(0.75, np.float32),
                    np.asarray(0.75 if shared else 1.25, np.float32),
                ),
            )
            assert native.left.calls[...] == np.float32(
                (index + 1) * (2 if shared else 1)
            )
            assert native.right.calls[...] == native.left.calls[...]
            assert (native.left is native.right) is shared
            references.append(
                tuple(np.asarray(value) for value in (*eager, *expected_state))
            )
            np.savez(
                tmp_path / f"native-{index}.npz",
                **{f"output{i}": value for i, value in enumerate(references[-1])},
            )
        with _observe_functions() as built, jax.checking_leaks():
            model = ui._to_onnx(
                prepared,
                _specs(prepared.initial),
                precision_policy=PrecisionPolicy.SOURCE,
                return_mode="proto",
                opset=23,
            )
        _assert_function_ports(model, built)
        assert _binding_context() == binding_context
        assert array_snapshot(prepared.original) == canonical
        _no_tracers(prepared.original)
    path = tmp_path / "full-state.onnx"
    onnx.save_model(model, path)
    onnx.checker.check_model(onnx.load_model(path), full_check=True)
    sessions = runs = 0
    for optimize in (False, True):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if optimize
            else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        sessions += 1
        assert session.get_providers() == ["CPUExecutionProvider"]
        names = [value.name for value in session.get_inputs()]
        for replay in range(2):
            state = prepared.initial
            for index, flag in enumerate(_FLAGS):
                physical_values = (x, np.asarray(flag), *state)
                numpy_values = tuple(np.asarray(value) for value in physical_values)
                _same_values(numpy_values, physical_values)
                outputs = session.run(None, dict(zip(names, numpy_values, strict=True)))
                np.savez(
                    tmp_path / f"target-{optimize}-{replay}-{index}.npz",
                    **{f"output{i}": value for i, value in enumerate(outputs)},
                    input=x,
                    flag=np.asarray(flag),
                    **{f"state-before-{i}": value for i, value in enumerate(state)},
                )
                runs += 1
                _same_results(outputs, references[index])
                state = outputs[5:]
    assert sessions == 2 and runs == 16
    assert array_snapshot(prepared.original) == canonical
    _no_tracers(prepared.original)


@pytest.mark.parametrize("failure", ["trace", "lowering"])
def test_nnx_function_failure_restores_owner_and_native_calls(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _runtime_configuration(False), _deny_runtime():
        prepared = _AllStateCall.from_owner(_TwoCalls(True))
        canonical = array_snapshot(prepared.original)
        configuration = _configuration()
        binding_context = _binding_context()
        if failure == "trace":
            original = FunctionPlugin._trace_function_body

            def failed(*args: Any, **kwargs: Any) -> Any:
                original(*args, **kwargs)
                raise RuntimeError("injected NNX Function trace failure")

            target, name = FunctionPlugin, "_trace_function_body"
        else:
            original = FunctionPlugin._emit_function_call

            def failed(*args: Any, **kwargs: Any) -> Any:
                original(*args, **kwargs)
                raise RuntimeError("injected NNX lowering failure")

            target, name = FunctionPlugin, "_emit_function_call"
        with monkeypatch.context() as context, jax.checking_leaks():
            context.setattr(
                target, name, staticmethod(failed) if failure == "lowering" else failed
            )
            with pytest.raises(RuntimeError, match="injected NNX"):
                ui._to_onnx(
                    prepared,
                    _specs(prepared.initial),
                    precision_policy=PrecisionPolicy.SOURCE,
                    return_mode="proto",
                    opset=23,
                )
        assert _configuration() == configuration
        assert _binding_context() == binding_context
        assert active_precision_policy() is PrecisionPolicy.LEGACY
        assert array_snapshot(prepared.original) == canonical
        _no_tracers(prepared.original)
        values = (jnp.ones((2, 10), np.float32), np.asarray(False), *prepared.initial)
        _same_results(
            prepared(*values), jax.jit(lambda *args: prepared(*args))(*values)
        )
        assert array_snapshot(prepared.original) == canonical


def test_nnx_owner_source_legacy_source_interleave_preserves_native_state() -> None:
    """Export-only interleave; these three artifacts receive no ORT execution."""
    with _runtime_configuration(False), _deny_runtime():
        prepared = PreparedMLP.from_original(_TwoDropoutOwner(), "nnx")
        canonical = array_snapshot(prepared.original)
        values = (
            jnp.ones((2, 10), np.float32),
            np.asarray(False),
            *prepared.initial_words,
        )
        native = prepared(*values)
        configuration = _configuration()
        binding_context = _binding_context()
        first = ui._to_onnx(
            prepared,
            _specs(prepared.initial_words),
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="proto",
            opset=23,
        )
        assert _binding_context() == binding_context
        # LEGACY keeps its original deterministic path; this call does not
        # request unsupported captured mutable-state execution under LEGACY.
        legacy_owner = prepared.fresh_original()
        ordinary = to_onnx(
            lambda x: legacy_owner(x, deterministic=True),
            [jax.ShapeDtypeStruct((2, 10), np.float32)],
            opset=23,
        )
        after_legacy_binding = _binding_context()
        last = ui._to_onnx(
            prepared,
            _specs(prepared.initial_words),
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="proto",
            opset=23,
        )
        assert first is not last
        assert get_precision_report(ordinary) is None
        assert (
            get_precision_report(first).profile
            == get_precision_report(last).profile
            == "source-static-v1"
        )
        assert _configuration() == configuration
        assert _binding_context() == after_legacy_binding
        assert array_snapshot(prepared.original) == canonical
        _no_tracers(prepared.original)
        _same_results(prepared(*values), native)
        _same_results(jax.jit(lambda *args: prepared(*args))(*values), native)
        _same_state(
            prepared.words_from_model(prepared.original), prepared.initial_words
        )


def test_unprepared_captured_mutable_nnx_owner_requires_trace_local_state() -> None:
    with _runtime_configuration(False), _deny_runtime():
        owner = _TwoCalls(True)
        canonical = array_snapshot(owner)
        children = (owner.left, owner.right)
        configuration = _configuration()
        binding_context = _binding_context()

        def unprepared(x: Any) -> Any:
            return owner(x, deterministic=False)

        with jax.checking_leaks():
            with pytest.raises(
                SourcePrecisionError,
                match="Mutable NNX Function state requires a trace-local prepared owner",
            ) as caught:
                ui._to_onnx(
                    unprepared,
                    [jax.ShapeDtypeStruct((2, 10), np.float32)],
                    precision_policy=PrecisionPolicy.SOURCE,
                    return_mode="proto",
                    opset=23,
                )
        assert caught.value.category == "profile-exclusion"
        assert caught.value.phase == "profile-validation"
        assert owner.left is children[0] and owner.right is children[1]
        assert owner.left is owner.right
        assert array_snapshot(owner) == canonical
        _no_tracers(owner)
        assert _configuration() == configuration
        assert _binding_context() == binding_context
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def test_captured_stateless_named_nnx_owner_never_updates_caller_state() -> None:
    with _runtime_configuration(False), _deny_runtime():
        owner = _StatelessOwner()
        canonical = array_snapshot(owner)
        parameter = owner.scale
        configuration = _configuration()
        binding_context = _binding_context()
        values = jnp.arange(20, dtype=np.float32).reshape(2, 10) / np.float32(16)

        def captured(x: Any) -> Any:
            return owner(x)

        native = captured(values)
        _same_values(jax.jit(captured)(values), native)
        updates = []

        def denied(*args: Any, **kwargs: Any) -> Any:
            updates.append(True)
            raise AssertionError(
                "Stateless captured NNX owner must not call nnx.update"
            )

        with patch.object(nnx, "update", denied), jax.checking_leaks():
            model = ui._to_onnx(
                captured,
                [jax.ShapeDtypeStruct((2, 10), np.float32)],
                precision_policy=PrecisionPolicy.SOURCE,
                return_mode="proto",
                opset=23,
            )
        assert model.functions and len(model.graph.output) == 4
        assert not updates
        assert owner.scale is parameter
        assert array_snapshot(owner) == canonical
        _no_tracers(owner)
        assert _configuration() == configuration
        assert _binding_context() == binding_context
        _same_values(captured(values), native)
        _same_values(jax.jit(captured)(values), native)


@pytest.mark.parametrize("failure_call", [1, 2], ids=["preflight", "partial-update"])
def test_nnx_state_update_failure_rolls_back_values_variable_identities_and_aliases(
    failure_call: int,
) -> None:
    with _runtime_configuration(False), _deny_runtime():
        owner = _UpdatePair()
        canonical = array_snapshot(owner)
        variables = (owner.first, owner.second, owner.alias)
        raw = owner.first.get_raw_value()
        nested = raw["nested"]
        pair = nested[1]
        shared_leaf, other_leaf = pair
        second_raw = owner.second.get_raw_value()
        metadata = tuple(variable.get_metadata() for variable in variables)
        witness_metadata = metadata[0]["witness_metadata"]
        metadata_tags = witness_metadata["tags"]
        metadata_items = tuple(metadata_tags)

        def original_identities() -> None:
            assert owner.first is variables[0]
            assert owner.second is variables[1]
            assert owner.alias is variables[2] is owner.first
            actual = owner.first.get_raw_value()
            assert actual is raw
            assert actual["nested"] is nested
            assert nested[1] is pair
            assert nested[0] is pair[0] is shared_leaf
            assert pair[1] is other_leaf
            assert owner.second.get_raw_value() is second_raw
            for variable, original in zip(variables, metadata, strict=True):
                current = variable.get_metadata()
                assert current.keys() == original.keys()
                assert all(current[name] is value for name, value in original.items())
            assert owner.first.get_metadata()["witness_metadata"] is witness_metadata
            assert witness_metadata["tags"] is metadata_tags
            assert len(metadata_tags) == len(metadata_items)
            assert all(
                a is b for a, b in zip(metadata_tags, metadata_items, strict=True)
            )

        original_identities()
        configuration = _configuration()
        prepared = prepare_nnx_function_state(owner, type(owner).__call__, 917)
        assert prepared is not None
        descriptor, initial = prepared
        assert len(initial) == 4
        positional = jnp.asarray(7, np.float32)
        increment = jnp.asarray(3, np.float32)
        with precision_policy_scope(PrecisionPolicy.SOURCE):
            arguments, parameters = plugin_system._pack_function_arguments(
                (positional,), {"increment": increment, "deterministic": False}
            )
        layout = FunctionOperandLayout(1, len(arguments), len(initial))
        ports = (*arguments, *initial)
        actual_args, actual_kwargs = plugin_system._unpack_function_arguments(
            ports, parameters, layout=layout
        )
        assert len(actual_args) == 1 and actual_args[0] is positional
        assert actual_kwargs["increment"] is increment
        assert actual_kwargs["deterministic"] is False
        assert all(
            value is original
            for value, original in zip(
                ports[layout.argument_count :], initial, strict=True
            )
        )
        invalid = dict(parameters)
        invalid["increment"] = type(parameters["increment"])(layout.argument_count)
        with pytest.raises(
            ValueError, match="keyword refers outside its argument ports"
        ):
            plugin_system._unpack_function_arguments(ports, invalid, layout=layout)
        descriptor = descriptor.with_effects(tuple(range(len(initial))))
        updated = tuple(value + np.float32(4) for value in initial)
        real_update = nnx.update
        calls = []
        changed_during_failure = []

        def injected(target: Any, state: Any, *states: Any) -> Any:
            assert target is owner and not states
            calls.append(True)
            if len(calls) == failure_call:
                # Perform a real public update before raising. On the second
                # call only its first state entry reaches the owner, simulating
                # an exception midway through the actual commit.
                payload = (
                    nnx.from_flat_state(tuple(nnx.to_flat_state(state))[:1])
                    if failure_call == 2
                    else state
                )
                real_update(target, payload)
                if failure_call == 1:
                    original_identities()
                changed_during_failure.append(
                    sum(
                        actual != expected
                        for actual, expected in zip(
                            array_snapshot(owner), canonical, strict=True
                        )
                    )
                )
                raise RuntimeError("injected NNX state update failure")
            result = real_update(target, state)
            # The successful no-advance preflight must retain the actual raw
            # dict/list/tuple objects, not just equal reconstructed leaves.
            original_identities()
            return result

        with patch.object(nnx, "update", injected):
            with pytest.raises(RuntimeError, match="injected NNX state update failure"):
                descriptor.update_owner(owner, updated)
        assert len(calls) == (1 if failure_call == 1 else 3)
        assert changed_during_failure == [0 if failure_call == 1 else 3]
        original_identities()
        assert array_snapshot(owner) == canonical
        _no_tracers(owner)
        assert _configuration() == configuration


def test_repeated_named_nnx_calls_forward_base_keys_and_reject_actual_key_reuse(
    tmp_path: Path,
) -> None:
    """Use JAX's experimental analyzer independently of the compatibility codec."""
    with _runtime_configuration(False), _deny_runtime():
        prepared = _AllStateCall.from_owner(_TwoCalls(True))
        canonical = array_snapshot(prepared.original)
        configuration = _configuration()
        binding_context = _binding_context()
        key_paths = (
            ("dropout1", "rngs", "key"),
            ("dropout2", "rngs", "key"),
        )
        assert prepared.original.left is prepared.original.right
        child = prepared.original.left
        streams = (child.dropout1.rngs, child.dropout2.rngs)
        assert streams[0] is not streams[1]
        assert streams[0].key is not streams[1].key
        assert streams[0].count is not streams[1].count
        native_entries = []
        native_leaves = []
        native_key_offsets = {}
        for path, variable in nnx.to_flat_state(nnx.state(child)):
            leaves, tree = jax.tree.flatten(variable)
            start = len(native_leaves)
            native_leaves.extend(leaves)
            native_entries.append((tuple(path), tree, start, len(native_leaves)))
            if any(
                jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
                for value in leaves
            ):
                assert len(leaves) == 1
                native_key_offsets[tuple(path)] = start
        assert tuple(native_key_offsets) == key_paths
        assert all(
            native_leaves[native_key_offsets[path]] is stream.key[...]
            for path, stream in zip(key_paths, streams, strict=True)
        )
        native_words = [
            _payload(jax.random.key_data(native_leaves[native_key_offsets[path]]))
            for path in key_paths
        ]
        assert native_words[0] != native_words[1]
        key_indices = [
            index
            for index, impl in enumerate(prepared.implementations)
            if impl is not None
        ]
        assert key_indices == [native_key_offsets[path] for path in key_paths]
        logical_key = native_leaves[native_key_offsets[key_paths[0]]]
        key_spec = jax.ShapeDtypeStruct((), logical_key.dtype)
        x = jnp.ones((2, 10), np.float32)
        observations = {
            "debug_key_reuse": True,
            "positive": [],
            "negative": [],
            "metadata_absent": [],
            "eager_source_calls": 0,
        }

        # This independent signature checker only reads consumption state.
        # JAX's eager assert_unconsumed identity primitive mutates that state.
        available_key = key_reuse_core.KeyReuseSignature(key_reuse_core.Sink(0))
        with jax.debug_key_reuse(True):
            assert jax.config.jax_debug_key_reuse
            native = prepared(x, np.asarray(False), *prepared.initial)
            assert np.all(np.isfinite(native[0]))
            _same_values(
                [native[5 + index] for index in key_indices],
                [prepared.initial[index] for index in key_indices],
            )
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                eager_owner = _TwoDropoutOwner()
                eager_streams = (eager_owner.dropout1.rngs, eager_owner.dropout2.rngs)
                assert eager_streams[0] is not eager_streams[1]
                assert eager_streams[0].key is not eager_streams[1].key
                assert eager_streams[0].count is not eager_streams[1].count
                eager_key_entries = {
                    tuple(path): jax.tree.leaves(variable)
                    for path, variable in nnx.to_flat_state(nnx.state(eager_owner))
                    if any(
                        jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
                        for value in jax.tree.leaves(variable)
                    )
                }
                assert tuple(eager_key_entries) == key_paths
                assert all(len(leaves) == 1 for leaves in eager_key_entries.values())
                original_keys = [eager_key_entries[path][0] for path in key_paths]
                assert all(
                    key is stream.key[...]
                    for key, stream in zip(original_keys, eager_streams, strict=True)
                )
                original_words = [
                    _payload(jax.random.key_data(value)) for value in original_keys
                ]
                assert original_keys[0] is not original_keys[1]
                assert original_words[0] != original_words[1]
                observations["native_key_paths"] = [list(path) for path in key_paths]
                observations["native_key_offsets"] = [
                    native_key_offsets[path] for path in key_paths
                ]
                for _ in range(2):
                    result = eager_owner(x, deterministic=False)
                    assert np.all(np.isfinite(result))
                    current_keys = [
                        value
                        for value in jax.tree.leaves(nnx.state(eager_owner))
                        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
                    ]
                    assert all(
                        a is b for a, b in zip(current_keys, original_keys, strict=True)
                    )
                    assert [
                        _payload(jax.random.key_data(value)) for value in current_keys
                    ] == original_words
                    for value in current_keys:
                        available_key.check_signature(value)
                    observations["eager_source_calls"] += 1
                _no_tracers(eager_owner)
                alias_owner = _KeyAliasOwner(original_keys[0])
                input_key = alias_owner.key[...]
                first_alias, second_alias = alias_owner()
                assert first_alias is second_alias
                assert first_alias is not input_key
                assert alias_owner.key[...] is input_key
                assert _payload(jax.random.key_data(first_alias)) == _payload(
                    jax.random.key_data(input_key)
                )
                available_key.check_signature(input_key)
                jax.random.uniform(first_alias, ())
                with pytest.raises(jax.errors.KeyReuseError):
                    available_key.check_signature(second_alias)
                with pytest.raises(jax.errors.KeyReuseError):
                    jax.random.uniform(second_alias, ())
                available_key.check_signature(input_key)
                jax.random.uniform(input_key, ())
                observations["duplicate_result_keys"] = (
                    "shared returned wrapper; input remains usable"
                )

                positive = jax.make_jaxpr(lambda *args: prepared(*args))(
                    *_specs(prepared.initial)
                )
                key_reuse_core.check_key_reuse_jaxpr(positive.jaxpr)
                conditions = [
                    equation
                    for equation in positive.jaxpr.eqns
                    if equation.primitive.name == "cond"
                ]
                assert len(conditions) == 1
                (condition,) = conditions
                for branch in condition.params["branches"]:
                    calls = [
                        equation
                        for equation in branch.jaxpr.eqns
                        if equation.primitive.name.startswith("onnx_fn::")
                    ]
                    assert len(calls) == 2
                    base_inputs = []
                    for call in calls:
                        signature = key_reuse_core.key_reuse_signature_from_eqn(call)
                        inputs = [
                            index
                            for index, variable in enumerate(call.invars)
                            if jax.dtypes.issubdtype(
                                variable.aval.dtype, jax.dtypes.prng_key
                            )
                        ]
                        adapter = call.params[plugin_system._SOURCE_STATE]
                        layout = call.params[plugin_system._FUNCTION_LAYOUT]
                        assert [
                            (entry.path, entry.tree, entry.start, entry.stop)
                            for entry in adapter.entries
                        ] == native_entries
                        assert (
                            layout.state_count
                            == adapter.state_count
                            == len(native_leaves)
                        )
                        assert layout.positional_count == 1
                        # x and the original increment remain separate from state.
                        assert layout.argument_count == 2
                        layout.check(call.invars)
                        result_count = len(call.outvars) - adapter.state_count
                        # Preserve the original _CountingChild result tuple.
                        assert result_count == 4
                        expected_ports = [
                            (
                                layout.argument_count + native_key_offsets[path],
                                result_count + native_key_offsets[path],
                            )
                            for path in key_paths
                        ]
                        assert inputs == [index for index, _ in expected_ports]
                        assert [
                            index
                            for index, variable in enumerate(call.outvars)
                            if jax.dtypes.issubdtype(
                                variable.aval.dtype, jax.dtypes.prng_key
                            )
                        ] == [index for _, index in expected_ports]
                        forwards = tuple(signature.forwards)
                        assert len(forwards) == len(expected_ports)
                        assert {
                            (item.in_idx, item.out_idx) for item in forwards
                        } == set(expected_ports)
                        for path, (index, output_index) in zip(
                            key_paths, expected_ports, strict=True
                        ):
                            assert not any(
                                item.idx == index for item in signature.sinks
                            )
                            output = call.outvars[output_index]
                            source = call.invars[index]
                            native_key = native_leaves[native_key_offsets[path]]
                            assert (
                                output.aval.dtype
                                == source.aval.dtype
                                == native_key.dtype
                            )
                            assert output.aval.shape == source.aval.shape == ()
                            assert (
                                not output.aval.weak_type and not source.aval.weak_type
                            )
                            assert not any(
                                item.idx == output_index for item in signature.sources
                            )
                        key_variables = tuple(call.invars[index] for index in inputs)
                        assert key_variables[0] is not key_variables[1]
                        base_inputs.append(key_variables)
                        observations["positive"].append(repr(signature))
                        absent_params = dict(call.params)
                        absent_params.pop(plugin_system._SOURCE_KEY_REUSE)
                        absent = call.replace(params=absent_params)
                        unknown = key_reuse_core.key_reuse_signature_from_eqn(absent)
                        for index in inputs:
                            assert any(
                                item.idx == index and np.any(item.mask)
                                for item in unknown.sinks
                            )
                        assert not tuple(unknown.sources) and not tuple(
                            unknown.forwards
                        )
                        observations["metadata_absent"].append(repr(unknown))
                    assert all(
                        first is second
                        for first, second in zip(*base_inputs, strict=True)
                    )

            for behavior in ("consume", "split"):

                def repeated(key: Any) -> Any:
                    owner = _ConsumedKeyOwner(key, behavior)
                    return owner(x), owner(x)

                # Retain intentionally invalid graphs before asking the
                # independent checker to reject them under restored debug=True.
                with jax.debug_key_reuse(False):
                    native_closed = jax.make_jaxpr(lambda key: repeated(key))(key_spec)
                assert jax.config.jax_debug_key_reuse
                with pytest.raises(jax.errors.KeyReuseError):
                    key_reuse_core.check_key_reuse_jaxpr(native_closed.jaxpr)
                with (
                    precision_policy_scope(PrecisionPolicy.SOURCE),
                    _activate_plugin_worlds(),
                ):
                    with jax.debug_key_reuse(False):
                        source_closed = jax.make_jaxpr(lambda key: repeated(key))(
                            key_spec
                        )
                    assert jax.config.jax_debug_key_reuse
                    assert (
                        len(
                            [
                                equation
                                for equation in source_closed.jaxpr.eqns
                                if equation.primitive.name.startswith("onnx_fn::")
                            ]
                        )
                        == 2
                    )
                    with pytest.raises(jax.errors.KeyReuseError):
                        key_reuse_core.check_key_reuse_jaxpr(source_closed.jaxpr)
                observations["negative"].append(behavior)

                def invalid_body(key: Any) -> Any:
                    return _ConsumedKeyOwner(key, behavior, repeat_inside=True)(x)

                with jax.debug_key_reuse(False):
                    native_invalid = jax.make_jaxpr(lambda key: invalid_body(key))(
                        key_spec
                    )
                    with (
                        precision_policy_scope(PrecisionPolicy.SOURCE),
                        _activate_plugin_worlds(),
                    ):
                        source_invalid = jax.make_jaxpr(lambda key: invalid_body(key))(
                            key_spec
                        )
                    invalid_calls = [
                        equation
                        for equation in source_invalid.jaxpr.eqns
                        if equation.primitive.name.startswith("onnx_fn::")
                    ]
                    assert len(invalid_calls) == 1
                    metadata = invalid_calls[0].params[plugin_system._SOURCE_KEY_REUSE]
                    assert isinstance(metadata.error, str) and metadata.error
                assert jax.config.jax_debug_key_reuse
                with pytest.raises(jax.errors.KeyReuseError):
                    key_reuse_core.check_key_reuse_jaxpr(native_invalid.jaxpr)
                with pytest.raises(jax.errors.KeyReuseError) as invalid_error:
                    key_reuse_core.check_key_reuse_jaxpr(source_invalid.jaxpr)
                assert metadata.error in str(invalid_error.value)
                observations["negative"].append(behavior + "-invalid-body-debug-off")
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
                patch.dict(globals(), {"_FIXED_TYPED_KEY": logical_key}),
            ):

                def captured(value: Any) -> Any:
                    return _CapturedKeyOwner()(value)

                with pytest.raises(
                    SourcePrecisionError, match="key-bearing captured constants"
                ) as caught:
                    jax.make_jaxpr(captured)(jax.ShapeDtypeStruct((2, 10), np.float32))
                assert caught.value.category == "profile-exclusion"
                observations["key_constvar"] = "explicit preparation required"
        assert _configuration() == configuration
        assert _binding_context() == binding_context
        assert _FIXED_TYPED_KEY is None
        assert array_snapshot(prepared.original) == canonical
        _no_tracers(prepared.original)
        (tmp_path / "key-analysis.json").write_text(
            json.dumps(observations, indent=2, sort_keys=True) + "\n"
        )


def _captured_state_function_call(call: Any, *, key_metadata: Any = None) -> Any:
    """Retain a real equation's state/formal order for actual delayed retracing."""
    plugin = plugin_system.PLUGIN_REGISTRY[call.primitive.name]
    assert isinstance(plugin, FunctionPlugin)
    parameters = dict(call.params)
    trace = parameters.pop(plugin_system._SOURCE_TRACE)
    actual_keys = parameters.pop(plugin_system._SOURCE_KEY_REUSE)
    adapter = parameters.pop(plugin_system._SOURCE_STATE)
    layout = parameters.pop(plugin_system._FUNCTION_LAYOUT)
    positional, keywords = plugin_system._unpack_function_arguments(
        call.invars, parameters, layout=layout
    )
    order = (
        *range(layout.positional_count),
        *range(layout.argument_count, layout.argument_count + layout.state_count),
        *range(layout.positional_count, layout.argument_count),
    )
    callee = FunctionStateCall(
        adapter,
        layout.positional_count,
        (actual_keys if key_metadata is None else key_metadata).reorder_inputs(order),
    )
    equation = call.replace(
        invars=[*positional, *call.invars[layout.argument_count :]], params=keywords
    )
    context = IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    for index, variable in enumerate(call.invars):
        representation = representation_for_aval(variable.aval)
        value = ir.Value(
            name=f"input_{index}",
            type=ir.TensorType(ir.DataType.from_numpy(representation.physical_dtype)),
            shape=ir.Shape(representation.physical_shape),
        )
        context.builder.inputs.append(value)
        context.bind_value_for_var(variable, value)
    captures = plugin._capture_arguments(context, callee, keywords)
    return plugin, callee, equation, captures, trace, context


def test_delayed_function_body_and_impl_preserve_local_key_debug_context(
    tmp_path: Path,
) -> None:
    """Opposite ambient flags cannot change this original numeric Python branch."""
    with _runtime_configuration(False), _deny_runtime():
        owner = _DebugFlagOwner()
        configuration = _configuration()
        binding_context = _binding_context()
        x = ((np.arange(20, dtype=np.float32) - 9) / np.float32(16)).reshape(2, 10)
        keys = []
        for flag in (False, True):
            with jax.debug_key_reuse(flag):
                native = owner(jnp.asarray(x))
                with (
                    precision_policy_scope(PrecisionPolicy.SOURCE),
                    _activate_plugin_worlds(),
                ):
                    captured = jax.make_jaxpr(lambda value: owner(value))(
                        jax.ShapeDtypeStruct(x.shape, np.float32)
                    )
            (call,) = captured.jaxpr.eqns
            plugin, callee, equation, captures, trace, context = (
                _captured_state_function_call(call)
            )
            assert trace.debug_key_reuse is flag
            keys.append(
                plugin._function_key(
                    callee, equation, captures, ctx=context, source_trace=trace
                )
            )
            with jax.debug_key_reuse(not flag):
                body = plugin._trace_function_body(
                    callee,
                    equation,
                    captures,
                    precision_policy=PrecisionPolicy.SOURCE,
                    source_trace=trace,
                )
                assert bool(jax.config.jax_debug_key_reuse) is not flag
                constants = tuple(_payload(value) for value in body.consts) + tuple(
                    _payload(atom.val)
                    for node in body.jaxpr.eqns
                    for atom in node.invars
                    if isinstance(atom, Literal)
                )
                assert _payload(np.float32(0.125 if flag else 0.25)) in constants
                implementation = plugin._source_primitive_impl(
                    jnp.asarray(x), **dict(call.params)
                )
                assert bool(jax.config.jax_debug_key_reuse) is not flag
                evaluator = jaxpr_as_fun(body)
                eager = evaluator(jnp.asarray(x))
                compiled = jax.jit(evaluator)(jnp.asarray(x))
            np.savez(
                tmp_path / f"debug-{flag}.npz",
                input=x,
                native=np.asarray(native),
                implementation=np.asarray(implementation[0]),
                body_eager=np.asarray(eager[0]),
                body_jit=np.asarray(compiled[0]),
            )
            assert np.all(np.isfinite(native))
            _same_values(implementation, (native,))
            _same_values(eager, (native,))
            _same_values(compiled, (native,))
            after = tuple(_payload(value) for value in body.consts) + tuple(
                _payload(atom.val)
                for node in body.jaxpr.eqns
                for atom in node.invars
                if isinstance(atom, Literal)
            )
            assert constants == after
        assert keys[0] != keys[1]
        assert _configuration() == configuration
        assert _binding_context() == binding_context
        _no_tracers(owner)


def test_invalid_key_failure_metadata_survives_actual_delayed_body_retrace(
    tmp_path: Path,
) -> None:
    """This is a native body regression; scan is still outside SOURCE admission."""
    with _runtime_configuration(False), _deny_runtime():
        configuration = _configuration()
        binding_context = _binding_context()
        key = jax.random.key(41)
        specs = (
            jax.ShapeDtypeStruct((), key.dtype),
            jax.ShapeDtypeStruct((), np.float32),
            jax.ShapeDtypeStruct((2, 10), np.float32),
        )

        def original(key: Any, offset: Any, x: Any) -> Any:
            return _LoopKeyReuseOwner(key)(x, offset=offset)

        with jax.debug_key_reuse(False):
            native = jax.make_jaxpr(lambda *args: original(*args))(*specs)
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                captured = jax.make_jaxpr(lambda *args: original(*args))(*specs)
        (call,) = captured.jaxpr.eqns
        metadata = call.params[plugin_system._SOURCE_KEY_REUSE]
        assert metadata.failed and metadata.error
        changed_diagnostic = replace(
            metadata,
            error=metadata.error + "\nEquivalent delayed-body variable rendering",
        )
        assert metadata == changed_diagnostic
        assert hash(metadata) == hash(changed_diagnostic)
        assert metadata != FunctionKeyReuse((), (), ())
        plugin, callee, equation, captures, trace, _ = _captured_state_function_call(
            call, key_metadata=changed_diagnostic
        )
        assert trace.debug_key_reuse is False
        with jax.debug_key_reuse(True):
            body = plugin._trace_function_body(
                callee,
                equation,
                captures,
                precision_policy=PrecisionPolicy.SOURCE,
                source_trace=trace,
            )
            assert bool(jax.config.jax_debug_key_reuse) is True
            actual = function_key_reuse_signature(body)
            assert actual.failed and actual.error
            assert actual == callee.key_reuse
            assert any(node.primitive.name == "scan" for node in body.jaxpr.eqns)
            for closed in (native, body, captured):
                with pytest.raises(jax.errors.KeyReuseError):
                    key_reuse_core.check_key_reuse_jaxpr(closed.jaxpr)
        (tmp_path / "delayed-failure.json").write_text(
            json.dumps(
                {
                    "caller_debug_key_reuse": trace.debug_key_reuse,
                    "ambient_debug_key_reuse": True,
                    "failed": metadata.failed,
                    "captured_diagnostic": metadata.error,
                    "changed_diagnostic": changed_diagnostic.error,
                    "actual_body_diagnostic": actual.error,
                    "canonical_equal": actual == callee.key_reuse,
                    "loop_admission": "not established; direct native body trace only",
                },
                indent=2,
            )
            + "\n"
        )
        assert _configuration() == configuration
        assert _binding_context() == binding_context
