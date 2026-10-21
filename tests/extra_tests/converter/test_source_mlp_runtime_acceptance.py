# tests/extra_tests/converter/test_source_mlp_runtime_acceptance.py

"""Finite CPU MLP state-feedback regression, separate from the full mask proof."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from flax import config as flax_config, linen, nnx
import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
from onnxruntime.capi import _pybind_state  # type: ignore[import-untyped]
import pytest

from jax2onnx import user_interface as ui
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
)
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter.source_precision_support import (
    _global_source_configuration,
)
from tests.extra_tests.converter.test_source_mlp_state_preparation import (
    PreparedMLP,
    _restore_configuration as _restore_configuration,  # noqa: F401
    array_snapshot,
    nonzero_mlp_input,
    original_mlp,
)

# Approved sample criteria provenance:
# e2959fccb041d2af5ce6af244b62094fdf7ce1d73bc4af7b63fd25e42d731d54.
# No external approval file is read at runtime. These bounds cover this sample.
_NONZERO_ATOL = {
    ("nnx", False, "float32"): 2.1e-5,
    ("nnx", True, "float32"): 2.3e-5,
    ("nnx", True, "float64"): 4.3e-14,
    ("linen", False, "float32"): 2.8e-5,
    ("linen", True, "float32"): 3.0e-5,
    ("linen", True, "float64"): 4.2e-7,
}
_FLAGS = (True, False, False, True)
_ROLES = ("original-eager", "original-nnx-jit", "prepared-eager", "prepared-jit")
# Zero Y requires positive-zero bytes; every uint32 state word requires exact
# dtype, shape and bytes, against all four independent native sequences.


def _callables() -> tuple[Any, ...]:
    return (
        nnx.Linear.__call__,
        nnx.BatchNorm.__call__,
        nnx.Dropout.__call__,
        linen.Dense.__call__,
        linen.BatchNorm.__call__,
        linen.Dropout.__call__,
        nnx.gelu,
        linen.gelu,
        jax.nn.gelu,
        jax.random.key,
        jax.random.split,
        jax.random.fold_in,
        jax.random.bernoulli,
        jax.random.uniform,
        jnp.shape,
        jnp.asarray,
        jnp.array,
        jax.jit,
    )


def _configuration() -> tuple[Any, ...]:
    states = (
        jax.enable_x64,
        jax.default_matmul_precision,
        jax.numpy_dtype_promotion,
        jax.numpy_rank_promotion,
    )
    return (
        _global_source_configuration(),
        tuple((state.get_global(), state.get_local()) for state in states),
        tuple(getattr(jax.config, name) for name in _global_source_configuration()),
        tuple(
            getattr(jax.config, name)
            for name in (
                "jax_default_prng_impl",
                "jax_threefry_partitionable",
                "jax_debug_key_reuse",
            )
        ),
        flax_config.flax_fix_rng_separator,
    )


@contextmanager
def _runtime_configuration(x64: bool) -> Iterator[None]:
    before, callables, policy = (
        _configuration(),
        _callables(),
        active_precision_policy(),
    )
    prng = {
        name: getattr(jax.config, name)
        for name in ("jax_default_prng_impl", "jax_threefry_partitionable")
    }
    try:
        assert policy is PrecisionPolicy.LEGACY
        jax.config.update("jax_default_prng_impl", "threefry2x32")
        jax.config.update("jax_threefry_partitionable", True)
        with (
            jax.enable_x64(x64),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            selected = _configuration()
            try:
                yield
            finally:
                assert _configuration() == selected
                assert _callables() == callables
                assert active_precision_policy() is policy
    finally:
        for name, value in prng.items():
            jax.config.update(name, value)
        assert _configuration() == before
        assert _callables() == callables
        assert active_precision_policy() is policy


def _host(value: Any) -> np.ndarray:
    result: np.ndarray = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _state(values: Any) -> tuple[np.ndarray, ...]:
    return tuple(_host(value) for value in values)


def _same_state(actual: Any, expected: Any) -> None:
    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected, strict=True):
        value, reference = np.asarray(value), np.asarray(reference)
        assert value.dtype == reference.dtype == np.dtype(np.uint32)
        assert value.shape == reference.shape
        assert value.tobytes() == reference.tobytes()


def _y(actual: Any, expected: Any, *, zero: bool, atol: float) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape == (2, 10)
    assert actual.dtype == expected.dtype
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    if zero:
        assert actual.tobytes() == expected.tobytes() == bytes(expected.nbytes)
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, equal_nan=False)


@dataclass(frozen=True)
class _NativeStep:
    y: np.ndarray
    before: tuple[np.ndarray, ...]
    after: tuple[np.ndarray, ...]


def _module(target: Any, kind: str) -> Any:
    return target if kind == "nnx" else target._model


def _native_references(
    prepared: PreparedMLP, x: np.ndarray, family: str, tmp_path: Path
) -> tuple[dict[str, tuple[_NativeStep, ...]], list[Any], list[dict[str, Any]]]:
    """Each role owns a clone and advances only its own actual state."""
    original_eager, original_jit = prepared.fresh_original(), prepared.fresh_original()
    prepared_eager = PreparedMLP.from_original(prepared.fresh_original(), prepared.kind)
    prepared_jit = PreparedMLP.from_original(prepared.fresh_original(), prepared.kind)
    owners = [original_eager, original_jit, prepared_eager, prepared_jit]
    targets = [
        original_eager,
        original_jit,
        prepared_eager.original,
        prepared_jit.original,
    ]
    assert len({id(_module(target, prepared.kind)) for target in targets}) == 4
    wrapper_type = type(prepared.original)

    def nnx_original(model: Any, values: Any, *, deterministic: bool) -> Any:
        return model(values, deterministic=deterministic)

    def linen_original(
        model: Any, key: Any, values: Any, *, deterministic: bool
    ) -> Any:
        # The bridge model is an NNX graph node. The original plain wrapper is
        # reconstructed with its unchanged key and original call contract.
        return wrapper_type(model, key)(values, deterministic=deterministic)

    original_compiled = nnx.jit(
        nnx_original if prepared.kind == "nnx" else linen_original,
        static_argnames=("deterministic",),
    )

    def prepared_call(values: Any, flag: Any, *words: Any) -> Any:
        return prepared_jit(values, flag, *words)

    prepared_compiled = jax.jit(prepared_call)
    values = jnp.asarray(x)
    references, identities = {}, []
    for role, owner, target in zip(_ROLES, owners, targets, strict=True):
        if isinstance(owner, PreparedMLP):
            words = owner.initial_words
        else:
            words = prepared.words_from_model(owner)
        _same_state(words, prepared.initial_words)
        steps = []
        for flag in _FLAGS:
            before = _state(words)
            if role == "original-eager":
                y = owner(values, deterministic=flag)
                words = prepared.words_from_model(owner)
            elif role == "original-nnx-jit":
                if prepared.kind == "nnx":
                    y = original_compiled(owner, values, deterministic=flag)
                else:
                    y = original_compiled(
                        owner._model, owner._rngs, values, deterministic=flag
                    )
                words = prepared.words_from_model(owner)
            else:
                runner = owner if role == "prepared-eager" else prepared_compiled
                y, *next_words = runner(values, jnp.asarray(flag), *words)
                words = tuple(next_words)
            y = _host(y)
            assert y.dtype == x.dtype
            _y(y, y, zero=family == "zero", atol=0)
            after = _state(words)
            if flag:
                _same_state(after, before)
            steps.append(_NativeStep(y, before, after))
        references[role] = tuple(steps)
        identities.append(
            {
                "role": role,
                "owner_id": id(owner),
                "original_callable_id": id(target),
                "module_id": id(_module(target, prepared.kind)),
                "jit_callable_id": (
                    id(original_compiled)
                    if role == "original-nnx-jit"
                    else id(prepared_compiled) if role == "prepared-jit" else None
                ),
            }
        )
        payload: dict[str, Any] = {f"step{i}-y": step.y for i, step in enumerate(steps)}
        payload.update(
            {
                f"step{i}-{position}-state{j}": word
                for i, step in enumerate(steps)
                for position, state in (("before", step.before), ("after", step.after))
                for j, word in enumerate(state)
            }
        )
        np.savez(tmp_path / f"native-{family}-{role}.npz", **payload)
    for step in range(len(_FLAGS)):
        for role in _ROLES:
            _same_state(
                references[role][step].before, references[_ROLES[0]][step].before
            )
            _same_state(references[role][step].after, references[_ROLES[0]][step].after)
    return references, owners, identities


def _port(value: Any, dtype: Any, shape: tuple[Any, ...]) -> None:
    tensor = value.type.tensor_type
    assert tensor.elem_type == onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    assert len(tensor.shape.dim) == len(shape)
    for dimension, expected in zip(tensor.shape.dim, shape, strict=True):
        if isinstance(expected, str):
            assert dimension.dim_param == expected and not dimension.HasField(
                "dim_value"
            )
        else:
            assert dimension.dim_value == expected and not dimension.dim_param


@pytest.mark.parametrize("kind", ["nnx", "linen"])
@pytest.mark.parametrize(
    "x64,dtype", [(False, np.float32), (True, np.float32), (True, np.float64)]
)
def test_original_mlp_cpu_state_feedback_and_reset_replay(
    kind: str, x64: bool, dtype: Any, tmp_path: Path, record_property: Any
) -> None:
    with _runtime_configuration(x64):
        prepared = PreparedMLP.from_original(original_mlp(kind), kind)
        module = _module(prepared.original, kind)
        canonical = array_snapshot(module)
        initial = _state(prepared.initial_words)
        families = {
            "nonzero": nonzero_mlp_input(2, dtype),
            "zero": np.zeros((2, 30), dtype=dtype),
        }
        references, owners, identities = {}, [], {}
        denied_attempts = []

        def canonical_unchanged() -> None:
            assert array_snapshot(module) == canonical
            _same_state(prepared.words_from_model(prepared.original), initial)

        def denied(*args: Any, **kwargs: Any) -> Any:
            denied_attempts.append((args, kwargs))
            pytest.fail("native reference capture must precede every target session")

        try:
            with (
                patch.object(ort, "InferenceSession", denied),
                patch.object(_pybind_state, "InferenceSession", denied),
            ):
                for family, x in families.items():
                    result, held, ids = _native_references(
                        prepared, x, family, tmp_path
                    )
                    references[family] = result
                    owners.extend(held)
                    identities[family] = ids
            assert not denied_attempts
            assert len(owners) == 8
            canonical_unchanged()
            (tmp_path / "native-reference-identities.json").write_text(
                json.dumps(identities, indent=2, sort_keys=True) + "\n"
            )
            B = jax_export.symbolic_shape("B")[0]
            specs = [
                jax.ShapeDtypeStruct((B, 30), dtype),
                jax.ShapeDtypeStruct((), np.bool_),
                *(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in initial),
            ]

            def production(x: Any, flag: Any, *words: Any) -> Any:
                return prepared(x, flag, *words)

            model = ui._to_onnx(
                production,
                specs,
                precision_policy=PrecisionPolicy.SOURCE,
                return_mode="proto",
                opset=23,
            )
            assert isinstance(model, onnx.ModelProto)
            path = tmp_path / "production.onnx"
            onnx.save_model(model, path)
            model = onnx.load_model(path)
            onnx.checker.check_model(model, full_check=True)
            assert {v.domain: v.version for v in model.opset_import}[""] == 23
            assert len(model.graph.input) == 2 + len(initial)
            assert len(model.graph.output) == 1 + len(initial)
            _port(model.graph.input[0], dtype, ("B", 30))
            _port(model.graph.input[1], np.bool_, ())
            _port(model.graph.output[0], dtype, ("B", 10))
            for index, word in enumerate(initial):
                assert word.dtype == np.dtype(np.uint32)
                _port(model.graph.input[index + 2], word.dtype, word.shape)
                _port(model.graph.output[index + 1], word.dtype, word.shape)
            report = get_precision_report(model)
            assert report is not None and report.profile == "source-static-v1"
            checks = {check.category: check.status for check in report.checks}
            assert (
                checks["representation"] is checks["lowering"] is CheckStatus.VALIDATED
            )
            assert (
                checks["runtime-execution"]
                is checks["numerical-acceptance"]
                is CheckStatus.UNVERIFIED
            )
            canonical_unchanged()
            blob = model.SerializeToString()
            names = [value.name for value in model.graph.input]
            atol = _NONZERO_ATOL[kind, x64, np.dtype(dtype).name]
            sessions = runs = 0
            for optimize in (False, True):
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    if optimize
                    else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                )
                session = ort.InferenceSession(
                    blob, options, providers=["CPUExecutionProvider"]
                )
                sessions += 1
                assert session.get_providers() == ["CPUExecutionProvider"]
                actual_options = session.get_session_options()
                assert (
                    actual_options.intra_op_num_threads
                    == actual_options.inter_op_num_threads
                    == 1
                )
                assert (
                    actual_options.graph_optimization_level
                    == options.graph_optimization_level
                )
                assert actual_options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
                assert [value.name for value in session.get_inputs()] == names
                for family, x in families.items():
                    for replay in range(2):
                        words = tuple(np.array(word, copy=True) for word in initial)
                        # Some state outputs alias their inputs. Keep the initial
                        # state arrays and every feed alive throughout this replay.
                        replay_owners: list[Any] = [words]
                        _same_state(words, initial)
                        for step_index, flag in enumerate(_FLAGS):
                            before = _state(words)
                            for role in _ROLES:
                                _same_state(
                                    words, references[family][role][step_index].before
                                )
                            feed = dict(
                                zip(
                                    names,
                                    (x, np.asarray(flag, dtype=np.bool_), *words),
                                    strict=True,
                                )
                            )
                            replay_owners.append(feed)
                            outputs = tuple(session.run(None, feed))
                            runs += 1
                            assert len(outputs) == 1 + len(initial)
                            assert outputs[0].dtype == np.dtype(dtype)
                            for role in _ROLES:
                                reference = references[family][role][step_index]
                                _y(
                                    outputs[0],
                                    reference.y,
                                    zero=family == "zero",
                                    atol=atol,
                                )
                                _same_state(outputs[1:], reference.after)
                            if flag:
                                _same_state(outputs[1:], before)
                            # Training uses the exact original progression already checked
                            # above; Linen has no invented generic counter increment rule.
                            words = outputs[1:]
                canonical_unchanged()
            assert sessions == 2 and runs == 32
            record_property(
                "source-mlp-runtime",
                json.dumps(
                    {
                        "kind": kind,
                        "x64": x64,
                        "dtype": np.dtype(dtype).name,
                        "exports": 1,
                        "sessions": sessions,
                        "target_runs": runs,
                        "native_records": len(_ROLES) * len(families) * len(_FLAGS),
                        "provider": "CPUExecutionProvider",
                        "intra_threads": 1,
                        "inter_threads": 1,
                        "rtol": 0,
                        "nonzero_atol": atol,
                        "equal_nan": False,
                        "zero_Y": "exact positive-zero bytes",
                        "state": "exact uint32 dtype/shape/bytes",
                        "scope": "bounded production Y/state sample; no direct mask equality claim",
                    },
                    sort_keys=True,
                ),
            )
        finally:
            canonical_unchanged()
