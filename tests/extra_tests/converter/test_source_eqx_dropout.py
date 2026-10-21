# tests/extra_tests/converter/test_source_eqx_dropout.py

"""Exact native Eqx forwarding; fixed samples do not widen shared PRNG admission."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat.jax import is_selected_prng_key_dtype, selected_prng_key_words
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins.equinox.eqx.nn.dropout import DropoutPlugin
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from scripts import source_precision_probe as probe
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.source_precision_support import SourceProvenance
from tests.extra_tests.converter.source_target_evidence import (
    report_evidence,
    require_target_pair,
)
from tests.extra_tests.converter.test_source_dropout_parameters import (
    EqxDropoutState,
    ExplicitStatePreparation,
    _dropout_named,
    _native_settings,
    _session,
    _target,
)
from tests.extra_tests.converter.test_source_predicate_families import (
    _globals,
    _settings,
    _restore_context as _restore_context,
)
from tests.extra_tests.converter.test_source_precision_raw_replay import (
    original_scope as original_scope,
)

_NATIVE_CALL = eqx.nn.Dropout.__call__
_BITWISE = SampleCriteria("bitwise")
_WIDTHS = ((np.float32, False), (np.float32, True), (np.float64, True))


@pytest.fixture(autouse=True)
def _dropout_binding_restored() -> Iterator[None]:
    assert eqx.nn.Dropout.__call__ is _NATIVE_CALL
    yield
    assert eqx.nn.Dropout.__call__ is _NATIVE_CALL
    assert active_precision_policy() is PrecisionPolicy.LEGACY


def _payload(value: Any) -> tuple[Any, ...]:
    if is_selected_prng_key_dtype(getattr(value, "dtype", None)):
        words = np.asarray(selected_prng_key_words(value))
        return (
            "native-prng-key",
            str(jax.random.key_impl(value)),
            str(value.dtype),
            tuple(value.shape),
            words.dtype.str,
            words.shape,
            words.tobytes(order="C"),
        )
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes(order="C")


def _same(actual: Any, expected: Any) -> None:
    assert _payload(actual) == _payload(expected)


def _fresh(fn: Any) -> Any:
    def call(*args: Any) -> Any:
        return fn(*args)

    return call


def _save(path: Path, values: Any) -> None:
    arrays, logical = {}, {}
    for index, value in enumerate(values):
        name = f"value_{index}"
        if is_selected_prng_key_dtype(getattr(value, "dtype", None)):
            arrays[name] = np.asarray(selected_prng_key_words(value))
            logical[name] = {
                "encoding": "selected-native-prng-key-words",
                "implementation": str(jax.random.key_impl(value)),
                "logical_dtype": str(value.dtype),
                "logical_shape": tuple(value.shape),
                "physical_dtype": arrays[name].dtype.str,
                "physical_shape": arrays[name].shape,
            }
        else:
            arrays[name] = np.asarray(value)
    np.savez(path, **arrays)
    path.with_suffix(".logical.json").write_text(json.dumps(logical, indent=2))


def _state(block: eqx.nn.Dropout) -> tuple[Any, ...]:
    return (
        id(block),
        id(block.p),
        type(block.p),
        _payload(block.p),
        id(block.inference),
        type(block.inference),
        _payload(block.inference),
    )


def _threefry(key: tuple[int, int], counter: int) -> tuple[int, int]:
    """Scalar Python Threefry2x32-20, calibrated against its published zero vector.

    No JAX random call, plugin helper, NumPy unsigned arithmetic, or ONNX graph
    is used to compute these expected integer words.
    """
    mask = (1 << 32) - 1
    schedule = (*key, key[0] ^ key[1] ^ 0x1BD11BDA)
    left, right = ((counter >> 32) + key[0]) & mask, (counter + key[1]) & mask
    rotations = (13, 15, 26, 6, 17, 29, 16, 24)
    for round_index in range(20):
        left = (left + right) & mask
        rotation = rotations[round_index % len(rotations)]
        right = (((right << rotation) | (right >> (32 - rotation))) & mask) ^ left
        if (round_index + 1) % 4 == 0:
            injection = (round_index + 1) // 4
            left = (left + schedule[injection % 3]) & mask
            right = (right + schedule[(injection + 1) % 3] + injection) & mask
    return left, right


def _oracle(words: Any, x: np.ndarray, probability_dtype: Any) -> tuple[Any, ...]:
    pair = tuple(int(word) for word in words)
    integers = [_threefry(pair, counter) for counter in range(x.size)]
    high = np.asarray([value[0] for value in integers], np.uint32)
    low = np.asarray([value[1] for value in integers], np.uint32)
    # The mantissa is an exact dyadic, including the FP64 route selected by a
    # weak Python probability under caller-x64-on with an FP32 input tensor.
    if np.dtype(probability_dtype) == np.dtype(np.float32):
        uniform = np.asarray(
            [((a ^ b) >> 9) * 2.0**-23 for a, b in integers], np.float32
        )
        bits = high ^ low
    else:
        uniform = np.asarray(
            [((a << 20) | (b >> 12)) * 2.0**-52 for a, b in integers], np.float64
        )
        bits = (high.astype(np.uint64) << np.uint64(32)) | low.astype(np.uint64)
    mask = uniform < np.asarray(0.5, probability_dtype)
    result = np.where(mask, x / np.asarray(0.5, x.dtype), np.asarray(0, x.dtype))
    return result.astype(x.dtype), mask, bits, uniform


def test_independent_threefry_zero_vector() -> None:
    assert _threefry((0, 0), 0) == (0x6B200159, 0x99BA4EFE)


def _function(block: Any, key: Any, runtime_key: bool, route: str) -> Any:
    alias = block.__call__

    def apply(x: Any, actual_key: Any) -> Any:
        if route == "alias":
            return alias(x=x, key=actual_key, inference=False)
        return block(x, key=actual_key, inference=False)

    called = jax.jit(apply) if route == "jit" else apply

    def fn(x: Any, *words: Any) -> Any:
        actual_key = (
            jax.random.wrap_key_data(words[0], impl="threefry2x32")
            if runtime_key
            else key
        )
        y = called(x, actual_key)
        physical = jax.random.key_data(actual_key)
        return y, x, x, physical, physical

    return fn


@pytest.mark.parametrize("dtype,x64", _WIDTHS)
@pytest.mark.parametrize(
    "extent,key_form,route",
    [
        (32, form, route)
        for form in ("typed-fixed", "physical-fixed", "runtime-words")
        for route in ("direct", "alias", "jit")
    ]
    + [(extent, "runtime-words", "direct") for extent in (0, 1, 3, 20, 31, 33)],
)
def test_native_training_exact_words_mask_storage_aliases_and_target_pair(
    dtype: Any, x64: bool, extent: int, key_form: str, route: str, tmp_path: Path
) -> None:
    with _native_settings(x64):
        block = eqx.nn.Dropout(p=0.5, inference=False)
        typed = jax.random.key(31, impl="threefry2x32")
        key = jax.random.key_data(typed) if key_form == "physical-fixed" else typed
        words = np.asarray(jax.random.key_data(typed)).copy()
        runtime_key = key_form == "runtime-words"
        fn = _function(block, key, runtime_key, route)
        x = ((np.arange(extent) % 13 - 6) / 8).astype(dtype)
        if extent >= 2:
            x[0], x[1] = dtype(0.0), dtype(-0.0)
        if dtype is np.float64 and extent >= 3:
            x[2] += 2.0**-35
        feed = (x, words) if runtime_key else (x,)
        probability_dtype = np.float64 if x64 else np.float32
        oracle, mask, bits, _ = _oracle(words, x, probability_dtype)
        if extent >= 20:
            assert mask.any() and (~mask).any()
        expected = (oracle, x, x, words, words)
        state, context = _state(block), (_globals(), _settings())
        spec = tuple(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in feed)
        warmed = jax.jit(fn)
        references = tuple(
            tuple(np.asarray(v).copy() for v in call(*feed))
            for call in (fn, warmed, warmed, jax.jit(_fresh(fn)))
        )
        native = jax.make_jaxpr(_fresh(fn))(*spec)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            active = jax.make_jaxpr(_fresh(fn))(*spec)
            activated = tuple(np.asarray(v) for v in _fresh(fn)(*feed))
            fresh_jit = tuple(np.asarray(v) for v in jax.jit(_fresh(fn))(*feed))
            warm_active = tuple(np.asarray(v) for v in warmed(*feed))
        assert typed_trace_manifest(active) == typed_trace_manifest(native)
        assert tuple(_payload(v) for v in active.consts) == tuple(
            _payload(v) for v in native.consts
        )
        for outputs in (*references, activated, fresh_jit, warm_active):
            for actual, reference in zip(outputs, expected, strict=True):
                _same(actual, reference)
        _save(tmp_path / "feeds.npz", feed)
        _save(tmp_path / "independent-mask-bits.npz", (mask, bits, oracle))
        _save(tmp_path / "original-key.npz", (words,))
        for number, outputs in enumerate(
            (*references, activated, fresh_jit, warm_active)
        ):
            _save(tmp_path / f"native-{number}.npz", outputs)
        (tmp_path / "native-trace.json").write_text(
            json.dumps(typed_trace_manifest(native), indent=2)
        )
        (tmp_path / "activated-trace.json").write_text(
            json.dumps(typed_trace_manifest(active), indent=2)
        )
        _save(tmp_path / "native-constants.npz", native.consts)
        _save(tmp_path / "activated-constants.npz", active.consts)
        model = ui._to_onnx(
            fn, list(spec), opset=23, precision_policy=PrecisionPolicy.SOURCE
        )
        onnx.checker.check_model(model, full_check=True)
        assert not any(node.op_type == "Dropout" for node in model.graph.node)
        (tmp_path / "source.onnx").write_bytes(model.SerializeToString())
        report = get_precision_report(model)
        assert report is not None
        assert (
            next(c.status for c in report.checks if c.category == "lowering")
            is CheckStatus.VALIDATED
        )
        samples = [SourceSample(feed, outputs, _BITWISE) for outputs in references]
        reports = []
        for mode in ("disabled", "all"):
            target = validate_source_samples(model, samples=samples, optimization=mode)
            (tmp_path / f"{mode}-report.json").write_text(
                json.dumps(report_evidence(target), indent=2)
            )
            reports.append(target)
            session = _session(model, mode == "all")
            actual = session.run(
                None, dict(zip((v.name for v in model.graph.input), feed, strict=True))
            )
            _save(tmp_path / f"actual-{mode}.npz", actual)
            for value, reference in zip(actual, expected, strict=True):
                _same(value, reference)
        require_target_pair(
            model, samples, *reports, evidence_path=tmp_path / "target-boundary.json"
        )
        assert _state(block) == state
        _same(jax.random.key_data(key), words)
        assert (_globals(), _settings()) == context


@pytest.mark.parametrize("dtype,x64", _WIDTHS)
@pytest.mark.parametrize("zero_rate", [False, True])
def test_static_inference_and_python_zero_rate_preserve_original_aliases(
    dtype: Any, x64: bool, zero_rate: bool, tmp_path: Path
) -> None:
    with _native_settings(x64):
        block = eqx.nn.Dropout(p=0.0 if zero_rate else 0.42, inference=not zero_rate)
        x = np.linspace(-1, 1, 32, dtype=dtype)
        x[0] = dtype(-0.0)

        def fn(value: Any) -> Any:
            result = block(value)
            return result, result, value

        value = jnp.asarray(x)
        assert block(value) is value
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        native = jax.make_jaxpr(_fresh(fn))(spec)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            assert block(value) is value
            active = jax.make_jaxpr(_fresh(fn))(spec)
        assert typed_trace_manifest(active) == typed_trace_manifest(native)
        assert not native.jaxpr.eqns
        model = ui._to_onnx(
            fn, [spec], opset=23, precision_policy=PrecisionPolicy.SOURCE
        )
        samples = [SourceSample((x,), (x, x, x), _BITWISE)]
        reports = [
            validate_source_samples(model, samples=samples, optimization=mode)
            for mode in ("disabled", "all")
        ]
        require_target_pair(
            model, samples, *reports, evidence_path=tmp_path / "target-boundary.json"
        )


def test_source_forwarding_preserves_exact_call_objects_warnings_and_exception_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    owner, value, key, flag, returned = (object() for _ in range(5))
    failure = TypeError("native original failure")

    def native(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((self, args, kwargs))
        warnings.warn("native warning", UserWarning)
        if kwargs.get("fail"):
            raise failure
        return returned

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("SOURCE reached the private key-dropping Eqx primitive")

    monkeypatch.setitem(vars(DropoutPlugin._PRIM), "bind", forbidden)
    wrapped = DropoutPlugin._patch_call(native)
    assert wrapped.__wrapped__ is native
    forms = [
        ((value,), {}),
        ((), {"x": value}),
        ((value,), {"key": key, "inference": flag}),
        ((value,), {"key": None, "deterministic": flag}),
        ((value, key), {"unknown": flag}),
    ]
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        for args, kwargs in forms:
            with pytest.warns(UserWarning, match="native warning"):
                assert wrapped(owner, *args, **kwargs) is returned
        with (
            pytest.warns(UserWarning, match="native warning"),
            pytest.raises(TypeError) as caught,
        ):
            wrapped(owner, x=value, key=key, fail=True)
    assert caught.value is failure
    assert calls == [(owner, args, kwargs) for args, kwargs in forms] + [
        (owner, (), {"x": value, "key": key, "fail": True})
    ]
    with (
        precision_policy_scope(PrecisionPolicy.SOURCE),
        pytest.raises(RuntimeError, match="Dropout.__call__ is not available"),
    ):
        DropoutPlugin._patch_call(None)(owner, value)


def _outcome(call: Any) -> tuple[Any, Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call()
            status = {"kind": "returned"}
        except Exception as exc:  # noqa: BLE001
            result = None
            status = {
                "kind": "raised",
                "type": type(exc).__qualname__,
                "detail": str(exc),
            }
    status["warnings"] = [
        (type(w.message).__qualname__, str(w.message)) for w in caught
    ]
    return status, result


@pytest.mark.parametrize("x64", [False, True])
def test_native_key_call_forms_errors_and_deprecated_flag_warnings(
    x64: bool, tmp_path: Path
) -> None:
    with _native_settings(x64):
        block = eqx.nn.Dropout(p=0.5, inference=False)
        value = jnp.ones(32, np.float32)
        key = jax.random.key(9)
        forms = {
            "missing-key": ((value,), {}),
            "none-key": ((value,), {"key": None}),
            "typed-key": ((value,), {"key": key}),
            "physical-key": ((value,), {"key": jax.random.key_data(key)}),
            "keyword-x": ((), {"x": value, "key": key}),
            "inference-ignores-key": ((value,), {"inference": True, "key": object()}),
            "deprecated": ((value,), {"deterministic": True}),
            "deprecated-precedence": (
                (value,),
                {"deterministic": True, "inference": False},
            ),
            "positional-key": ((value, key), {}),
            "duplicate-x": ((value,), {"x": value}),
            "unknown": ((value,), {"unknown": None}),
            "missing-x": ((), {}),
            "malformed-key": ((value,), {"key": np.ones((3,), np.uint32)}),
        }
        wrapped = DropoutPlugin._patch_call(_NATIVE_CALL)
        outcomes = {}
        for name, (args, kwargs) in forms.items():
            expected, native = _outcome(lambda: _NATIVE_CALL(block, *args, **kwargs))
            with precision_policy_scope(PrecisionPolicy.SOURCE):
                actual, active = _outcome(lambda: wrapped(block, *args, **kwargs))
            assert actual == expected
            if native is not None:
                _same(active, native)
            native_trace, native_graph = _outcome(
                lambda: jax.make_jaxpr(lambda: _NATIVE_CALL(block, *args, **kwargs))()
            )
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                active_trace, active_graph = _outcome(
                    lambda: jax.make_jaxpr(lambda: wrapped(block, *args, **kwargs))()
                )
            assert active_trace == native_trace
            if native_graph is not None:
                assert typed_trace_manifest(active_graph) == typed_trace_manifest(
                    native_graph
                )
                assert tuple(_payload(v) for v in active_graph.consts) == tuple(
                    _payload(v) for v in native_graph.consts
                )
            outcomes[name] = {**actual, "native_trace": native_trace}
        assert (
            outcomes["missing-key"]["kind"]
            == outcomes["malformed-key"]["kind"]
            == "raised"
        )
        assert outcomes["deprecated"]["warnings"]
        (tmp_path / "actual-native-call-forms.json").write_text(
            json.dumps(outcomes, indent=2)
        )


@pytest.mark.parametrize("route", ["vmap", "cond", "loop"])
@pytest.mark.parametrize("x64", [False, True])
def test_original_and_activated_nested_native_capture_paths(
    route: str, x64: bool
) -> None:
    with _native_settings(x64):
        block = eqx.nn.Dropout(0.5)
        key = jax.random.key(31)

        def apply(value: Any) -> Any:
            return block(value, inference=False, key=key)

        if route == "vmap":
            fn, value = jax.vmap(apply), jnp.ones((2, 32), np.float32)
        elif route == "cond":

            def fn(value: Any) -> Any:
                return jax.lax.cond(value[0] > 0, apply, lambda x: -apply(-x), value)

            value = jnp.ones(32, np.float32)
        else:

            def fn(value: Any) -> Any:
                return jax.lax.fori_loop(0, 2, lambda _, x: apply(x), value)

            value = jnp.ones(32, np.float32)
        native = jax.make_jaxpr(_fresh(fn))(value)
        expected = fn(value), jax.jit(_fresh(fn))(value)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(_fresh(fn))(value)
            actual = fn(value), jax.jit(_fresh(fn))(value)
        assert typed_trace_manifest(activated) == typed_trace_manifest(native)
        assert tuple(_payload(v) for v in activated.consts) == tuple(
            _payload(v) for v in native.consts
        )
        for a, b in zip(actual, expected, strict=True):
            _same(a, b)


@pytest.mark.parametrize("x64", [False, True])
def test_dynamic_original_python_flag_requires_explicit_native_branch_preparation(
    x64: bool,
) -> None:
    with _native_settings(x64):
        target = _target("eqx", 0.5)
        value = jnp.ones(32, np.float32)
        with pytest.raises(jax.errors.TracerBoolConversionError):
            jax.make_jaxpr(target)(value, jax.ShapeDtypeStruct((), np.bool_))
        preparation = ExplicitStatePreparation(
            EqxDropoutState.from_target(target), _dropout_named
        )
        compiled = jax.jit(lambda x, flag: preparation(x, flag))
        for flag in (True, False, False, True):
            original = target(value, flag)
            (eager,) = preparation(value, jnp.asarray(flag))
            (jit,) = compiled(value, jnp.asarray(flag))
            _same(eager, original)
            _same(jit, original)
        preparation.adapter.original.assert_original()


@pytest.mark.parametrize("reverse", [False, True])
def test_source_legacy_and_native_cache_contexts_restore_original_semantics(
    reverse: bool,
) -> None:
    with _native_settings(False):
        block = eqx.nn.Dropout(0.5)
        key = jax.random.key(31)
        value = jnp.ones(32, np.float32)
        fn = _function(block, key, False, "direct")
        warmed = jax.jit(fn)
        expected = fn(value)
        for actual, reference in zip(warmed(value), expected, strict=True):
            _same(actual, reference)
        phases = (
            (PrecisionPolicy.SOURCE, PrecisionPolicy.LEGACY, PrecisionPolicy.SOURCE)
            if reverse
            else (
                PrecisionPolicy.LEGACY,
                PrecisionPolicy.SOURCE,
                PrecisionPolicy.LEGACY,
            )
        )
        for policy in phases:
            with precision_policy_scope(policy), _activate_plugin_worlds():
                trace = jax.make_jaxpr(_fresh(fn))(value)
                names = {eqn.primitive.name for eqn in trace.jaxpr.eqns}
                assert ("eqx.nn.dropout" in names) is (policy is PrecisionPolicy.LEGACY)
                if policy is PrecisionPolicy.SOURCE:
                    for actual, reference in zip(
                        jax.jit(_fresh(fn))(value), expected, strict=True
                    ):
                        _same(actual, reference)
                with _activate_plugin_worlds():
                    assert active_precision_policy() is policy
            assert eqx.nn.Dropout.__call__ is _NATIVE_CALL
            for actual, reference in zip(warmed(value), expected, strict=True):
                _same(actual, reference)

        def legacy_default(x: Any) -> Any:
            return block(x, key=key)

        legacy = ui.to_onnx(
            legacy_default, [jax.ShapeDtypeStruct((32,), np.float32)], opset=23
        )
        assert any(node.op_type == "Dropout" for node in legacy.graph.node)


def test_source_keeps_private_dropout_carrier_excluded() -> None:
    DropoutPlugin.ensure_abstract_eval_bound()

    def private(value: Any) -> Any:
        return DropoutPlugin._PRIM.bind(value, True, p=0.5, call_time=False)

    with pytest.raises(SourcePrecisionError) as caught:
        ui._to_onnx(
            private,
            [jax.ShapeDtypeStruct((32,), np.float32)],
            opset=23,
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert caught.value.category == "profile-exclusion"
    assert caught.value.primitive == "eqx.nn.dropout"


@pytest.mark.parametrize("fault", ["rbg", "unpartitionable", "opset16", "scalar-rank"])
def test_forwarding_cannot_waive_existing_native_prng_guards(fault: str) -> None:
    with _native_settings(False):
        key = jax.random.key(31, impl="rbg" if fault == "rbg" else "threefry2x32")
        block = eqx.nn.Dropout(0.5)
        shape = () if fault == "scalar-rank" else (32,)

        def fn(value: Any) -> Any:
            return block(value, key=key)

        if fault == "unpartitionable":
            jax.config.update("jax_threefry_partitionable", False)
        with pytest.raises(SourcePrecisionError) as caught:
            ui._to_onnx(
                fn,
                [jax.ShapeDtypeStruct(shape, np.float32)],
                opset=16 if fault == "opset16" else 23,
                precision_policy=PrecisionPolicy.SOURCE,
            )
        assert caught.value.category == "profile-exclusion"
        assert caught.value.primitive != "eqx.nn.dropout"


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("x64", [False, True])
def test_unchanged_original_mlp_raw_recipes_keep_observed_outcomes(
    ordinal: int,
    x64: bool,
    original_scope: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, plan, registrations, adapters = original_scope
    origin = "example::examples.eqx::MlpExample"
    row = next(
        row for row in plan.rows if (row.origin, row.ordinal) == (origin, ordinal)
    )
    registered = {name: metadata for name, _, metadata in registrations}
    metadata = probe._resolve_case(row, registered)
    target, callback = metadata["callable"], metadata.get("post_check_onnx_graph")
    before = _globals(), _settings()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Original product-only rows must not invent runtime acceptance")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    result = probe.execute_probe_plan(
        plan,
        registrations,
        stage="product",
        raw_replay=True,
        limit=1,
        cases=frozenset({SourceProvenance(origin, ordinal)}),
        scenario=probe.SCENARIOS[int(x64)],
        replay_adapters=adapters,
        evidence_directory=tmp_path / "raw",
    )
    cell = next(
        cell
        for cell in result.cells
        if cell.identity == (origin, ordinal, probe.SCENARIOS[int(x64)])
    )
    (evidence,) = cell.history
    (tmp_path / "original-outcome.json").write_text(
        json.dumps(
            {
                "scope": "product-only; runtime/numerics UNVERIFIED",
                "row": asdict(row),
                "cell": asdict(cell),
            },
            indent=2,
        )
    )
    capture = evidence.replay_capture
    assert cell.source_reference is CheckStatus.VALIDATED
    assert (
        cell.runtime
        is cell.numerics
        is evidence.runtime
        is evidence.numerics
        is CheckStatus.UNVERIFIED
    )
    assert row.criteria is None
    assert evidence.row_contract_sha256 == row.contract_sha256
    assert capture["raw_callback"]["status"] == evidence.raw_callback_status
    for role in (
        "eager",
        "fresh-jit",
        "concrete-typed-trace",
        "declared-typed-trace",
        "original-recipe",
        "instantiated-target-state",
    ):
        assert capture[role]["status"] == "captured"
        assert Path(capture[role]["path"]).is_file()
    for role in (
        "eager-state",
        "fresh-jit-state",
        "concrete-typed-trace-state",
        "declared-typed-trace-state",
        "after-product-export-state",
    ):
        assert capture[role]["status"] == "unchanged"
    failure = capture.get("source_error")
    if failure is not None:
        assert failure["category"] == "profile-exclusion"
        assert failure["primitive"] != "eqx.nn.dropout"
        assert all(failure[key] for key in ("phase", "requirement", "action"))
    else:
        assert cell.artifact is cell.schema is CheckStatus.VALIDATED
        assert evidence.raw_callback_status in {"passed", "failed", "error"}
    assert probe._resolve_case(row, registered)["callable"] is target
    assert probe._resolve_case(row, registered).get("post_check_onnx_graph") is callback
    assert result.rows == plan.rows
    assert [after for after in result.cells if after.identity != cell.identity] == [
        before for before in plan.cells if before.identity != cell.identity
    ]
    assert (_globals(), _settings()) == before


@pytest.mark.parametrize("storage", [np.float32, np.float64])
@pytest.mark.parametrize("x64", [False, True])
def test_probability_storage_and_aliases_survive_actual_native_canonicalization(
    storage: Any, x64: bool
) -> None:
    with jax.enable_x64(True):
        probability = jnp.asarray(0.5, dtype=storage)
        block = eqx.nn.Dropout(p=probability)
        key = jax.random.key(31)
    state, key_payload = _state(block), _payload(jax.random.key_data(key))
    with _native_settings(x64):
        value = jnp.ones(32, np.float32)

        def fn(x: Any) -> Any:
            return block(x, key=key), block.p, block.p, x, x

        native = jax.make_jaxpr(_fresh(fn))(value)
        expected = fn(value), jax.jit(_fresh(fn))(value)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(_fresh(fn))(value)
            actual = fn(value), jax.jit(_fresh(fn))(value)
        assert typed_trace_manifest(activated) == typed_trace_manifest(native)
        assert tuple(_payload(v) for v in activated.consts) == tuple(
            _payload(v) for v in native.consts
        )
        for actual_outputs, expected_outputs in zip(actual, expected, strict=True):
            for a, b in zip(actual_outputs, expected_outputs, strict=True):
                _same(a, b)
    assert _state(block) == state
    assert _payload(jax.random.key_data(key)) == key_payload


def test_static_rank_one_counter_limit_remains_strict_before_emission() -> None:
    from jax2onnx.plugins.jax.random._source_prng import (
        validate_selected_random_equation,
    )

    with _native_settings(False):
        key = jax.random.key(31)
        graph = jax.make_jaxpr(lambda value: jax.random.bits(value, (3,), np.uint32))(
            key
        )
        bits = next(
            eqn for eqn in graph.jaxpr.eqns if eqn.primitive.name == "random_bits"
        )
        validate_selected_random_equation(bits)
        oversized = bits.replace(params={**bits.params, "shape": (1 << 63,)})
        with pytest.raises(ValueError, match="signed-int64 physical tensor limit"):
            validate_selected_random_equation(oversized)
        validate_selected_random_equation(bits)


@pytest.mark.parametrize("dtype,x64", _WIDTHS)
@pytest.mark.parametrize("extent", [0, 1, 3, 20, 31, 32, 33])
def test_shared_static_rank_one_uniform_compares_every_native_mantissa_bit(
    dtype: Any, x64: bool, extent: int, tmp_path: Path
) -> None:
    # A separate original function has one native uniform consumer per key/run.
    # Dropout's original caller and fixed external key are never modified to
    # expose a second random draw merely for test observability.
    with _native_settings(x64):
        words = np.asarray([0xA1B2C3D4, 0x1BD11BDA], np.uint32)
        _, _, expected_bits, oracle = _oracle(words, np.zeros(extent, dtype), dtype)

        def original(key_words: Any) -> Any:
            key = jax.random.wrap_key_data(key_words, impl="threefry2x32")
            return jax.random.uniform(key, (extent,), dtype=dtype)

        spec = jax.ShapeDtypeStruct((2,), np.uint32)
        warmed = jax.jit(original)
        references = [
            np.asarray(call(words)).copy()
            for call in (original, warmed, warmed, jax.jit(_fresh(original)))
        ]
        native = jax.make_jaxpr(_fresh(original))(spec)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            active = jax.make_jaxpr(_fresh(original))(spec)
            actual = original(words), jax.jit(_fresh(original))(words)
        assert typed_trace_manifest(active) == typed_trace_manifest(native)
        for result in (*references, *actual):
            _same(result, oracle)
        _save(tmp_path / "independent-words-and-uniform.npz", (expected_bits, oracle))
        _save(tmp_path / "key-words.npz", (words,))
        _save(tmp_path / "native-results.npz", references)
        (tmp_path / "native-trace.json").write_text(
            json.dumps(typed_trace_manifest(native), indent=2)
        )
        (tmp_path / "activated-trace.json").write_text(
            json.dumps(typed_trace_manifest(active), indent=2)
        )
        model = ui._to_onnx(
            original, [spec], opset=23, precision_policy=PrecisionPolicy.SOURCE
        )
        onnx.checker.check_model(model, full_check=True)
        (tmp_path / "source.onnx").write_bytes(model.SerializeToString())
        samples = [
            SourceSample((words,), (reference,), _BITWISE) for reference in references
        ]
        reports = []
        for mode in ("disabled", "all"):
            target = validate_source_samples(model, samples=samples, optimization=mode)
            reports.append(target)
            (tmp_path / f"{mode}-report.json").write_text(
                json.dumps(report_evidence(target), indent=2)
            )
            session = _session(model, mode == "all")
            result = session.run(None, {model.graph.input[0].name: words})
            _save(tmp_path / f"actual-{mode}.npz", result)
            assert len(result) == 1
            _same(result[0], oracle)
        require_target_pair(
            model, samples, *reports, evidence_path=tmp_path / "target-boundary.json"
        )
