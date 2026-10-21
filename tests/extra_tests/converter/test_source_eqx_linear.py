# tests/extra_tests/converter/test_source_eqx_linear.py

"""Exact native Equinox Linear calls; original replay outcomes stay observations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import user_interface as ui
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins.equinox.eqx.nn.linear import LinearPlugin
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from scripts import source_precision_probe as probe
from tests.extra_tests.converter.source_target_evidence import require_target_pair
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.source_precision_support import SourceProvenance
from tests.extra_tests.converter.test_source_predicate_families import (
    _globals,
    _settings,
    _restore_context as _restore_context,
)
from tests.extra_tests.converter.test_source_precision_raw_replay import (
    original_scope as original_scope,
)

_NATIVE_CALL = eqx.nn.Linear.__call__
_BITWISE = SampleCriteria("bitwise")
_WIDTHS = (
    ("float32", "float32", False),
    ("float32", "float32", True),
    ("float32", "float64", True),
    ("float64", "float32", True),
    ("float64", "float64", True),
)
_GEOMETRIES = (
    (4, 3, "direct"),
    (4, 3, "alias"),
    (4, 3, "native-jit"),
    (4, 3, "vmap"),
    (4, 3, "nested-vmap"),
    ("scalar", 3, "direct"),
    (4, "scalar", "direct"),
    ("scalar", "scalar", "direct"),
    (0, 3, "direct"),
    (4, 0, "direct"),
)


@pytest.fixture(autouse=True)
def _linear_binding_restored() -> Iterator[None]:
    assert eqx.nn.Linear.__call__ is _NATIVE_CALL
    yield
    assert eqx.nn.Linear.__call__ is _NATIVE_CALL
    assert active_precision_policy() is PrecisionPolicy.LEGACY


def _payload(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes(order="C")


def _same(actual: Any, expected: Any) -> None:
    assert _payload(actual) == _payload(expected)


def _state(layer: eqx.nn.Linear) -> Any:
    return (
        id(layer),
        layer.in_features,
        layer.out_features,
        layer.use_bias,
        tuple(
            (id(v), type(v), _payload(v), bool(jax.typeof(v).weak_type))
            for v in jax.tree.leaves(layer)
        ),
    )


def _fresh(fn: Any) -> Any:
    def call(*args: Any) -> Any:
        return fn(*args)

    return call


def _save_arrays(path: Path, values: Any) -> None:
    np.savez(path, **{f"value_{i}": np.asarray(v) for i, v in enumerate(values)})


def _oracle(layer: eqx.nn.Linear, value: Any) -> np.ndarray:
    """Coordinate sums over exact dyadics, independent of JAX/NumPy matmul."""
    weight = np.asarray(layer.weight)
    x = np.asarray(value)
    if layer.in_features == "scalar":
        x = np.broadcast_to(x, (1,))
    dtype = np.result_type(
        weight.dtype, x.dtype, *([] if layer.bias is None else [layer.bias.dtype])
    )
    output = np.empty((*x.shape[:-1], weight.shape[0]), dtype)
    for index in np.ndindex(output.shape):
        total = Fraction(0)
        for k in range(weight.shape[1]):
            total += Fraction.from_float(
                float(weight[index[-1], k])
            ) * Fraction.from_float(float(x[(*index[:-1], k)]))
        if layer.bias is not None:
            total += Fraction.from_float(float(layer.bias[index[-1]]))
        output[index] = float(total)
    return np.squeeze(output, axis=-1) if layer.out_features == "scalar" else output


def _fixture(
    in_features: Any,
    out_features: Any,
    storage: str,
    input_dtype: str,
    bias: bool,
    route: str,
) -> tuple[Any, Any, Any]:
    # These are separate exact-dyadic fixtures. Original registry weights below
    # are never replaced, rerounded, reconstructed, or assigned these criteria.
    layer = eqx.nn.Linear(
        in_features, out_features, use_bias=bias, dtype=storage, key=jax.random.key(917)
    )
    rows = 1 if out_features == "scalar" else out_features
    columns = 1 if in_features == "scalar" else in_features
    weight = ((np.arange(rows * columns).reshape(rows, columns) % 11 - 5) / 8).astype(
        storage
    )
    if storage == "float64" and weight.size:
        weight.flat[0] += 2.0**-35
    layer = eqx.tree_at(lambda model: model.weight, layer, jnp.asarray(weight))
    if bias:
        offset = ((np.arange(rows) - 1) / 16).astype(storage)
        layer = eqx.tree_at(lambda model: model.bias, layer, jnp.asarray(offset))
    shape = () if in_features == "scalar" else (columns,)
    if route == "vmap":
        shape = (2, *shape)
    elif route == "nested-vmap":
        shape = (2, 3, *shape)
    value = (
        (np.arange(np.prod(shape, dtype=np.int64)).reshape(shape) % 7 - 3) / 8
    ).astype(input_dtype)
    key = jax.random.key(918)
    return layer, jnp.asarray(value), key


def _function(layer: eqx.nn.Linear, key: Any, route: str) -> Any:
    alias = layer.__call__

    def direct(value: Any) -> Any:
        return layer(value, key=key)

    if route == "native-jit":
        apply = jax.jit(direct)
    elif route == "vmap":
        apply = jax.vmap(direct)
    elif route == "nested-vmap":
        apply = jax.vmap(jax.vmap(direct))
    elif route == "alias":

        def apply(value: Any) -> Any:
            return alias(x=value, key=key)

    else:
        apply = direct

    def fn(value: Any) -> Any:
        # Physical parameter and alias outputs make original storage types and
        # bytes observable even when native arithmetic promotes its operands.
        native = apply(value)
        consumed = native.astype(np.float32) + np.float32(0.5)
        result = (native, consumed, layer.weight, layer.weight, value, value)
        return result if layer.bias is None else (*result, layer.bias)

    return fn


@pytest.mark.parametrize("storage,input_dtype,x64", _WIDTHS)
@pytest.mark.parametrize("bias", [False, True])
def test_native_linear_geometry_storage_aliases_and_context(
    storage: str,
    input_dtype: str,
    x64: bool,
    bias: bool,
    tmp_path: Path,
) -> None:
    with jax.enable_x64(x64):
        for number, (in_features, out_features, route) in enumerate(_GEOMETRIES):
            directory = tmp_path / f"geometry-{number}-{route}"
            directory.mkdir()
            layer, value, key = _fixture(
                in_features, out_features, storage, input_dtype, bias, route
            )
            fn = _function(layer, key, route)
            state, feed = _state(layer), np.asarray(value).copy()
            _save_arrays(directory / "feeds.npz", (feed,))
            _save_arrays(directory / "parameters.npz", jax.tree.leaves(layer))
            key_payload = _payload(jax.random.key_data(key))
            context = _globals(), _settings()
            oracle = _oracle(layer, value)
            expected = (
                oracle,
                oracle.astype(np.float32) + np.float32(0.5),
                np.asarray(layer.weight),
                np.asarray(layer.weight),
                feed,
                feed,
            )
            if layer.bias is not None:
                expected = (*expected, np.asarray(layer.bias))
            references = tuple(
                tuple(np.asarray(v).copy() for v in jax.tree.leaves(call(value)))
                for call in (fn, jax.jit(_fresh(fn)))
            )
            spec = jax.ShapeDtypeStruct(
                value.shape, value.dtype, weak_type=bool(jax.typeof(value).weak_type)
            )
            native = jax.make_jaxpr(_fresh(fn))(spec)
            before_constants = tuple(_payload(v) for v in native.consts)
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                active = jax.make_jaxpr(_fresh(fn))(spec)
                active_outputs = tuple(jax.tree.leaves(_fresh(fn)(value)))
            (directory / "native-trace.json").write_text(
                json.dumps(typed_trace_manifest(native), indent=2)
            )
            (directory / "activated-trace.json").write_text(
                json.dumps(typed_trace_manifest(active), indent=2)
            )
            _save_arrays(directory / "native-constants.npz", native.consts)
            _save_arrays(directory / "activated-constants.npz", active.consts)
            for role, outputs in zip(
                ("eager", "fresh-jit", "active"),
                (*references, active_outputs),
                strict=True,
            ):
                _save_arrays(directory / (role + ".npz"), outputs)
            assert typed_trace_manifest(active) == typed_trace_manifest(native)
            assert tuple(_payload(v) for v in active.consts) == before_constants
            for outputs in (*references, active_outputs):
                for actual, reference in zip(outputs, expected, strict=True):
                    _same(actual, reference)
            model = ui._to_onnx(
                fn, [spec], opset=23, precision_policy=PrecisionPolicy.SOURCE
            )
            (directory / "source.onnx").write_bytes(model.SerializeToString())
            onnx.checker.check_model(model, full_check=True)
            provenance = get_precision_report(model)
            assert provenance is not None
            assert (
                next(
                    check.status
                    for check in provenance.checks
                    if check.category == "lowering"
                )
                is CheckStatus.VALIDATED
            )
            samples = [
                SourceSample((feed,), outputs, _BITWISE) for outputs in references
            ]
            reports = []
            for mode in ("disabled", "all"):
                report = validate_source_samples(
                    model, samples=samples, optimization=mode
                )
                reports.append(report)
                (directory / f"{mode}-report.json").write_text(
                    json.dumps(
                        {
                            "artifact_sha256": report.artifact_sha256,
                            "target": (
                                None if report.target is None else asdict(report.target)
                            ),
                            "checks": [asdict(check) for check in report.checks],
                            "samples": [asdict(sample) for sample in report.samples],
                            "status": report.status,
                            "scope": report.scope,
                        },
                        indent=2,
                    )
                )
            require_target_pair(
                model,
                samples,
                reports[0],
                reports[1],
                evidence_path=directory / "target-boundary.json",
            )
            assert _state(layer) == state
            _same(value, feed)
            assert _payload(jax.random.key_data(key)) == key_payload
            assert (_globals(), _settings()) == context


def test_source_forwarding_preserves_objects_arguments_and_exception_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    model, value, key, returned = (object() for _ in range(4))
    failure = TypeError("original native callback failure")

    def native(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((self, args, kwargs))
        if kwargs.get("fail"):
            raise failure
        return returned

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("SOURCE reached the private Linear primitive")

    monkeypatch.setitem(vars(LinearPlugin._PRIM), "bind", forbidden)
    wrapper = LinearPlugin._patch_call(native)
    forms = [
        ((value,), {}),
        ((), {"x": value}),
        ((value,), {"key": None}),
        ((value,), {"key": key}),
        ((value, key), {"unknown": key}),
    ]
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        for args, kwargs in forms:
            assert wrapper(model, *args, **kwargs) is returned
        with pytest.raises(TypeError) as caught:
            wrapper(model, x=value, key=key, fail=True)
    assert caught.value is failure
    assert calls == [(model, args, kwargs) for args, kwargs in forms] + [
        (model, (), {"x": value, "key": key, "fail": True})
    ]
    with (
        precision_policy_scope(PrecisionPolicy.SOURCE),
        pytest.raises(RuntimeError, match="Linear.__call__ is not available"),
    ):
        LinearPlugin._patch_call(None)(model, value)


def _outcome(call: Any) -> tuple[Any, Any]:
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001
        return {
            "kind": "raised",
            "type": type(exc).__qualname__,
            "detail": str(exc),
        }, None
    return {"kind": "returned"}, result


def _call_form(method: Any, layer: Any, form: str, key: Any) -> Any:
    def call(value: Any) -> Any:
        args, kwargs = (value,), {}
        if form == "keyword-x":
            args, kwargs = (), {"x": value}
        elif form == "none-key":
            kwargs = {"key": None}
        elif form in {"actual-key", "ignored-object-key"}:
            kwargs = {"key": key}
        elif form == "positional-key":
            args = value, key
        elif form == "positional-none":
            args = value, None
        elif form == "duplicate-x":
            kwargs = {"x": value}
        elif form == "unknown":
            kwargs = {"out_sharding": None}
        elif form == "missing-x":
            args = ()
        elif form == "bad-shape":
            args = (jnp.ones((5,), dtype=np.float32),)
        return method(layer, *args, **kwargs)

    return call


def test_original_native_call_forms_and_errors_are_preserved(tmp_path: Path) -> None:
    layer = eqx.nn.Linear(4, 3, key=jax.random.key(81))
    value = jnp.asarray([0.25, -0.5, 0.75, 1.0], dtype=np.float32)
    key = jax.random.key(82)
    state, feed, key_before = (
        _state(layer),
        _payload(value),
        _payload(jax.random.key_data(key)),
    )
    wrapper = LinearPlugin._patch_call(_NATIVE_CALL)
    forms = (
        "omitted",
        "keyword-x",
        "none-key",
        "actual-key",
        "ignored-object-key",
        "positional-key",
        "positional-none",
        "duplicate-x",
        "unknown",
        "missing-x",
        "bad-shape",
    )
    outcomes = {}
    for name in forms:
        form_key = object() if name == "ignored-object-key" else key
        original = _call_form(_NATIVE_CALL, layer, name, form_key)
        active_call = _call_form(wrapper, layer, name, form_key)
        expected, native = _outcome(lambda call=original: call(value))
        with precision_policy_scope(PrecisionPolicy.SOURCE):
            actual, active = _outcome(lambda call=active_call: call(value))
        assert actual == expected
        if native is not None:
            _same(active, native)
        # Fresh native/activated traces retain old errors as well as successful
        # actual key omission. Invalid forms are never silently normalized.
        native_trace, closed = _outcome(
            lambda call=original: jax.make_jaxpr(_fresh(call))(value)
        )
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            active_trace, activated = _outcome(
                lambda call=active_call: jax.make_jaxpr(_fresh(call))(value)
            )
        assert active_trace == native_trace
        expected_kind = (
            "returned"
            if name
            in {"omitted", "keyword-x", "none-key", "actual-key", "ignored-object-key"}
            else "raised"
        )
        assert expected["kind"] == native_trace["kind"] == expected_kind
        if closed is not None:
            assert typed_trace_manifest(activated) == typed_trace_manifest(closed)
            assert tuple(_payload(v) for v in activated.consts) == tuple(
                _payload(v) for v in closed.consts
            )
        outcomes[name] = {"eager": actual, "trace": active_trace}
    (tmp_path / "native-call-forms.json").write_text(json.dumps(outcomes, indent=2))
    assert _state(layer) == state and _payload(value) == feed
    assert _payload(jax.random.key_data(key)) == key_before


def test_stored_fp64_native_canonicalization_with_caller_x64_off_is_preserved() -> None:
    with jax.enable_x64(True):
        layer = eqx.nn.Linear(4, 3, dtype=np.float64, key=jax.random.key(83))
    state = _state(layer)
    value = jnp.asarray([0.25, -0.5, 0.75, 1.0], dtype=np.float32)
    with jax.enable_x64(False):
        native = _NATIVE_CALL(layer, value)
        trace = jax.make_jaxpr(_fresh(lambda x: _NATIVE_CALL(layer, x)))(value)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = layer(value)
            active = jax.make_jaxpr(_fresh(lambda x: layer(x)))(value)
        _same(actual, native)
        assert typed_trace_manifest(active) == typed_trace_manifest(trace)
        assert tuple(_payload(v) for v in active.consts) == tuple(
            _payload(v) for v in trace.consts
        )
    assert _state(layer) == state


@pytest.mark.parametrize(
    "in_features,out_features,shape", [("scalar", 3, (1,)), (4, "scalar", (4, 2))]
)
def test_native_scalar_shape_errors_are_preserved(
    in_features: Any, out_features: Any, shape: Any
) -> None:
    layer = eqx.nn.Linear(in_features, out_features, key=jax.random.key(84))
    value = jnp.ones(shape, dtype=np.float32)
    state = _state(layer)
    expected, _ = _outcome(lambda: _NATIVE_CALL(layer, value))
    assert expected["kind"] == "raised"
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        actual, _ = _outcome(lambda: layer(value))
    assert actual == expected
    assert _state(layer) == state


def test_source_still_rejects_the_private_linear_carrier() -> None:
    weight = jnp.ones((3, 4), dtype=np.float32)
    bias = jnp.zeros((3,), dtype=np.float32)

    def private(value: Any) -> Any:
        return LinearPlugin._PRIM.bind(value, weight, bias)

    LinearPlugin.ensure_abstract_eval_bound()
    trace = jax.make_jaxpr(private)(jax.ShapeDtypeStruct((4,), np.float32))
    body = getattr(trace, "jaxpr", trace)
    assert len(body.eqns) == 1 and body.eqns[0].primitive is LinearPlugin._PRIM
    with pytest.raises(SourcePrecisionError) as caught:
        ui._to_onnx(
            private,
            [jax.ShapeDtypeStruct((4,), np.float32)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert caught.value.category == "profile-exclusion"
    assert caught.value.primitive == "eqx.nn.linear"


@pytest.mark.parametrize(
    "origin,ordinal",
    [("plugin::eqx.nn.linear", ordinal) for ordinal in range(5)]
    + [("example::examples.eqx::MlpExample", ordinal) for ordinal in (0, 1)],
)
@pytest.mark.parametrize("x64", [False, True])
def test_unchanged_original_linear_and_mlp_recipes_keep_all_raw_outcomes(
    origin: str,
    ordinal: int,
    x64: bool,
    original_scope: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, plan, registrations, adapters = original_scope
    row = next(
        row for row in plan.rows if (row.origin, row.ordinal) == (origin, ordinal)
    )
    registered = {name: metadata for name, _, metadata in registrations}
    metadata = probe._resolve_case(row, registered)
    target, callback = metadata["callable"], metadata.get("post_check_onnx_graph")
    before = _globals(), _settings()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Original product-only recipes must not invent runtime acceptance")

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
                "plan_rows": len(result.rows),
                "plan_cells": len(result.cells),
                "row": asdict(row),
                "cell": asdict(cell),
            },
            indent=2,
        )
    )
    capture = evidence.replay_capture
    assert capture["raw_callback"]["status"] == evidence.raw_callback_status
    assert cell.source_reference is CheckStatus.VALIDATED
    assert (
        cell.runtime
        is cell.numerics
        is evidence.runtime
        is evidence.numerics
        is CheckStatus.UNVERIFIED
    )
    assert row.criteria is None  # No criterion is manufactured for these rows.
    assert evidence.row_contract_sha256 == row.contract_sha256
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
    if origin == "plugin::eqx.nn.linear" and ordinal in (2, 3, 4):
        assert failure is None, failure
        assert (
            cell.artifact is cell.schema is cell.lowering_audit is CheckStatus.VALIDATED
        )
        assert evidence.raw_callback_status in {"passed", "failed", "error"}
        if evidence.raw_callback_status in {"failed", "error"}:
            assert cell.category == "raw-callback-failure"
            assert capture["raw-callback"]["status"] == "failed"
    else:
        # Native rhs-batched symbolic Linear rows and later MLP exclusions remain
        # actual captured failures. Their source is never swapped/reordered to
        # fit the current leading-lhs-B admission or another family's contract.
        if failure is not None:
            assert failure["category"] == "profile-exclusion"
            assert failure["primitive"] != "eqx.nn.linear"
            assert all(failure[key] for key in ("phase", "requirement", "action"))
        else:
            assert cell.artifact is cell.schema is CheckStatus.VALIDATED
    assert probe._resolve_case(row, registered)["callable"] is target
    assert probe._resolve_case(row, registered).get("post_check_onnx_graph") is callback
    assert result.rows == plan.rows
    assert [after for after in result.cells if after.identity != cell.identity] == [
        before for before in plan.cells if before.identity != cell.identity
    ]
    assert (_globals(), _settings()) == before
