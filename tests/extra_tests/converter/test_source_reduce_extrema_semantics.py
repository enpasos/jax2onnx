# tests/extra_tests/converter/test_source_reduce_extrema_semantics.py

"""Numerical SOURCE extrema acceptance, including observable zero signs.

All full-feed native references and artifacts precede the first ORT session in
each fresh worker. The default diagnostic records failures without claiming
acceptance; the explicit FTZ target must pass the unchanged strict criteria.
NaN payload equality is deliberately absent. Mathematical subnormal words are
recorded separately from native acceptance.

The original maximum/reciprocal composition at opset21 remains a separately
rejected, numerically unverified SOURCE obligation. The distinct max-only21
worker case does not qualify that composition or establish an ONNX limitation.

The unchanged default-target acceptance obligation remains executable with:
python tests/extra_tests/converter/test_source_reduce_extrema_semantics.py \
    --profile default-strict --report /tmp/extrema-default-strict.json
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import user_interface as ui
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError


def _subnormal_mask(value: np.ndarray) -> np.ndarray:
    words = (
        np.ascontiguousarray(value)
        .view(f"uint{value.itemsize * 8}")
        .reshape(value.shape)
    )
    fraction = (1 << np.finfo(value.dtype).nmant) - 1
    exponent = ((1 << (value.itemsize * 8 - 1)) - 1) ^ fraction
    return ((words & exponent) == 0) & ((words & fraction) != 0)


def _assert_output(
    actual: Any, expected: np.ndarray, *, role: str, nominal_oracle: bool = False
) -> None:
    actual = np.asarray(actual)
    assert actual.dtype == expected.dtype, (role, actual.dtype, expected.dtype)
    assert actual.shape == expected.shape, (role, actual.shape, expected.shape)
    nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), nan, err_msg=role)
    compared = ~nan
    if nominal_oracle:
        # JAX documents backend-dependent FTZ semantics. A mathematical word
        # oracle cannot replace the actual official native reference there.
        compared &= ~_subnormal_mask(expected)
    word_dtype = np.dtype(f"uint{expected.itemsize * 8}")
    np.testing.assert_array_equal(
        actual[compared].view(word_dtype),
        expected[compared].view(word_dtype),
        err_msg=role,
    )
    zeros = (expected == 0) & ~nan
    np.testing.assert_array_equal(
        np.signbit(actual[zeros]), np.signbit(expected[zeros]), err_msg=role
    )


def _record_subnormal_conformance(
    record_property: Any, actual: Any, nominal: Any, *, role: str, feed: int
) -> None:
    rows = []
    for index, (value, expected) in enumerate(zip(actual, nominal, strict=True)):
        mask = _subnormal_mask(expected)
        if not np.any(mask):
            continue
        value = np.asarray(value)
        if value.dtype != expected.dtype or value.shape != expected.shape:
            # Keep diagnostics from preempting the strict per-output failures.
            rows.append(
                {
                    "output": index,
                    "diagnostic_status": "output-contract-mismatch",
                    "actual_dtype": value.dtype.name,
                    "actual_shape": list(value.shape),
                    "expected_dtype": expected.dtype.name,
                    "expected_shape": list(expected.shape),
                }
            )
            continue
        word_dtype = np.dtype(f"uint{expected.itemsize * 8}")
        nominal_words = expected[mask].view(word_dtype)
        actual_words = value[mask].view(word_dtype)
        rows.append(
            {
                "output": index,
                "dtype": expected.dtype.name,
                "shape": list(expected.shape),
                "flat_coordinates": np.flatnonzero(mask).tolist(),
                "nominal_words": [hex(int(word)) for word in nominal_words],
                "actual_words": [hex(int(word)) for word in actual_words],
                "same_words": bool(np.array_equal(actual_words, nominal_words)),
            }
        )
    if rows:
        record_property(
            f"subnormal_conformance_{role}_feed{feed}",
            json.dumps({"authority": "diagnostic-only", "observations": rows}),
        )


def _oracle(
    values: np.ndarray, axes: tuple[int, ...], *, minimum: bool, keepdims: bool = False
) -> np.ndarray:
    """Nominal word-selection oracle; native FTZ keeps its own authority."""
    retained = tuple(axis for axis in range(values.ndim) if axis not in axes)
    retained_shape = tuple(values.shape[axis] for axis in retained)
    width = int(np.prod([values.shape[axis] for axis in axes], dtype=np.int64))
    rows = np.transpose(values, (*retained, *axes)).reshape((*retained_shape, width))
    result = np.empty(retained_shape, dtype=values.dtype)
    for index in np.ndindex(retained_shape):
        row = rows[index]
        if not row.size:
            result[index] = np.inf if minimum else -np.inf
        elif np.any(np.isnan(row)):
            result[index] = np.nan
        else:
            value = (min if minimum else max)(row)
            if value == 0:
                zero_signs = np.signbit(row[row == 0])
                negative = bool(np.any(zero_signs) if minimum else np.all(zero_signs))
                value = np.asarray(-0.0 if negative else 0.0, dtype=values.dtype)
            result[index] = value
    if keepdims:
        result = result.reshape(
            tuple(
                1 if axis in axes else values.shape[axis] for axis in range(values.ndim)
            )
        )
    return result


def _model(frontend: str) -> Any:
    minimum, maximum = (
        (jnp.min, jnp.max)
        if frontend == "numpy"
        else (jax.lax.reduce_min, jax.lax.reduce_max)
    )

    def model(x: Any, empty: Any) -> Any:
        if frontend == "numpy":
            low, high = minimum(x, axis=2), maximum(x, axis=2)
            multi = minimum(x, axis=(0, 2), keepdims=True), maximum(
                x, axis=(0, 2), keepdims=True
            )
            scalar = minimum(x), maximum(x)
            identity = minimum(x, axis=()), maximum(x, axis=())
            identities = minimum(empty, axis=1, initial=np.inf), maximum(
                empty, axis=1, initial=-np.inf
            )
        else:
            low, high = minimum(x, (2,)), maximum(x, (2,))
            multi = minimum(x, (0, 2)), maximum(x, (0, 2))
            scalar = minimum(x, (0, 1, 2)), maximum(x, (0, 1, 2))
            identity = minimum(x, ()), maximum(x, ())
            identities = minimum(empty, (1,)), maximum(empty, (1,))
        return (
            (low, high, jnp.reciprocal(low), jnp.reciprocal(high)),
            multi,
            scalar,
            identity,
            identities,
        )

    return model


def _expected(values: np.ndarray, empty: np.ndarray, frontend: str) -> Any:
    def pair(x: np.ndarray, axes: tuple[int, ...], *, keepdims: bool = False) -> Any:
        return tuple(
            _oracle(x, axes, minimum=minimum, keepdims=keepdims)
            for minimum in (True, False)
        )

    low, high = pair(values, (2,))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        reciprocal = np.reciprocal(low), np.reciprocal(high)
    return (
        (low, high, *reciprocal),
        pair(values, (0, 2), keepdims=frontend == "numpy"),
        pair(values, (0, 1, 2)),
        pair(values, ()),
        pair(empty, (1,)),
    )


def _feeds(dtype: Any) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    dtype = np.dtype(dtype)
    word_dtype = np.dtype(f"uint{dtype.itemsize * 8}")
    tiny = np.asarray(1, dtype=word_dtype).view(dtype)[()]
    two_tiny = np.asarray(2, dtype=word_dtype).view(dtype)[()]
    boundary = np.nextafter(np.asarray(1, dtype), np.asarray(2, dtype))
    rows = np.asarray(
        [
            [0.0, -0.0],
            [-0.0, 0.0],
            [-0.0, -0.0],
            [0.0, 0.0],
            [1, np.nan],
            [np.nan, 1],
            [np.nan, np.nan],
            [-np.inf, -0.0],
            [np.inf, 0.0],
            [-np.inf, np.inf],
            [1, boundary],
            [tiny, two_tiny],
            [-tiny, -two_tiny],
        ],
        dtype=dtype,
    )
    values = np.stack([rows, rows[:, ::-1]])
    finite = np.where(np.isnan(values), np.asarray(2, dtype), values)
    zero_ties = np.empty_like(values)
    zero_ties[..., 0], zero_ties[..., 1] = 0.0, -0.0
    negative_zeros = np.full_like(values, -0.0)
    empty = np.empty((2, 0, 3), dtype=dtype)
    return (
        (values, empty),
        (values[:, :, ::-1].copy(), empty),
        (finite, empty),
        (zero_ties, empty),
        (negative_zeros, empty),
    )


@dataclass(frozen=True)
class _PreparedFunction:
    artifact: bytes
    feeds: Any
    expected: Any
    references: Any
    native_outcomes: Any
    agreement_outcomes: Any


def _prepare_function(
    model: Any,
    feeds: Any,
    expected: Any,
    *,
    opset: int,
    record_property: Any,
) -> _PreparedFunction:
    tree = jax.tree.structure(expected[0])
    references = []
    native_outcomes, agreement_outcomes = [], []
    for index, (feed, wanted) in enumerate(zip(feeds, expected, strict=True)):
        native = []
        fresh_jit = jax.jit(lambda x, empty: model(x, empty))
        for role, result in (("eager", model(*feed)), ("jit", fresh_jit(*feed))):
            assert jax.tree.structure(result) == tree
            arrays = [np.asarray(value).copy() for value in jax.tree.leaves(result)]
            nominal = jax.tree.leaves(wanted)
            native.append(arrays)
            for output, (actual, reference) in enumerate(
                zip(arrays, nominal, strict=True)
            ):
                _assert_output(
                    actual,
                    reference,
                    role=f"native {role} nominal oracle feed={index}",
                    nominal_oracle=True,
                )
                native_outcomes.append([index, output, role, "PASS"])
            _record_subnormal_conformance(
                record_property, arrays, nominal, role=role, feed=index
            )
        for output, (eager, compiled) in enumerate(zip(*native, strict=True)):
            _assert_output(
                eager, compiled, role=f"native eager/JIT agreement feed={index}"
            )
            agreement_outcomes.append([index, output, "PASS"])
        references.append(native)
    proto = ui._to_onnx(
        model,
        [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in feeds[0]],
        opset=opset,
        return_mode="proto",
        precision_policy=PrecisionPolicy.SOURCE,
    )
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    typed = onnx.shape_inference.infer_shapes(proto, strict_mode=True)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*typed.graph.input, *typed.graph.value_info, *typed.graph.output)
    }
    # Integer selection is safe; no floating Where may repair signed zeros.
    assert all(
        types[node.output[0]] == onnx.TensorProto.INT64
        for node in typed.graph.node
        if node.op_type == "Where"
    )
    return _PreparedFunction(
        proto.SerializeToString(),
        feeds,
        expected,
        references,
        native_outcomes,
        agreement_outcomes,
    )


def _execute_function(
    prepared: _PreparedFunction,
    *,
    record_property: Any,
    session_config: dict[str, str] | None = None,
    comparison: Any = None,
) -> None:
    for mode in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        for name, value in (session_config or {}).items():
            options.add_session_config_entry(name, value)
        options.graph_optimization_level = mode
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            prepared.artifact, options, providers=["CPUExecutionProvider"]
        )
        for index, (feed, wanted) in enumerate(
            zip(prepared.feeds, prepared.expected, strict=True)
        ):
            results = session.run(
                None,
                {
                    meta.name: value
                    for meta, value in zip(session.get_inputs(), feed, strict=True)
                },
            )
            _record_subnormal_conformance(
                record_property,
                results,
                jax.tree.leaves(wanted),
                role=mode.name,
                feed=index,
            )
            for role, native in zip(
                ("eager", "jit"), prepared.references[index], strict=True
            ):
                assert len(results) == len(native)
                for output, (actual, reference) in enumerate(
                    zip(results, native, strict=True)
                ):
                    if comparison is None:
                        _assert_output(
                            actual,
                            reference,
                            role=f"ORT {mode.name} native={role} feed={index}",
                        )
                    else:
                        comparison(
                            actual,
                            reference,
                            role=role,
                            mode=mode.name,
                            feed=index,
                            output=output,
                        )


def _accept_function(
    model: Any,
    feeds: Any,
    expected: Any,
    *,
    opset: int,
    record_property: Any,
    session_config: dict[str, str] | None = None,
) -> None:
    """Original unmasked strict acceptance, also available to direct callers."""
    prepared = _prepare_function(
        model, feeds, expected, opset=opset, record_property=record_property
    )
    _execute_function(
        prepared, record_property=record_property, session_config=session_config
    )


def _accept(dtype: Any, frontend: str, *, opset: int, record_property: Any) -> None:
    feeds = _feeds(dtype)
    _accept_function(
        _model(frontend),
        feeds,
        [_expected(*feed, frontend) for feed in feeds],
        opset=opset,
        record_property=record_property,
    )


def _scientific_maximum_case() -> Any:
    # The scientific21 profile admits reduce_max. Its reduce_min exclusion is
    # unrelated to this numerical repair and is deliberately preserved.
    def model(x: Any, empty: Any) -> Any:
        maximum = jax.lax.reduce_max(x, (2,))
        return (
            maximum,
            jnp.reciprocal(maximum),
            jax.lax.reduce_max(x, (0, 2)),
            jax.lax.reduce_max(x, (0, 1, 2)),
            jax.lax.reduce_max(x, ()),
            jax.lax.reduce_max(empty, (1,)),
        )

    feeds = _feeds(np.float64)
    expected = []
    for feed in feeds:
        main, multi, scalar, identity, empty = _expected(*feed, "lax")
        expected.append((main[1], main[3], multi[1], scalar[1], identity[1], empty[1]))
    return model, feeds, expected


def _scientific_maximum_only_case() -> Any:
    """Separate admitted maximum witness; original reciprocal case stays intact."""
    _, feeds, original_expected = _scientific_maximum_case()

    def model(x: Any, empty: Any) -> Any:
        return (
            jax.lax.reduce_max(x, (2,)),
            jax.lax.reduce_max(x, (0, 2)),
            jax.lax.reduce_max(x, (0, 1, 2)),
            jax.lax.reduce_max(x, ()),
            jax.lax.reduce_max(empty, (1,)),
        )

    expected = [(values[0], *values[2:]) for values in original_expected]
    return model, feeds, expected


_UNIMPLEMENTED_COMPOSITION = {
    "id": "original-maximum-reciprocal-float64-21",
    "owner": "test_scientific_maximum_original_reciprocal_composition_rejected",
    "feeds": 5,
    "outputs": 6,
    "numerical_status": "UNVERIFIED",
    "target_comparisons": 0,
    "obligation": "unimplemented SOURCE composition; no ONNX-limit claim",
}


_CASES = (
    ("lax-float32-23", "float32", "lax", 23, 12),
    ("lax-float64-23", "float64", "lax", 23, 12),
    ("numpy-float32-23", "float32", "numpy", 23, 12),
    ("numpy-float64-23", "float64", "numpy", 23, 12),
    ("lax-float64-16", "float64", "lax", 16, 12),
    ("lax-float64-26", "float64", "lax", 26, 12),
    ("maximum-only-float64-21", "float64", "maximum-only", 21, 5),
    ("lax-float16-23", "float16", "lax", 23, 12),
)
_ROLES = ("eager", "jit")
_MODES = ("ORT_DISABLE_ALL", "ORT_ENABLE_ALL")
_BOUNDARY = "native-signed-zero-to-target-same-sign-subnormal"


def _known_default_boundary(actual: Any, native: np.ndarray) -> dict[str, Any] | None:
    """Classify only the observed failure; this never changes strict criteria."""
    actual = np.asarray(actual)
    if actual.dtype != native.dtype or actual.shape != native.shape:
        return None
    nan = np.isnan(native)
    if not np.array_equal(np.isnan(actual), nan):
        return None
    word_dtype = np.dtype(f"uint{native.itemsize * 8}")
    actual_words = np.ascontiguousarray(actual).view(word_dtype).reshape(actual.shape)
    native_words = np.ascontiguousarray(native).view(word_dtype).reshape(native.shape)
    different = (actual_words != native_words) & ~nan
    if not np.any(different):
        return None
    sign = 1 << (native.itemsize * 8 - 1)
    native_zero = (native_words & (sign - 1)) == 0
    same_sign = (actual_words & sign) == (native_words & sign)
    allowed = native_zero & _subnormal_mask(actual) & same_sign
    if not np.all(allowed[different]):
        return None
    return {
        "kind": _BOUNDARY,
        "dtype": native.dtype.name,
        "flat_coordinates": np.flatnonzero(different).tolist(),
        "native_words": [hex(int(word)) for word in native_words[different]],
        "target_words": [hex(int(word)) for word in actual_words[different]],
    }


def _expected_coverage() -> tuple[set[Any], set[Any], set[Any]]:
    native, agreement, target = set(), set(), set()
    for identity, _, _, _, outputs in _CASES:
        for feed in range(5):
            for output in range(outputs):
                agreement.add((identity, feed, output))
                for role in _ROLES:
                    native.add((identity, feed, output, role))
                    for mode in _MODES:
                        target.add((identity, feed, output, role, mode))
    return native, agreement, target


def _feed_hash(feed: Any) -> str:
    digest = hashlib.sha256()
    for value in feed:
        digest.update(json.dumps([value.dtype.str, value.shape]).encode())
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _run_profile(profile: str, report_path: Path) -> int:
    session_config = (
        {"session.set_denormal_as_zero": "1"} if profile == "ftz-strict" else None
    )
    report: dict[str, Any] = {
        "profile": profile,
        "session_config": session_config,
        "strict_acceptance": profile != "default-diagnostic",
        "criteria": {
            "native_oracle": "normal/special words; subnormal words diagnostic-only",
            "native_agreement": "strict shape/dtype/NaN-mask/all non-NaN words",
            "target": "strict shape/dtype/NaN-mask/all non-NaN words",
        },
        "cases": [],
        "prepared_before_first_session": [],
        "native_outcomes": [],
        "agreement_outcomes": [],
        "target_outcomes": [],
        "subnormal_conformance": [],
        "unimplemented_compositions": [dict(_UNIMPLEMENTED_COMPOSITION)],
    }
    try:
        prepared_cases = []
        # No ORT session exists until every native reference and schema-checked
        # artifact is materialized. FTZ target settings cannot affect references.
        for identity, dtype, frontend, opset, outputs in _CASES:

            def record(name: str, value: str, case: str = identity) -> None:
                report["subnormal_conformance"].append([case, name, json.loads(value)])

            with jax.enable_x64(dtype == "float64"):
                if frontend == "maximum-only":
                    model, feeds, expected = _scientific_maximum_only_case()
                else:
                    model, feeds = _model(frontend), _feeds(np.dtype(dtype))
                    expected = [_expected(*feed, frontend) for feed in feeds]
                prepared = _prepare_function(
                    model, feeds, expected, opset=opset, record_property=record
                )
            assert len(prepared.feeds) == 5
            assert all(
                len(pair) == outputs for refs in prepared.references for pair in refs
            )
            prepared_cases.append((identity, prepared))
            report["cases"].append(
                {
                    "id": identity,
                    "dtype": dtype,
                    "frontend": frontend,
                    "opset": opset,
                    "outputs": outputs,
                    "schema": "PASS",
                    "artifact_sha256": hashlib.sha256(prepared.artifact).hexdigest(),
                    "input_sha256": [_feed_hash(feed) for feed in prepared.feeds],
                }
            )
            report["native_outcomes"].extend(
                [identity, *row] for row in prepared.native_outcomes
            )
            report["agreement_outcomes"].extend(
                [identity, *row] for row in prepared.agreement_outcomes
            )
        report["prepared_before_first_session"] = [case for case, _ in prepared_cases]
        assert len(prepared_cases) == len(_CASES)

        for identity, prepared in prepared_cases:

            def record(name: str, value: str, case: str = identity) -> None:
                report["subnormal_conformance"].append([case, name, json.loads(value)])

            def compare(
                actual: Any,
                native: np.ndarray,
                *,
                role: str,
                mode: str,
                feed: int,
                output: int,
            ) -> None:
                status, detail = "PASS", None
                try:
                    # The original, unmasked comparison executes for every
                    # role/mode/output, including all known failing coordinates.
                    _assert_output(
                        actual,
                        native,
                        role=f"{identity}/{feed}/{output}/{role}/{mode}",
                    )
                except AssertionError as exc:
                    status = "FAILED"
                    detail = _known_default_boundary(actual, native)
                    if detail is None:
                        detail = {
                            "kind": "unexpected-mismatch",
                            "error": str(exc)[:240],
                        }
                report["target_outcomes"].append(
                    [identity, feed, output, role, mode, status, detail]
                )

            _execute_function(
                prepared,
                record_property=record,
                session_config=session_config,
                comparison=compare,
            )
    except Exception as exc:
        report["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}

    native_ids, agreement_ids, target_ids = _expected_coverage()
    complete = (
        {tuple(row[:4]) for row in report["native_outcomes"]} == native_ids
        and len(report["native_outcomes"]) == len(native_ids)
        and {tuple(row[:3]) for row in report["agreement_outcomes"]} == agreement_ids
        and len(report["agreement_outcomes"]) == len(agreement_ids)
        and {tuple(row[:5]) for row in report["target_outcomes"]} == target_ids
        and len(report["target_outcomes"]) == len(target_ids)
        and len(report["prepared_before_first_session"]) == len(_CASES)
    )
    failed = [row for row in report["target_outcomes"] if row[5] == "FAILED"]
    report["numerical_status"] = (
        "FAILED" if failed else "PASS" if complete else "UNVERIFIED"
    )
    allowed = all(
        profile == "default-diagnostic" and row[6]["kind"] == _BOUNDARY
        for row in failed
    )
    gate_pass = complete and "fatal_error" not in report and allowed
    report["gate_status"] = "PASS" if gate_pass else "FAILED"
    report["acceptance_claim"] = (
        gate_pass and profile != "default-diagnostic" and not failed
    )
    report_path.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    return 0 if gate_pass else 1


def _check_profile_report(report: dict[str, Any], profile: str) -> None:
    assert report["profile"] == profile
    assert report["session_config"] == (
        {"session.set_denormal_as_zero": "1"} if profile == "ftz-strict" else None
    )
    assert report["strict_acceptance"] is (profile != "default-diagnostic")
    assert report["unimplemented_compositions"] == [_UNIMPLEMENTED_COMPOSITION]
    assert "fatal_error" not in report, report.get("fatal_error")
    identities = [case[0] for case in _CASES]
    assert report["prepared_before_first_session"] == identities
    assert len(report["cases"]) == len(_CASES)
    for actual, expected in zip(report["cases"], _CASES, strict=True):
        assert (
            tuple(
                actual[key] for key in ("id", "dtype", "frontend", "opset", "outputs")
            )
            == expected
        )
        assert actual["schema"] == "PASS"
        assert len(actual["artifact_sha256"]) == 64
        assert len(actual["input_sha256"]) == 5
        assert all(len(digest) == 64 for digest in actual["input_sha256"])
    expected_native, expected_agreement, expected_target = _expected_coverage()
    for key, width, expected in (
        ("native_outcomes", 4, expected_native),
        ("agreement_outcomes", 3, expected_agreement),
        ("target_outcomes", 5, expected_target),
    ):
        rows = report[key]
        assert len(rows) == len(expected)
        assert {tuple(row[:width]) for row in rows} == expected
    assert all(len(row) == 5 and row[4] == "PASS" for row in report["native_outcomes"])
    assert all(
        len(row) == 4 and row[3] == "PASS" for row in report["agreement_outcomes"]
    )
    failed = []
    for row in report["target_outcomes"]:
        assert len(row) == 7 and row[5] in {"PASS", "FAILED"}
        if row[5] == "PASS":
            assert row[6] is None
        else:
            failed.append(row)
            assert profile == "default-diagnostic"
            assert row[6]["kind"] == _BOUNDARY
            assert row[6]["flat_coordinates"]
            assert len(row[6]["native_words"]) == len(row[6]["target_words"])
    assert report["numerical_status"] == ("FAILED" if failed else "PASS")
    assert report["gate_status"] == "PASS"
    assert report["acceptance_claim"] is (profile == "ftz-strict")


def test_source_extrema_full_matrix_isolated_target_profiles(
    tmp_path: Path, record_property: Any
) -> None:
    """Diagnostic success is separate from FTZ-target numerical acceptance."""
    profiles = ("default-diagnostic", "ftz-strict")
    reports, processes = [], []
    for profile in profiles:
        path = tmp_path / f"{profile}.json"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--profile",
                profile,
                "--report",
                str(path),
            ],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        report = json.loads(path.read_text()) if path.is_file() else None
        if report is not None:
            record_property(
                f"extrema_{profile}", json.dumps(report, separators=(",", ":"))
            )
        processes.append(result)
        reports.append(report)
    for profile, result, report in zip(profiles, processes, reports, strict=True):
        assert report is not None, (result.returncode, result.stdout, result.stderr)
        assert result.returncode == 0, (
            report.get("fatal_error"),
            result.stdout,
            result.stderr,
        )
        _check_profile_report(report, profile)
    assert [case["input_sha256"] for case in reports[0]["cases"]] == [
        case["input_sha256"] for case in reports[1]["cases"]
    ]


def test_default_extrema_boundary_diagnostic_rejects_other_corruption() -> None:
    native = np.asarray([0.0, -0.0, 1.0, np.inf, np.nan], dtype=np.float32)
    words = np.asarray([1, 0x80000001, 0x3F800000, 0x7F800000, 0x7FC00000], np.uint32)
    target = words.view(np.float32)
    with pytest.raises(AssertionError):
        _assert_output(target, native, role="known boundary remains strict failure")
    assert _known_default_boundary(target, native) is not None
    for index, word in (
        (0, 0x80000001),
        (0, 0x00800000),
        (0, 0x80000000),
        (2, 0x40000000),
        (3, 0xFF800000),
        (4, 0),
        (0, 0x7FC00000),
    ):
        corrupted = words.copy()
        corrupted[index] = word
        assert _known_default_boundary(corrupted.view(np.float32), native) is None
    assert _known_default_boundary(target.reshape((1, 5)), native) is None
    assert _known_default_boundary(target.astype(np.float64), native) is None
    assert _known_default_boundary(native, native) is None
    alternate_nan = words.copy()
    alternate_nan[4] = 0x7FC00001
    assert _known_default_boundary(alternate_nan.view(np.float32), native) is not None


def test_scientific_maximum_original_reciprocal_composition_rejected(
    record_property: Any,
) -> None:
    """Rejection contains an open composition; it proves no numerical acceptance."""
    evidence = {
        **_UNIMPLEMENTED_COMPOSITION,
        "native_oracle_comparisons": 0,
        "native_agreement_comparisons": 0,
    }
    try:
        with jax.enable_x64(True):
            model, feeds, expected = _scientific_maximum_case()
            assert len(feeds) == len(expected) == 5
            for index, (feed, wanted) in enumerate(zip(feeds, expected, strict=True)):
                references = []
                for role, result in (
                    ("eager", model(*feed)),
                    ("jit", jax.jit(lambda x, empty: model(x, empty))(*feed)),
                ):
                    assert jax.tree.structure(result) == jax.tree.structure(wanted)
                    arrays = [
                        np.asarray(value).copy() for value in jax.tree.leaves(result)
                    ]
                    nominal = jax.tree.leaves(wanted)
                    assert len(arrays) == len(nominal) == 6
                    for actual, reference in zip(arrays, nominal, strict=True):
                        _assert_output(
                            actual,
                            reference,
                            role=f"unimplemented composition {role} feed={index}",
                            nominal_oracle=True,
                        )
                        evidence["native_oracle_comparisons"] += 1
                    _record_subnormal_conformance(
                        record_property, arrays, nominal, role=role, feed=index
                    )
                    references.append(arrays)
                for eager, compiled in zip(*references, strict=True):
                    _assert_output(
                        eager,
                        compiled,
                        role=f"unimplemented composition native agreement feed={index}",
                    )
                    evidence["native_agreement_comparisons"] += 1

            with pytest.raises(SourcePrecisionError) as rejected:
                ui._to_onnx(
                    model,
                    [
                        jax.ShapeDtypeStruct(value.shape, value.dtype)
                        for value in feeds[0]
                    ],
                    opset=21,
                    return_mode="proto",
                    precision_policy=PrecisionPolicy.SOURCE,
                )
            error = rejected.value
            assert error.phase == "profile-validation"
            assert error.category == "profile-exclusion"
            assert error.primitive == "integer_pow" and error.opset == 21
            assert set(error.requested_dtypes) == {"float64"}
            assert error.detail == (
                "dtype is outside the original opset18/21 typed expression"
            )
            evidence["source_error"] = str(error)
            assert evidence["native_oracle_comparisons"] == 60
            assert evidence["native_agreement_comparisons"] == 30
    finally:
        record_property(
            "original_maximum_reciprocal_unimplemented", json.dumps(evidence)
        )


def test_source_bfloat16_extrema_artifact_without_cpu_acceptance_claim() -> None:
    """BF16 schema coverage does not assert a CPU kernel exists."""
    with jax.enable_x64(False):
        dtype = jnp.bfloat16
        proto = ui._to_onnx(
            lambda x: (jax.lax.reduce_min(x, (1,)), jax.lax.reduce_max(x, (1,))),
            [jax.ShapeDtypeStruct((2, 3), dtype)],
            opset=23,
            return_mode="proto",
            precision_policy=PrecisionPolicy.SOURCE,
        )
    onnx.checker.check_model(proto, full_check=True)
    assert all(
        value.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
        for value in (*proto.graph.input, *proto.graph.output)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("frontend", ["lax", "numpy"])
def test_source_extrema_original_nan_signed_zero_counterexample(
    dtype: Any, frontend: str
) -> None:
    """Accept the narrow counterexample; full subnormal feeds remain separate."""
    values = np.asarray(
        [
            [0.0, -0.0],
            [-0.0, 0.0],
            [-0.0, -0.0],
            [0.0, 0.0],
            [1.0, np.nan],
            [np.nan, 1.0],
            [-np.inf, np.inf],
        ],
        dtype=dtype,
    )
    expected = tuple(
        np.asarray(row, dtype=dtype)
        for row in (
            [-0.0, -0.0, -0.0, 0.0, np.nan, np.nan, -np.inf],
            [0.0, 0.0, -0.0, 0.0, np.nan, np.nan, np.inf],
            [-np.inf, -np.inf, -np.inf, np.inf, np.nan, np.nan, -0.0],
            [np.inf, np.inf, -np.inf, np.inf, np.nan, np.nan, 0.0],
        )
    )

    def model(x: Any) -> Any:
        if frontend == "numpy":
            minimum, maximum = jnp.min(x, axis=1), jnp.max(x, axis=1)
        else:
            minimum = jax.lax.reduce_min(x, (1,))
            maximum = jax.lax.reduce_max(x, (1,))
        return minimum, maximum, jnp.reciprocal(minimum), jnp.reciprocal(maximum)

    with jax.enable_x64(dtype is np.float64):
        references = []
        for role, result in (
            ("eager", model(values)),
            ("jit", jax.jit(lambda x: model(x))(values)),
        ):
            assert jax.tree.structure(result) == jax.tree.structure(expected)
            arrays = tuple(np.asarray(value).copy() for value in result)
            for actual, reference in zip(arrays, expected, strict=True):
                _assert_output(actual, reference, role=f"counterexample native {role}")
            references.append(arrays)
        for eager, compiled in zip(*references, strict=True):
            _assert_output(eager, compiled, role="counterexample native agreement")
        proto = ui._to_onnx(
            model,
            [jax.ShapeDtypeStruct((7, 2), dtype)],
            opset=23,
            return_mode="proto",
            precision_policy=PrecisionPolicy.SOURCE,
        )
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    for mode in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = mode
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        results = session.run(None, {session.get_inputs()[0].name: values})
        for role, native in zip(("eager", "jit"), references, strict=True):
            assert len(results) == len(native)
            for actual, reference in zip(results, native, strict=True):
                _assert_output(
                    actual,
                    reference,
                    role=f"counterexample ORT {mode.name} native={role}",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        choices=("default-diagnostic", "default-strict", "ftz-strict"),
    )
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    raise SystemExit(_run_profile(arguments.profile, arguments.report))
