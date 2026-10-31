# tests/extra_tests/converter/source_erfc_data/configured_cpu.py

"""Fresh-process CPU worker for the production Erfc regression, never imported.

The explicit denormal setting is bounded test configuration. It is not exporter
configuration, general provider acceptance, or evidence for FP16/BF16 kernels.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Any

HERE = Path(__file__).resolve().parent
OUTPUTS = (
    "direct_erfc",
    "t",
    "half_x",
    "gelu_erfc",
    "gelu",
    "consumer",
    "consumer_rounded",
)
EXACT = {"t", "half_x", "consumer", "consumer_rounded"}
ORDINARY_LABELS = (
    "common:1/2",
    "common:-1/2",
    "common:1",
    "common:-1",
    "common:2",
    "common:-2",
    "common:4",
    "common:-4",
    "common:6",
    "common:-6",
)
# Only these exact typed prefix word pairs can describe the reviewed negative
# fixture. They are usable only when independent eager/JIT actually contradict.
KNOWN_FLOAT_PREFIX = {
    "t": {
        72: ("80000001", "80000000"),
        73: ("00000001", "00000000"),
        74: ("805a827a", "80000000"),
        75: ("005a827a", "00000000"),
    },
    "half_x": {
        74: ("00400000", "00000000"),
        75: ("80400000", "80000000"),
    },
}
KNOWN_PREFIX_BY_DTYPE = {
    "float32": KNOWN_FLOAT_PREFIX,
    "float64": {
        "t": {
            74: ("8000000000000001", "8000000000000000"),
            75: ("0000000000000001", "0000000000000000"),
            76: ("800b504f333f9de6", "8000000000000000"),
            77: ("000b504f333f9de6", "0000000000000000"),
        },
        "half_x": {
            76: ("0008000000000000", "0000000000000000"),
            77: ("8008000000000000", "8000000000000000"),
        },
    },
}
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)
GRID_SHA256 = {
    "grid-float32.json": "4d5260f9c1dc5a4dc332d0ec50f5ae86b6c8d31d79685b1055511ebd9630b0e3",
    "grid-float64.json": "04e3d27e71ccd0228157b9ede1f0b428b7897c1363403286459518f8c316a8a6",
    "grid-float32.bin": "758160da2ba508755a43edbf875924ae12a0901184d742de4d0b845c1f457a44",
    "grid-float64.bin": "d1f0d48f426dc1be01b3765da7d5cc6e14110b87528e12fccff59b2936a0e3c4",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def identity() -> tuple[int, int, int]:
    assert threading.current_thread() is threading.main_thread()
    assert threading.active_count() == 1
    return os.getpid(), threading.get_ident(), threading.get_native_id()


def save_report(output: Path, report: Any) -> None:
    (output / "configured-cpu.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


def compare_outputs(raw: Any, references: Any, dtype: str, gate: Any) -> Any:
    results = {
        role: {
            name: gate.compare_bytes(
                raw[name], reference[name], dtype, exact=name in EXACT
            )
            for name in OUTPUTS
        }
        for role, reference in references.items()
    }
    return results


def status(passed: bool) -> str:
    return "PASSED" if passed else "FAILED"


def byte_differences(actual: bytes, reference: bytes, dtype: str) -> list[int]:
    width = 4 if dtype == "float32" else 8
    assert len(actual) == len(reference) and len(actual) % width == 0
    return [
        index
        for index in range(len(actual) // width)
        if actual[index * width : (index + 1) * width]
        != reference[index * width : (index + 1) * width]
    ]


def reference_diagnostics(
    refs: Any, dtype: str, gate: Any, *, named: bool = False
) -> Any:
    pairs = {
        "independent-eager-vs-jit": ("independent-eager", "independent-jit"),
        "independent-jit-vs-eager": ("independent-jit", "independent-eager"),
    }
    for role in (
        ("replayed-native", "actual-function-body")
        if named
        else ("activated", "replayed")
    ):
        for mode in ("eager", "jit"):
            pairs[f"{role}-{mode}"] = (f"{role}-{mode}", f"independent-{mode}")
    result = {}
    for label, (actual_role, reference_role) in pairs.items():
        actual = {"direct_erfc": refs[actual_role]} if named else refs[actual_role]
        reference = (
            {"direct_erfc": refs[reference_role]} if named else refs[reference_role]
        )
        checks = {
            name: gate.compare_bytes(
                actual[name], reference[name], dtype, exact=name in EXACT
            )
            for name in actual
        }
        result[label] = {
            "actual_role": actual_role,
            "reference_role": reference_role,
            "status": status(all(check["pass"] for check in checks.values())),
            "outputs": checks,
            "raw_difference_indices": {
                name: byte_differences(actual[name], reference[name], dtype)
                for name in actual
            },
        }
    return result


def failed_elements(checks: Any) -> list[tuple[str, Any]]:
    return [
        (name, element)
        for name, check in checks.items()
        for element in check["elements"]
        if not element["pass"]
    ]


def known_prefix_element(dtype: str, name: str, element: Any) -> bool:
    pair = KNOWN_PREFIX_BY_DTYPE.get(dtype, {}).get(name, {}).get(element["index"])
    return (
        dtype in KNOWN_PREFIX_BY_DTYPE
        and pair is not None
        and {element["target_word"], element["reference_word"]} == set(pair)
        and element["reasons"]
        == (
            ["exact_signed_zero"]
            if element["reference_class"] == "zero"
            else ["nonzero_reference_flushed_to_zero"]
        )
    )


def inspect_native_conflict(diagnostics: Any, dtype: str) -> tuple[bool, list[Any]]:
    """Classify the exact observed contradiction; never erase a failed gate."""
    expected = {
        (name, index)
        for name, indices in KNOWN_PREFIX_BY_DTYPE.get(dtype, {}).items()
        for index in indices
    }
    independent = failed_elements(diagnostics["independent-jit-vs-eager"]["outputs"])
    conflict = bool(independent)
    known = (
        dtype in KNOWN_PREFIX_BY_DTYPE
        and {(name, item["index"]) for name, item in independent} == expected
        and all(known_prefix_element(dtype, name, item) for name, item in independent)
    )
    unexpected = []
    for role, comparison in diagnostics.items():
        failures = failed_elements(comparison["outputs"])
        if not failures:
            continue
        permitted_role = role in {
            "independent-eager-vs-jit",
            "independent-jit-vs-eager",
            "replayed-eager",
        }
        if not (
            known
            and permitted_role
            and {(name, item["index"]) for name, item in failures} == expected
            and all(known_prefix_element(dtype, name, item) for name, item in failures)
        ):
            unexpected.append(
                {"scope": "native", "dtype": dtype, "role": role, "failures": failures}
            )
    return conflict and known, unexpected


def ordinary_composition(comparisons: Any, grid: Any) -> Any:
    indices = [
        next(item["index"] for item in grid["entries"] if label in item["labels"])
        for label in ORDINARY_LABELS
    ]
    assert len(set(indices)) == 10
    roles = {
        role: {
            name: {
                "elements": [checks[name]["elements"][index] for index in indices],
                "pass": all(
                    checks[name]["elements"][index]["pass"] for index in indices
                ),
            }
            for name in OUTPUTS[:5]
        }
        for role, checks in comparisons.items()
    }
    return {
        "labels": ORDINARY_LABELS,
        "indices": indices,
        "roles": roles,
        "pass": all(
            check["pass"] for checks in roles.values() for check in checks.values()
        ),
    }


def fixed_contract(raw: Any, dtype: str, grid: Any, gate: Any) -> Any:
    width = 4 if dtype == "float32" else 8

    def word(label: str) -> int:
        index = next(
            item["index"] for item in grid["entries"] if label in item["labels"]
        )
        return int.from_bytes(
            raw["direct_erfc"][width * index : width * (index + 1)], "little"
        )

    one = 0x3F800000 if dtype == "float32" else 0x3FF0000000000000
    two = 0x40000000 if dtype == "float32" else 0x4000000000000000
    exceptions = {
        label: {
            "word": word(label),
            "expected": expected,
            "pass": word(label) == expected,
        }
        for label, expected in (
            ("+0", one),
            ("-0", one),
            ("+infinity", 0),
            ("-infinity", two),
        )
    }
    exceptions["quiet_nan"] = {
        "word": word("quiet_nan"),
        "pass": gate.decode(word("quiet_nan"), dtype)[0] == "nan",
    }
    mantissa = []
    if dtype == "float64":
        for left, right in (
            ("common:1", "double_discriminator:1+2^-30"),
            ("+0", "double_discriminator:1*2^-40"),
            ("+0", "double_discriminator:-1*2^-40"),
        ):
            mantissa.append(
                {
                    "left": left,
                    "right": right,
                    "left_word": word(left),
                    "right_word": word(right),
                    "pass": word(left) != word(right),
                }
            )
    expected = {
        "consumer": "00000000" if dtype == "float32" else "000000000000f03f",
        "consumer_rounded": "0000804b" if dtype == "float32" else "0000001000007041",
    }
    consumer = {
        name: gate.compare_bytes(raw[name], bytes.fromhex(payload), dtype, exact=True)
        for name, payload in expected.items()
    }
    return {
        "exceptions": exceptions,
        "mantissa": mantissa,
        "consumer": consumer,
        "pass": all(
            item["pass"]
            for item in (*exceptions.values(), *mantissa, *consumer.values())
        ),
    }


def gather_references(
    tests: Any, specs: Any, feeds: Any, output: Path, report: Any, dtype: str, gate: Any
) -> Any:
    """Finish independent, activated and preserved-trace execution before ORT."""
    jax, np = tests.jax, tests.np
    fn = tests._seven_outputs
    native = tests._trace_roles(fn, specs)
    manifest = tests.typed_trace_manifest(native)
    constants = tuple(tests._payload(v) for v in native.consts)

    def run(role: str, runner: Any) -> dict[str, bytes]:
        input_payload = tuple(tests._payload(value) for value in feeds)
        report["native_attempts"] += 1
        save_report(output, report)
        actual = tuple(np.asarray(v).copy() for v in jax.tree.leaves(runner(*feeds)))
        report["native_completed"] += 1
        np.savez(
            output / f"fixed-{dtype}-{role}.npz",
            input=feeds[0],
            zero_input=feeds[1],
            **dict(zip(OUTPUTS, actual, strict=True)),
        )
        save_report(output, report)
        assert len(actual) == len(OUTPUTS)
        for name, value in zip(OUTPUTS, actual, strict=True):
            assert value.dtype == np.dtype(dtype)
            assert value.shape == (
                (1,) if name in {"consumer", "consumer_rounded"} else feeds[0].shape
            )
        assert tuple(tests._payload(value) for value in feeds) == input_payload
        return dict(zip(OUTPUTS, (v.tobytes() for v in actual), strict=True))

    refs = {
        "independent-eager": run("independent-eager", tests._fresh(fn)),
        "independent-jit": run("independent-jit", jax.jit(tests._fresh(fn))),
    }
    with (
        tests.precision_policy_scope(tests.PrecisionPolicy.SOURCE),
        tests._activate_plugin_worlds(),
    ):
        activated = jax.make_jaxpr(tests._fresh(fn))(*specs)
        assert tests.typed_trace_manifest(activated) == manifest
        refs["activated-eager"] = run("activated-eager", tests._fresh(fn))
        refs["activated-jit"] = run("activated-jit", jax.jit(tests._fresh(fn)))
    replayed = tests.jaxpr_as_fun(native)
    refs["replayed-eager"] = run("replayed-eager", tests._fresh(replayed))
    refs["replayed-jit"] = run("replayed-jit", jax.jit(tests._fresh(replayed)))
    diagnostics = reference_diagnostics(refs, dtype, gate)
    path = f"native-fixed-{dtype}-comparison.json"
    (output / path).write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n"
    )
    known, unexpected = inspect_native_conflict(diagnostics, dtype)
    report["native_diagnostics"][f"fixed-{dtype}"] = {
        "path": path,
        "statuses": {role: check["status"] for role, check in diagnostics.items()},
        "known_prefix_conflict": known,
    }
    report["unexpected_numerical_failures"].extend(unexpected)
    assert tests.typed_trace_manifest(native) == manifest
    assert tuple(tests._payload(v) for v in native.consts) == constants
    return refs


def assert_seven_output_structure(tests: Any, model: Any, dtype: str) -> None:
    """Consumer casts are outside each typed Erfc body and do actual rounding."""
    onnx = tests.onnx
    code = onnx.TensorProto.FLOAT if dtype == "float32" else onnx.TensorProto.DOUBLE
    payload = json.loads((HERE / "coefficients.json").read_text())[dtype]

    def constant(key: str) -> Any:
        return "constant", code, payload["constants"][key]["raw_le_hex"]

    x, zero = "x", "zero"
    t = ("Mul", ("Neg", x), constant("sqrt_half"))
    half_x = ("Mul", constant("half"), x)
    gelu_erfc = tests._expected_expression(t, dtype)
    at_zero = tests._expected_expression(zero, dtype)
    big = ("constant", onnx.TensorProto.DOUBLE, "0000000000007041")
    if dtype == "float32":
        rounded = (
            "Cast",
            onnx.TensorProto.FLOAT,
            ("Add", ("Cast", onnx.TensorProto.DOUBLE, at_zero), big),
        )
        consumer = (
            "Cast",
            onnx.TensorProto.FLOAT,
            ("Sub", ("Cast", onnx.TensorProto.DOUBLE, rounded), big),
        )
    else:
        rounded = ("Add", at_zero, big)
        consumer = ("Sub", rounded, big)
    expected = (
        tests._expected_expression(x, dtype),
        t,
        half_x,
        gelu_erfc,
        ("Mul", half_x, gelu_erfc),
        consumer,
        rounded,
    )
    inputs = {model.graph.input[0].name: x, model.graph.input[1].name: zero}
    assert len(model.graph.output) == len(expected)
    for value, expression in zip(model.graph.output, expected, strict=True):
        assert tests._expression(model.graph, value.name, inputs) == expression
    # Erfc bodies themselves contribute no Cast; only the F32 consumer does.
    assert sum(node.op_type == "Cast" for node in model.graph.node) == (
        4 if dtype == "float32" else 0
    )


def prepare(tests: Any, output: Path, report: Any, gate: Any) -> list[Any]:
    jax, np = tests.jax, tests.np
    prepared = []
    for name, digest in GRID_SHA256.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
    report["frozen_grids_sha256"] = GRID_SHA256
    for dtype, count in (("float32", 81), ("float64", 86)):
        grid = json.loads((HERE / f"grid-{dtype}.json").read_text())
        raw = (HERE / f"grid-{dtype}.bin").read_bytes()
        assert grid["shape"] == [count] and len(grid["entries"]) == count
        assert raw == b"".join(
            bytes.fromhex(item["raw_le_hex"]) for item in grid["entries"]
        )
        inputs = np.frombuffer(raw, dtype=dtype).copy()
        zero = np.frombuffer(bytes(4 if dtype == "float32" else 8), dtype=dtype).copy()
        specs = [
            jax.ShapeDtypeStruct(inputs.shape, dtype),
            jax.ShapeDtypeStruct(zero.shape, dtype),
        ]
        feeds = (inputs, zero)
        refs = gather_references(tests, specs, feeds, output, report, dtype, gate)
        report["export_attempts"] += 1
        save_report(output, report)
        model = tests._artifact(tests._seven_outputs, specs)
        report["exports_completed"] += 1
        payload = model.SerializeToString()
        (output / f"erfc-{dtype}.onnx").write_bytes(payload)
        save_report(output, report)
        assert_seven_output_structure(tests, model, dtype)
        prepared.append(("fixed", dtype, payload, [(feeds, refs, grid)]))
        # One named Function and one shared-B artifact cover both concrete B's.
        report["export_attempts"] += 1
        save_report(output, report)
        named, native, body = tests._named_case(np.dtype(dtype).type)
        report["exports_completed"] += 1
        (output / f"named-erfc-{dtype}.onnx").write_bytes(named.SerializeToString())
        save_report(output, report)
        manifest = tests.typed_trace_manifest(body)
        named_cases = []
        for batch in (2, 5):
            values = np.linspace(-2.5, 2.5, batch * 6, dtype=dtype).reshape(batch, 2, 3)
            reference = {}
            for role, runner in (
                ("independent", tests._named_erfc_entry),
                ("replayed-native", tests.jaxpr_as_fun(native)),
                ("actual-function-body", tests.jaxpr_as_fun(body)),
            ):
                for mode, callable_ in (
                    ("eager", tests._fresh(runner)),
                    ("jit", jax.jit(tests._fresh(runner))),
                ):
                    input_payload = tests._payload(values)
                    report["native_attempts"] += 1
                    save_report(output, report)
                    outputs = tuple(
                        np.asarray(v).copy() for v in jax.tree.leaves(callable_(values))
                    )
                    report["native_completed"] += 1
                    np.savez(
                        output / f"named-{dtype}-B{batch}-{role}-{mode}.npz",
                        input=values,
                        output=outputs[0],
                    )
                    save_report(output, report)
                    assert len(outputs) == 1 and outputs[0].dtype == np.dtype(dtype)
                    assert outputs[0].shape == (batch, 2, 3)
                    assert tests._payload(values) == input_payload
                    reference[f"{role}-{mode}"] = outputs[0].tobytes()
            diagnostics = reference_diagnostics(reference, dtype, gate, named=True)
            path = f"native-named-{dtype}-B{batch}-comparison.json"
            (output / path).write_text(
                json.dumps(diagnostics, indent=2, allow_nan=False) + "\n"
            )
            report["native_diagnostics"][f"named-{dtype}-B{batch}"] = {
                "path": path,
                "statuses": {
                    role: check["status"] for role, check in diagnostics.items()
                },
            }
            report["unexpected_numerical_failures"].extend(
                {
                    "scope": "native-named",
                    "dtype": dtype,
                    "batch": batch,
                    "role": role,
                    "failures": failed_elements(check["outputs"]),
                }
                for role, check in diagnostics.items()
                if check["status"] == "FAILED"
            )
            named_cases.append(((values,), reference, {"batch": batch}))
        assert tests.typed_trace_manifest(body) == manifest
        prepared.append(("named", dtype, named.SerializeToString(), named_cases))
    report["references_complete_before_first_session"] = True
    report["trace_roles"] = [
        "independent",
        "activated",
        "replayed-native",
        "actual-function-body",
    ]
    report["named_activated_whole_carrier_runtime"] = "NOT_ESTABLISHED"
    assert report["native_attempts"] == report["native_completed"] == 36
    assert report["export_attempts"] == report["exports_completed"] == 4
    save_report(output, report)
    return prepared


def execute(test_path: Path, output: Path, report: Any) -> None:
    assert sys.flags.isolated and not sys.flags.optimize and sys.byteorder == "little"
    assert all(os.environ.get(key) == "1" for key in THREAD_KEYS)
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=false"
    assert "onnxruntime" not in sys.modules
    # Bound native thread creation too, before any framework import.
    if hasattr(os, "sched_getaffinity"):
        available = set(os.sched_getaffinity(0))
        cpus = [4, 5, 6, 7] if {4, 5, 6, 7} <= available else sorted(available)[:4]
        os.sched_setaffinity(0, cpus)
        report["cpus"] = cpus
    origin = identity()
    checkout = test_path.parents[3]
    assert (checkout / "jax2onnx").is_dir()
    sys.path.insert(0, str(checkout))
    tests = load_module("source_erfc_child_tests", test_path)
    gate = load_module("source_erfc_exact_gate", HERE / "numeric_gate.py")
    assert (
        "onnxruntime" not in sys.modules
    ), "no ORT import before fresh-child preparation"
    with (
        tests.jax.enable_x64(True),
        tests.jax.default_matmul_precision(None),
        tests.jax.numpy_dtype_promotion("standard"),
        tests.jax.numpy_rank_promotion("allow"),
    ):
        prepared = prepare(tests, output, report, gate)
    assert identity() == origin
    import onnxruntime as ort

    report["ort_version"] = ort.__version__
    report["thread_environment"] = {key: os.environ[key] for key in THREAD_KEYS}
    report["session_config"] = {"session.set_denormal_as_zero": "1"}
    report["thread_identity"] = list(origin)
    stderr = os.dup(2)
    log_path = output / "ort-initialization.log"
    try:
        with log_path.open("wb", buffering=0) as log:
            os.dup2(log.fileno(), 2)
            for kind, dtype, payload, cases in prepared:
                for optimization in ("disabled", "all"):
                    assert identity() == origin
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = options.inter_op_num_threads = 1
                    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                        if optimization == "disabled"
                        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    options.add_session_config_entry(
                        "session.set_denormal_as_zero", "1"
                    )
                    assert (
                        options.get_session_config_entry("session.set_denormal_as_zero")
                        == "1"
                    )
                    options.log_severity_level = 0 if report["sessions"] == 0 else 3
                    options.log_verbosity_level = 1
                    report["session_attempts"] += 1
                    save_report(output, report)
                    session = ort.InferenceSession(
                        payload,
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                    report["sessions"] += 1
                    save_report(output, report)
                    assert session.get_providers() == ["CPUExecutionProvider"]
                    if report["sessions"] == 1:
                        assert (
                            "Flush-to-zero and denormal-as-zero are on"
                            in log_path.read_text(errors="replace")
                        )
                        report["first_session_ftz_daz_on_log"] = True
                    for feeds, references, grid in cases:
                        assert identity() == origin
                        original_inputs = tuple(
                            tests._payload(value) for value in feeds
                        )
                        report["run_attempts"] += 1
                        save_report(output, report)
                        actual = session.run(
                            None,
                            dict(
                                zip(
                                    (v.name for v in session.get_inputs()),
                                    feeds,
                                    strict=True,
                                )
                            ),
                        )
                        report["runs"] += 1
                        label = f"target-{kind}-{dtype}-{optimization}"
                        if kind == "named":
                            label += f"-B{grid['batch']}"
                        tests.np.savez(
                            output / f"{label}.npz",
                            **{f"input-{i}": value for i, value in enumerate(feeds)},
                            **{f"output-{i}": value for i, value in enumerate(actual)},
                        )
                        save_report(output, report)
                        if kind == "fixed":
                            assert len(actual) == 7
                            raw = {}
                            for name, value in zip(OUTPUTS, actual, strict=True):
                                assert value.dtype == tests.np.dtype(dtype)
                                assert list(value.shape) == (
                                    [1]
                                    if name in {"consumer", "consumer_rounded"}
                                    else grid["shape"]
                                )
                                raw[name] = value.tobytes()
                            comparison = {
                                "roles": compare_outputs(raw, references, dtype, gate),
                                "fixed_contract": fixed_contract(
                                    raw, dtype, grid, gate
                                ),
                            }
                            comparison["full_seven_output_pass"] = all(
                                check["pass"]
                                for checks in comparison["roles"].values()
                                for check in checks.values()
                            )
                            comparison["ordinary_composition"] = ordinary_composition(
                                comparison["roles"], grid
                            )
                            path = f"{dtype}-{optimization}-comparison.json"
                            (output / path).write_text(
                                json.dumps(comparison, indent=2, allow_nan=False) + "\n"
                            )
                            report["fixed_comparisons"].append(path)
                            report["fixed_results"].append(
                                {
                                    "dtype": dtype,
                                    "optimization": optimization,
                                    "full_seven_output_pass": comparison[
                                        "full_seven_output_pass"
                                    ],
                                    "outputs": {
                                        role: {
                                            name: status(check["pass"])
                                            for name, check in checks.items()
                                        }
                                        for role, checks in comparison["roles"].items()
                                    },
                                    "direct_erfc": all(
                                        checks["direct_erfc"]["pass"]
                                        for checks in comparison["roles"].values()
                                    )
                                    and all(
                                        item["pass"]
                                        for item in (
                                            *comparison["fixed_contract"][
                                                "exceptions"
                                            ].values(),
                                            *comparison["fixed_contract"]["mantissa"],
                                        )
                                    ),
                                    "consumer": all(
                                        checks[name]["pass"]
                                        for checks in comparison["roles"].values()
                                        for name in ("consumer", "consumer_rounded")
                                    )
                                    and all(
                                        item["pass"]
                                        for item in comparison["fixed_contract"][
                                            "consumer"
                                        ].values()
                                    ),
                                    "fixed_contract": comparison["fixed_contract"][
                                        "pass"
                                    ],
                                    "ordinary_composition": comparison[
                                        "ordinary_composition"
                                    ]["pass"],
                                }
                            )
                            known = report["native_diagnostics"][f"fixed-{dtype}"][
                                "known_prefix_conflict"
                            ]
                            for role, checks in comparison["roles"].items():
                                report["unexpected_numerical_failures"].extend(
                                    {
                                        "scope": "target-fixed",
                                        "dtype": dtype,
                                        "optimization": optimization,
                                        "role": role,
                                        "output": name,
                                        "element": element,
                                    }
                                    for name, element in failed_elements(checks)
                                    if not (
                                        known
                                        and known_prefix_element(dtype, name, element)
                                    )
                                )
                        else:
                            assert len(actual) == 1
                            assert actual[0].dtype == tests.np.dtype(dtype)
                            assert actual[0].shape == (grid["batch"], 2, 3)
                            comparisons = {
                                role: gate.compare_bytes(
                                    actual[0].tobytes(), reference, dtype
                                )
                                for role, reference in references.items()
                            }
                            path = f"{label}-comparison.json"
                            (output / path).write_text(
                                json.dumps(comparisons, indent=2, allow_nan=False)
                                + "\n"
                            )
                            report["named_comparisons"].append(path)
                            report["named_results"].append(
                                {
                                    "dtype": dtype,
                                    "optimization": optimization,
                                    "batch": grid["batch"],
                                    "pass": all(
                                        check["pass"] for check in comparisons.values()
                                    ),
                                }
                            )
                            report["unexpected_numerical_failures"].extend(
                                {
                                    "scope": "target-named",
                                    "dtype": dtype,
                                    "optimization": optimization,
                                    "batch": grid["batch"],
                                    "role": role,
                                    "elements": [
                                        item
                                        for item in check["elements"]
                                        if not item["pass"]
                                    ],
                                }
                                for role, check in comparisons.items()
                                if not check["pass"]
                            )
                        assert (
                            tuple(tests._payload(value) for value in feeds)
                            == original_inputs
                        )
                        report["schedule"].append(
                            [kind, dtype, optimization, grid.get("batch")]
                        )
                        save_report(output, report)
                    del session
    finally:
        os.dup2(stderr, 2)
        os.close(stderr)
    assert report["sessions"] == 8 and report["runs"] == 12
    assert report["session_attempts"] == 8 and report["run_attempts"] == 12
    assert report["native_attempts"] == report["native_completed"] == 36
    assert report["export_attempts"] == report["exports_completed"] == 4
    assert identity() == origin
    assert len(report["fixed_results"]) == len(report["fixed_comparisons"]) == 4
    assert len(report["named_results"]) == len(report["named_comparisons"]) == 8
    report["diagnostic_execution"] = "COMPLETE"
    report["native_reference_agreement"] = status(
        all(
            value == "PASSED"
            for check in report["native_diagnostics"].values()
            for value in check["statuses"].values()
        )
    )
    report["replayed_eager_preservation"] = {
        dtype: report["native_diagnostics"][f"fixed-{dtype}"]["statuses"][
            "replayed-eager"
        ]
        for dtype in ("float32", "float64")
    }
    report["known_native_prefix_conflict"] = any(
        check.get("known_prefix_conflict", False)
        for check in report["native_diagnostics"].values()
    )
    full_pass = all(
        check["full_seven_output_pass"] for check in report["fixed_results"]
    )
    report["full_seven_output_gate"] = status(full_pass)
    for field, key in (
        ("direct_erfc", "direct_erfc"),
        ("consumer", "consumer"),
        ("fixed_exception_mantissa_consumer", "fixed_contract"),
        ("ordinary_composition", "ordinary_composition"),
    ):
        report[field] = status(all(check[key] for check in report["fixed_results"]))
    report["named_shared_batch"] = status(
        all(check["pass"] for check in report["named_results"])
    )
    consistent = (full_pass and report["native_reference_agreement"] == "PASSED") or (
        not full_pass
        and report["known_native_prefix_conflict"]
        and report["native_reference_agreement"] == "FAILED"
    )
    positives = all(
        report[field] == "PASSED"
        for field in (
            "direct_erfc",
            "consumer",
            "fixed_exception_mantissa_consumer",
            "ordinary_composition",
            "named_shared_batch",
        )
    )
    report["diagnostic_regression"] = status(
        consistent and positives and not report["unexpected_numerical_failures"]
    )
    report["status"] = "DIAGNOSTIC_COMPLETE"
    save_report(output, report)
    assert (
        report["diagnostic_regression"] == "PASSED"
    ), "unexpected numerical diagnostic; bounded collection retained"


def main() -> int:
    test_path, output = (Path(value).resolve() for value in sys.argv[1:])
    report: dict[str, Any] = {
        "status": "STARTING",
        "sessions": 0,
        "runs": 0,
        "schedule": [],
        "session_attempts": 0,
        "run_attempts": 0,
        "native_attempts": 0,
        "native_completed": 0,
        "export_attempts": 0,
        "exports_completed": 0,
        "native_diagnostics": {},
        "fixed_comparisons": [],
        "fixed_results": [],
        "named_comparisons": [],
        "named_results": [],
        "unexpected_numerical_failures": [],
        "diagnostic_execution": "PARTIAL",
        "low_precision_numerical_acceptance": "UNVERIFIED",
        "default_cpu_fixed_grid": "FAILED_RECORDED_SEPARATE_ENVIRONMENT",
    }
    result = 1
    try:
        execute(test_path, output, report)
        result = 0
    except Exception:
        report.update(
            status=(
                "DIAGNOSTIC_COMPLETE_WITH_UNEXPECTED_FAILURE"
                if report["diagnostic_execution"] == "COMPLETE"
                else "HARD_ERROR_PARTIAL_DIAGNOSTIC"
            ),
            traceback=traceback.format_exc(),
        )
    finally:
        (output / "configured-cpu.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    if result:
        print(report.get("traceback", report["status"]), file=sys.stderr)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
