# internal_docs/precision/poc/fixed_source_preparation.py

"""Portable reproduction of the finite fixed-source preparation proof.

Run from the repository with ``python -m
internal_docs.precision.poc.fixed_source_preparation --output /outside/repo/run``.
The output directory must be new. All arrays, traces, ONNX models and reports
are written there. This is an internal PoC, not a public preparation API or a
generic transformer. It intentionally changes one selected cast before tracing;
the changed computation has its own native reference, not equivalence to FP32.

The accepted historical stage-1/stage-2 manifests remain separate and immutable.
Their failed indexed parameter assignment and SOURCE profile barriers are not
reclassified by this fresh execution epoch. HIGHEST is imposed by this PoC's
settings; it is not an intrinsic requirement of every NNX Linear.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import textwrap
import traceback
from typing import Any, Callable, Iterator, cast
import uuid

import jax
from jax._src import config as jax_config
from jax._src.lib import _jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax2onnx import user_interface
from jax2onnx._source_preparation import prepare_nnx_linear
from jax2onnx.converter import source_profile
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.precision import (
    CheckStatus,
    get_precision_report,
    precision_report_to_json,
)

ROOT = Path(__file__).resolve().parents[3]
EXACT_PATH = ROOT / "tests/extra_tests/converter/precision_ux_support.py"
MINIMUM_PATH = ROOT / "internal_docs/precision/precision_target_state.md"
SETTING_NAMES = (
    "jax_enable_x64",
    "jax_default_matmul_precision",
    "jax_platforms",
    "jax_numpy_dtype_promotion",
    "jax_numpy_rank_promotion",
)
INTENDED_SETTINGS = (True, "highest", "cpu", "standard", "allow")
UNSET = _jax.config.unset
EXACT_SOURCE = (
    "def exact_region(x: jax.Array) -> jax.Array:\n"
    "    y = x.astype(jnp.float32)\n"
    "    return (y + 1) - y\n"
)
MINIMUM_SOURCE = (
    "def centered_values(x):\n"
    "    compute_dtype = jnp.promote_types(x.dtype, jnp.float32)\n"
    "    y = x.astype(compute_dtype)\n"
    "    return y - jnp.mean(y, axis=-1, keepdims=True)\n"
)
HISTORY = {
    "native_manifest_sha256": "07cf51be62b3b25542dd016c2933ce6db9f765d95d036fd5ae637bd43afb803f",
    "target_manifest_sha256": "cd2dfab320e2203ed18fccf03b2cc9c33d75f389cfd21d3a5f01e1e8c131e03a",
    "failed_indexed_assignment": "historical failure: retained FP32 storage; not replayed",
    "repair": "full Variable.set_value replacement of actual captured weights",
    "target_counts": "24 exports, 8 sessions/runs across two stacks; two exact models only",
    "barriers": "8 per stack: 2 protected sums, 4 minimum means, 2 PoC HIGHEST dots",
    "scope": "no generic transformer, broader bitcast or all-model acceptance",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_texts() -> tuple[str, str]:
    """Bind unchanged existing source expressions, without importing their runner."""
    source = EXACT_PATH.read_text()
    main = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    fn = next(
        node
        for node in main.body
        if isinstance(node, ast.FunctionDef) and node.name == "exact_region"
    )
    exact = textwrap.dedent(
        "".join(source.splitlines(True)[fn.lineno - 1 : fn.end_lineno])
    )
    document = MINIMUM_PATH.read_text()
    start = document.index("def centered_values(x):\n")
    minimum = document[start : document.index("\n```", start)] + "\n"
    if (exact, minimum) != (EXACT_SOURCE, MINIMUM_SOURCE):
        raise ValueError("The finite proof's original source preimage changed")
    return exact, minimum


def compile_source(source: str) -> Callable[[jax.Array], Any]:
    namespace: dict[str, Any] = {"jax": jax, "jnp": jnp}
    exec(compile(source, "<fixed-source-preparation-poc>", "exec"), namespace)
    function = cast(ast.FunctionDef, ast.parse(source).body[0])
    return cast(Callable[[jax.Array], Any], namespace[function.name])


def prepare_exact(
    source: str, *, replace_selected_cast: bool = False
) -> tuple[Callable[[jax.Array], Any], str]:
    """Only the explicitly selected immutable exact-region cast is supported."""
    if not replace_selected_cast:
        raise ValueError("No explicit authorization to replace the exact FP32 cast")
    if source != EXACT_SOURCE:
        raise ValueError("Unsupported source, scope, barrier or algorithm change")
    tree = ast.parse(source)
    selected = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    assert (selected.lineno, selected.col_offset) == (2, 8)
    assert ast.unparse(selected) == "x.astype(jnp.float32)"
    before = ast.dump(tree)
    cast(ast.Attribute, selected.args[0]).attr = "float64"
    assert ast.dump(tree) == before.replace("attr='float32'", "attr='float64'", 1)
    prepared = ast.unparse(tree) + "\n"
    return compile_source(prepared), prepared


def snapshot() -> dict[str, Any]:
    """Include hidden globals/locals; effective values alone miss leaks."""
    states = [jax_config.config_states[name] for name in SETTING_NAMES]
    local = []
    for state in states:
        value = state.swap_local(UNSET)
        state.set_local(value)
        local.append(value)
    return {
        "global": tuple(state.get_global() for state in states),
        "local": tuple(local),
        "effective": tuple(state.value for state in states),
        "callables": (
            jnp.asarray,
            jnp.astype,
            jnp.sum,
            jnp.promote_types,
            nnx.Linear.__call__,
        ),
        "policy": active_precision_policy(),
    }


def settings_record(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ["<unset>" if item is UNSET else item for item in value[key]]
        for key in ("global", "local", "effective")
    } | {"policy": value["policy"].value}


@contextmanager
def preserved_settings() -> Iterator[None]:
    before = snapshot()
    try:
        with ExitStack() as stack:
            for name, value in zip(SETTING_NAMES, INTENDED_SETTINGS, strict=True):
                stack.enter_context(jax_config.config_states[name](value))
            try:
                yield
            finally:
                for name, value in zip(SETTING_NAMES, before["global"], strict=True):
                    jax.config.update(name, value)
    finally:
        assert (
            snapshot() == before
        ), "Hidden/effective settings, callable or policy leaked"


def protected_consumers(region: Callable[[jax.Array], Any], x: jax.Array) -> Any:
    selected = region(x)
    cast_back = selected.astype(jnp.float32)
    fp32 = x.astype(jnp.float32)
    fp16 = jnp.asarray([1.0001, 1.0007], dtype=jnp.float64).astype(jnp.float16)
    bf16 = jnp.asarray([1.001, 1.005], dtype=jnp.float64).astype(jnp.bfloat16)
    required_sum32 = jnp.sum(fp32, dtype=jnp.float32)
    rounded_consumer = (fp32 + jnp.float32(1)) - fp32
    bits = jax.lax.bitcast_convert_type(fp32, jnp.uint32)
    return (
        selected,
        cast_back,
        fp32,
        rounded_consumer,
        bits,
        fp16,
        bf16,
        required_sum32,
    )


def linear_models() -> tuple[Any, Any]:
    """Ordinary exposed configuration, separate from the external cast adapter."""
    with preserved_settings():
        recipe = construct_and_call(
            nnx.Linear, 128, 64, dtype=with_requested_dtype(), rngs=with_rng_seed(0)
        )
        original = recipe.with_dtype(jnp.float32).instantiate()
        prepared = prepare_nnx_linear(
            original, compute_dtype=jnp.float64, parameter_dtype=jnp.float64
        )
    return original, prepared


def same_bits(a: Any, b: Any) -> bool:
    aa, bb = np.asarray(a), np.asarray(b)
    return (
        str(a.dtype) == str(b.dtype)
        and aa.shape == bb.shape
        and aa.tobytes() == bb.tobytes()
    )


@dataclass(frozen=True, eq=False)
class Case:
    function: Callable[[jax.Array], Any]
    feed: jax.Array
    bound: Any = None
    expected: Any = None

    def __call__(self, x: jax.Array) -> Any:
        with preserved_settings():
            return self.function(x)


def cases_and_weights() -> tuple[dict[str, Case], dict[str, Any]]:
    exact_source, minimum_source = source_texts()
    original = compile_source(exact_source)
    prepared, _ = prepare_exact(exact_source, replace_selected_cast=True)
    minimum = compile_source(minimum_source)
    result = {}
    weights = {}
    with preserved_settings():
        feed = jnp.asarray([1e8, 1e8 + 1, -1e8, 0, 16777216], dtype=jnp.float64)
        for label, fn, dtype in (
            ("exact_original", original, jnp.float32),
            ("exact_prepared", prepared, jnp.float64),
        ):
            result[label] = Case(
                fn,
                feed,
                expected=(
                    np.array([0, 0, 0, 1, 0], np.float32)
                    if label == "exact_original"
                    else np.ones(5, np.float64)
                ),
            )

            def taps(x: jax.Array, fn: Any = fn, dtype: Any = dtype) -> Any:
                y = x.astype(dtype)
                return y, y + 1, fn(x)

            def protected(x: jax.Array, fn: Any = fn) -> Any:
                return protected_consumers(fn, x)

            result[label + "_intermediates"] = Case(taps, feed)
            result[label + "_protected"] = Case(protected, feed)
        for dtype, values in (
            (jnp.float16, [2048, 2050, 2052, 2054]),
            (jnp.bfloat16, [256, 258, 260, 262]),
            (jnp.float32, [1e8, 1e8 + 1, 1e8 + 2, 1e8 + 3]),
            (jnp.float64, [1e8, 1e8 + 1, 1e8 + 2, 1e8 + 3]),
        ):
            result["minimum_" + jnp.dtype(dtype).name] = Case(
                minimum,
                jnp.asarray([values], dtype=dtype),
                expected=(
                    np.array([[-1.5, -0.5, 0.5, 1.5]], np.float64)
                    if dtype == jnp.float64
                    else (
                        np.zeros((1, 4), np.float32)
                        if dtype == jnp.float32
                        else np.array([[-3, -1, 1, 3]], np.float32)
                    )
                ),
            )
        linear_feed = jnp.asarray(
            np.arange(256).reshape(2, 128) / 256 - 0.5, dtype=jnp.float32
        )
        for label, model in zip(("original", "prepared"), linear_models(), strict=True):
            weights[label] = {
                key: np.array(getattr(model, key)[...]) for key in ("kernel", "bias")
            }
            graphdef, state = nnx.split(model)

            def apply(
                x: jax.Array, graphdef: Any = graphdef, state: Any = state
            ) -> Any:
                return nnx.merge(graphdef, state)(x)

            dtype = np.float32 if label == "original" else np.float64
            eps = np.finfo(dtype).eps
            gamma = 128 * eps / (1 - 128 * eps)
            bound = 2 * gamma * (
                np.abs(np.asarray(linear_feed).astype(dtype)).astype(np.float64)
                @ np.abs(weights[label]["kernel"]).astype(np.float64)
            ) + 4 * eps * np.abs(weights[label]["bias"])
            result["linear_" + label] = Case(apply, linear_feed, bound)
    return result, weights


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def array_record(output: Path, name: str, value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    path = output / (name + ".npy")
    np.save(path, array, allow_pickle=False)
    loaded = np.load(path, allow_pickle=False)
    assert loaded.shape == array.shape and loaded.tobytes() == array.tobytes()
    # NPY stores BF16 as void2; retain its logical dtype independently.
    return {
        "path": path.name,
        "sha256": digest(path.read_bytes()),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "weak_type": bool(getattr(value, "weak_type", False)),
        "value_sha256": digest(array.tobytes()),
    }


def native_case(
    output: Path, name: str, case: Case
) -> tuple[dict[str, Any], list[Any]]:
    with preserved_settings():
        eager = jax.tree_util.tree_leaves(case(case.feed))
        compiled = jax.jit(case)
        jit = jax.tree_util.tree_leaves(compiled(case.feed))
        repeated = jax.tree_util.tree_leaves(case(case.feed))
        repeated_jit = jax.tree_util.tree_leaves(compiled(case.feed))
        traced = jax.make_jaxpr(case)(case.feed)
    trace_path = output / (name + ".jaxpr.txt")
    trace_path.write_text(str(traced) + "\n")
    row = {
        "native": "validated",
        "trace": {"status": "captured", "path": trace_path.name},
        "input": array_record(output, name + ".input", case.feed),
        "eager": [
            array_record(output, f"{name}.eager.{i}", v) for i, v in enumerate(eager)
        ],
        "jit": [array_record(output, f"{name}.jit.{i}", v) for i, v in enumerate(jit)],
        "criterion": "bitwise" if case.bound is None else "gamma128 pointwise bound",
        "repeated_eager_and_jit_bitwise": True,
    }
    if case.bound is not None:
        row["bound"] = array_record(output, name + ".bound", case.bound)
    # Preserve the actual observations even if a later acceptance check fails.
    for a, b, c, d in zip(eager, jit, repeated, repeated_jit, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert bool(a.weak_type) == bool(b.weak_type)
        assert same_bits(a, c) and same_bits(b, d)
        if case.bound is None:
            assert same_bits(a, b), name
        else:
            assert np.all(np.abs(np.asarray(a) - np.asarray(b)) <= case.bound), name
    if case.expected is not None:
        assert len(eager) == 1 and same_bits(eager[0], case.expected), name
    return row, eager


def error_record(exc: Exception) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    if isinstance(exc, source_profile.SourcePrecisionError):
        record["fields"] = {
            key: getattr(exc, key)
            for key in (
                "phase",
                "category",
                "location",
                "primitive",
                "precision_params",
                "requirement",
            )
        }
    return record


def export_case(
    output: Path, name: str, case: Case, references: list[Any]
) -> dict[str, Any]:
    import onnx

    assert user_interface.validate_source_jaxpr is source_profile.validate_source_jaxpr
    with preserved_settings():
        model = user_interface._to_onnx(
            case,
            [
                jax.ShapeDtypeStruct(
                    case.feed.shape, case.feed.dtype, weak_type=case.feed.weak_type
                )
            ],
            opset=23,
            model_name="fixed_source_preparation_" + name,
            input_names=["X"],
            output_names=[f"Y{i}" for i in range(len(references))],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert isinstance(model, onnx.ModelProto)
    precision = get_precision_report(model)
    assert precision is not None and precision.opset == 23
    assert (
        precision.caller_x64_enabled
        and precision.caller_default_matmul_precision == "highest"
    )
    assert precision.caller_numpy_dtype_promotion == "standard"
    assert precision.caller_numpy_rank_promotion == "allow"
    checks = {check.category: check.status for check in precision.checks}
    assert checks["representation"] is CheckStatus.VALIDATED
    assert checks["lowering"] is CheckStatus.VALIDATED
    assert checks["runtime-execution"] is CheckStatus.UNVERIFIED
    assert checks["numerical-acceptance"] is CheckStatus.UNVERIFIED
    for traced, reference in zip(precision.traced_outputs, references, strict=True):
        assert (traced.dtype, tuple(traced.shape), traced.weak_type) == (
            str(reference.dtype),
            reference.shape,
            bool(reference.weak_type),
        )
    onnx.checker.check_model(model, full_check=True)
    path = output / (name + ".onnx")
    path.write_bytes(model.SerializeToString())
    reloaded = onnx.load_model(path, load_external_data=False)
    onnx.checker.check_model(reloaded, full_check=True)
    assert reloaded.SerializeToString() == path.read_bytes()
    assert get_precision_report(reloaded) == precision
    report_path = output / (name + ".precision-report.json")
    report_path.write_text(precision_report_to_json(precision) + "\n")
    return {
        "export": "admitted",
        "artifact": {"path": path.name, "sha256": digest(path.read_bytes())},
        "precision_report": report_path.name,
        "ordinary_default_audited_validator": True,
        "schema_and_serialization_roundtrip": True,
    }


def target_case(
    output: Path,
    name: str,
    artifact: dict[str, str],
    feed: Any,
    references: list[Any],
    optimization: str,
    counts: dict[str, int],
) -> dict[str, Any]:
    import onnxruntime as ort  # type: ignore[import-untyped]

    path = output / artifact["path"]
    assert digest(path.read_bytes()) == artifact["sha256"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if optimization == "disabled"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    result: dict[str, Any] = {
        "artifact": artifact,
        "optimization": options.graph_optimization_level.name,
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "execution_mode": options.execution_mode.name,
        "runtime_version": ort.__version__,
        "runtime_load": "unverified",
        "runtime_execution": "unverified",
        "numerical_acceptance": None,
    }
    try:
        counts["sessions"] += 1
        session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        session.disable_fallback()
        result["providers"] = session.get_providers()
        result["provider_options"] = session.get_provider_options()
        assert result["providers"] == ["CPUExecutionProvider"]
        actual_options = session.get_session_options()
        result["actual_session_options"] = {
            "intra_op_threads": actual_options.intra_op_num_threads,
            "inter_op_threads": actual_options.inter_op_num_threads,
            "execution_mode": actual_options.execution_mode.name,
            "optimization": actual_options.graph_optimization_level.name,
        }
        result["runtime_load"] = "validated"
        counts["runs"] += 1
        values = session.run(None, {"X": np.asarray(feed)})
        result["runtime_execution"] = "validated"
        result["actual_outputs"] = [
            array_record(output, f"{name}.{optimization}.Y{i}", v)
            for i, v in enumerate(values)
        ]
        result["numerical_acceptance"] = (
            "validated"
            if len(values) == len(references)
            and all(same_bits(a, b) for a, b in zip(values, references, strict=True))
            else "failed"
        )
    except Exception as exc:
        result["error"] = error_record(exc)
        result["numerical_acceptance"] = "failed"
        failed_stage = (
            "runtime_load"
            if result["runtime_load"] != "validated"
            else "runtime_execution"
        )
        result[failed_stage] = "failed"
    assert digest(path.read_bytes()) == artifact["sha256"]
    write_json(output / f"{name}.{optimization}.target-report.json", result)
    return result


def run(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Generated output must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    import jax2onnx

    assert Path(jax2onnx.__file__).resolve().is_relative_to(ROOT)
    before = snapshot()
    source_paths = (EXACT_PATH, MINIMUM_PATH, Path(__file__).resolve())
    source_hashes = {
        str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in source_paths
    }
    cases, weights = cases_and_weights()
    contract: dict[str, Any] = {
        "epoch": str(uuid.uuid4()),
        "history": HISTORY,
        "sources": source_hashes,
        "original_exact": EXACT_SOURCE,
        "original_minimum": MINIMUM_SOURCE,
        "prepared_exact": prepare_exact(EXACT_SOURCE, replace_selected_cast=True)[1],
        "selected_edit": "only exact_region line 2 x.astype(float32) -> float64, before tracing",
        "settings": dict(zip(SETTING_NAMES, INTENDED_SETTINGS, strict=True)),
        "cases": {
            name: {
                "shape": list(case.feed.shape),
                "dtype": str(case.feed.dtype),
                "input_bytes_sha256": digest(np.asarray(case.feed).tobytes()),
                "native_criterion": (
                    "bitwise" if case.bound is None else "gamma128 pointwise bound"
                ),
                "known_expected_output": (
                    None
                    if case.expected is None
                    else {
                        "dtype": str(case.expected.dtype),
                        "shape": list(case.expected.shape),
                        "bytes_hex": case.expected.tobytes().hex(),
                    }
                ),
            }
            for name, case in cases.items()
        },
        "linear_bound": "2*gamma128*sum(abs(x*k)) + 4*eps*abs(bias); gamma128=128*eps/(1-128*eps)",
        "target_programs": ["exact_original", "exact_prepared"],
        "target_criterion": "exact dtype, shape and bytes against frozen eager/JIT references",
        "optimizations": ["disabled", "all"],
        "opset": 23,
        "all_other_target_numerical_acceptance": None,
    }
    # This contract is persisted before any target execution; later rows only
    # report observations. A new invocation creates a new epoch/output directory.
    write_json(output / "contract.json", contract)
    for label, parameters in weights.items():
        for key, value in parameters.items():
            array_record(output, f"weights.{label}.{key}", value)
            if label == "prepared":
                assert same_bits(value, weights["original"][key].astype(np.float64))
    report: dict[str, Any] = {
        "epoch": contract["epoch"],
        "python": platform.python_version(),
        "executable": sys.executable,
        "jax": jax.__version__,
        "numpy": np.__version__,
        "flax": __import__("flax").__version__,
        "jax2onnx_import": jax2onnx.__file__,
        "affinity": sorted(os.sched_getaffinity(0)),
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "before": settings_record(before),
        "contract_sha256": digest((output / "contract.json").read_bytes()),
        "counts": {"exports": 0, "sessions": 0, "runs": 0},
        "cases": {},
        "all_model_compatibility": False,
        "generic_transformer": False,
    }
    references = {}
    for name, case in cases.items():
        try:
            row, references[name] = native_case(output, name, case)
        except Exception as exc:
            row = {"native": "failed", "error": error_record(exc)}
        report["cases"][name] = row | {
            "export": "unverified",
            "runtime": [],
            "numerical_target_acceptance": None,
        }
        write_json(output / "report.json", report)
    # All native references and criteria are fixed before the first export/run.
    for name, case in cases.items():
        row = report["cases"][name]
        current = snapshot()
        if row["native"] == "validated":
            try:
                report["counts"]["exports"] += 1
                row.update(export_case(output, name, case, references[name]))
            except source_profile.SourcePrecisionError as exc:
                row.update(export="source-rejected", first_barrier=error_record(exc))
            except Exception as exc:
                row.update(export="failed", first_barrier=error_record(exc))
            if name in contract["target_programs"] and row["export"] == "admitted":
                row["runtime"] = [
                    target_case(
                        output,
                        name,
                        row["artifact"],
                        case.feed,
                        references[name],
                        opt,
                        report["counts"],
                    )
                    for opt in contract["optimizations"]
                ]
                row["numerical_target_acceptance"] = (
                    "validated"
                    if all(
                        r["numerical_acceptance"] == "validated" for r in row["runtime"]
                    )
                    else "failed"
                )
        assert snapshot() == current
        row["settings_callables_policy_restored"] = True
        write_json(output / "report.json", report)
    assert snapshot() == before
    assert source_hashes == {
        str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in source_paths
    }
    report["after"] = settings_record(snapshot())
    report["status"] = (
        "validated"
        if (
            all(
                r["native"] == "validated"
                and r["export"] in ("admitted", "source-rejected")
                for r in report["cases"].values()
            )
            and all(
                report["cases"][name]["numerical_target_acceptance"] == "validated"
                for name in contract["target_programs"]
            )
        )
        else "failed"
    )
    write_json(output / "report.json", report)
    write_json(
        output / "manifest.json",
        {
            p.name: digest(p.read_bytes())
            for p in sorted(output.iterdir())
            if p.is_file()
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args().output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "epoch": report["epoch"],
                "counts": report["counts"],
            }
        )
    )
    return 0 if report["status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
