# tests/extra_tests/converter/test_source_deployment_target.py

"""Explicit reviewed deployment requirements; runtime and numerical gates stay separate."""

from dataclasses import asdict, replace
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx.inliner
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_deployment import (
    DeploymentRequirements,
    MatrixPrecisionRequirement,
    RuntimeTargetObservation,
    deployment_requirements_from_json,
    deployment_requirements_to_json,
    matrix_precision_contract,
    ort_cpu_deployment_target,
    require_runtime_target,
    reviewed_fp32_contract,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins.plugin_system import FunctionPlugin
from jax2onnx.precision import (
    CheckStatus,
    attach_precision_report,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    _deployment_graph_boundary,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_precision_evidence import _identity, _report
from tests.extra_tests.converter.test_source_ordinary_matrix import (
    _SUBNORMAL_OUTPUTS,
    _SUBNORMAL_TARGETS,
    _restore_settings as _restore_settings,
)


_DIMENSIONS = (((1,), (0,)), ((), ()))


def _target() -> Any:
    return ort_cpu_deployment_target(
        ort.__version__, sys.implementation.cache_tag or ""
    )


@onnx_function
def _highest(left: Any, right: Any) -> Any:
    return jax.lax.dot_general(
        left,
        right,
        _DIMENSIONS,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=np.float32,
    )


@onnx_function
def _nested(left: Any, right: Any) -> Any:
    return _highest(left, right)


def _function(scope: str) -> Any:
    def function(left: Any, right: Any, predicate: Any) -> Any:
        if scope == "inline":
            return jax.lax.dot_general(
                left, right, _DIMENSIONS, precision=jax.lax.Precision.HIGHEST
            )
        if scope == "named":
            return _highest(left, right)
        if scope == "nested":
            return _nested(left, right)
        if scope == "jit":
            return jax.jit(_nested)(left, right)
        return jax.lax.cond(
            predicate,
            lambda a, b: _nested(a, b),
            lambda a, b: _nested(-a, b),
            left,
            right,
        )

    return function


def _spec(value: Any) -> Any:
    return jax.ShapeDtypeStruct(value.shape, value.dtype)


def _references(
    function: Any, values: tuple[Any, ...], directory: Path
) -> tuple[np.ndarray, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    args = tuple(jnp.asarray(value) for value in values)
    compiled = jax.jit(function)
    references = []
    events = []
    for label, call in (("fresh", compiled), ("warmed", compiled), ("eager", function)):
        try:
            returned = call(*args)
        except BaseException as exc:
            events.append(
                dict(
                    stage=label,
                    status="raised",
                    exception_type=type(exc).__qualname__,
                    message=str(exc),
                )
            )
            (directory / "native-events.json").write_text(json.dumps(events))
            raise
        events.append(dict(stage=label, status="returned"))
        (directory / "native-events.json").write_text(json.dumps(events))
        # Keep the actual return before validation or a later native call.
        np.savez(directory / f"native-{label}.npz", actual=jax.device_get(returned))
        actual = np.asarray(returned)
        assert actual.dtype == np.float32 and actual.ndim == 2
        references.append(actual.copy())
    return tuple(references)


@pytest.mark.parametrize("scope", ["inline", "named", "nested", "jit", "if"])
@pytest.mark.parametrize("predicate", [False, True])
def test_explicit_target_reaches_nested_scopes_and_actual_cpu(
    scope: str, predicate: bool, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        values = (
            np.asarray([[1 + 2**-11, 2**-12]], np.float32),
            np.asarray([[1], [2]], np.float32),
            np.asarray(predicate, np.bool_),
        )
        function = _function(scope)
        references = _references(function, values, tmp_path)
        original = jax.make_jaxpr(function)(*map(_spec, values))
        before = typed_trace_manifest(original)
        model = ui._to_onnx(
            function,
            list(map(_spec, values)),
            precision_policy=PrecisionPolicy.SOURCE,
            deployment_target=_target(),
            opset=23,
        )
        assert typed_trace_manifest(original) == before
        report = get_precision_report(model)
        assert report is not None and report.deployment_requirements is not None
        requirements = report.deployment_requirements
        assert requirements.target == _target() and requirements.matrix_precision
        assert json.loads(precision_report_to_json(report))[
            "deployment_requirements"
        ] == json.loads(deployment_requirements_to_json(requirements))
        assert all(
            site.source_scope and "dot_general" in site.source_equation
            for site in requirements.matrix_precision
        )
        assert (
            get_precision_report(onnx.load_model_from_string(model.SerializeToString()))
            == report
        )
        assert {item.category: item.status for item in report.checks}[
            "numerical-acceptance"
        ] is CheckStatus.UNVERIFIED
        for optimization in ("disabled", "all"):
            checked = validate_source_samples(
                model,
                samples=[
                    SourceSample(values, [reference], SampleCriteria("bitwise"))
                    for reference in references
                ],
                optimization=optimization,
            )
            (tmp_path / f"runtime-{optimization}.json").write_text(
                json.dumps(asdict(checked.target), indent=2)
            )
            checked.require_sample_acceptance()
            assert (
                checked.target is not None
                and checked.target.deployment_observation is not None
            )


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "runtime",
        "build",
        "cpu",
        "isa",
        "library",
        "provider",
        "fastmath",
        "denormals",
    ],
)
def test_unqualified_target_retains_original_highest_rejection(fault: str) -> None:
    target = _target()
    if fault == "none":
        target = None
    elif fault == "runtime":
        target = replace(target, runtime_version="unknown")
    elif fault == "build":
        target = replace(target, runtime_build="unreviewed")
    elif fault == "cpu":
        target = replace(target, cpu_model="unreviewed")
    elif fault == "isa":
        target = replace(target, cpu_features=target.cpu_features + ("avx512f",))
    elif fault == "library":
        target = replace(
            target,
            libraries=((target.libraries[0][0], "0" * 64), *target.libraries[1:]),
        )
    elif fault == "provider":
        target = replace(target, providers=("CUDAExecutionProvider",))
    else:
        key = (
            "mlas.enable_gemm_fastmath_arm64_bfloat16"
            if fault == "fastmath"
            else "session.set_denormal_as_zero"
        )
        target = replace(
            target,
            session_config=tuple(
                (name, "1" if name == key else value)
                for name, value in target.session_config
            ),
        )
    specs = [
        jax.ShapeDtypeStruct((1, 2), np.float32),
        jax.ShapeDtypeStruct((2, 1), np.float32),
    ]
    native = jax.make_jaxpr(_highest)(*specs)
    before = typed_trace_manifest(native)
    with pytest.raises(SourcePrecisionError, match="matrix precision/algorithm"):
        validate_source_jaxpr(native.jaxpr, 23, deployment_target=target)
    with pytest.raises(SourcePrecisionError, match="matrix precision/algorithm"):
        ui._to_onnx(
            _highest,
            specs,
            precision_policy=PrecisionPolicy.SOURCE,
            deployment_target=target,
        )
    assert typed_trace_manifest(native) == before


@pytest.mark.parametrize("precision", ["high", "mixed", "foreign", "algorithm"])
def test_caller_precision_or_algorithm_strings_cannot_grant_capability(
    precision: str,
) -> None:
    class ForeignPrecision(Enum):
        HIGHEST = 2

    request = {
        "high": (jax.lax.Precision.HIGH,) * 2,
        "mixed": (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST),
        "foreign": (ForeignPrecision.HIGHEST,) * 2,
        "algorithm": jax.lax.DotAlgorithmPreset.F32_F32_F32,
    }[precision]
    assert matrix_precision_contract(_target(), request, ("float32",) * 3, None) is None


def test_deployment_declaration_roundtrip_and_child_cache_ownership() -> None:
    target = _target()
    site = MatrixPrecisionRequirement(
        "graph",
        "dot_general",
        "output",
        ("HIGHEST",) * 2,
        ("float32",) * 3,
        None,
        reviewed_fp32_contract(target),
    )
    requirements = DeploymentRequirements(target, (site,))
    assert (
        deployment_requirements_from_json(deployment_requirements_to_json(requirements))
        == requirements
    )
    root = IRContext(
        opset=23,
        enable_double_precision=False,
        precision_policy=PrecisionPolicy.SOURCE,
        deployment_target=target,
    )
    for kind in ("function", "control_flow"):
        child = root.create_child_context(scope=kind, prefix="deployment")
        assert child.deployment_target is target
    original = FunctionPlugin._function_key
    observed: dict[Any, list[Any]] = {}

    def observe(self: Any, *args: Any, **kwargs: Any) -> Any:
        key = original(self, *args, **kwargs)
        observed.setdefault(kwargs["ctx"].deployment_target, []).append(key)
        return key

    other = ort_cpu_deployment_target(
        "1.24.1" if ort.__version__ != "1.24.1" else "1.31.0", "cpython-311"
    )
    specs = [
        jax.ShapeDtypeStruct((1, 2), np.float32),
        jax.ShapeDtypeStruct((2, 1), np.float32),
    ]
    with patch.object(FunctionPlugin, "_function_key", observe):
        for declaration in (target, other, target):
            model = ui._to_onnx(
                _nested,
                specs,
                precision_policy=PrecisionPolicy.SOURCE,
                deployment_target=declaration,
            )
            assert (
                get_precision_report(model).deployment_requirements.target
                == declaration
            )
    assert set(observed[target]).isdisjoint(observed[other])
    assert len(set(observed[target])) < len(observed[target])
    with pytest.raises(SourcePrecisionError):
        ui._to_onnx(_nested, specs, precision_policy=PrecisionPolicy.SOURCE)


@pytest.mark.parametrize(
    "fault",
    [
        "runtime",
        "provider",
        "options",
        "threads",
        "optimization",
        "boundary",
        "distribution",
    ],
)
def test_actual_runtime_mismatches_never_acquire_acceptance(fault: str) -> None:
    target = _target()
    required = DeploymentRequirements(target, ())
    actual = RuntimeTargetObservation(
        target,
        "ORT_ENABLE_ALL",
        "ORT_SEQUENTIAL",
        1,
        1,
        target.runtime_version,
        "a" * 64,
        "matmul-transpose-and-original-casts-v1",
        None,
    )
    if fault == "runtime":
        actual = replace(actual, target=replace(target, runtime_version="unknown"))
    elif fault == "provider":
        actual = replace(
            actual, target=replace(target, providers=("CUDAExecutionProvider",))
        )
    elif fault == "options":
        actual = replace(actual, target=replace(target, session_config=()))
    elif fault == "threads":
        actual = replace(actual, intra_op_threads=2)
    elif fault == "optimization":
        actual = replace(actual, optimization="unobserved")
    elif fault == "boundary":
        actual = replace(
            actual, boundary_contract=None, boundary_error="uninspected Gemm"
        )
    else:
        actual = replace(actual, distribution_version="unknown")
    with pytest.raises(ValueError, match="Actual runtime"):
        require_runtime_target(required, actual)


@pytest.mark.parametrize("fault", ["gemm", "fused-alpha", "cast"])
def test_actual_optimized_boundary_changes_stay_unfinished(fault: str) -> None:
    tensor = onnx.TensorProto.FLOAT

    def value(name: str) -> Any:
        return onnx.helper.make_tensor_value_info(name, tensor, [1, 1])

    original = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "MatMul", ["left", "right"], ["out"], name="jax2onnx_src_000000"
                )
            ],
            "boundary",
            [value("left"), value("right")],
            [value("out")],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 23)],
    )
    optimized = onnx.load_model_from_string(original.SerializeToString())
    node = optimized.graph.node[0]
    if fault == "gemm":
        node.op_type = "Gemm"
    elif fault == "fused-alpha":
        node.domain = "com.microsoft"
        node.op_type = "FusedMatMul"
        node.attribute.extend([onnx.helper.make_attribute("alpha", 0.5)])
        optimized.opset_import.extend([onnx.helper.make_opsetid("com.microsoft", 1)])
    else:
        original.graph.node.append(
            onnx.helper.make_node(
                "Cast",
                ["out"],
                ["cast"],
                to=onnx.TensorProto.DOUBLE,
                name="jax2onnx_src_000001",
            )
        )
        original.graph.output[0].CopyFrom(
            onnx.helper.make_tensor_value_info("cast", onnx.TensorProto.DOUBLE, [1, 1])
        )
    required = DeploymentRequirements(_target(), ())
    with pytest.raises(RuntimeError, match="support remains unfinished"):
        _deployment_graph_boundary(original, optimized, required)


@pytest.mark.parametrize(
    "probe", ["mantissa", "cancellation", "finite_range", "subnormal"]
)
def test_original_sensitive_matrix_criteria_with_highest_target(
    probe: str, tmp_path: Path
) -> None:
    info = np.finfo(np.float32)
    left, right = {
        "mantissa": ([1 + 2**-11, 0], [1, 1]),
        "cancellation": ([1024, -1024, 0.25], [1, 1, 1]),
        "finite_range": ([info.max / 4, info.max / 4], [1, 1]),
        "subnormal": ([info.smallest_subnormal, info.smallest_subnormal], [1, 1]),
    }[probe]
    values = (
        np.asarray([left], np.float32),
        np.asarray(right, np.float32).reshape(-1, 1),
    )
    references = _references(_highest, values, tmp_path)
    model = ui._to_onnx(
        _highest,
        list(map(_spec, values)),
        precision_policy=PrecisionPolicy.SOURCE,
        deployment_target=_target(),
    )
    for optimization in ("disabled", "all"):
        report = validate_source_samples(
            model,
            samples=[
                SourceSample(values, [reference], SampleCriteria("bitwise"))
                for reference in references
            ],
            optimization=optimization,
        )
        (tmp_path / f"sensitive-{optimization}.json").write_text(
            json.dumps(
                dict(
                    target=asdict(report.target),
                    samples=[asdict(value) for value in report.samples],
                ),
                indent=2,
            )
        )
        if report.status is CheckStatus.FAILED:
            assert (
                probe == "subnormal"
                and (jax.__version__, ort.__version__) in _SUBNORMAL_TARGETS
            )
            assert all(reference.tobytes() == bytes(4) for reference in references)
            assert all(
                sample.outputs[0].sha256 == _SUBNORMAL_OUTPUTS["float32"]
                for sample in report.samples
            )
            with pytest.raises(SampleAcceptanceError, match="bitwise"):
                report.require_sample_acceptance()
        else:
            report.require_sample_acceptance()


def test_historical_report_reads_without_target_or_new_default_claim() -> None:
    report = _report()
    payload = json.loads(precision_report_to_json(report))
    assert "deployment_requirements" not in payload
    model = _identity()
    attach_precision_report(model, report)
    recovered = get_precision_report(model)
    assert recovered == report and recovered.deployment_requirements is None


@pytest.mark.parametrize(
    "fault", ["target-string", "unknown-kernel-claim", "site-contract"]
)
def test_malformed_or_caller_claimed_contract_metadata_is_rejected(fault: str) -> None:
    target = _target()
    site = MatrixPrecisionRequirement(
        "graph",
        "dot_general",
        "out",
        ("HIGHEST",) * 2,
        ("float32",) * 3,
        None,
        reviewed_fp32_contract(target),
    )
    data = json.loads(
        deployment_requirements_to_json(DeploymentRequirements(target, (site,)))
    )
    if fault == "target-string":
        data["target"]["cpu_features"] = "avx2"
    elif fault == "unknown-kernel-claim":
        data["target"]["kernel_contract"] = "caller-says-highest"
    else:
        data["matrix_precision"][0]["kernel_contract"] = "caller-says-highest"
    with pytest.raises(ValueError):
        deployment_requirements_from_json(json.dumps(data))


@jax.custom_jvp
def _custom_highest(left: Any, right: Any) -> Any:
    return _highest(left, right)


@_custom_highest.defjvp
def _custom_highest_jvp(primals: Any, tangents: Any) -> Any:
    left, right = primals
    dleft, dright = tangents
    return _highest(left, right), _highest(dleft, right) + _highest(left, dright)


@pytest.mark.parametrize("scope", ["scan", "custom-jvp", "remat"])
def test_target_reaches_external_recursive_native_owners(scope: str) -> None:
    def function(left: Any, right: Any) -> Any:
        if scope == "custom-jvp":
            return _custom_highest(left, right)
        if scope == "remat":
            return jax.checkpoint(_highest)(left, right)

        def body(carry: Any, _: Any) -> Any:
            return carry + _highest(left, right), None

        return jax.lax.scan(body, jnp.zeros((1, 1), np.float32), None, length=2)[0]

    specs = (
        jax.ShapeDtypeStruct((1, 2), np.float32),
        jax.ShapeDtypeStruct((2, 1), np.float32),
    )
    original = jax.make_jaxpr(function)(*specs)
    before = typed_trace_manifest(original)
    validate_source_jaxpr(original.jaxpr, 23, deployment_target=_target())
    with pytest.raises(SourcePrecisionError, match="matrix precision/algorithm"):
        validate_source_jaxpr(original.jaxpr, 23)
    assert typed_trace_manifest(original) == before


def test_parallel_cast_rewiring_with_same_counts_and_op_names_is_rejected() -> None:
    def value(name: str, dtype: int) -> Any:
        return onnx.helper.make_tensor_value_info(name, dtype, [1, 1])

    nodes = [
        onnx.helper.make_node(
            "Cast",
            ["x1"],
            ["c1"],
            to=onnx.TensorProto.FLOAT,
            name="jax2onnx_src_000000",
        ),
        onnx.helper.make_node(
            "Cast",
            ["x2"],
            ["c2"],
            to=onnx.TensorProto.FLOAT,
            name="jax2onnx_src_000001",
        ),
        onnx.helper.make_node(
            "Add", ["c1", "weight"], ["a1"], name="jax2onnx_src_000002"
        ),
        onnx.helper.make_node(
            "Add", ["c2", "weight"], ["a2"], name="jax2onnx_src_000003"
        ),
        onnx.helper.make_node(
            "MatMul", ["a1", "weight"], ["y1"], name="jax2onnx_src_000004"
        ),
        onnx.helper.make_node(
            "MatMul", ["a2", "weight"], ["y2"], name="jax2onnx_src_000005"
        ),
    ]
    original = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            "parallel",
            [
                value("x1", onnx.TensorProto.DOUBLE),
                value("x2", onnx.TensorProto.DOUBLE),
                value("weight", onnx.TensorProto.FLOAT),
            ],
            [value("y1", onnx.TensorProto.FLOAT), value("y2", onnx.TensorProto.FLOAT)],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 23)],
    )
    optimized = onnx.load_model_from_string(original.SerializeToString())
    optimized.graph.node[2].input[0] = "c2"
    optimized.graph.node[3].input[0] = "c1"
    assert [node.op_type for node in original.graph.node] == [
        node.op_type for node in optimized.graph.node
    ]
    required = DeploymentRequirements(
        _target(),
        tuple(
            MatrixPrecisionRequirement(
                "graph",
                "dot_general",
                output,
                ("HIGHEST",) * 2,
                ("float32",) * 3,
                None,
                reviewed_fp32_contract(_target()),
                matrix_output=output,
                matrix_scope="graph:parallel",
                matrix_tag=f"jax2onnx_src_{index:06d}",
            )
            for index, output in ((4, "y1"), (5, "y2"))
        ),
    )
    with pytest.raises(RuntimeError, match="bound dataflow path"):
        _deployment_graph_boundary(original, optimized, required)


@pytest.mark.parametrize("mode", ["proto", "ir", "file"])
def test_named_output_and_export_forms_retain_actual_matrix_binding(
    mode: str, tmp_path: Path
) -> None:
    values = (np.asarray([[1, 2]], np.float32), np.asarray([[3], [4]], np.float32))

    def plain(left: Any, right: Any) -> Any:
        return jax.lax.dot_general(
            left, right, _DIMENSIONS, precision=jax.lax.Precision.HIGHEST
        )

    references = _references(plain, values, tmp_path / "references")
    options: dict[str, Any] = dict(
        return_mode=mode, input_names=["left", "right"], output_names=["answer"]
    )
    if mode == "file":
        options["output_path"] = tmp_path / "qualified.onnx"
    artifact = ui._to_onnx(
        plain,
        list(map(_spec, values)),
        precision_policy=PrecisionPolicy.SOURCE,
        deployment_target=_target(),
        **options,
    )
    model = (
        ir.to_proto(artifact)
        if mode == "ir"
        else onnx.load_model(artifact) if mode == "file" else artifact
    )
    report = get_precision_report(model)
    assert report is not None and report.deployment_requirements is not None
    assert model.graph.output[0].name == "answer"
    assert any(
        site.output == "answer"
        for site in report.deployment_requirements.matrix_precision
    )
    validated = validate_source_samples(
        model,
        samples=[
            SourceSample(values, [reference], SampleCriteria("bitwise"))
            for reference in references
        ],
        optimization="disabled",
    )
    validated.require_sample_acceptance()


def _import_fixture(policy: PrecisionPolicy) -> tuple[Any, Any, Any, Any]:
    from jax2onnx.converter.conversion_api import _attach_ir_functions

    def graph(name: str, domain: str, op: str, imports: dict[str, int]) -> Any:
        left = ir.val(name + "_input", dtype=ir.DataType.FLOAT, shape=(1,))
        output = ir.val(name + "_output", dtype=ir.DataType.FLOAT, shape=(1,))
        node = ir.Node(domain, op, [left], outputs=[output], name=name + "_node")
        return ir.Graph(
            [left], [output], nodes=[node], opset_imports=imports, name=name
        )

    leaf = ir.Function(
        "local.leaf",
        "Leaf",
        graph=graph("leaf", "", "Identity", {"": 23, "local.leaf": 23, "foreign": 7}),
        attributes={},
    )
    outer = ir.Function(
        "local.outer",
        "Outer",
        graph=graph(
            "outer",
            "local.leaf",
            "Leaf",
            {"": 23, "local.outer": 23, "local.leaf": 23, "foreign": 7},
        ),
        attributes={},
    )
    model = ir.Model(
        graph("main", "local.outer", "Outer", {"": 23, "local.leaf": 5, "foreign": 9}),
        ir_version=10,
    )
    context = IRContext(
        opset=23, enable_double_precision=False, precision_policy=policy
    )
    context.ir_functions.extend([outer, leaf])
    return _attach_ir_functions, model, context, (outer, leaf)


@pytest.mark.parametrize("policy", [PrecisionPolicy.SOURCE, PrecisionPolicy.LEGACY])
def test_function_attachment_uses_owned_model_versions_and_keeps_external_imports(
    policy: PrecisionPolicy,
) -> None:
    attach, model, context, functions = _import_fixture(policy)
    before = tuple(dict(function.graph.opset_imports) for function in functions)
    nodes = tuple(tuple(function.graph) for function in functions)
    attach(model, context)
    assert model.opset_imports == {
        "": 23,
        "local.leaf": 5,
        "local.outer": 1,
        "foreign": 9,
    }
    assert tuple(tuple(function.graph) for function in functions) == nodes
    assert all(
        model.functions[function.identifier()] is function for function in functions
    )
    for function in functions:
        assert function.graph.opset_imports[""] == 23
        assert function.graph.opset_imports["foreign"] == 7
    if policy is PrecisionPolicy.SOURCE:
        assert functions[0].graph.opset_imports["local.outer"] == 1
        assert functions[0].graph.opset_imports["local.leaf"] == 5
        assert functions[1].graph.opset_imports["local.leaf"] == 5
    else:
        assert (
            tuple(dict(function.graph.opset_imports) for function in functions)
            == before
        )


@pytest.mark.parametrize("location", ["root", "function"])
@pytest.mark.parametrize("fault", ["name", "overload"])
def test_source_local_domain_collision_fails_before_attachment(
    location: str, fault: str
) -> None:
    attach, model, context, functions = _import_fixture(PrecisionPolicy.SOURCE)
    node = list(model.graph if location == "root" else functions[0].graph)[0]
    node.domain = "local.leaf"
    node.op_type = "External" if fault == "name" else "Leaf"
    node.overload = "" if fault == "name" else "unresolved-overload"
    imports = tuple(dict(function.graph.opset_imports) for function in functions)
    before = dict(model.opset_imports)
    with pytest.raises(
        SourcePrecisionError, match="local Function domain.*unresolved external"
    ):
        attach(model, context)
    assert not model.functions
    assert dict(model.opset_imports) == before
    assert (
        tuple(dict(function.graph.opset_imports) for function in functions) == imports
    )


@onnx_function
def _cast_transpose_highest(left: Any, right: Any) -> Any:
    rounded = jax.lax.convert_element_type(left, np.float32)
    return jax.lax.dot_general(
        jax.lax.transpose(rounded, (1, 0)),
        right,
        _DIMENSIONS,
        precision=jax.lax.Precision.HIGHEST,
    )


def test_named_repeated_cast_matrix_calls_keep_exact_correspondence(
    tmp_path: Path,
) -> None:
    from jax2onnx.precision import (
        _configure_deployment_session,
        _observe_deployment_session,
    )

    def function(left1: Any, right1: Any, left2: Any, right2: Any) -> Any:
        return _cast_transpose_highest(left1, right1), _cast_transpose_highest(
            left2, right2
        )

    with jax.enable_x64(True):
        specs = [
            jax.ShapeDtypeStruct(shape, dtype)
            for shape, dtype in (((3, 2), np.float64), ((3, 4), np.float32))
        ] * 2
        model = ui._to_onnx(
            function,
            specs,
            precision_policy=PrecisionPolicy.SOURCE,
            deployment_target=_target(),
            opset=23,
        )
    report = get_precision_report(model)
    assert report is not None and report.deployment_requirements is not None
    flat = onnx.inliner.inline_local_functions(model)
    assert not flat.functions
    assert sum(node.op_type == "MatMul" for node in flat.graph.node) == 2
    assert sum(node.op_type == "Cast" for node in flat.graph.node) == 2
    assert (
        _deployment_graph_boundary(model, flat, report.deployment_requirements)
        == "matmul-transpose-and-original-casts-v1"
    )
    for label, level in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        path = tmp_path / f"optimized-{label}.onnx"
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _configure_deployment_session(options, report.deployment_requirements, path)
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        session.disable_fallback()
        observation = _observe_deployment_session(model, session, ort, path)
        (tmp_path / f"observation-{label}.json").write_text(
            json.dumps(asdict(observation), indent=2)
        )
        require_runtime_target(report.deployment_requirements, observation)
        # Structural/kernel correspondence only; no session.run or numerics here.


@onnx_function
def _cast_bias_highest(left: Any) -> Any:
    rounded = jax.lax.convert_element_type(left, np.float32)
    weights = np.asarray([[1.0, 2.0], [3.0, 4.0]], np.float32)
    bias = np.asarray([0.25, -0.5], np.float32)
    product = jax.lax.dot_general(
        rounded,
        weights,
        _DIMENSIONS,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jax.lax.convert_element_type(product + bias, np.float64)


def test_actual_repeated_named_gemm_keeps_original_casts_and_v2_observation(
    tmp_path: Path,
) -> None:
    from jax2onnx.precision import (
        _configure_deployment_session,
        _observe_deployment_session,
    )

    def function(first: Any, second: Any) -> Any:
        return _cast_bias_highest(first), _cast_bias_highest(second), first, second

    with jax.enable_x64(True):
        model = ui._to_onnx(
            function,
            [jax.ShapeDtypeStruct((2, 2), np.float64)] * 2,
            precision_policy=PrecisionPolicy.SOURCE,
            deployment_target=_target(),
            opset=23,
        )
    report = get_precision_report(model)
    assert report is not None and report.deployment_requirements is not None
    original = model.SerializeToString()
    for label, level in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        path = tmp_path / f"gemm-optimized-{label}.onnx"
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _configure_deployment_session(options, report.deployment_requirements, path)
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        session.disable_fallback()
        observation = _observe_deployment_session(model, session, ort, path)
        (tmp_path / f"gemm-observation-{label}.json").write_text(
            json.dumps(asdict(observation), indent=2)
        )
        require_runtime_target(report.deployment_requirements, observation)
        if label == "all":
            optimized = onnx.load_model(path)
            assert sum(node.op_type == "Gemm" for node in optimized.graph.node) == 2
            assert (
                observation.boundary_contract
                == "matmul-gemm-quickgelu-and-original-casts-v2"
            )
        else:
            assert (
                observation.boundary_contract
                == "matmul-transpose-and-original-casts-v1"
            )
    assert model.SerializeToString() == original


def test_fusion_metadata_presence_distinguishes_unknown_rank_and_scalar() -> None:
    from jax2onnx._deployment_fusion_bridge import _type

    unknown = onnx.ValueInfoProto(name="unknown")
    unknown.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    scalar = onnx.ValueInfoProto(name="scalar")
    scalar.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    scalar.type.tensor_type.shape.SetInParent()
    dynamic = onnx.helper.make_tensor_value_info(
        "dynamic", onnx.TensorProto.FLOAT, ["B", None, 0]
    )
    assert _type(unknown).shape is None
    assert _type(scalar).shape == ()
    assert _type(dynamic).shape == ("B", None, 0)
