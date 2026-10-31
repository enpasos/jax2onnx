# tests/extra_tests/converter/test_source_leading_batch_matrix_bias.py

"""Finite leading-batch artifacts and binding; target numerics stay unverified."""

from collections.abc import Iterator
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime
import pytest

from jax2onnx import onnx_function, to_onnx, user_interface as ui
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_representation import normalized_shape_dimensions
from jax2onnx.plugins.jax.lax.dot_general import DotGeneralPlugin
from jax2onnx.precision import (
    CheckStatus,
    PRECISION_METADATA_KEY,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)
from scripts import source_precision_probe as probe
from scripts.source_precision_inventory import raw_registrations
from tests.extra_tests.converter.source_nnx_linear_fixtures import (
    assert_original_linear,
    expected_original_linear_kernel,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_native_contract import _forged_aval

_CASES = (
    (
        "plugin::jax.numpy.matmul",
        6,
        "56ba2b5913b56b5778c9d42b0c8d5fccf12a21c5ed1e7059502ed2f3df6f4b63",
    ),
    (
        "plugin::jax.numpy.matmul",
        5,
        "620115ebfa6606a2399d8d58866de0f736ee1e7f3e1a756a05c5445bf9124acc",
    ),
    (
        "example::examples.gpt::GPT_broadcast_add",
        0,
        "9e20b6c83c9e519b7e51a85d92c3f7a89a9ec759212f661eeb21c70cfc1c489c",
    ),
    (
        "plugin::nnx.linear",
        0,
        "8c70628910c9d9bc30737e3bb959dafc3dea41c24eecfd7cc1853f076519a515",
    ),
    (
        "plugin::nnx.linear",
        3,
        "06c91c76e73a3daced2f70b65cb56d9e59797008354dababdcb3d4d2bebc77ef",
    ),
)
_KERNEL_SHA = "d443c758235aff0c56e9af2d5862ae9807f6ca8e96d9aaf5289450663e4d3ef9"
_BIAS_SHA = "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1"
_MODEL: Any = None


@onnx_function(name="LeadingBatchLinear")
def _named(x: Any) -> Any:
    return _MODEL(x)


@onnx_function(name="NestedLeadingBatchLinear")
def _nested(x: Any) -> Any:
    return _named(x)


def _globals() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


def _settings() -> tuple[Any, ...]:
    return tuple(getattr(jax.config, name) for name in _globals())


@pytest.fixture(autouse=True)
def _restore_and_deny_target_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    hidden, effective = _globals(), _settings()
    originals = jnp.matmul, jnp.reshape, jnp.add

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "Artifact/binding tests do not execute the separately released numerical packet"
        )

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    try:
        with (
            jax.enable_x64(False),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in hidden.items():
            jax.config.update(name, value)
        assert (_globals(), _settings()) == (hidden, effective)
        assert (jnp.matmul, jnp.reshape, jnp.add) == originals
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _body(graph: Any) -> Any:
    inner = getattr(graph, "jaxpr", graph)
    return inner if inner is not graph else graph


def _sha(value: Any) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _signature(variables: Any) -> Any:
    return tuple(
        (
            np.dtype(v.aval.dtype).name,
            normalized_shape_dimensions(v.aval.shape),
            bool(v.aval.weak_type),
        )
        for v in variables
    )


def _original(case: Any) -> tuple[Any, Any]:
    origin, ordinal, digest = case
    registrations, _ = raw_registrations()
    blob = probe.DEFAULT_BASELINE.read_bytes()
    plan = probe.build_probe_plan(
        json.loads(blob), baseline_sha256=hashlib.sha256(blob).hexdigest()
    )
    row = next(
        row for row in plan.rows if (row.origin, row.ordinal) == (origin, ordinal)
    )
    assert row.contract_sha256 == digest
    metadata = probe._resolve_case(row, {key: value for key, _, value in registrations})
    fn = metadata["callable"]
    if origin == "plugin::nnx.linear":
        assert fn._args == (128, 64) and fn._dtype is None and fn._instance is None
        fn = fn.instantiate()
        assert_original_linear(fn, metadata["callable"], ordinal)
        assert (_sha(fn.bias[...]) if fn.bias is not None else None) == (
            _BIAS_SHA if ordinal == 0 else None
        )
    return fn, metadata


def _specs(shapes: Any, dtype: Any = np.float32, weak: bool = False) -> list[Any]:
    (batch,) = export.symbolic_shape("B")
    return [
        jax.ShapeDtypeStruct(
            tuple(batch if dim == "B" else dim for dim in shape), dtype, weak_type=weak
        )
        for shape in shapes
    ]


def _feeds(shapes: Any, batch: int, dtype: Any = np.float32) -> tuple[np.ndarray, ...]:
    result = []
    for index, shape in enumerate(shapes):
        concrete = tuple(batch if dim == "B" else dim for dim in shape)
        result.append(
            (
                np.arange(np.prod(concrete)).reshape(concrete) / 32
                + (batch + index) / 16
            ).astype(dtype)
        )
    return tuple(result)


def _native(fn: Any, specs: Any, feeds: Any) -> Any:
    trace = jax.make_jaxpr(fn)(*specs)
    references = []
    for values in feeds:
        eager = tuple(np.asarray(v) for v in jax.tree.leaves(fn(*values)))
        compiled = tuple(np.asarray(v) for v in jax.tree.leaves(jax.jit(fn)(*values)))
        assert [(v.dtype, v.shape) for v in eager] == [
            (v.dtype, v.shape) for v in compiled
        ]
        references.append((eager, compiled))
    return trace, references


def _artifact(
    fn: Any, specs: Any, *, mode: str = "ir", path: Path | None = None
) -> Any:
    before = _settings(), _globals()
    options: dict[str, Any] = {"return_mode": mode}
    if path is not None:
        options["output_path"] = path
    try:
        artifact = ui._to_onnx(
            fn, specs, precision_policy=PrecisionPolicy.SOURCE, **options
        )
    finally:
        assert (_settings(), _globals()) == before
    proto = (
        ir.to_proto(artifact)
        if isinstance(artifact, ir.Model)
        else onnx.load(artifact) if isinstance(artifact, str) else artifact
    )
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-static-v1"
    assert {c.category: c.status for c in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    return artifact, proto, report


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("x64", [False, True])
def test_five_unchanged_raw_sources_keep_one_symbolic_artifact(
    case: Any, x64: bool, tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        fn, metadata = _original(case)
        shapes = metadata["input_shapes"]
        specs = _specs(shapes)
        feeds = [_feeds(shapes, batch) for batch in (2, 5)]
        native, references = _native(fn, specs, feeds)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = jax.make_jaxpr(fn)(*specs)
            for values, (eager, compiled) in zip(feeds, references, strict=True):
                for left, right in zip(
                    jax.tree.leaves(fn(*values)), eager, strict=True
                ):
                    assert np.asarray(left).tobytes() == right.tobytes()
                for left, right in zip(
                    jax.tree.leaves(jax.jit(fn)(*values)), compiled, strict=True
                ):
                    assert np.asarray(left).tobytes() == right.tobytes()
        assert typed_trace_manifest(actual) == typed_trace_manifest(native)
        artifact, proto, report = _artifact(fn, specs)
        assert isinstance(artifact, ir.Model) and not proto.functions
        assert tuple(
            (v.dtype, v.shape, v.weak_type) for v in report.traced_outputs
        ) == _signature(_body(native).outvars)
        names = [node.op_type for node in proto.graph.node]
        assert not {"Cast", "Expand", "Transpose", "Gemm", "ConstantOfShape"} & set(
            names
        )
        if case[0].startswith("example::"):
            assert names == ["Add"] and not proto.graph.initializer
        else:
            assert names.count("MatMul") == 1
            assert names.count("Add") == int(case[:2] == ("plugin::nnx.linear", 0))
        assert len(proto.graph.input) == len(shapes)
        for index, shape in enumerate(shapes):
            expected = tuple(shape)
            observed = report.inputs[index].exported.shape
            assert observed == expected
            if shape[0] != "B":
                assert feeds[0][index].tobytes() != feeds[1][index].tobytes()
                assert proto.graph.input[index].name in proto.graph.node[0].input
        assert report.exported_outputs[0].shape[0] == "B"
        if case[0] == "plugin::nnx.linear":
            arrays = [onnx.numpy_helper.to_array(t) for t in proto.graph.initializer]
            expected_kernel = expected_original_linear_kernel(
                metadata["callable"], case[1]
            )
            assert any(
                v.shape == (128, 64) and _sha(v) == expected_kernel for v in arrays
            )
            assert _sha(fn.kernel[...]) == expected_kernel
            if case[1] == 0:
                assert any(v.size == 64 and _sha(v) == _BIAS_SHA for v in arrays)
                assert _sha(fn.bias[...]) == _BIAS_SHA
            else:
                assert fn.bias is None and names == ["MatMul"] and len(arrays) == 1
        # Serialization retains B; the same bytes describe both native references.
        saved = tmp_path / "original-symbolic.onnx"
        onnx.save_model(proto, saved)
        reloaded = onnx.load(saved)
        assert reloaded.SerializeToString() == proto.SerializeToString()
        for batch, (eager, _) in zip((2, 5), references, strict=True):
            assert eager[0].shape[0] == batch
        assert get_precision_report(reloaded) == report


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("kind", ["same-B", "left-1", "right-1"])
def test_rank_two_three_equal_tail_add_in_both_operand_orders(
    rank: int, kind: str
) -> None:
    tail = (3,) if rank == 2 else (4, 5)
    shapes = [("B", *tail)] * 2
    if kind != "same-B":
        shapes[0 if kind == "left-1" else 1] = (1, *tail)
    specs = _specs(shapes)
    _, proto, report = _artifact(lambda x, y: jax.lax.add(x, y), specs)
    assert [node.op_type for node in proto.graph.node] == ["Add"]
    assert report.exported_outputs[0].shape == ("B", *tail)


@pytest.mark.parametrize("mode", ["ir", "proto", "file"])
def test_real_fp64_mantissa_and_static_source_casts_keep_symbolic_ports(
    mode: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        weight = jnp.asarray([[1, 0], [1, 1]], np.float32)

        def model(x: Any) -> Any:
            return (
                jax.lax.dot_general(
                    x, weight.astype(jnp.float64), (((1,), (0,)), ((), ()))
                ),
                weight,
            )

        specs = _specs([("B", 2)], np.float64)
        for batch in (2, 5):
            value = np.full((batch, 2), 2**24, np.float64)
            value[:, 1] = 1
            assert np.all(np.asarray(model(value)[0])[:, 0] == 2**24 + 1)
        _, proto, report = _artifact(
            model,
            specs,
            mode=mode,
            path=tmp_path / "wide.onnx" if mode == "file" else None,
        )
        assert report.exported_outputs[0].dtype == "float64"
        assert next(node for node in proto.graph.node if node.op_type == "MatMul")
        assert any(node.op_type == "Cast" for node in proto.graph.node)
        assert any(
            onnx.numpy_helper.to_array(t).dtype == np.float32
            and _sha(onnx.numpy_helper.to_array(t)) == _sha(weight)
            for t in proto.graph.initializer
        )


def test_native_weak_matrix_operands_keep_the_actual_result_weakness() -> None:
    specs = _specs([("B", 2), (2, 2)], weak=True)

    def model(x: Any, weight: Any) -> Any:
        return jax.lax.dot_general(x, weight, (((1,), (0,)), ((), ())))

    native = jax.make_jaxpr(model)(*specs)
    assert _body(native).outvars[0].aval.weak_type
    _, _, report = _artifact(model, specs)
    assert all(entry.traced.weak_type for entry in report.inputs)
    assert report.traced_outputs[0].weak_type is True


@pytest.mark.parametrize("nested", [False, True])
def test_nonzero_bias_named_captures_aliases_and_cache_settings(
    nested: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = _original(_CASES[3])
    model.bias[...] = jnp.asarray(np.linspace(-0.5, 0.5, 64), np.float32)
    monkeypatch.setattr(__import__(__name__, fromlist=["_MODEL"]), "_MODEL", model)
    kernel, bias = model.kernel[...], model.bias[...]
    frozen = _sha(kernel), _sha(bias)

    def call(x: Any) -> Any:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("strict"),
            jax.numpy_rank_promotion("raise"),
        ):
            result = _nested(x) if nested else _named(x)
            return result, kernel, bias

    specs = _specs([("B", 128)])
    feeds = [_feeds([("B", 128)], batch) for batch in (2, 5)]
    _native(call, specs, feeds)
    for phase in ("native", "legacy", "source", "native"):
        if phase == "native":
            _native(call, specs, feeds)
        elif phase == "legacy":
            previous = _globals()
            try:
                to_onnx(call, specs, return_mode="ir")
            finally:
                for name, value in previous.items():
                    jax.config.update(name, value)
        else:
            artifact, proto, _ = _artifact(call, specs)
            assert isinstance(artifact, ir.Model)
            functions = [
                f
                for f in artifact.functions.values()
                if any(node.op_type == "MatMul" for node in f.graph)
            ]
            assert len(functions) == 1
            child = functions[0].graph
            assert child.inputs[0].dtype is ir.DataType.FLOAT
            assert child.outputs[0].dtype is ir.DataType.FLOAT
            assert normalized_shape_dimensions(child.inputs[0].shape) == ("B", 128)
            assert normalized_shape_dimensions(child.outputs[0].shape) == ("B", 64)
            assert not child.initializers
            for node in child:
                assert node.graph is child
                assert all(value.producer().graph is child for value in node.outputs)
            assert any(node.op_type == "Add" for node in child)
            arrays = [onnx.numpy_helper.to_array(t) for t in proto.graph.initializer]
            assert any(v.shape == (128, 64) and _sha(v) == frozen[0] for v in arrays)
            assert any(v.shape == (64,) and _sha(v) == frozen[1] for v in arrays)
    assert (_sha(kernel), _sha(bias)) == frozen


@pytest.mark.parametrize(
    "kind", ["batch-broadcast", "K", "N", "static-1", "reference-B"]
)
def test_shared_binding_and_static_singleton_reject_bad_samples_before_ort(
    kind: str,
) -> None:
    add = kind == "static-1"
    shapes = [("B", 3, 4), (1, 3, 4)] if add else [("B", 3, 4), ("B", 4, 5)]
    fn = (
        (lambda a, b: jax.lax.add(a, b))
        if add
        else (lambda a, b: jax.lax.dot_general(a, b, (((2,), (1,)), ((0,), (0,)))))
    )
    _, proto, _ = _artifact(fn, _specs(shapes))
    values = list(_feeds(shapes, 5))
    expected = np.zeros((5, 3, 4 if add else 5), np.float32)
    if kind == "batch-broadcast":
        values[0] = values[0][:1]
    elif kind == "K":
        values[1] = np.zeros((5, 6, 5), np.float32)
    elif kind == "N":
        values[1] = np.zeros((5, 4, 6), np.float32)
    elif kind == "static-1":
        values[1] = np.zeros((5, 3, 4), np.float32)
    else:
        expected = expected[:2]
    result = validate_source_samples(
        proto,
        samples=[SourceSample(values, [expected], SampleCriteria("bitwise"))],
        optimization="disabled",
    )
    assert result.status is CheckStatus.FAILED
    assert any(
        check.category == "sample-input" and check.status is CheckStatus.FAILED
        for sample in result.samples
        for check in sample.checks
    )
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


@pytest.mark.parametrize("corruption", ["report", "artifact", "unknown"])
def test_rank_three_stale_boundaries_fail_before_target_session(
    corruption: str,
) -> None:
    shapes = [("B", 3, 4), ("B", 4, 5)]

    def matrix(a: Any, b: Any) -> Any:
        return jax.lax.dot_general(a, b, (((2,), (1,)), ((0,), (0,))))

    _, proto, report = _artifact(matrix, _specs(shapes))
    if corruption == "report":
        metadata = json.loads(precision_report_to_json(report))
        metadata["inputs"][1]["exported"]["shape"][0] = "Other"
        for entry in proto.metadata_props:
            if entry.key == PRECISION_METADATA_KEY:
                entry.value = json.dumps(metadata)
    else:
        dim = proto.graph.output[0].type.tensor_type.shape.dim[0]
        if corruption == "artifact":
            dim.dim_value = 2
        else:
            dim.Clear()
    result = validate_source_samples(
        proto,
        samples=[
            SourceSample(
                _feeds(shapes, 2),
                [np.zeros((2, 3, 5), np.float32)],
                SampleCriteria("bitwise"),
            )
        ],
        optimization="disabled",
    )
    assert result.status is CheckStatus.FAILED
    assert any(
        check.category == "source-shape-contract" and check.status is CheckStatus.FAILED
        for check in result.checks
    )
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


def test_local_setting_failure_restores_hidden_and_effective_source_state() -> None:
    specs = _specs([("B", 3, 4)])

    def original(x: Any) -> Any:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision("highest"),
            jax.numpy_dtype_promotion("strict"),
            jax.numpy_rank_promotion("raise"),
        ):
            return jax.lax.neg(x)

    before = _globals(), _settings()
    native = jax.make_jaxpr(original)(*specs)
    validate_source_jaxpr(native, 23)
    assert _signature(_body(native).outvars) == (("float32", ("B", 3, 4), False),)
    _, proto, _ = _artifact(original, specs)
    assert [node.op_type for node in proto.graph.node] == ["Neg"]
    for value in (*proto.graph.input, *proto.graph.output):
        tensor = value.type.tensor_type
        assert tensor.elem_type == onnx.TensorProto.FLOAT
        assert [dim.dim_param or dim.dim_value for dim in tensor.shape.dim] == [
            "B",
            3,
            4,
        ]
    assert (_globals(), _settings()) == before

    def unsupported(x: Any) -> Any:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision("highest"),
            jax.numpy_dtype_promotion("strict"),
            jax.numpy_rank_promotion("raise"),
        ):
            return jax.lax.reduce_sum(x, axes=(0,))

    _, reduced, _ = _artifact(unsupported, specs)
    assert any(node.op_type == "ReduceSum" for node in reduced.graph.node)
    result_type = reduced.graph.output[0].type.tensor_type
    assert result_type.elem_type == onnx.TensorProto.FLOAT
    assert tuple(dim.dim_value for dim in result_type.shape.dim) == (3, 4)
    assert (_globals(), _settings()) == before

    def unadmitted(x: Any) -> Any:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision("highest"),
            jax.numpy_dtype_promotion("strict"),
            jax.numpy_rank_promotion("raise"),
        ):
            return jax.lax.erf(x)

    with pytest.raises(SourcePrecisionError) as caught:
        _artifact(unadmitted, specs)
    assert caught.value.primitive == "erf"
    assert caught.value.category == "profile-exclusion"
    assert (_globals(), _settings()) == before


@pytest.mark.parametrize(
    "defect",
    [
        "scope",
        "output-scope",
        "dtype",
        "weak",
        "axes",
        "bool-axis",
        "float-axis",
        "algorithm",
        "placement",
    ],
)
def test_symbolic_dot_scope_and_typed_metadata_fail_before_lowering(
    defect: str,
) -> None:
    specs = _specs([("B", 3, 4), ("B", 4, 5)])
    native = jax.make_jaxpr(
        lambda a, b: jax.lax.dot_general(a, b, (((2,), (1,)), ((0,), (0,))))
    )(*specs)
    body = _body(native)
    original = body.eqns[0]
    inputs = [SimpleNamespace(aval=_forged_aval(v.aval)) for v in original.invars]
    output = SimpleNamespace(aval=_forged_aval(original.outvars[0].aval))
    params = original.params.copy()
    equation = SimpleNamespace(
        primitive=original.primitive,
        invars=inputs,
        outvars=[output],
        params=params,
        effects=original.effects,
        ctx=original.ctx,
    )
    graph = SimpleNamespace(
        invars=inputs, outvars=[output], constvars=(), eqns=[equation], effects=()
    )
    validate_source_jaxpr(graph, 23)
    (foreign,) = export.symbolic_shape("B")
    if defect == "scope":
        inputs[1].aval = _forged_aval(inputs[1].aval, shape=(foreign, 4, 5))
    elif defect == "output-scope":
        output.aval = _forged_aval(output.aval, shape=(foreign, 3, 5))
    elif defect == "dtype":
        output.aval = _forged_aval(output.aval, dtype=np.dtype(np.float64))
    elif defect == "weak":
        output.aval = _forged_aval(output.aval, weak_type=True)
    elif defect == "axes":
        params["dimension_numbers"] = (((1,), (2,)), ((0,), (0,)))
    elif defect in {"bool-axis", "float-axis"}:
        params["dimension_numbers"] = (
            ((2,), (True if defect == "bool-axis" else 1.0,)),
            ((0,), (0,)),
        )
    elif defect == "algorithm":
        params["precision"] = (jax.lax.Precision.HIGHEST,) * 2
    else:
        params["out_sharding"] = object()
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.category == (
        "lowering-defect"
        if defect in {"dtype", "weak", "axes", "bool-axis", "float-axis"}
        else "profile-exclusion"
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("logical corruption must fail before reading or emitting IR")

    context = SimpleNamespace(
        deployment_target=None,
        require_value_for_var=forbidden,
        allocate_value_for_var=forbidden,
        bind_value_for_var=forbidden,
        builder=SimpleNamespace(opset=23, op=forbidden),
    )
    if defect in {"scope", "output-scope"}:
        with pytest.raises(ValueError, match="scope"):
            DotGeneralPlugin()._lower_source(context, equation)
    elif defect in {"bool-axis", "float-axis"}:
        with pytest.raises(ValueError, match="axes"):
            DotGeneralPlugin()._lower_source(context, equation)


@pytest.mark.parametrize(
    "defect",
    ["shape", "weak", "dtype", "permutation", "unknown", "placement", "inferred"],
)
def test_only_resolved_static_bias_reshape_is_admitted(defect: str) -> None:
    source = SimpleNamespace(aval=jax.ShapeDtypeStruct((64,), np.float32))
    target = SimpleNamespace(aval=jax.ShapeDtypeStruct((1, 64), np.float32))
    params = {"new_sizes": (1, 64), "dimensions": None, "sharding": None}
    if defect == "shape":
        target.aval = jax.ShapeDtypeStruct((2, 32), np.float32)
    elif defect == "weak":
        target.aval = jax.ShapeDtypeStruct((1, 64), np.float32, weak_type=True)
    elif defect == "dtype":
        target.aval = jax.ShapeDtypeStruct((1, 64), np.float64)
    elif defect == "permutation":
        params["dimensions"] = (0,)
    elif defect == "unknown":
        params["extra"] = None
    elif defect == "placement":
        params["sharding"] = object()
    else:
        params["new_sizes"] = (1, -1)
    equation = SimpleNamespace(
        primitive=jax.lax.reshape_p,
        invars=[source],
        outvars=[target],
        params=params,
        effects=(),
    )
    graph = SimpleNamespace(
        invars=[source], constvars=(), outvars=[target], eqns=[equation], effects=()
    )
    if defect == "permutation":
        # (0,) is the valid identity permutation; common tensor/index CPU
        # controls already check the more general permuted reshape bitwise.
        validate_source_jaxpr(graph, 23)
        return
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.category == (
        "lowering-defect"
        if defect in {"shape", "weak", "dtype"}
        else "profile-exclusion"
    )


@pytest.mark.parametrize("port", ["lhs", "rhs", "output"])
@pytest.mark.parametrize("defect", ["dtype", "symbol", "tail", "unknown"])
def test_symbolic_matmul_checks_canonical_physical_values_before_emission(
    port: str, defect: str
) -> None:
    native = jax.make_jaxpr(
        lambda a, b: jax.lax.dot_general(a, b, (((2,), (1,)), ((0,), (0,))))
    )(*_specs([("B", 3, 4), ("B", 4, 5)]))
    equation = _body(native).eqns[0]
    validate_source_jaxpr(native, 23)
    values = {
        name: ir.Value(type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(shape))
        for name, shape in (
            ("lhs", ("B", 3, 4)),
            ("rhs", ("B", 4, 5)),
            ("output", ("B", 3, 5)),
        )
    }
    emissions, bindings = [], []

    def emit(operator: str, operands: Any, *, output: Any) -> Any:
        emissions.append((operator, operands, output))
        return output

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid physical values must fail before emitting or binding")

    context = SimpleNamespace(
        deployment_target=None,
        fresh_name=lambda hint: hint,
        require_value_for_var=lambda var: values[
            "lhs" if var is equation.invars[0] else "rhs"
        ],
        allocate_value_for_var=lambda *args, **kwargs: values["output"],
        bind_value_for_var=lambda var, value: bindings.append((var, value)),
        builder=SimpleNamespace(opset=23, op=emit),
    )
    clean = tuple((v.type.dtype, tuple(v.shape)) for v in values.values())
    DotGeneralPlugin()._lower_source(context, equation)
    assert len(emissions) == len(bindings) == 1
    operator, operands, output = emissions[0]
    assert operator == "MatMul" and len(operands) == 2
    assert operands[0] is values["lhs"] and operands[1] is values["rhs"]
    assert output is values["output"]
    assert bindings[0][0] is equation.outvars[0] and bindings[0][1] is output
    assert tuple((v.type.dtype, tuple(v.shape)) for v in values.values()) == clean
    emissions.clear()
    bindings.clear()
    context.builder.op = context.bind_value_for_var = forbidden
    value = values[port]
    if defect == "dtype":
        value.type = ir.TensorType(ir.DataType.DOUBLE)
    else:
        shape = list(value.shape)
        if defect == "symbol":
            shape[0] = "Other"
        elif defect == "tail":
            shape[-1] = 7
        else:
            shape[0] = None
        value.shape = ir.Shape(shape)
    snapshot = tuple((v.type.dtype, tuple(v.shape)) for v in values.values())
    expected = "requires a known named dimension" if defect == "unknown" else "physical"
    with pytest.raises(ValueError, match=expected):
        DotGeneralPlugin()._lower_source(context, equation)
    assert not emissions and not bindings
    assert tuple((v.type.dtype, tuple(v.shape)) for v in values.values()) == snapshot
