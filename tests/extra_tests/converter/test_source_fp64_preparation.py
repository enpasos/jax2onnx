# tests/extra_tests/converter/test_source_fp64_preparation.py

"""Explicit preparation of the actual full Linear and finite FP64 CPU samples.

Raw rows 0/3 retain their factory, shapes, seed, opset, callback and criteria=None.
The exact rational roundoff criterion below is separate and fixed before target
observations. It covers only finite ordinary FP64 arithmetic on these samples.
"""

from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from flax import nnx
import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._source_preparation import SourcePreparationError, prepare_nnx_linear
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins.jax.lax.dot_general import DotGeneralPlugin
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter.source_nnx_linear_fixtures import (
    assert_original_linear,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_native_contract import _forged_aval
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _globals,
    _settings,
    _restore_state as _restore_state,
)

_CANONICAL = (((1,), (0,)), ((), ()))
# Round-to-nearest binary64 unit roundoff. Each implementation's 128-product
# dot plus optional bias has at most 129 rounding steps along any summand.
# The triangle inequality therefore gives 2*gamma129*(sumabs+abs(bias)) between
# two implementations. FMA or a shallower reduction tree cannot increase this
# bound. There is no empirical tolerance or floating-point bound evaluation.
_U64 = Fraction(1, 2**53)
_GAMMA129 = 129 * _U64 / (1 - 129 * _U64)
_TARGET_RULE = (
    "2*gamma129*(sum_i(abs(x_i*w_i))+abs(b)); u=2^-53; exact rational comparison"
)
_KERNEL_SHA = "d443c758235aff0c56e9af2d5862ae9807f6ca8e96d9aaf5289450663e4d3ef9"


def _body(graph: Any) -> Any:
    inner = getattr(graph, "jaxpr", graph)
    return inner if inner is not graph else graph


def _trace(fn: Any, specs: Any) -> Any:
    # A new identity per patch epoch prevents JAX from returning a cached graph.
    def fresh(*values: Any) -> Any:
        return fn(*values)

    return jax.make_jaxpr(fresh)(*specs)


def _spec(dtype: Any = np.float32, weak: bool = False) -> jax.ShapeDtypeStruct:
    (batch,) = export.symbolic_shape("B")
    return jax.ShapeDtypeStruct((batch, 128), dtype, weak_type=weak)


def _feed(batch: int) -> np.ndarray:
    # B=2 is the original PoC feed, with every original value/dtype unchanged.
    return (np.arange(batch * 128).reshape(batch, 128) / 256 - 0.5).astype(np.float32)


def _raw(ordinal: int) -> tuple[Any, dict[str, Any]]:
    metadata = source.raw_source_case("plugin::nnx.linear", ordinal)
    assert metadata["testcase"] == (
        "linear_symbolic_batch" if ordinal == 0 else "linear_no_bias"
    )
    assert metadata["input_shapes"] == [("B", 128)]
    assert metadata["expected_output_shapes"] == [("B", 64)]
    assert metadata.get("opset_version", 23) == 23
    assert all(
        scenario.criteria is None
        for scenario in source.raw_source_scenarios(
            metadata, source.SourceProvenance("plugin::nnx.linear", ordinal)
        )
    )
    recipe = metadata["callable"]
    assert recipe._args == (128, 64) and recipe._dtype is recipe._instance is None
    model = recipe.instantiate()
    assert_original_linear(model, recipe, ordinal)
    assert (model.bias is None) == (ordinal == 3)
    return model, metadata


def _state(model: Any) -> tuple[Any, ...]:
    return (
        tuple(
            (
                str(path),
                np.asarray(value).dtype.str,
                np.asarray(value).shape,
                np.asarray(value).tobytes(),
            )
            for path, value in jax.tree_util.tree_leaves_with_path(nnx.state(model))
        ),
        model.dtype,
        model.param_dtype,
        model.precision,
        model.preferred_element_type,
        model.promote_dtype,
        model.dot_general,
    )


def _artifact(fn: Any, spec: Any) -> tuple[ir.Model, onnx.ModelProto]:
    before = _globals(), _settings()
    artifact = ui._to_onnx(
        fn, [spec], opset=23, precision_policy=PrecisionPolicy.SOURCE, return_mode="ir"
    )
    assert isinstance(artifact, ir.Model)
    assert (_globals(), _settings()) == before
    proto = ir.to_proto(artifact)
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-static-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    assert report.exported_outputs[0].shape == ("B", 64)
    return artifact, proto


def _native(fn: Any, spec: Any, feeds: Any) -> tuple[Any, list[Any]]:
    trace = _trace(fn, [spec])
    references = []
    for values in feeds:
        # The references precede patch activation and export.
        eager = np.asarray(fn(values))
        compiled = np.asarray(jax.jit(lambda x: fn(x))(values))
        assert eager.dtype == compiled.dtype and eager.shape == compiled.shape
        references.append((eager, compiled))
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        actual = _trace(fn, [spec])
    assert typed_trace_manifest(actual) == typed_trace_manifest(trace)
    return trace, references


def _fraction(value: Any) -> Fraction:
    return Fraction.from_float(float(value))


def _bounds(feed: np.ndarray, kernel: np.ndarray, bias: Any) -> list[list[Fraction]]:
    assert feed.dtype == np.float32 and kernel.dtype == np.float64
    assert feed.shape[1] == kernel.shape[0] == 128 and kernel.shape[1] == 64
    if bias is None:
        bias = np.zeros(64, np.float64)
    assert bias.dtype == np.float64 and bias.shape == (64,)
    assert all(np.isfinite(value).all() for value in (feed, kernel, bias))
    normal_min = Fraction(1, 2**1022)
    maximum = _fraction(np.finfo(np.float64).max)
    weights = [[_fraction(value) for value in row] for row in kernel]
    biases = [_fraction(value) for value in bias]
    operands = [
        *map(_fraction, feed.flat),
        *(value for row in weights for value in row),
        *biases,
    ]
    assert Fraction(1, max(value.denominator for value in operands)) >= normal_min
    result = []
    for row in feed:
        inputs = [_fraction(value) for value in row]
        bounds = []
        for column in range(64):
            products = [inputs[index] * weights[index][column] for index in range(128)]
            terms = [*products, biases[column]]
            # These fixed dyadics share a normal-or-larger quantum: any nonzero
            # exact partial sum is normal. sumabs also excludes overflow under
            # any ordering. This justifies the relative-rounding assumptions.
            assert all(term.denominator & (term.denominator - 1) == 0 for term in terms)
            assert Fraction(1, max(term.denominator for term in terms)) >= normal_min
            magnitude = sum(map(abs, terms), Fraction())
            assert magnitude / (1 - 129 * _U64) < maximum
            bounds.append(2 * _GAMMA129 * magnitude)
        result.append(bounds)
    return result


def _compare(actual: Any, expected: Any, bounds: Any) -> None:
    assert actual.dtype == expected.dtype == np.float64
    assert actual.shape == expected.shape == (len(bounds), 64)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    for row, limits in enumerate(bounds):
        for column, limit in enumerate(limits):
            assert (
                abs(_fraction(actual[row, column]) - _fraction(expected[row, column]))
                <= limit
            )


def _saved_targets(
    proto: Any, feeds: Any, references: Any, kernel: Any, bias: Any, path: Path
) -> None:
    # Freeze the sample-dependent exact bounds and statement before any session.
    limits = [_bounds(value, kernel, bias) for value in feeds]
    (path / "criterion.json").write_text(
        json.dumps({"rule": _TARGET_RULE, "raw_metadata_criteria": None})
    )
    saved = path / "linear.onnx"
    onnx.save_model(proto, saved)
    payload = saved.read_bytes()
    reloaded = onnx.load(saved)
    onnx.checker.check_model(reloaded, full_check=True)
    assert reloaded.SerializeToString() == payload
    original_report = get_precision_report(reloaded)
    first = []
    for mode, level in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            payload, options, providers=["CPUExecutionProvider"]
        )
        assert session.get_providers() == ["CPUExecutionProvider"]
        for index, (values, pair, bound) in enumerate(
            zip(feeds, references, limits, strict=True)
        ):
            actual = session.run(None, {session.get_inputs()[0].name: values})[0]
            np.save(path / f"actual-{mode}-{index}.npy", actual, allow_pickle=False)
            for expected in pair:
                _compare(actual, expected, bound)
            if mode == "disabled":
                first.append(actual.copy())
            else:
                # Additional finite sample control. The approved arithmetic
                # bound above remains the numerical acceptance criterion;
                # no portable bitwise-FP64 guarantee follows from this check.
                assert actual.tobytes() == first[index].tobytes()
    assert (
        saved.read_bytes() == payload
        and get_precision_report(reloaded) == original_report
    )


@pytest.mark.parametrize("ordinal", [0, 3])
@pytest.mark.parametrize("caller_x64", [False, True])
def test_original_full_linear_default_and_raw_gemm_callback_remain_distinct(
    ordinal: int, caller_x64: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with jax.enable_x64(caller_x64), jax.default_matmul_precision(None):
        model, metadata = _raw(ordinal)
        before = _state(model)
        trace, refs = _native(model, _spec(), [_feed(2), _feed(5)])
        artifact, proto = _artifact(model, _spec())
        assert all(ref.dtype == np.float32 for pair in refs for ref in pair)
        assert all(value.aval.dtype == np.float32 for value in _body(trace).outvars)
        names = [node.op_type for node in artifact.graph]
        assert names.count("MatMul") == 1 and names.count("Add") == int(ordinal == 0)
        assert "Gemm" not in names
        # Invoke the untouched raw callback and retain its actual diagnostic;
        # native SOURCE MatMul(+Add) does not satisfy its legacy Gemm count.
        capsys.readouterr()
        outcome = metadata["post_check_onnx_graph"](proto)
        diagnostic = capsys.readouterr().out
        assert outcome is False and "Gemm" in diagnostic and "FAILED" in diagnostic
        (tmp_path / "raw-callback.txt").write_text(diagnostic)
        assert _state(model) == before


@pytest.mark.parametrize("ordinal", [0, 3])
@pytest.mark.parametrize("caller_x64", [False, True])
@pytest.mark.parametrize(
    "precision",
    [jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST],
)
@pytest.mark.parametrize("convert_storage", [False, True])
def test_actual_prepared_full_linear_has_one_symbolic_saved_cpu_artifact(
    ordinal: int,
    caller_x64: bool,
    precision: Any,
    convert_storage: bool,
    tmp_path: Path,
) -> None:
    with jax.enable_x64(caller_x64):
        original, metadata = _raw(ordinal)
        original.precision = precision
        before = _state(original), _globals(), _settings()
        with jax.enable_x64(True), jax.default_matmul_precision(None):
            baseline = np.asarray(original(_feed(2)))
            prepared = prepare_nnx_linear(
                original,
                compute_dtype=np.float64,
                parameter_dtype=np.float64 if convert_storage else None,
            )
            assert type(prepared) is nnx.Linear and prepared is not original
            assert prepared.kernel is not original.kernel
            assert prepared.precision is precision
            assert (
                (prepared.bias is None)
                if ordinal == 3
                else (prepared.bias is not original.bias)
            )
            assert prepared.kernel[...].dtype == (
                np.float64 if convert_storage else np.float32
            )
            feeds = [_feed(2), _feed(5)]
            trace, references = _native(prepared, _spec(), feeds)
            assert all(ref.dtype == np.float64 for pair in references for ref in pair)
            assert _body(trace).outvars[0].aval.dtype == np.float64
            _, proto = _artifact(prepared, _spec())
            report = get_precision_report(proto)
            assert report is not None and report.traced_outputs[0].dtype == "float64"
            kernel = np.asarray(original.kernel[...]).astype(np.float64)
            bias = (
                None
                if original.bias is None
                else np.asarray(original.bias[...]).astype(np.float64)
            )
            arrays = [
                onnx.numpy_helper.to_array(value) for value in proto.graph.initializer
            ]
            assert any(
                value.shape == kernel.shape
                and value.dtype in (np.float32, np.float64)
                and value.astype(np.float64).tobytes() == kernel.tobytes()
                for value in arrays
            )
            _saved_targets(proto, feeds, references, kernel, bias, tmp_path)
            assert np.asarray(original(_feed(2))).tobytes() == baseline.tobytes()
        assert (_state(original), _globals(), _settings()) == before
        assert metadata["post_check_onnx_graph"](proto) is False


@pytest.mark.parametrize("convert_storage", [False, True])
@pytest.mark.parametrize("raise_error", [False, True])
def test_preparation_preserves_graph_state_metadata_callables_and_original_calls(
    convert_storage: bool, raise_error: bool
) -> None:
    with jax.enable_x64(True), jax.default_matmul_precision(None):
        model, _ = _raw(0)
        model.bias.set_value(jnp.arange(64, dtype=jnp.float32) / 16 - 1)
        model.kernel.set_metadata(tag="captured-kernel")
        model.alias = model.kernel
        model.counter = nnx.BatchStat(jnp.asarray([7], np.int32))
        calls = []
        original_promote = model.promote_dtype

        def promote(values: Any, *, dtype: Any) -> Any:
            calls.append(dtype)
            return original_promote(values, dtype=dtype)

        model.promote_dtype = promote
        model.preferred_element_type = np.float64
        before = _state(model), _globals(), _settings()
        try:
            prepared = prepare_nnx_linear(
                model,
                compute_dtype=np.float64,
                parameter_dtype=np.float64 if convert_storage else None,
            )
            assert prepared.promote_dtype is model.promote_dtype
            assert prepared.dot_general is model.dot_general
            assert prepared.preferred_element_type is np.float64
            assert prepared.alias is prepared.kernel and model.alias is model.kernel
            assert prepared.counter is not model.counter
            assert prepared.kernel.get_metadata("tag") == "captured-kernel"
            np.testing.assert_array_equal(prepared.counter[...], model.counter[...])
            assert (
                np.asarray(prepared.bias[...]).astype(np.float64).tobytes()
                == np.asarray(model.bias[...]).astype(np.float64).tobytes()
            )
            onehot = np.eye(1, 128, 19, dtype=np.float64)
            expected = np.asarray(model.kernel[...])[19:20].astype(
                np.float64
            ) + np.asarray(model.bias[...]).astype(np.float64)
            for fn in (prepared, jax.jit(lambda x: prepared(x))):
                assert np.asarray(fn(onehot)).tobytes() == expected.tobytes()
            spec = _spec(np.float64)
            native = _trace(prepared, [spec])
            count = len(calls)
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                actual = _trace(prepared, [spec])
            assert len(calls) > count
            assert typed_trace_manifest(actual) == typed_trace_manifest(native)
            artifact, proto = _artifact(prepared, spec)
            names = {node.op_type for node in proto.graph.node}
            assert "Gemm" not in names
            if convert_storage:
                assert "Cast" not in names
            # Retained FP32 storage legitimately promotes before FP64 work.
            # Constant folding may remove these casts; typed operands, not a
            # guessed optimization result, establish the actual computation.
            matrices = [node for node in artifact.graph if node.op_type == "MatMul"]
            assert len(matrices) == 1
            assert all(
                value.dtype == ir.DataType.DOUBLE for value in matrices[0].inputs
            )
            assert matrices[0].outputs[0].dtype == ir.DataType.DOUBLE
            old = np.asarray(model.kernel[...]).copy()
            prepared.kernel.set_value(prepared.kernel[...] + 1)
            prepared.bias.set_value(prepared.bias[...] + 2)
            prepared.counter.set_value(jnp.asarray([9], np.int32))
            assert np.asarray(model.kernel[...]).tobytes() == old.tobytes()
            if raise_error:
                raise RuntimeError("prepared caller exit")
        except RuntimeError as exc:
            assert raise_error and str(exc) == "prepared caller exit"
        assert (_state(model), _globals(), _settings()) == before


@pytest.mark.parametrize("weak", [False, True])
def test_already_fp64_payloads_and_native_weak_input_survive_without_rerounding(
    weak: bool, tmp_path: Path
) -> None:
    with jax.enable_x64(True), jax.default_matmul_precision(None):
        original, _ = _raw(0)
        kernel = np.asarray(original.kernel[...]).astype(np.float64) + 2**-40
        bias = np.arange(64, dtype=np.float64) / 8 + 2**-40
        original.kernel.set_value(jnp.asarray(kernel))
        original.bias.set_value(jnp.asarray(bias))
        prepared = prepare_nnx_linear(
            original, compute_dtype=np.float64, parameter_dtype=np.float64
        )
        for name, expected in (("kernel", kernel), ("bias", bias)):
            assert (
                np.asarray(getattr(prepared, name)[...]).tobytes() == expected.tobytes()
            )
            assert (
                expected.astype(np.float32).astype(np.float64).tobytes()
                != expected.tobytes()
            )
        values = (
            jax.lax.broadcast(jnp.asarray(0.25), (2, 128))
            if weak
            else jnp.full((2, 128), 0.25, np.float64)
        )
        assert bool(values.weak_type) is weak
        trace, references = _native(prepared, _spec(np.float64, weak), [values])
        assert _body(trace).invars[0].aval.weak_type is weak
        assert not _body(trace).outvars[0].aval.weak_type
        _, proto = _artifact(prepared, _spec(np.float64, weak))
        assert not {"Cast", "Gemm"} & {node.op_type for node in proto.graph.node}
        # A captured one-hot row selects original full-sized weights exactly;
        # it supplements the unchanged complete input above.
        onehot = np.eye(1, 128, 19, dtype=np.float64)
        expected = kernel[19:20] + bias
        for fn in (prepared, jax.jit(lambda x: prepared(x))):
            assert np.asarray(fn(onehot)).tobytes() == expected.tobytes()
        assert all(value.dtype == np.float64 for pair in references for value in pair)
        saved = tmp_path / "captured-onehot.onnx"
        onnx.save_model(proto, saved)
        payload = saved.read_bytes()
        reloaded = onnx.load(saved)
        onnx.checker.check_model(reloaded, full_check=True)
        assert reloaded.SerializeToString() == payload
        for optimization in ("disabled", "all"):
            evidence = validate_source_samples(
                reloaded,
                samples=[SourceSample([onehot], [expected], SampleCriteria("bitwise"))],
                optimization=optimization,
            )
            evidence.require_sample_acceptance()
        assert saved.read_bytes() == payload


@pytest.mark.parametrize(
    "defect",
    [
        "x64",
        "compute",
        "compute-none",
        "storage",
        "model",
        "subclass",
        "shape",
        "bias-state",
        "storage-family",
        "setter-hook",
        "param-subclass",
        "preferred",
        "promotion",
        "dot",
    ],
)
def test_private_preparation_rejects_unsupported_requests_without_changes(
    defect: str,
) -> None:
    with jax.enable_x64(True):
        model, _ = _raw(0)
        kwargs = {"compute_dtype": np.float64, "parameter_dtype": np.float64}
        target = model
        if defect == "compute":
            kwargs["compute_dtype"] = np.float32
        elif defect == "compute-none":
            kwargs["compute_dtype"] = None
        elif defect == "storage":
            kwargs["parameter_dtype"] = np.float32
        elif defect == "model":
            target = object()
        elif defect == "subclass":

            class DifferentLinear(nnx.Linear):
                pass

            target = DifferentLinear(128, 64, rngs=nnx.Rngs(0))
        elif defect == "shape":
            model.kernel.set_value(jnp.zeros((127, 64), np.float32))
        elif defect == "bias-state":
            model.use_bias = False
        elif defect == "storage-family":
            model.kernel.set_value(jnp.zeros((128, 64), np.float16))
        elif defect == "setter-hook":
            model.kernel.set_metadata(on_set_value=lambda variable, value: value)
        elif defect == "param-subclass":

            class DifferentParam(nnx.Param):
                def set_value(self, value: Any, **kwargs: Any) -> None:
                    pytest.fail("unreviewed Param subclass setter must not run")

            model.kernel = DifferentParam(model.kernel.get_raw_value())
        elif defect == "preferred":
            model.preferred_element_type = np.float32
        elif defect == "promotion":
            model.promote_dtype = None
        elif defect == "dot":
            model.dot_general = None
        before = _state(model), _globals(), _settings()
        with jax.enable_x64(defect != "x64"):
            with pytest.raises(SourcePreparationError):
                prepare_nnx_linear(target, **kwargs)
        assert (_state(model), _globals(), _settings()) == before


@pytest.mark.parametrize("left", list(jax.lax.Precision))
@pytest.mark.parametrize("right", list(jax.lax.Precision))
@pytest.mark.parametrize("preferred", [None, np.float64])
def test_only_native_all_fp64_enum_pairs_are_additionally_admitted(
    left: Any, right: Any, preferred: Any
) -> None:
    with jax.enable_x64(True):

        def model(x: Any, w: Any) -> Any:
            return jax.lax.dot_general(
                x,
                w,
                _CANONICAL,
                precision=(left, right),
                preferred_element_type=preferred,
            )

        specs = [
            jax.ShapeDtypeStruct((2, 128), np.float64),
            jax.ShapeDtypeStruct((128, 64), np.float64),
        ]
        trace = _trace(model, specs)
        validate_source_jaxpr(trace, 23)
        artifact = ui._to_onnx(
            model,
            specs,
            opset=23,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="proto",
        )
        onnx.checker.check_model(artifact, full_check=True)
        assert [node.op_type for node in artifact.graph.node] == ["MatMul"]


@pytest.mark.parametrize(
    "defect",
    [
        "fp32-high",
        "fp32-highest",
        "mixed",
        "result",
        "preferred",
        "preset",
        "algorithm",
        "list",
        "string-pair",
        "one-enum",
        "unknown",
        "placement",
        "axes",
        "weak",
        "shape",
    ],
)
def test_fp64_enum_admission_keeps_semantic_shape_and_weakness_boundaries(
    defect: str,
) -> None:
    with jax.enable_x64(True):
        native = _trace(
            lambda x, w: jax.lax.dot_general(
                x, w, _CANONICAL, precision=jax.lax.Precision.HIGHEST
            ),
            [
                jax.ShapeDtypeStruct((2, 128), np.float64),
                jax.ShapeDtypeStruct((128, 64), np.float64),
            ],
        )
    original = _body(native).eqns[0]
    inputs = [
        SimpleNamespace(aval=_forged_aval(value.aval)) for value in original.invars
    ]
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
        invars=inputs, constvars=(), outvars=[output], eqns=[equation], effects=()
    )
    validate_source_jaxpr(graph, 23)
    if defect.startswith("fp32-"):
        for value in (*inputs, output):
            value.aval = _forged_aval(value.aval, dtype=np.dtype(np.float32))
        params["precision"] = (getattr(jax.lax.Precision, defect[5:].upper()),) * 2
    elif defect == "mixed":
        inputs[0].aval = _forged_aval(inputs[0].aval, dtype=np.dtype(np.float32))
        # Mixed FP32/FP64 with no preferred dtype computes FP64 natively;
        # the forged FP32 result contradicts that admitted computation.
        output.aval = _forged_aval(output.aval, dtype=np.dtype(np.float32))
        params["preferred_element_type"] = None
    elif defect == "result":
        output.aval = _forged_aval(output.aval, dtype=np.dtype(np.float32))
    elif defect == "preferred":
        params["preferred_element_type"] = np.float32
    elif defect == "preset":
        params["precision"] = jax.lax.DotAlgorithmPreset.F64_F64_F64
    elif defect == "algorithm":
        params["precision"] = jax.lax.DotAlgorithm(np.float64, np.float64, np.float64)
    elif defect == "list":
        params["precision"] = [jax.lax.Precision.HIGHEST] * 2
    elif defect == "string-pair":
        params["precision"] = ("highest", "highest")
    elif defect == "one-enum":
        params["precision"] = (jax.lax.Precision.HIGHEST,)
    elif defect == "unknown":
        params["unreviewed_option"] = None
    elif defect == "placement":
        params["out_sharding"] = object()
    elif defect == "axes":
        params["dimension_numbers"] = (((True,), (0,)), ((), ()))
    elif defect == "weak":
        output.aval = _forged_aval(output.aval, weak_type=True)
    else:
        output.aval = _forged_aval(output.aval, shape=(2, 63))
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.primitive == "dot_general"
    assert caught.value.phase == "profile-validation"
    if defect == "mixed":
        assert caught.value.category == "lowering-defect"

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("logical corruption must fail before reading or emitting IR")

    context = SimpleNamespace(
        deployment_target=None,
        require_value_for_var=forbidden,
        allocate_value_for_var=forbidden,
        bind_value_for_var=forbidden,
        builder=SimpleNamespace(opset=23, op=forbidden),
    )
    with pytest.raises((NotImplementedError, ValueError)):
        DotGeneralPlugin()._lower_source(context, equation)


@pytest.mark.parametrize("port", ["lhs", "rhs", "output"])
@pytest.mark.parametrize("defect", ["dtype", "shape"])
def test_new_fp64_precision_path_still_checks_physical_storage_before_emission(
    port: str, defect: str
) -> None:
    with jax.enable_x64(True):
        graph = _trace(
            lambda x, w: jax.lax.dot_general(
                x, w, _CANONICAL, precision=jax.lax.Precision.HIGH
            ),
            [
                jax.ShapeDtypeStruct((2, 128), np.float64),
                jax.ShapeDtypeStruct((128, 64), np.float64),
            ],
        )
    equation = _body(graph).eqns[0]
    validate_source_jaxpr(graph, 23)
    values = {
        name: ir.Value(type=ir.TensorType(ir.DataType.DOUBLE), shape=ir.Shape(shape))
        for name, shape in (("lhs", (2, 128)), ("rhs", (128, 64)), ("output", (2, 64)))
    }
    emissions, bindings = [], []

    def emit(operator: str, operands: Any, *, output: Any) -> Any:
        emissions.append((operator, operands, output))
        return output

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("physical mismatch must fail before emission or binding")

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
    clean = tuple((value.type.dtype, tuple(value.shape)) for value in values.values())
    DotGeneralPlugin()._lower_source(context, equation)
    assert len(emissions) == len(bindings) == 1
    operator, operands, output = emissions[0]
    assert operator == "MatMul" and len(operands) == 2
    assert operands[0] is values["lhs"] and operands[1] is values["rhs"]
    assert output is values["output"]
    assert bindings[0][0] is equation.outvars[0] and bindings[0][1] is output
    assert (
        tuple((value.type.dtype, tuple(value.shape)) for value in values.values())
        == clean
    )
    emissions.clear()
    bindings.clear()
    context.builder.op = context.bind_value_for_var = forbidden
    values[port].type = (
        ir.TensorType(ir.DataType.FLOAT) if defect == "dtype" else values[port].type
    )
    if defect == "shape":
        values[port].shape = ir.Shape((1, 1))
    snapshot = tuple(
        (value.type.dtype, tuple(value.shape)) for value in values.values()
    )
    with pytest.raises(ValueError, match="physical"):
        DotGeneralPlugin()._lower_source(context, equation)
    assert not emissions and not bindings
    assert (
        tuple((value.type.dtype, tuple(value.shape)) for value in values.values())
        == snapshot
    )


def test_retained_getter_custom_promotion_and_dot_keep_their_actual_native_semantics() -> (
    None
):
    with jax.enable_x64(True), jax.default_matmul_precision(None):
        model, _ = _raw(0)
        calls = {"get": 0, "promote": 0, "dot": 0, "set": 0}
        promote_original, dot_original = model.promote_dtype, model.dot_general

        def getter(variable: Any, value: Any) -> Any:
            calls["get"] += 1
            return value

        def setter(variable: Any, value: Any) -> Any:
            calls["set"] += 1
            return value

        def promote(values: Any, *, dtype: Any) -> Any:
            calls["promote"] += 1
            # An explicit custom function owns its actual computation; setting
            # the declared model dtype does not override this FP32 decision.
            return promote_original(values, dtype=np.float32)

        def dot(*args: Any, **kwargs: Any) -> Any:
            calls["dot"] += 1
            return dot_original(*args, **kwargs)

        model.kernel.set_metadata(on_get_value=getter, on_set_value=setter)
        model.promote_dtype, model.dot_general = promote, dot
        snapshot = calls.copy()
        prepared = prepare_nnx_linear(model, compute_dtype=np.float64)
        assert calls == snapshot
        assert prepared.kernel.get_metadata("on_get_value") is getter
        assert prepared.kernel.get_metadata("on_set_value") is setter
        assert prepared.promote_dtype is promote and prepared.dot_general is dot
        spec = _spec(np.float64)
        trace = _trace(prepared, [spec])
        assert _body(trace).outvars[0].aval.dtype == np.float32
        counts = calls.copy()
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = _trace(prepared, [spec])
        assert all(calls[name] > counts[name] for name in ("get", "promote", "dot"))
        assert calls["set"] == 0
        assert typed_trace_manifest(actual) == typed_trace_manifest(trace)
        _, proto = _artifact(prepared, spec)
        assert (
            proto.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        )
        with pytest.raises(SourcePreparationError, match="hooks"):
            prepare_nnx_linear(
                model, compute_dtype=np.float64, parameter_dtype=np.float64
            )
        assert calls["set"] == 0


def test_parameter_dtype_declaration_does_not_replace_actual_stored_values() -> None:
    with jax.enable_x64(True):
        model, _ = _raw(0)
        model.param_dtype = np.float64
        captured = np.asarray(model.kernel[...]).copy()
        assert captured.dtype == np.float32
        computation_only = prepare_nnx_linear(model, compute_dtype=np.float64)
        converted = prepare_nnx_linear(
            model, compute_dtype=np.float64, parameter_dtype=np.float64
        )
        assert computation_only.param_dtype is model.param_dtype
        assert computation_only.kernel.get_raw_value().dtype == np.float32
        assert np.asarray(computation_only.kernel[...]).tobytes() == captured.tobytes()
        assert converted.kernel.get_raw_value().dtype == np.float64
        assert (
            np.asarray(converted.kernel[...]).tobytes()
            == captured.astype(np.float64).tobytes()
        )
        assert np.asarray(model.kernel[...]).tobytes() == captured.tobytes()


@pytest.mark.parametrize("defect", ["beyond-bound", "nonfinite", "dtype"])
def test_frozen_full_model_comparison_gate_rejects_corrupted_actuals(
    defect: str,
) -> None:
    with jax.enable_x64(True), jax.default_matmul_precision(None):
        original, _ = _raw(0)
        prepared = prepare_nnx_linear(
            original, compute_dtype=np.float64, parameter_dtype=np.float64
        )
        feed = _feed(2)
        limits = _bounds(
            feed, np.asarray(prepared.kernel[...]), np.asarray(prepared.bias[...])
        )
        reference = np.asarray(prepared(feed))
    _compare(reference, reference, limits)
    changed = reference.copy()
    if defect == "beyond-bound":
        changed[0, 0] += 1
        assert abs(_fraction(changed[0, 0]) - _fraction(reference[0, 0])) > limits[0][0]
    elif defect == "nonfinite":
        changed[0, 0] = np.inf
    else:
        changed = changed.astype(np.float32)
    with pytest.raises(AssertionError):
        _compare(changed, reference, limits)
