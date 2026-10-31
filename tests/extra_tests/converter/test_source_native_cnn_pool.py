# tests/extra_tests/converter/test_source_native_cnn_pool.py

"""Unchanged full CNN SOURCE export and separate finite native pool controls.

General nonzero CNN target criteria remain unchosen. Pool special-value
comparisons retain every actual return before assertions, including failures.
"""

from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from flax import nnx
import jax
from jax import export
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import ShapedArray, Var
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_pooling import validate_native_window_sum
from jax2onnx.converter.source_profile import SourcePrecisionError, _Site, _tensor
from jax2onnx.plugins.flax.nnx.avg_pool import AvgPoolPlugin
from jax2onnx.plugins.flax.nnx.relu import ReluPlugin
from jax2onnx.plugins.jax.lax.reduce_window_sum import ReduceWindowSumPlugin
from jax2onnx.plugins.plugin_system import FunctionPlugin, _SourceFunctionTrace
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter import source_cnn_acceptance as cnn
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_native_conv import (
    _product,
    _saved_samples,
    _trace,
)
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _globals,
    _settings,
)


_CNN: Any = None


@onnx_function(name="OriginalCNNSource")
def _named_cnn(x: Any) -> Any:
    return _CNN(x)


@onnx_function(name="NativeDisjointPool")
def _named_pool(x: Any) -> Any:
    return nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))


def _pool(x: Any) -> Any:
    return nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))


def _parameters(model: Any) -> list[dict[str, Any]]:
    return [
        {
            "path": list(map(str, path)),
            "dtype": np.asarray(value).dtype.str,
            "shape": list(np.asarray(value).shape),
            "sha256": hashlib.sha256(np.asarray(value).tobytes()).hexdigest(),
        }
        for path, value in jax.tree_util.tree_flatten_with_path(
            nnx.state(model, nnx.Param)
        )[0]
    ]


def _source_artifact(
    fn: Any, specs: Any, path: Path
) -> tuple[ir.Model, onnx.ModelProto]:
    before = _globals(), _settings()
    artifact = ui._to_onnx(
        fn,
        specs,
        opset=23,
        precision_policy=PrecisionPolicy.SOURCE,
        return_mode="ir",
    )
    assert isinstance(artifact, ir.Model)
    assert (_globals(), _settings()) == before
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    proto = ir.to_proto(artifact)
    path.write_bytes(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-static-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    return artifact, proto


def _cnn_product(
    fn: Any, specs: Any, model: Any, batch: int | str, named: bool, path: Path
) -> tuple[ir.Model, onnx.ModelProto]:
    artifact, proto = _source_artifact(fn, specs, path)
    # This exact matcher owns the complete signed-zero selectors and their
    # BOOL-to-INT64 casts; it rejects other data casts and altered parameters.
    cnn.ArtifactContract(cnn.parameters(model), batch, named=named).check(proto)
    return artifact, proto


def assert_original_cnn_source(
    ordinal: int,
    x64: bool,
    tmp_path: Path,
    record_property: Any,
    *,
    named: bool = False,
) -> None:
    global _CNN
    metadata = source.raw_source_case("example::examples.nnx::CNN", ordinal)
    assert metadata["testcase"] == (
        "simple_cnn_static" if ordinal == 0 else "simple_cnn"
    )
    assert metadata["input_shapes"] == [(3 if ordinal == 0 else "B", 28, 28, 1)]
    assert metadata["expected_output_shapes"] == [(3 if ordinal == 0 else "B", 10)]
    assert metadata["run_only_f32_variant"] is True
    assert metadata.get("opset_version", 23) == 23
    assert metadata.get("run_only_dynamic", False) is bool(ordinal)
    recipe = metadata["callable"]
    assert recipe._dtype is None and recipe._instance is None
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("example::examples.nnx::CNN", ordinal)
    )[int(x64)]
    if ordinal == 1:
        scenario = replace(scenario, concrete_shapes=((3, 28, 28, 1),))
    prepared = source.prepare_source_case(metadata, scenario)
    assert scenario.criteria is prepared.criteria is None
    model = prepared.fn
    parameters = _parameters(model)
    assert len(parameters) == 8 and all(item["dtype"] == "<f4" for item in parameters)
    assert {tuple(item["shape"]) for item in parameters} >= {
        (3, 3, 1, 32),
        (3, 3, 32, 64),
        (3136, 256),
        (256, 10),
    }
    (tmp_path / "original-parameters.json").write_text(json.dumps(parameters, indent=2))
    previous = _CNN
    try:
        _CNN = model

        def named_call(x: Any) -> Any:
            return _named_cnn(x)

        call = named_call if named else model
        with source.source_configuration(scenario):
            native = _trace(model, prepared.input_specs)
            native_manifest = typed_trace_manifest(native)
            assert Counter(item["sha256"] for item in parameters) <= Counter(
                item["sha256"] for item in native_manifest["const_payloads"]
            )
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                actual = _trace(model, prepared.input_specs)
            assert typed_trace_manifest(actual) == native_manifest
            # Capture full original nonzero eager/JIT returns, without choosing a target tolerance.
            eager = tuple(np.array(v, copy=True) for v in prepared.reference_outputs)
            compiled = tuple(
                np.array(v, copy=True)
                for v in jax.tree.leaves(nnx.jit(model)(*prepared.arguments))
            )
            np.savez(
                tmp_path / "original-nonzero-native.npz",
                eager=eager[0],
                jit=compiled[0],
                input=prepared.arguments[0],
            )
            for value in (*eager, *compiled):
                assert (
                    value.dtype == np.float32
                    and value.shape == (3, 10)
                    and np.all(np.isfinite(value))
                )
            caller_settings = {
                "x64_enabled": bool(jax.config.x64_enabled),
                "default_matmul_precision": jax.config.jax_default_matmul_precision,
                "numpy_dtype_promotion": jax.config.jax_numpy_dtype_promotion,
                "numpy_rank_promotion": jax.config.jax_numpy_rank_promotion,
                "debug_key_reuse": bool(jax.config.jax_debug_key_reuse),
            }
            original_body = FunctionPlugin._trace_function_body
            original_callee = _named_cnn
            observed_bodies = []

            def observe_body(
                plugin: Any, callee: Any, equation: Any, captures: Any, **kwargs: Any
            ) -> Any:
                if callee is original_callee:
                    trace = kwargs["source_trace"]
                    assert type(trace) is _SourceFunctionTrace
                    assert {
                        "x64_enabled": trace.x64_enabled,
                        "default_matmul_precision": trace.default_matmul_precision,
                        "numpy_dtype_promotion": trace.numpy_dtype_promotion,
                        "numpy_rank_promotion": trace.numpy_rank_promotion,
                        "debug_key_reuse": trace.debug_key_reuse,
                    } == caller_settings
                actual_body = original_body(
                    plugin, callee, equation, captures, **kwargs
                )
                if callee is original_callee:
                    assert kwargs["precision_policy"] is PrecisionPolicy.SOURCE
                    assert captures.dynamic == () and dict(captures.static_kwargs) == {}
                    actual_manifest = typed_trace_manifest(actual_body)
                    assert actual_manifest == native_manifest
                    observed_bodies.append(actual_manifest)
                return actual_body  # Preserve the exact original returned object.

            with (
                patch.object(FunctionPlugin, "_trace_function_body", observe_body)
                if named
                else nullcontext()
            ):
                artifact, proto = _cnn_product(
                    call,
                    prepared.input_specs,
                    model,
                    3 if ordinal == 0 else "B",
                    named,
                    tmp_path / "original-source.onnx",
                )
            assert FunctionPlugin._trace_function_body is original_body
            if named:
                assert observed_bodies and all(
                    body == native_manifest for body in observed_bodies
                )
                (tmp_path / "actual-exported-function-native-body.json").write_text(
                    json.dumps(observed_bodies, indent=2)
                )
            onnx.save_model(proto, tmp_path / "original-source.onnx")
            # The legacy structural callback remains unchanged and fails independently.
            callback = metadata["post_check_onnx_graph"](proto)
            assert callback is False
            record_property("raw_callback_status", "failed-retained")
            record_property(
                "general_nonzero_target_numerics", "unverified; original criteria=None"
            )
            if named:
                assert proto.functions and any(
                    function.name == "OriginalCNNSource" for function in proto.functions
                )
            # Establish the actual original zero-bias premise before the zero-input control.
            biases = [
                np.asarray(getattr(model, name).bias[...])
                for name in ("conv1", "conv2", "linear1", "linear2")
            ]
            assert len(biases) == 4 and all(
                value.dtype == np.float32 and value.tobytes() == bytes(value.nbytes)
                for value in biases
            )
            feeds, expected = [], []
            for batch in (3,) if ordinal == 0 else (2, 5):
                x = np.zeros((batch, 28, 28, 1), np.float32)
                y, compiled_y = np.asarray(model(x)), np.asarray(nnx.jit(model)(x))
                np.savez(
                    tmp_path / f"original-zero-B{batch}-native.npz",
                    input=x,
                    eager=y,
                    jit=compiled_y,
                )
                assert (
                    y.dtype == compiled_y.dtype == np.float32
                    and y.shape == compiled_y.shape == (batch, 10)
                )
                assert y.tobytes() == compiled_y.tobytes() == bytes(y.nbytes)
                feeds.append((x,))
                expected.append((y,))
            _saved_samples(
                proto, feeds, expected, tmp_path / "same-original-source.onnx"
            )
            assert _parameters(model) == parameters
            assert (
                metadata["post_check_onnx_graph"]
                is prepared.metadata["post_check_onnx_graph"]
            )
            assert prepared.criteria is None
    finally:
        _CNN = previous


@pytest.mark.parametrize("x64", [False, True])
def test_named_complete_original_cnn_keeps_same_symbolic_artifact(
    x64: bool, tmp_path: Path, record_property: Any
) -> None:
    assert_original_cnn_source(1, x64, tmp_path, record_property, named=True)


@pytest.mark.parametrize("plugin", ["relu", "avg_pool"])
def test_source_wrappers_forward_exact_original_calls_after_nested_patch(
    plugin: str,
) -> None:
    marker, value, argument, option = object(), object(), object(), object()
    calls = []

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return marker

    factory = (
        ReluPlugin.patch_info()["patch_function"]
        if plugin == "relu"
        else AvgPoolPlugin.get_monkey_patch
    )
    previous = AvgPoolPlugin._ORIG_CALL
    try:
        patched = factory(factory(original))
        with precision_policy_scope(PrecisionPolicy.SOURCE):
            assert patched(value, argument, unknown=option) is marker
    finally:
        AvgPoolPlugin._ORIG_CALL = previous
    assert calls == [((value, argument), {"unknown": option})]


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("x64", [False, True])
def test_native_pool_dyadic_window_arrangement_same_dynamic_artifact(
    named: bool, x64: bool, tmp_path: Path
) -> None:
    def named_call(x: Any) -> Any:
        return _named_pool(x)

    call = named_call if named else _pool
    with jax.enable_x64(x64):
        (batch,) = export.symbolic_shape("B")
        specs = [jax.ShapeDtypeStruct((batch, 4, 6, 3), np.float32)]
        native = _trace(_pool, specs)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = _trace(_pool, specs)
        assert typed_trace_manifest(actual) == typed_trace_manifest(native)
        _, proto = _product(call, specs)
        nodes = [
            *proto.graph.node,
            *(node for function in proto.functions for node in function.node),
        ]
        assert sum(node.op_type == "ReduceSum" for node in nodes) == 1
        assert sum(node.op_type == "Div" for node in nodes) == 1
        assert not {"AveragePool", "Conv", "Relu"} & {node.op_type for node in nodes}
        feeds, expected = [], []
        for size in (2, 5):
            for family in ("zero", "dyadic"):
                x = (
                    np.zeros((size, 4, 6, 3), np.float32)
                    if family == "zero"
                    else (
                        (np.arange(size * 4 * 6 * 3) % 23 - 11).reshape(size, 4, 6, 3)
                        / 16
                    ).astype(np.float32)
                )
                manual = x.reshape(size, 2, 2, 3, 2, 3).sum(
                    axis=(2, 4), dtype=np.float32
                ) / np.float32(4)
                eager, compiled = np.asarray(_pool(x)), np.asarray(jax.jit(_pool)(x))
                np.savez(
                    tmp_path / f"B{size}-{family}-native.npz",
                    input=x,
                    eager=eager,
                    jit=compiled,
                    manual=manual,
                )
                assert eager.dtype == compiled.dtype == manual.dtype == np.float32
                assert eager.tobytes() == compiled.tobytes() == manual.tobytes()
                feeds.append((x,))
                expected.append((eager,))
        _saved_samples(proto, feeds, expected, tmp_path / "same-pool.onnx")


@pytest.mark.parametrize(
    "family", ["positive-zero", "negative-zero", "nan", "cancellation", "overflow"]
)
def test_native_pool_special_sum_and_separate_division_capture_before_comparison(
    family: str, tmp_path: Path
) -> None:
    import onnxruntime as ort

    values = {
        "positive-zero": [0.0, 0.0, 0.0, 0.0],
        "negative-zero": [-0.0, -0.0, -0.0, -0.0],
        "nan": [np.nan, 1.0, -1.0, 0.0],
        "cancellation": [2.0**24, 1.0, -(2.0**24), 1.0],
        "overflow": [np.finfo(np.float32).max] * 4,
    }

    def original(x: Any) -> Any:
        total = jax.lax.reduce_window_sum_p.bind(
            x,
            window_dimensions=(1, 2, 2, 1),
            window_strides=(1, 2, 2, 1),
            padding=((0, 0),) * 4,
            base_dilation=(1,) * 4,
            window_dilation=(1,) * 4,
        )
        return total, total / np.float32(4)

    x = np.asarray(values[family], np.float32).reshape(1, 2, 2, 1)
    eager = tuple(np.asarray(value) for value in original(x))
    compiled = tuple(np.asarray(value) for value in jax.jit(original)(x))
    np.savez(
        tmp_path / "native-before-target.npz",
        input=x,
        eager_sum=eager[0],
        eager_mean=eager[1],
        jit_sum=compiled[0],
        jit_mean=compiled[1],
    )
    _, proto = _product(original, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    assert any(node.op_type == "Div" for node in proto.graph.node)
    onnx.save_model(proto, tmp_path / "pool-special-source.onnx")
    target_returns = {}
    for mode in ("disabled", "all"):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if mode == "disabled"
            else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session = ort.InferenceSession(
            proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        returned = tuple(session.run(None, {session.get_inputs()[0].name: x}))
        np.savez(
            tmp_path / f"{mode}-returned-before-assertions.npz",
            **{f"out{i}": value for i, value in enumerate(returned)},
        )
        target_returns[mode] = returned
    # Both modes are retained before any reduction-order/signed-zero comparison.
    for returned in target_returns.values():
        for reference in (eager, compiled):
            for expected, actual in zip(reference, returned, strict=True):
                assert actual.dtype == expected.dtype and actual.shape == expected.shape
                assert np.array_equal(np.isnan(actual), np.isnan(expected))
                assert (
                    actual[~np.isnan(actual)].tobytes()
                    == expected[~np.isnan(expected)].tobytes()
                )


@pytest.mark.parametrize(
    "defect",
    [
        "identity",
        "missing",
        "unknown",
        "overlap",
        "padding",
        "dilation",
        "channel",
        "result",
        "weak",
    ],
)
def test_native_window_sum_rejects_unimplemented_or_malformed_contract(
    defect: str,
) -> None:
    graph = jax.make_jaxpr(_pool)(jax.ShapeDtypeStruct((2, 4, 6, 3), np.float32))
    eqn = next(
        e for e in graph.jaxpr.eqns if e.primitive is jax.lax.reduce_window_sum_p
    )
    changed = SimpleNamespace(
        primitive=eqn.primitive,
        params=dict(eqn.params),
        invars=list(eqn.invars),
        outvars=list(eqn.outvars),
    )
    if defect == "identity":
        changed.primitive = SimpleNamespace(name="reduce_window_sum")
    elif defect == "missing":
        changed.params.pop("padding")
    elif defect == "unknown":
        changed.params["unreviewed"] = None
    elif defect == "overlap":
        changed.params["window_strides"] = (1, 1, 1, 1)
    elif defect == "padding":
        changed.params["padding"] = ((0, 0), (0, 1), (0, 0), (0, 0))
    elif defect == "dilation":
        changed.params["window_dilation"] = (1, 2, 1, 1)
    elif defect == "channel":
        changed.params["window_dimensions"] = (1, 2, 2, 3)
    elif defect == "result":
        changed.outvars = [Var(ShapedArray((2, 3, 3, 3), np.float32))]
    elif defect == "weak":
        changed.outvars = [Var(ShapedArray((2, 2, 3, 3), np.float32, weak_type=True))]
    site = _Site("pool-control", 23, "reduce_window_sum")
    with pytest.raises(SourcePrecisionError):
        validate_native_window_sum(
            changed,
            [_tensor(v, site) for v in changed.invars],
            [_tensor(v, site) for v in changed.outvars],
            site,
        )


@pytest.mark.parametrize(
    "dtype", [np.int32, np.complex64, np.float16, "bfloat16", np.float64]
)
def test_native_pool_other_dtypes_remain_implementation_obligations(dtype: Any) -> None:
    graph = jax.make_jaxpr(_pool)(jax.ShapeDtypeStruct((2, 4, 6, 3), np.float32))
    eqn = next(
        e for e in graph.jaxpr.eqns if e.primitive is jax.lax.reduce_window_sum_p
    )
    changed = SimpleNamespace(
        primitive=eqn.primitive,
        params=dict(eqn.params),
        invars=[Var(ShapedArray((2, 4, 6, 3), dtype))],
        outvars=[Var(ShapedArray((2, 2, 3, 3), dtype))],
    )
    site = _Site("pool-dtype-control", 23, "reduce_window_sum")
    with pytest.raises(SourcePrecisionError) as error:
        validate_native_window_sum(
            changed,
            [_tensor(v, site) for v in changed.invars],
            [_tensor(v, site) for v in changed.outvars],
            site,
        )
    assert error.value.category == "profile-exclusion"


def _pool_artifact_contract(proto: onnx.ModelProto, native: Any) -> None:
    body = native.jaxpr
    promoted = np.dtype(body.outvars[0].aval.dtype) == np.dtype(np.float64)
    assert [equation.primitive for equation in body.eqns] == (
        [jax.lax.reduce_window_sum_p]
        + ([jax.lax.convert_element_type_p] if promoted else [])
        + [jax.lax.div_p]
    )
    window, division = body.eqns[0], body.eqns[-1]
    assert window.invars[0] is body.invars[0]
    assert division.outvars[0] is body.outvars[0]
    if promoted:
        conversion = body.eqns[1]
        assert conversion.invars[0] is window.outvars[0]
        assert division.invars[0] is conversion.outvars[0]
        assert np.dtype(conversion.invars[0].aval.dtype) == np.dtype(np.float32)
        assert np.dtype(conversion.outvars[0].aval.dtype) == np.dtype(np.float64)
    else:
        assert division.invars[0] is window.outvars[0]
    divisor = division.invars[1]
    assert hasattr(divisor, "val") and np.asarray(divisor.val).item() == 4
    assert tuple(divisor.aval.shape) == ()
    assert np.dtype(divisor.aval.dtype) == np.dtype(body.outvars[0].aval.dtype)

    def signature(value: onnx.ValueInfoProto) -> tuple[Any, ...]:
        tensor = value.type.tensor_type
        assert tensor.HasField("shape")
        assert all(dim.HasField("dim_value") for dim in tensor.shape.dim)
        return tensor.elem_type, tuple(dim.dim_value for dim in tensor.shape.dim)

    def declared(variable: Any) -> tuple[Any, ...]:
        return (
            onnx.helper.np_dtype_to_tensor_dtype(np.dtype(variable.aval.dtype)),
            tuple(variable.aval.shape),
        )

    assert not proto.functions
    assert len(proto.graph.input) == len(proto.graph.output) == 1
    assert signature(proto.graph.input[0]) == declared(body.invars[0])
    assert signature(proto.graph.output[0]) == declared(body.outvars[0])
    nodes = list(proto.graph.node)
    assert [node.op_type for node in nodes] == (
        ["Reshape", "ReduceSum"] + (["Cast"] if promoted else []) + ["Div"]
    )
    assert all(node.domain == "" and len(node.output) == 1 for node in nodes)
    assert len({node.output[0] for node in nodes}) == len(nodes)
    reshape, reduction, divide = nodes[0], nodes[1], nodes[-1]
    assert len(reshape.input) == len(reduction.input) == len(divide.input) == 2
    assert reshape.input[0] == proto.graph.input[0].name
    assert reduction.input[0] == reshape.output[0]
    assert divide.output[0] == proto.graph.output[0].name
    assert {attr.name: attr.i for attr in reshape.attribute} == {"allowzero": 0}
    assert {attr.name: attr.i for attr in reduction.attribute} == {"keepdims": 0}
    assert not divide.attribute
    numerator = reduction.output[0]
    if promoted:
        cast = nodes[2]
        assert list(cast.input) == [numerator]
        assert {attr.name: attr.i for attr in cast.attribute} == {
            "to": onnx.TensorProto.DOUBLE
        }
        numerator = cast.output[0]
    assert divide.input[0] == numerator
    initializers = {value.name: value for value in proto.graph.initializer}
    assert set(initializers) == {reshape.input[1], reduction.input[1], divide.input[1]}
    for name, expected in (
        (reshape.input[1], np.asarray([2, 2, 2, 3, 2, 3], np.int64)),
        (reduction.input[1], np.asarray([2, 4], np.int64)),
        (divide.input[1], np.asarray(4, dtype=divisor.aval.dtype)),
    ):
        value = onnx.numpy_helper.to_array(initializers[name])
        assert value.dtype == expected.dtype and value.shape == expected.shape
        assert value.tobytes() == expected.tobytes()
    infos = {
        value.name: value
        for value in (*proto.graph.input, *proto.graph.output, *proto.graph.value_info)
    }
    assert signature(infos[reshape.output[0]]) == (
        onnx.TensorProto.FLOAT,
        (2, 2, 2, 3, 2, 3),
    )
    assert signature(infos[reduction.output[0]]) == declared(window.outvars[0])
    assert signature(infos[numerator]) == declared(division.invars[0])


def _pool_artifact_corruptions(
    proto: onnx.ModelProto, native: Any, directory: Path
) -> None:
    directory.mkdir()
    original = proto.SerializeToString()
    promoted = np.dtype(native.jaxpr.outvars[0].aval.dtype) == np.dtype(np.float64)
    families = ["input-dtype", "output-dtype", "div-bypass", "divisor", "extra-cast"]
    if promoted:
        families += ["missing-conversion", "conversion-before-sum"]
    for family in families:
        damaged = onnx.load_model_from_string(original)
        graph = damaged.graph
        if family == "input-dtype":
            graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
        elif family == "output-dtype":
            graph.output[0].type.tensor_type.elem_type = (
                onnx.TensorProto.FLOAT if promoted else onnx.TensorProto.DOUBLE
            )
        elif family == "div-bypass":
            graph.output[0].name = graph.node[-1].input[0]
        elif family == "divisor":
            value = next(
                value
                for value in graph.initializer
                if value.name == graph.node[-1].input[1]
            )
            value.CopyFrom(
                onnx.numpy_helper.from_array(
                    np.asarray(8, dtype=native.jaxpr.outvars[0].aval.dtype),
                    name=value.name,
                )
            )
        elif family == "extra-cast":
            graph.node.append(
                onnx.helper.make_node(
                    "Cast",
                    [graph.output[0].name],
                    ["extra-floating-cast"],
                    to=graph.output[0].type.tensor_type.elem_type,
                )
            )
        elif family == "missing-conversion":
            graph.node[-1].input[0] = graph.node[1].output[0]
            del graph.node[2]
        else:
            cast = onnx.NodeProto()
            cast.CopyFrom(graph.node[2])
            cast.input[0] = graph.node[0].output[0]
            graph.node[1].input[0] = cast.output[0]
            graph.node[-1].input[0] = graph.node[1].output[0]
            del graph.node[2]
            graph.node.insert(1, cast)
        (directory / f"{family}.onnx").write_bytes(damaged.SerializeToString())
        with pytest.raises(AssertionError):
            _pool_artifact_contract(damaged, native)
    assert proto.SerializeToString() == original


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("weak", [False, True])
def test_native_pool_retains_actual_input_and_sum_weakness(
    weak: bool, x64: bool, tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        specs = [jax.ShapeDtypeStruct((2, 4, 6, 3), np.float32, weak_type=weak)]
        native = _trace(_pool, specs)
        equation = next(
            e for e in native.jaxpr.eqns if e.primitive is jax.lax.reduce_window_sum_p
        )
        site = _Site("native-pool-weakness", 23, "reduce_window_sum")
        assert (
            equation.invars[0].aval.weak_type
            is equation.outvars[0].aval.weak_type
            is weak
        )
        assert all(
            np.dtype(variable.aval.dtype) == np.dtype(np.float32)
            for variable in (*equation.invars, *equation.outvars)
        )
        assert validate_native_window_sum(
            equation,
            [_tensor(v, site) for v in equation.invars],
            [_tensor(v, site) for v in equation.outvars],
            site,
        ) == (2, 2)
        assert np.dtype(native.jaxpr.outvars[0].aval.dtype) == np.dtype(
            np.float64 if weak and x64 else np.float32
        )
        assert native.jaxpr.outvars[0].aval.weak_type is weak
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = _trace(_pool, specs)
        assert typed_trace_manifest(actual) == typed_trace_manifest(native)
        _, proto = _source_artifact(_pool, specs, tmp_path / "native-pool-source.onnx")
        report = get_precision_report(proto)
        assert report is not None
        for traced, variable in (
            (report.inputs[0].traced, native.jaxpr.invars[0]),
            (report.traced_outputs[0], native.jaxpr.outvars[0]),
        ):
            assert traced is not None
            assert traced.dtype == np.dtype(variable.aval.dtype).name
            assert traced.shape == tuple(variable.aval.shape)
            assert traced.weak_type is variable.aval.weak_type
        _pool_artifact_contract(proto, native)
        _pool_artifact_corruptions(proto, native, tmp_path / "artifact-corruptions")


@pytest.mark.parametrize("opset", [16, 18, 21, 24, 26])
def test_native_pool_other_opsets_require_separate_implementation(opset: int) -> None:
    native = _trace(_pool, [jax.ShapeDtypeStruct((2, 4, 6, 3), np.float32)])
    equation = next(
        e for e in native.jaxpr.eqns if e.primitive is jax.lax.reduce_window_sum_p
    )
    site = _Site("native-pool-opset", opset, "reduce_window_sum")
    with pytest.raises(SourcePrecisionError) as error:
        validate_native_window_sum(
            equation,
            [_tensor(v, site) for v in equation.invars],
            [_tensor(v, site) for v in equation.outvars],
            site,
        )
    assert error.value.category == "profile-exclusion"


@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("defect", ["dtype", "extent", "unknown"])
def test_source_window_sum_rejects_physical_drift_before_emission(
    operand: int, defect: str
) -> None:
    native = _trace(_pool, [jax.ShapeDtypeStruct((2, 4, 6, 3), np.float32)])
    equation = next(
        e for e in native.jaxpr.eqns if e.primitive is jax.lax.reduce_window_sum_p
    )
    variables = (*equation.invars, *equation.outvars)
    values = [
        ir.val(f"pool{index}", dtype=ir.DataType.FLOAT, shape=var.aval.shape)
        for index, var in enumerate(variables)
    ]
    if defect == "dtype":
        values[operand].type = ir.TensorType(ir.DataType.DOUBLE)
    elif defect == "extent":
        values[operand].shape = ir.Shape((2, 5, 6, 3))
    else:
        values[operand].shape = None
    bindings = dict(zip(map(id, variables), values, strict=True))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Physical mismatch must fail before any builder emission")

    context = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=SimpleNamespace(
            opset=23, op=forbidden, add_initializer_from_array=forbidden
        ),
        fresh_name=lambda hint: hint,
        require_value_for_var=lambda var: bindings[id(var)],
        allocate_value_for_var=lambda var, **kwargs: bindings[id(var)],
        bind_value_for_var=forbidden,
    )
    with pytest.raises(TypeError, match="SOURCE window sum physical storage"):
        ReduceWindowSumPlugin().lower(context, equation)
