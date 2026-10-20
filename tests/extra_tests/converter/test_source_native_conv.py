# tests/extra_tests/converter/test_source_native_conv.py

"""Native Conv admission, unchanged raw anchors and separate finite CPU samples.

Raw metadata criteria remain None. Exact numerical controls below belong only
to their declared dyadic samples, after original unpatched eager/JIT references.
The unchanged complete CNN is maintained by the native pooling follow-up tests.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from flax import linen as nn, nnx
import jax
from jax import export
import jax.numpy as jnp
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
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins.flax.linen.conv import ConvPlugin as LinenConvPlugin
from jax2onnx.plugins.flax.nnx.conv import ConvPlugin as NNXConvPlugin
from jax2onnx.plugins.jax.lax.conv import ConvGeneralDilatedPlugin
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _globals,
    _settings,
    _restore_state as _restore_state,
)

_BITS = SampleCriteria("bitwise")
_ORIGINAL_CONV_NAMES = (
    "conv",
    "conv2",
    "conv_nchw",
    "conv_general_dilated_1d",
    "conv_nhwc",
    "conv_general_dilated_nhwc_output",
    "conv_complex64",
    "conv_complex64_nhwc",
    "conv_complex128_grouped",
    "conv_transpose_lhs_dilation_nchw",
    "conv_transpose_nhwc_same",
    "conv_transpose_valid_stride_gt_kernel",
    "conv_transpose_valid_stride_gt_kernel_dynamic",
    "conv_transpose_1d_leading_output_pad",
    "conv_transpose_grouped",
    "conv_transpose_grouped_nhwc",
    "conv_transpose_asymmetric",
)


@onnx_function(name="NativeFP32Conv1D")
def _named_native_conv(x: Any, kernel: Any) -> Any:
    return jax.lax.conv_general_dilated(
        x, kernel, (1,), "VALID", dimension_numbers=("NCW", "OIW", "NCW")
    )


def _signature(variables: Any) -> Any:
    return tuple(
        (
            np.dtype(var.aval.dtype).name,
            tuple(map(str, var.aval.shape)),
            var.aval.weak_type,
        )
        for var in variables
    )


def _trace(fn: Any, specs: Any) -> Any:
    # A fresh forwarding callable forces each patch epoch to trace the same
    # original target again instead of reusing a cached make_jaxpr/JIT result.
    def fresh(*values: Any) -> Any:
        return fn(*values)

    return jax.make_jaxpr(fresh)(*specs)


def _product(fn: Any, specs: Any) -> tuple[ir.Model, onnx.ModelProto]:
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
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-static-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    assert all(value.dtype == ir.DataType.FLOAT for value in artifact.graph.inputs)
    assert all(value.dtype == ir.DataType.FLOAT for value in artifact.graph.outputs)
    assert not {"Cast", "CastLike", "ConvTranspose"} & {
        node.op_type for node in proto.graph.node
    }
    return artifact, proto


def _saved_samples(
    proto: onnx.ModelProto, feeds: Any, expected: Any, path: Path
) -> onnx.ModelProto:
    report = get_precision_report(proto)
    onnx.save_model(proto, path)
    saved = onnx.load(path)
    assert saved.SerializeToString() == proto.SerializeToString()
    for optimization in ("disabled", "all"):
        evidence = validate_source_samples(
            saved,
            samples=[
                SourceSample(
                    dict(
                        zip(
                            (port.name for port in saved.graph.input),
                            values,
                            strict=True,
                        )
                    ),
                    references,
                    _BITS,
                )
                for values, references in zip(feeds, expected, strict=True)
            ],
            optimization=optimization,
        )
        evidence.require_sample_acceptance()
    assert get_precision_report(saved) == report
    return saved


@pytest.mark.parametrize("ordinal", range(6))
@pytest.mark.parametrize("x64", [False, True])
def test_original_native_conv_rows_reach_normal_source_unchanged(
    ordinal: int, x64: bool
) -> None:
    metadata = source.raw_source_case("plugin::conv_general_dilated", ordinal)
    assert metadata["testcase"] == _ORIGINAL_CONV_NAMES[ordinal]
    scenarios = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::conv_general_dilated", ordinal)
    )
    scenario = scenarios[int(x64)]
    assert scenario.criteria is None
    result = source.run_source_case(metadata, scenario, optimization=None)
    assert result.model is not None, result.audit.detail
    assert result.audit.status is result.schema is CheckStatus.VALIDATED
    assert result.prepared is not None and result.prepared.criteria is None
    assert result.prepared.fn is metadata["callable"]
    assert result.raw_callback_status == "passed"
    assert result.numerical_status is CheckStatus.UNVERIFIED
    with source.source_configuration(scenario):
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = _trace(result.prepared.fn, result.prepared.input_specs)
    assert typed_trace_manifest(actual) == typed_trace_manifest(
        result.prepared.original_jaxpr
    )
    report = get_precision_report(result.model)
    assert report is not None and report.profile == "source-static-v1"
    assert metadata["post_check_onnx_graph"](result.model) is True


@pytest.mark.parametrize("ordinal", [3, 5])
@pytest.mark.parametrize("x64", [False, True])
def test_original_dyadic_conv_samples_keep_exact_saved_cpu_evidence(
    ordinal: int, x64: bool, tmp_path: Path
) -> None:
    metadata = source.raw_source_case("plugin::conv_general_dilated", ordinal)
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::conv_general_dilated", ordinal)
    )[int(x64)]
    prepared = source.prepare_source_case(metadata, scenario)
    assert scenario.criteria is prepared.criteria is None
    with source.source_configuration(scenario):
        compiled = tuple(
            np.asarray(value)
            for value in jax.tree.leaves(jax.jit(prepared.fn)(*prepared.arguments))
        )
        if ordinal == 3:
            manual = np.asarray([[[-1.5, -2.0, -2.5]]], np.float32)
        else:
            manual = np.full((1, 5, 5, 4), 12, np.float32)
            manual[:, -1, :, :] = 6
            manual[:, :, -1, :] = 6
            manual[:, -1, -1, :] = 3
        for value in (prepared.reference_outputs[0], compiled[0]):
            assert value.dtype == manual.dtype and value.shape == manual.shape
            assert value.tobytes() == manual.tobytes()
        artifact, proto = _product(prepared.fn, prepared.input_specs)
    assert sum(node.op_type == "Conv" for node in artifact.graph) == 1
    saved = _saved_samples(
        proto,
        [prepared.arguments],
        [prepared.reference_outputs],
        tmp_path / "original-conv.onnx",
    )
    assert metadata["post_check_onnx_graph"](saved) is True
    assert prepared.criteria is None


@pytest.mark.parametrize("spatial", [1, 2])
@pytest.mark.parametrize("x64", [False, True])
def test_grouped_dilated_conv_retains_one_actual_symbolic_batch_artifact(
    spatial: int, x64: bool, tmp_path: Path
) -> None:
    if spatial == 1:
        lhs_shape, rhs_shape = ("B", 7, 4), (2, 2, 6)
        layouts, strides, dilation, padding = (
            ("NWC", "WIO", "NWC"),
            (2,),
            (2,),
            ((1, 0),),
        )
    else:
        lhs_shape, rhs_shape = ("B", 4, 5, 6), (2, 2, 2, 6)
        layouts, strides, dilation, padding = (
            ("NCHW", "HWIO", "NHWC"),
            (2, 1),
            (1, 2),
            ((1, 0), (0, 1)),
        )

    def target(x: Any, kernel: Any) -> Any:
        return jax.lax.conv_general_dilated(
            x,
            kernel,
            strides,
            padding,
            rhs_dilation=dilation,
            feature_group_count=2,
            dimension_numbers=layouts,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=np.float32,
        )

    with jax.enable_x64(x64):
        (batch,) = export.symbolic_shape("B")
        specs = [
            jax.ShapeDtypeStruct(
                tuple(batch if dim == "B" else dim for dim in shape), np.float32
            )
            for shape in (lhs_shape, rhs_shape)
        ]
        native = _trace(target, specs)
        feeds, expected = [], []
        for size in (1, 3):
            values = tuple(
                ((np.arange(np.prod(shape)).reshape(shape) % 5 + 1) / 8).astype(
                    np.float32
                )
                for shape in (
                    tuple(size if dim == "B" else dim for dim in lhs_shape),
                    rhs_shape,
                )
            )
            eager, compiled = np.asarray(target(*values)), np.asarray(
                jax.jit(target)(*values)
            )
            assert eager.tobytes() == compiled.tobytes()
            feeds.append(values)
            expected.append([eager])
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = _trace(target, specs)
        assert typed_trace_manifest(actual) == typed_trace_manifest(native)
        _, proto = _product(target, specs)
    report = get_precision_report(proto)
    assert report is not None and report.exported_outputs[0].shape[0] == "B"
    conv = next(node for node in proto.graph.node if node.op_type == "Conv")
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in conv.attribute
    }
    assert attributes["group"] == 2
    assert tuple(attributes["strides"]) == strides
    assert tuple(attributes["dilations"]) == dilation
    assert tuple(attributes["pads"]) == tuple(pair[0] for pair in padding) + tuple(
        pair[1] for pair in padding
    )
    _saved_samples(proto, feeds, expected, tmp_path / "symbolic-conv.onnx")


@pytest.mark.parametrize("scope", ["function", "if", "scan"])
@pytest.mark.parametrize("x64", [False, True])
def test_native_conv_preserves_real_function_and_control_flow_scopes(
    scope: str, x64: bool, tmp_path: Path
) -> None:
    def target(x: Any, kernel: Any) -> Any:
        if scope == "function":
            return _named_native_conv(x, kernel)
        if scope == "if":
            return jax.lax.cond(
                x[0, 0, 0] > np.float32(0),
                lambda values: _named_native_conv(*values),
                lambda values: -_named_native_conv(*values),
                (x, kernel),
            )
        return jax.lax.scan(
            lambda state, _: (state, _named_native_conv(x, kernel)),
            np.int32(0),
            None,
            length=2,
        )[1]

    values = (
        np.asarray([[[1, 2, 3, 4]]], np.float32),
        np.asarray([[[0.5, -1]]], np.float32),
    )
    with jax.enable_x64(x64):
        eager, compiled = np.asarray(target(*values)), np.asarray(
            jax.jit(target)(*values)
        )
        assert eager.tobytes() == compiled.tobytes()
        _, proto = _product(
            target, [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
        )
    assert (
        sum(
            node.op_type == "Conv"
            for function in proto.functions
            for node in function.node
        )
        == 1
    )
    _saved_samples(proto, [values], [[eager]], tmp_path / "scoped-conv.onnx")


@pytest.mark.parametrize("family", ["nnx", "linen"])
def test_source_conv_wrappers_forward_actual_original_arguments(family: str) -> None:
    plugin = NNXConvPlugin if family == "nnx" else LinenConvPlugin
    marker, argument, extra, sharding = object(), object(), object(), object()
    calls = []

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return marker

    previous = plugin._ORIGINAL_CALL
    try:
        patched = plugin._make_patch(original)
        patched = plugin._make_patch(patched)
        with precision_policy_scope(PrecisionPolicy.SOURCE):
            assert (
                patched(marker, argument, extra=extra, out_sharding=sharding) is marker
            )
    finally:
        plugin._ORIGINAL_CALL = previous
    assert calls == [((marker, argument), {"extra": extra, "out_sharding": sharding})]


@pytest.mark.parametrize("family", ["nnx", "linen"])
@pytest.mark.parametrize("x64", [False, True])
def test_actual_framework_conv_keeps_mask_bias_promotion_and_native_trace(
    family: str, x64: bool
) -> None:
    calls = []

    def promote(*values: Any, dtype: Any = None, **kwargs: Any) -> Any:
        calls.append(dtype)
        if len(values) == 1 and isinstance(values[0], tuple):
            values = values[0]
        return tuple(
            None if value is None else jnp.asarray(value, dtype=dtype)
            for value in values
        )

    with jax.enable_x64(x64):
        mask = np.ones((2, 2, 2, 4), np.float32)
        mask[0, 0, :, :] = 0
        options = dict(
            kernel_size=(2, 2),
            padding="SAME",
            dtype=np.float32,
            param_dtype=np.float32,
            mask=mask,
            promote_dtype=promote,
        )
        values = np.asarray(
            np.arange(2 * 4 * 5 * 2).reshape(2, 4, 5, 2) / 16, np.float32
        )
        if family == "nnx":
            layer = nnx.Conv(2, 4, rngs=nnx.Rngs(0), **options)
            layer.bias[...] = np.asarray([0.25, 0.5, 0.75, 1], np.float32)
            target = layer
            kernel_before, bias_before = (
                np.asarray(layer.kernel[...]).copy(),
                np.asarray(layer.bias[...]).copy(),
            )
        else:
            layer = nn.Conv(features=4, **options)
            parameters = layer.init(jax.random.key(0), values)
            parameters["params"]["bias"] = jnp.asarray([0.25, 0.5, 0.75, 1], np.float32)

            def target(x: Any) -> Any:
                return layer.apply(parameters, x)

            kernel_before, bias_before = (
                np.asarray(parameters["params"]["kernel"]).copy(),
                np.asarray(parameters["params"]["bias"]).copy(),
            )
        specs = [jax.ShapeDtypeStruct(values.shape, values.dtype)]
        before_trace = len(calls)
        native = _trace(target, specs)
        assert len(calls) > before_trace
        eager = np.asarray(target(values))
        before_trace = len(calls)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = _trace(target, specs)
        assert len(calls) > before_trace
        before_trace = len(calls)
        with precision_policy_scope(PrecisionPolicy.LEGACY), _activate_plugin_worlds():
            legacy = _trace(target, specs)
        assert len(calls) > before_trace
        assert calls and all(dtype == np.float32 for dtype in calls)
        assert typed_trace_manifest(actual) == typed_trace_manifest(native)
        assert not {"nnx.conv", "linen.conv"} & {
            eqn.primitive.name for eqn in actual.jaxpr.eqns
        }
        assert f"{family}.conv" in {eqn.primitive.name for eqn in legacy.jaxpr.eqns}
        assert _signature(legacy.jaxpr.outvars) == _signature(native.jaxpr.outvars)
        before_trace = len(calls)
        assert typed_trace_manifest(_trace(target, specs)) == typed_trace_manifest(
            native
        )
        assert len(calls) > before_trace
        _, proto = _product(target, specs)
        assert _signature(native.jaxpr.outvars) == _signature(actual.jaxpr.outvars)
        assert eager.shape == (2, 4, 5, 4) and eager.dtype == np.float32
        if family == "nnx":
            kernel_after, bias_after = np.asarray(layer.kernel[...]), np.asarray(
                layer.bias[...]
            )
        else:
            kernel_after, bias_after = np.asarray(
                parameters["params"]["kernel"]
            ), np.asarray(parameters["params"]["bias"])
    assert kernel_before.tobytes() == kernel_after.tobytes()
    assert bias_before.tobytes() == bias_after.tobytes()
    assert sum(node.op_type == "Conv" for node in proto.graph.node) == 1


@pytest.mark.parametrize("family", ["nnx", "linen"])
def test_framework_high_precision_reaches_native_source_rejection(family: str) -> None:
    values = np.ones((1, 4, 4, 2), np.float32)
    options = dict(kernel_size=(2, 2), precision=jax.lax.Precision.HIGHEST)
    if family == "nnx":
        target = nnx.Conv(2, 4, rngs=nnx.Rngs(0), **options)
    else:
        layer = nn.Conv(features=4, **options)
        parameters = layer.init(jax.random.key(0), values)

        def target(x: Any) -> Any:
            return layer.apply(parameters, x)

    with pytest.raises(SourcePrecisionError, match="precision") as error:
        ui._to_onnx(
            target,
            [jax.ShapeDtypeStruct(values.shape, values.dtype)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert error.value.primitive == "conv_general_dilated"
    assert error.value.category == "profile-exclusion"


def _native_equation() -> Any:
    target = source.raw_source_case("plugin::conv_general_dilated", 3)["callable"]
    graph = jax.make_jaxpr(target)(
        jax.ShapeDtypeStruct((1, 1, 4), np.float32),
        jax.ShapeDtypeStruct((1, 1, 2), np.float32),
    )
    return graph.jaxpr.eqns[0]


@pytest.mark.parametrize(
    "left_weak,right_weak", [(False, False), (False, True), (True, False), (True, True)]
)
def test_native_conv_preserves_actual_weak_operand_and_result_contract(
    left_weak: bool, right_weak: bool
) -> None:
    target = source.raw_source_case("plugin::conv_general_dilated", 3)["callable"]
    specs = [
        jax.ShapeDtypeStruct((1, 1, 4), np.float32, weak_type=left_weak),
        jax.ShapeDtypeStruct((1, 1, 2), np.float32, weak_type=right_weak),
    ]
    native = _trace(target, specs)
    _, proto = _product(target, specs)
    report = get_precision_report(proto)
    assert report is not None
    assert report.traced_outputs[0].weak_type is native.jaxpr.outvars[0].aval.weak_type
    assert report.traced_outputs[0].weak_type == (left_weak and right_weak)


@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.float64, np.complex64])
def test_native_conv_keeps_other_precision_routes_excluded(dtype: Any) -> None:
    def target(x: Any, kernel: Any) -> Any:
        return jax.lax.conv_general_dilated(x, kernel, (1,), "VALID")

    with jax.enable_x64(True):
        with pytest.raises(SourcePrecisionError) as error:
            ui._to_onnx(
                target,
                [
                    jax.ShapeDtypeStruct((1, 1, 4), dtype),
                    jax.ShapeDtypeStruct((1, 1, 2), dtype),
                ],
                precision_policy=PrecisionPolicy.SOURCE,
            )
    assert error.value.primitive == "conv_general_dilated"
    assert error.value.category == "profile-exclusion"


@pytest.mark.parametrize("opset", [16, 18, 21, 24, 26, 27])
def test_native_conv_keeps_other_opset_routes_excluded(opset: int) -> None:
    equation = _native_equation()
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(
            SimpleNamespace(
                invars=equation.invars,
                constvars=(),
                outvars=equation.outvars,
                eqns=[equation],
                effects=(),
            ),
            opset,
        )
    assert error.value.primitive == "conv_general_dilated"
    assert error.value.category == "profile-exclusion"


@pytest.mark.parametrize(
    "mutation",
    [
        "foreign-primitive",
        "missing-field",
        "extra-field",
        "input-arity",
        "output-arity",
        "mixed-dtype",
        "output-dtype",
        "output-shape",
        "output-weakness",
        "dimension-type",
        "duplicate-axis",
        "unsupported-layout",
        "precision-high",
        "precision-algorithm",
        "preferred-wide",
        "placement",
        "stride-boolean",
        "stride-zero",
        "stride-rank",
        "rhs-dilation-zero",
        "lhs-dilation",
        "negative-padding",
        "string-padding",
        "group-boolean",
        "group-zero",
        "group-channels",
        "batch-groups",
    ],
)
def test_native_conv_schema_mutations_fail_before_lowering(mutation: str) -> None:
    native = _native_equation()
    primitive, params = native.primitive, dict(native.params)
    inputs, outputs = list(native.invars), list(native.outvars)
    if mutation == "foreign-primitive":
        primitive = SimpleNamespace(name="conv_general_dilated")
    elif mutation == "missing-field":
        params.pop("padding")
    elif mutation == "extra-field":
        params["unreviewed"] = None
    elif mutation == "input-arity":
        inputs.pop()
    elif mutation == "output-arity":
        outputs.append(Var(ShapedArray((1, 1, 3), np.float32)))
    elif mutation == "mixed-dtype":
        inputs[1] = Var(ShapedArray((1, 1, 2), np.float64))
    elif mutation.startswith("output-"):
        outputs[0] = Var(
            ShapedArray(
                (1, 1, 4) if mutation == "output-shape" else (1, 1, 3),
                np.float64 if mutation == "output-dtype" else np.float32,
                weak_type=mutation == "output-weakness",
            )
        )
    elif mutation == "dimension-type":
        params["dimension_numbers"] = tuple(params["dimension_numbers"])
    elif mutation == "duplicate-axis":
        params["dimension_numbers"] = jax.lax.ConvDimensionNumbers(
            (0, 1, 1), (0, 1, 2), (0, 1, 2)
        )
    elif mutation == "unsupported-layout":
        params["dimension_numbers"] = jax.lax.ConvDimensionNumbers(
            (1, 0, 2), (0, 1, 2), (0, 1, 2)
        )
    elif mutation == "precision-high":
        params["precision"] = (jax.lax.Precision.HIGH,) * 2
    elif mutation == "precision-algorithm":
        params["precision"] = "F32_F32_F32"
    elif mutation == "preferred-wide":
        params["preferred_element_type"] = np.dtype(np.float64)
    elif mutation == "placement":
        params["out_sharding"] = object()
    elif mutation.startswith("stride-"):
        params["window_strides"] = {
            "stride-boolean": (True,),
            "stride-zero": (0,),
            "stride-rank": (1, 1),
        }[mutation]
    elif mutation == "rhs-dilation-zero":
        params["rhs_dilation"] = (0,)
    elif mutation == "lhs-dilation":
        params["lhs_dilation"] = (2,)
    elif mutation == "negative-padding":
        params["padding"] = ((-1, 0),)
    elif mutation == "string-padding":
        params["padding"] = "SAME"
    elif mutation.startswith("group-"):
        params["feature_group_count"] = {
            "group-boolean": True,
            "group-zero": 0,
            "group-channels": 2,
        }[mutation]
    else:
        params["batch_group_count"] = 2
    equation = SimpleNamespace(
        primitive=primitive, invars=inputs, outvars=outputs, params=params, effects=()
    )
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(
            SimpleNamespace(
                invars=inputs,
                constvars=(),
                outvars=outputs,
                eqns=[equation],
                effects=(),
            ),
            23,
        )
    assert error.value.primitive == "conv_general_dilated"
    assert error.value.category in {"profile-exclusion", "lowering-defect"}


@pytest.mark.parametrize("defect", ["dtype", "shape", "missing-shape"])
def test_source_conv_rejects_physical_storage_drift_before_emission(
    defect: str,
) -> None:
    equation = _native_equation()
    values = [
        ir.val(f"value{index}", dtype=ir.DataType.FLOAT, shape=var.aval.shape)
        for index, var in enumerate((*equation.invars, *equation.outvars))
    ]
    if defect == "dtype":
        values[0].type = ir.TensorType(ir.DataType.DOUBLE)
    elif defect == "shape":
        values[0].shape = ir.Shape((1, 1, 5))
    else:
        values[0].shape = None
    bindings = dict(
        zip(map(id, (*equation.invars, *equation.outvars)), values, strict=True)
    )
    context = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        fresh_name=lambda name: name,
        get_value_for_var=lambda var, **kwargs: bindings[id(var)],
    )
    with pytest.raises(TypeError, match="SOURCE convolution storage"):
        ConvGeneralDilatedPlugin().lower(context, equation)


@pytest.mark.parametrize("ordinal", range(6, 17))
def test_all_other_original_conv_rows_remain_explicit_unaccepted_obligations(
    ordinal: int,
) -> None:
    metadata = source.raw_source_case("plugin::conv_general_dilated", ordinal)
    assert metadata["testcase"] == _ORIGINAL_CONV_NAMES[ordinal]
    assert metadata["post_check_onnx_graph"] is not None
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance("plugin::conv_general_dilated", ordinal)
    )[1]
    if ordinal == 12:
        scenario = replace(scenario, concrete_shapes=((2, 2, 3, 3), (3, 2, 2, 2)))
    prepared = source.prepare_source_case(metadata, scenario)
    assert scenario.criteria is prepared.criteria is None
    with source.source_configuration(scenario):
        with pytest.raises(SourcePrecisionError) as error:
            ui._to_onnx(
                prepared.fn,
                prepared.input_specs,
                opset=23,
                precision_policy=PrecisionPolicy.SOURCE,
            )
    assert error.value.primitive == "conv_general_dilated"
    assert error.value.category == "profile-exclusion"


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("x64", [False, True])
def test_unchanged_complete_cnn_exports_source_with_separate_raw_callback(
    ordinal: int, x64: bool, tmp_path: Path, record_property: Any
) -> None:
    from tests.extra_tests.converter.test_source_native_cnn_pool import (
        assert_original_cnn_source,
    )

    assert_original_cnn_source(ordinal, x64, tmp_path, record_property)
