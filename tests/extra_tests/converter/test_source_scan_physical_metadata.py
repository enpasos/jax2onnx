# tests/extra_tests/converter/test_source_scan_physical_metadata.py

"""Retain native physical metadata until Scan consumers have finished lowering."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import jax
from jax import export
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.converter.source_representation import (
    normalized_shape_dimensions,
    source_shape_contract,
)
from jax2onnx.plugins.jax.lax.conv import ConvGeneralDilatedPlugin
from jax2onnx.plugins.jax.lax.scan import _relax_value_for_var
from jax2onnx.plugins.plugin_system import (
    FunctionPlugin,
    PLUGIN_REGISTRY,
    _SourceFunctionTrace,
    apply_monkey_patches,
)
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter import (
    test_source_complex_loop_precision as complex_cases,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_loop_admission import _loop_interfaces
from tests.extra_tests.converter.test_source_native_conv import _native_equation
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _globals,
    _settings,
    _restore_state as _restore_state,
)


@onnx_function(name="SourceScanPhysicalConv")
def _conv(x: Any, kernel: Any) -> Any:
    return jax.lax.conv_general_dilated(
        x, kernel, (1,), "VALID", dimension_numbers=("NCW", "OIW", "NCW")
    )


@onnx_function(name="SourceScanPhysicalFP64Matrix")
def _matrix(x: Any, kernel: Any) -> Any:
    return jax.lax.dot_general(x, kernel, (((1,), (0,)), ((), ())))


@onnx_function(name="SourceScanPhysicalComplexChannels")
def _channels(value: Any) -> Any:
    real, imaginary = jax.lax.real(value), jax.lax.imag(value)
    return jax.lax.complex(imaginary, real), real, imaginary


def _trace(fn: Any, specs: Any) -> Any:
    def fresh(*values: Any) -> Any:
        return fn(*values)

    return jax.make_jaxpr(fresh)(*specs)


def _signature(variables: Any) -> Any:
    return tuple(
        (np.dtype(value.aval.dtype).name, tuple(value.aval.shape), value.aval.weak_type)
        for value in variables
    )


def _trace_facts(trace: Any) -> Any:
    assert type(trace) is _SourceFunctionTrace
    assert set(vars(trace)) == {
        "output_avals",
        "x64_enabled",
        "default_matmul_precision",
        "numpy_dtype_promotion",
        "numpy_rank_promotion",
        "debug_key_reuse",
    }
    return {
        "output_avals": _signature(
            [SimpleNamespace(aval=aval) for aval in trace.output_avals]
        ),
        "x64_enabled": trace.x64_enabled,
        "default_matmul_precision": trace.default_matmul_precision,
        "numpy_dtype_promotion": trace.numpy_dtype_promotion,
        "numpy_rank_promotion": trace.numpy_rank_promotion,
        "debug_key_reuse": trace.debug_key_reuse,
    }


def _carrier_manifest(closed: Any) -> Any:
    # Preserve the complete trace; normalize only the known Function context
    # record that generic typed_trace_manifest intentionally leaves unresolved.
    def normalize(value: Any) -> Any:
        if isinstance(value, _SourceFunctionTrace):
            return _trace_facts(value)
        body = getattr(value, "jaxpr", value)
        if all(hasattr(body, name) for name in ("eqns", "invars", "outvars")):
            return SimpleNamespace(
                consts=getattr(value, "consts", ()),
                jaxpr=SimpleNamespace(
                    invars=body.invars,
                    outvars=body.outvars,
                    constvars=body.constvars,
                    effects=body.effects,
                    eqns=[
                        SimpleNamespace(
                            primitive=eqn.primitive,
                            invars=eqn.invars,
                            outvars=eqn.outvars,
                            effects=eqn.effects,
                            params=normalize(eqn.params),
                        )
                        for eqn in body.eqns
                    ],
                ),
            )
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(normalize(item) for item in value)
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return typed_trace_manifest(normalize(closed))


def _native_function_references(boundary: Any, originals: Any) -> Any:
    # These finite fixtures contain one Scan whose child calls only the three
    # named free functions above. Trace each original callee before full binding
    # activation, retaining its complete native attributes/constants/aliases.
    assert len(boundary.jaxpr.eqns) == 1
    scan = boundary.jaxpr.eqns[0]
    assert scan.primitive is jax.lax.scan_p
    references: dict[str, Any] = {}
    expected_primitives = (
        (jax.lax.conv_general_dilated_p,),
        (jax.lax.dot_general_p,),
        (jax.lax.real_p, jax.lax.imag_p, jax.lax.complex_p),
    )
    for equation in scan.params["jaxpr"].jaxpr.eqns:
        assert set(equation.params) == {"__jax2onnx_source_trace"}
        plugin = PLUGIN_REGISTRY.get(equation.primitive.name)
        assert isinstance(plugin, FunctionPlugin)
        assert equation.primitive is plugin._source_primitive
        index = next(i for i, target in enumerate(originals) if plugin.target is target)
        trace = equation.params["__jax2onnx_source_trace"]
        specs = [
            jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype, weak_type=v.aval.weak_type)
            for v in equation.invars
        ]
        # The boundary record is checked against the independent caller, rather
        # than letting that record select its own native reference settings.
        assert (
            trace.x64_enabled,
            trace.default_matmul_precision,
            trace.numpy_dtype_promotion,
            trace.numpy_rank_promotion,
            trace.debug_key_reuse,
        ) == (
            bool(jax.config.x64_enabled),
            jax.config.jax_default_matmul_precision,
            str(jax.config.jax_numpy_dtype_promotion),
            str(jax.config.jax_numpy_rank_promotion),
            bool(jax.config.jax_debug_key_reuse),
        )
        native = _trace(originals[index], specs)
        assert (
            tuple(eqn.primitive for eqn in native.jaxpr.eqns)
            == expected_primitives[index]
        )
        manifest = typed_trace_manifest(native)
        assert manifest["unresolved_parameters"] == []
        assert _signature(native.jaxpr.invars) == _signature(equation.invars)
        assert _signature(native.jaxpr.outvars) == _signature(equation.outvars)
        reference = (
            originals[index],
            _signature(equation.invars),
            _trace_facts(trace),
            manifest,
        )
        if plugin.name in references:
            assert references[plugin.name] == reference
        references[plugin.name] = reference
    assert references
    return references


def _check_native_scan_interface(native: Any, boundary: Any) -> None:
    # Named wrapping changes child equation grouping, while the native outer
    # topology, operands, roles, constants, settings and typed ports must agree.
    inline, carried = typed_trace_manifest(native), _carrier_manifest(boundary)
    for field in (
        "inputs",
        "consts",
        "const_payloads",
        "outputs",
        "effects",
        "unresolved_parameters",
    ):
        assert inline[field] == carried[field]
    assert len(inline["equations"]) == len(carried["equations"]) == 1
    left, right = inline["equations"][0], carried["equations"][0]
    for field in ("primitive", "inputs", "outputs", "effects", "scan_role_arity"):
        assert left[field] == right[field]
    assert left["params"].keys() == right["params"].keys()
    for name in left["params"]:
        if name != "jaxpr":
            assert left["params"][name] == right["params"][name]
    left_body, right_body = left["params"]["jaxpr"], right["params"]["jaxpr"]
    for field in ("inputs", "consts", "const_payloads", "effects"):
        assert left_body[field] == right_body[field]
    assert [value["aval"] for value in left_body["outputs"]] == [
        value["aval"] for value in right_body["outputs"]
    ]


def _loop_bodies(nodes: Any) -> Any:
    for node in nodes:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                if node.op_type == "Loop":
                    yield attribute.g
                yield from _loop_bodies(attribute.g.node)


def _product(fn: Any, values: Any, path: Path) -> tuple[Any, Any]:
    specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
    native = _trace(fn, specs)
    eager = [np.asarray(value) for value in jax.tree.leaves(fn(*values))]
    compiled = [
        np.asarray(value)
        for value in jax.tree.leaves(jax.jit(lambda *args: fn(*args))(*values))
    ]
    for actual, expected in zip(compiled, eager, strict=True):
        complex_cases._exact(actual, expected)
    originals = (_conv, _matrix, _channels)
    # Existing patch_info boundaries package the native callees; binding_specs
    # remain inactive. These three callees use only actual lax primitives.
    with precision_policy_scope(PrecisionPolicy.SOURCE), apply_monkey_patches():
        boundary = _trace(fn, specs)
    _check_native_scan_interface(native, boundary)
    references = _native_function_references(boundary, originals)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        actual_trace = _trace(fn, specs)
    # This compares all ordered call-site bindings and output aliases, rather
    # than equating inline native operations with an opaque Function marker.
    assert _carrier_manifest(actual_trace) == _carrier_manifest(boundary)
    observed = set()
    original_trace_body = FunctionPlugin._trace_function_body

    def observe_body(
        plugin: Any, callee: Any, equation: Any, captures: Any, **kwargs: Any
    ) -> Any:
        target, signature, facts, expected = references[plugin.name]
        assert callee is target
        assert captures.dynamic == () and dict(captures.static_kwargs) == {}
        assert kwargs["precision_policy"] is PrecisionPolicy.SOURCE
        assert _signature(equation.invars) == signature
        assert _trace_facts(kwargs["source_trace"]) == facts
        actual = original_trace_body(plugin, callee, equation, captures, **kwargs)
        assert typed_trace_manifest(actual) == expected
        observed.add(plugin.name)
        return actual  # Exactly the original returned object, unchanged.

    before = _globals(), _settings()
    with patch.object(FunctionPlugin, "_trace_function_body", observe_body):
        artifact = ui._to_onnx(
            fn,
            specs,
            opset=23,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
        )
    assert FunctionPlugin._trace_function_body is original_trace_body
    assert observed == references.keys()
    assert isinstance(artifact, ir.Model)
    assert (_globals(), _settings()) == before
    proto = ir.to_proto(artifact)
    onnx.save_model(proto, path)
    payload = path.read_bytes()
    saved = onnx.load(path)
    assert saved.SerializeToString() == payload
    onnx.checker.check_model(saved, full_check=True)
    _loop_interfaces(saved)
    report = get_precision_report(saved)
    assert report is not None and report.profile == "source-static-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    sample = SourceSample(
        [complex_cases._physical(value) for value in values],
        [complex_cases._physical(value) for value in eager],
        SampleCriteria("bitwise"),
    )
    for optimization in ("disabled", "all"):
        evidence = validate_source_samples(
            path, samples=[sample], optimization=optimization
        )
        evidence.require_sample_acceptance()
    assert path.read_bytes() == payload and get_precision_report(saved) == report
    # The existing late pass still loosens internal Loop value_info. Interfaces
    # and Function formals keep the physical contract needed by their consumers.
    bodies = list(_loop_bodies(saved.graph.node))
    assert bodies
    for body in bodies:
        for value in body.value_info:
            assert all(
                not dimension.HasField("dim_value")
                and not dimension.HasField("dim_param")
                for dimension in value.type.tensor_type.shape.dim
            )
    return artifact, saved


@pytest.mark.parametrize("role", ["carry", "xs"])
@pytest.mark.parametrize("x64", [False, True])
def test_real_scan_function_conv_preserves_carry_and_sliced_input_shapes(
    role: str, x64: bool, tmp_path: Path
) -> None:
    def target(x: Any, kernel: Any) -> Any:
        if role == "carry":

            def step(state: Any, unused: Any) -> Any:
                result = _conv(state, kernel)
                return result, result

            return jax.lax.scan(step, x, None, length=2)
        return jax.lax.scan(
            lambda state, item: (state, _conv(item, kernel)), np.int32(0), x
        )

    x = np.asarray([[[1, 2, 3, 4]]], np.float32)
    values = (
        x if role == "carry" else np.stack([x, 2 * x]),
        np.asarray([[[0.5]]], np.float32),
    )
    with jax.enable_x64(x64):
        artifact, proto = _product(target, values, tmp_path / "scan-conv.onnx")
    functions = [
        function
        for function in artifact.functions.values()
        if any(node.op_type == "Conv" for node in function.graph)
    ]
    assert len(functions) == 1
    assert [
        normalized_shape_dimensions(value.shape.dims)
        for value in functions[0].graph.inputs
    ] == [(1, 1, 4), (1, 1, 1)]
    assert (
        sum(
            node.op_type == "Conv"
            for function in proto.functions
            for node in function.node
        )
        == 1
    )
    if role == "xs":
        # The Gather result was known while the Function lowered, and is an
        # internal rank-only value after the existing late postprocessing pass.
        bodies = list(_loop_bodies(proto.graph.node))
        gathered = {
            name
            for body in bodies
            for node in body.node
            if node.op_type == "Gather"
            for name in node.output
        }
        assert any(
            value.name in gathered for body in bodies for value in body.value_info
        )


@pytest.mark.parametrize("with_xs", [False, True])
def test_named_fp64_matrix_in_scan_keeps_actual_double_physical_shapes(
    with_xs: bool, tmp_path: Path
) -> None:
    def target(x: Any, kernel: Any, sequence: Any = None) -> Any:
        def step(state: Any, item: Any) -> Any:
            result = _matrix(state, kernel)
            emitted = result if item is None else _matrix(item, kernel)
            return result, emitted

        return jax.lax.scan(step, x, sequence, length=2)

    values = (
        np.asarray([[1 + 2**-40, -2 - 2**-39, 3, 4]], np.float64),
        np.diag(np.asarray([1, -1, 0.5, 2], np.float64)),
        np.asarray([[[1 + 2**-40, -2 - 2**-39, 3, 4]], [[2, 3, 4, 5]]], np.float64),
    )
    with jax.enable_x64(True):
        artifact, proto = _product(
            target, values if with_xs else values[:2], tmp_path / "scan-fp64.onnx"
        )
    functions = [
        function
        for function in artifact.functions.values()
        if any(node.op_type == "MatMul" for node in function.graph)
    ]
    assert len(functions) == 1
    assert all(value.dtype == ir.DataType.DOUBLE for value in functions[0].graph.inputs)
    assert [
        normalized_shape_dimensions(value.shape.dims)
        for value in functions[0].graph.inputs
    ] == [(1, 4), (4, 4)]
    assert not {"Cast", "CastLike"} & {
        node.op_type for function in proto.functions for node in function.node
    }


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("with_xs", [False, True])
def test_packed_complex_scan_carry_and_slices_retain_both_named_function_channels(
    dtype: Any, with_xs: bool, tmp_path: Path
) -> None:
    def target(initial: Any, other: Any) -> Any:
        def step(state: Any, item: Any) -> Any:
            carry, real, imaginary = _channels(state)
            emitted = _channels(other if item is None else item)[0]
            return carry, (emitted, real, imaginary)

        return jax.lax.scan(step, initial, other if with_xs else None, length=2)

    first = complex_cases._payload(dtype, (3,))
    second = complex_cases._payload(dtype, (3,), other=True)
    with jax.enable_x64(True):
        artifact, _ = _product(
            target,
            (first, np.stack([first, second]) if with_xs else second),
            tmp_path / "scan-complex.onnx",
        )
    functions = [
        function
        for function in artifact.functions.values()
        if function.name.startswith("SourceScanPhysicalComplexChannels")
    ]
    assert functions
    for function in functions:
        assert normalized_shape_dimensions(function.graph.inputs[0].shape.dims) == (
            3,
            2,
        )
        assert function.graph.inputs[0].dtype == (
            ir.DataType.FLOAT if dtype == np.complex64 else ir.DataType.DOUBLE
        )


@pytest.mark.parametrize("x64", [False, True])
def test_scan_keeps_shared_symbolic_batch_as_an_explicit_unadmitted_contract(
    x64: bool,
) -> None:
    def target(x: Any, kernel: Any) -> Any:
        return jax.lax.scan(
            lambda state, unused: (state, _conv(x, kernel)), np.int32(0), None, length=2
        )[1]

    (batch,) = export.symbolic_shape("B")
    specs = [
        jax.ShapeDtypeStruct((batch, 1, 4), np.float32),
        jax.ShapeDtypeStruct((1, 1, 1), np.float32),
    ]
    with jax.enable_x64(x64):
        native = _trace(target, specs)
        with pytest.raises(SourcePrecisionError) as error:
            ui._to_onnx(
                target, specs, opset=23, precision_policy=PrecisionPolicy.SOURCE
            )
    assert (
        error.value.primitive == "scan" and error.value.category == "profile-exclusion"
    )
    shape = tuple(native.jaxpr.outvars[0].aval.shape)
    assert shape == (2, batch, 1, 4)
    with pytest.raises(ValueError):
        source_shape_contract(shape)
    assert str(shape) in error.value.detail


@pytest.mark.parametrize("defect", ["extent", "unknown", "dtype"])
def test_preserving_source_scan_metadata_does_not_repair_actual_conv_storage_drift(
    defect: str,
) -> None:
    equation = _native_equation()
    values = [
        ir.Value(type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(var.aval.shape))
        for var in (*equation.invars, *equation.outvars)
    ]
    if defect == "extent":
        values[0].shape = ir.Shape((1, 1, 5))
    elif defect == "unknown":
        values[0].shape = ir.Shape((None, None, None))
    else:
        values[0].type = ir.TensorType(ir.DataType.DOUBLE)
    bindings = dict(
        zip(map(id, (*equation.invars, *equation.outvars)), values, strict=True)
    )
    context = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        fresh_name=lambda name: name,
        get_value_for_var=lambda var, **kwargs: bindings[id(var)],
    )
    before = values[0].type, values[0].shape
    _relax_value_for_var(context, values[0], equation.invars[0])
    assert (values[0].type, values[0].shape) == before
    with pytest.raises((TypeError, ValueError)):
        ConvGeneralDilatedPlugin().lower(context, equation)


def test_legacy_scan_shape_relaxation_is_unchanged() -> None:
    value = ir.Value(type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 1, 4)))
    _relax_value_for_var(
        SimpleNamespace(precision_policy=PrecisionPolicy.LEGACY), value, None
    )
    assert value.dtype == ir.DataType.FLOAT
    assert len(value.shape) == 3 and all(
        dimension.value is None for dimension in value.shape
    )
