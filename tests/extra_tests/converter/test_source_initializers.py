# tests/extra_tests/converter/test_source_initializers.py

"""Canonical SOURCE initializer payloads and one audited plugin regression.

The owning-plugin initializer fixture is explicit and test-only. Native
F16/F32/F64 exports remain separate; native BF16 product rejection is retained.
"""

from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import onnx_function
from jax2onnx.converter import conversion_api
from jax2onnx.converter.function_scope import FunctionScope
from jax2onnx.converter.ir_builder import IRBuilder
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.ir_utils import const_value_to_numpy, tensor_attr
from jax2onnx.plugins.jax.nn.relu6 import Relu6Plugin


@pytest.fixture(autouse=True)
def _restore_jax_settings() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    previous_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", None)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)
        jax.config.update("jax_default_matmul_precision", previous_precision)


_FLOAT_DTYPES = (np.float16, jnp.bfloat16, np.float32, np.float64)
_DTYPES = (*_FLOAT_DTYPES, np.int32, np.int64)


def _builder(entrypoint: str = "helper") -> IRBuilder:
    builder = IRBuilder(
        opset=23,
        enable_double_precision=False,
        precision_policy=PrecisionPolicy.SOURCE,
    )
    builder._function_mode = entrypoint == "function"
    return builder


def _context() -> IRContext:
    return IRContext(
        opset=23,
        enable_double_precision=False,
        input_specs=[],
        precision_policy=PrecisionPolicy.SOURCE,
    )


def _assert_payload(actual: np.ndarray | None, expected: np.ndarray) -> None:
    assert actual is not None
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("scope", ["graph", "function", "control_flow"])
@pytest.mark.parametrize("kind", ["array", "scalar"])
def test_source_initializer_preserves_typed_payload_in_every_builder_scope(
    dtype: Any, scope: str, kind: str
) -> None:
    ctx = _context()
    if scope != "graph":
        ctx = ctx.create_child_context(scope=scope)
    data = np.asarray([1 + 2**-30, -0.0, -2.5], dtype=dtype)
    expected = data if kind == "array" else data[0:1].reshape(())
    if kind == "array":
        value = ctx.builder.add_initializer_from_array(name="typed", array=expected)
    else:
        value = ctx.builder.add_initializer_from_scalar(
            name="typed", value=expected[()]
        )
    assert value.dtype == ir.DataType.from_numpy(expected.dtype)
    _assert_payload(const_value_to_numpy(value), expected)
    if scope == "graph":
        assert ctx.builder.graph.initializers["typed"] is value
    else:
        assert not ctx.builder.initializers
        assert value.producer().op_type == "Constant"


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_source_serialization_and_function_clone_preserve_exact_payload(
    dtype: Any,
) -> None:
    payload = np.asarray([1 + 2**-30, -0.0, -2.5], dtype=dtype)
    ctx = _context()
    value = ctx.builder.add_initializer_from_array(name="typed", array=payload)
    ctx.builder.outputs.append(value)
    model = ctx.builder.to_ir_model(name="typed_initializer")
    _assert_payload(const_value_to_numpy(model.graph.outputs[0]), payload)
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    _assert_payload(onnx.numpy_helper.to_array(proto.graph.initializer[0]), payload)

    scope = FunctionScope(_context(), name="typed_function", domain="test.source")
    scope.begin([])
    constant = scope.ctx.builder.add_initializer_from_array(name="typed", array=payload)
    function = scope.end([constant])
    cloned = next(iter(function.graph)).attributes["value"].as_tensor().numpy()
    _assert_payload(cloned, payload)


@pytest.mark.parametrize("factory", [IRBuilder, IRContext])
def test_source_rejects_legacy_double_promotion(factory: Any) -> None:
    with pytest.raises(ValueError, match="Source precision.*legacy double"):
        factory(
            opset=23,
            enable_double_precision=True,
            precision_policy=PrecisionPolicy.SOURCE,
        )


@pytest.mark.parametrize(
    "value", [1.0, [1.0, 2.0], (1, 2.0), [np.float32(1), np.float32(2)]]
)
@pytest.mark.parametrize("entrypoint", ["scalar", "array"])
def test_source_floating_coefficients_require_an_explicit_container_dtype(
    value: Any, entrypoint: str
) -> None:
    builder = _builder()
    with pytest.raises(TypeError, match="explicitly typed.*match the traced"):
        if entrypoint == "scalar":
            builder.add_initializer_from_scalar(name="untyped", value=value)
        else:
            builder.add_initializer_from_array(name="untyped", array=value)
    assert not builder.initializers
    assert not list(builder.nodes)


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_source_accepts_explicitly_typed_jax_scalar(dtype: Any) -> None:
    scalar = jnp.asarray(1 + 2**-30, dtype=dtype)
    value = _builder().add_initializer_from_scalar(name="typed", value=scalar)
    _assert_payload(const_value_to_numpy(value), np.asarray(scalar))


def _insert(builder: IRBuilder, payload: np.ndarray, entrypoint: str) -> ir.Value:
    if entrypoint != "list":
        return builder.add_initializer_from_array(name="typed", array=payload)
    value = ir.Value(
        name="typed",
        shape=ir.Shape(payload.shape),
        type=ir.TensorType(ir.DataType.from_numpy(payload.dtype)),
        const_value=ir.tensor(payload),
    )
    builder.initializers.append(value)
    return builder.graph.initializers["typed"]


@pytest.mark.parametrize("entrypoint", ["helper", "list", "function"])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_source_duplicate_reuses_exact_typed_payload_and_connections(
    dtype: Any, entrypoint: str
) -> None:
    builder = _builder(entrypoint)
    payload = np.asarray([[1, 2], [3, 4]], dtype=dtype)
    original = _insert(builder, payload, entrypoint)
    consumer = builder.op("Identity", [original])
    # Physical array layout does not alter the serialized logical payload.
    reused = _insert(builder, np.asfortranarray(payload), entrypoint)
    assert reused is original
    assert consumer.producer().inputs[0] is original
    assert len(builder.initializers) == (0 if entrypoint == "function" else 1)


@pytest.mark.parametrize("entrypoint", ["helper", "list", "function"])
@pytest.mark.parametrize("difference", ["dtype", "shape", "negative-zero", "precision"])
def test_source_duplicate_rejects_distinct_typed_payloads(
    entrypoint: str, difference: str
) -> None:
    builder = _builder(entrypoint)
    original = np.asarray([0.0, 1.0], dtype=np.float64)
    changed = {
        "dtype": original.astype(np.float32),
        "shape": original.reshape(1, 2),
        "negative-zero": np.asarray([-0.0, 1.0], dtype=np.float64),
        "precision": np.asarray([0.0, 1 + 2**-30], dtype=np.float64),
    }[difference]
    stored = _insert(builder, original, entrypoint)
    with pytest.raises(ValueError, match="already exists with different data"):
        _insert(builder, changed, entrypoint)
    if entrypoint != "function":
        assert builder.graph.initializers["typed"] is stored
    _assert_payload(const_value_to_numpy(stored), original)


@pytest.mark.parametrize("entrypoint", ["helper", "list", "function"])
@pytest.mark.parametrize(
    ("uint_dtype", "float_dtype", "bits"),
    [
        (np.uint16, np.float16, 0x7E01),
        (np.uint16, jnp.bfloat16, 0x7FC1),
        (np.uint32, np.float32, 0x7FC00001),
        (np.uint64, np.float64, 0x7FF8000000000001),
    ],
)
def test_source_duplicate_compares_nan_payload_bits(
    entrypoint: str, uint_dtype: Any, float_dtype: Any, bits: int
) -> None:
    builder = _builder(entrypoint)
    payload = np.asarray([bits], dtype=uint_dtype).view(float_dtype)
    original = _insert(builder, payload, entrypoint)
    assert _insert(builder, payload.copy(), entrypoint) is original
    changed = np.asarray([bits + 1], dtype=uint_dtype).view(float_dtype)
    with pytest.raises(ValueError, match="already exists with different data"):
        _insert(builder, changed, entrypoint)


@pytest.mark.parametrize("collision", ["input", "operation"])
def test_source_function_constant_rejects_existing_nonconstant_value(
    collision: str,
) -> None:
    builder = _builder("function")
    existing = ir.val("typed", dtype=ir.DataType.FLOAT, shape=())
    if collision == "input":
        builder.inputs.append(existing)
    else:
        builder.op(
            "Neg",
            [ir.val("x", dtype=ir.DataType.FLOAT, shape=())],
            output=existing,
        )
    before = len(builder.nodes)
    with pytest.raises(ValueError, match="nonconstant function value"):
        builder.add_initializer_from_scalar(name="typed", value=np.float32(1))
    assert len(builder.nodes) == before


def test_source_function_constant_reuse_tracks_actual_graph_mutations() -> None:
    builder = _builder("function")
    original = builder.add_initializer_from_scalar(name="typed", value=np.float32(1))
    builder.graph.remove(original.producer())
    replacement = builder.add_initializer_from_scalar(name="typed", value=np.float32(2))
    assert replacement is not original
    assert len(builder.nodes) == 1
    _assert_payload(const_value_to_numpy(replacement), np.asarray(2, dtype=np.float32))


@pytest.mark.parametrize(
    "mutation",
    [
        "attribute-payload",
        "attribute-dtype",
        "attribute-shape",
        "cached-payload",
        "declared-dtype",
        "declared-shape",
    ],
)
def test_source_function_constant_reuse_rejects_inconsistent_serialized_value(
    mutation: str,
) -> None:
    builder = _builder("function")
    original = builder.add_initializer_from_scalar(name="typed", value=np.float32(1))
    if mutation.startswith("attribute"):
        changed = {
            "attribute-payload": np.asarray(2, dtype=np.float32),
            "attribute-dtype": np.asarray(1, dtype=np.float64),
            "attribute-shape": np.asarray([1], dtype=np.float32),
        }[mutation]
        original.producer().attributes["value"] = tensor_attr(
            "value", ir.tensor(changed)
        )
    elif mutation == "cached-payload":
        original.const_value = ir.tensor(np.asarray(2, dtype=np.float32))
    elif mutation == "declared-dtype":
        original.type = ir.TensorType(ir.DataType.DOUBLE)
    else:
        original.shape = ir.Shape([1])
    with pytest.raises(ValueError, match="already exists with different data"):
        builder.add_initializer_from_scalar(name="typed", value=np.float32(1))
    assert len(builder.nodes) == 1


def test_source_function_constant_reuse_reads_attribute_without_cached_payload() -> (
    None
):
    builder = _builder("function")
    original = builder.add_initializer_from_scalar(name="typed", value=np.float32(1))
    original.const_value = None
    assert (
        builder.add_initializer_from_scalar(name="typed", value=np.float32(1))
        is original
    )


@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_legacy_initializer_normalization_remains_unchanged(
    double: bool, dtype: Any
) -> None:
    """Retain main's float64-only narrowing and exact low-precision storage."""
    builder = IRBuilder(opset=23, enable_double_precision=double)
    payload = np.asarray([1 + 2**-30, -0.0], dtype=dtype)
    expected = (
        payload.astype(np.float32)
        if not double and payload.dtype == np.float64
        else payload
    )
    value = builder.add_initializer_from_array(name="legacy", array=payload)
    _assert_payload(const_value_to_numpy(value), expected)


def test_legacy_duplicate_numeric_cast_rules_remain_unchanged() -> None:
    builder = IRBuilder(opset=23, enable_double_precision=False)
    original = builder.add_initializer_from_array(
        name="legacy", array=np.asarray([0.0, 1.0], dtype=np.float32)
    )
    reused = builder.add_initializer_from_array(
        name="legacy", array=np.asarray([-0.0, 1 + 2**-30], dtype=np.float64)
    )
    assert reused is original
    builder.add_initializer_from_array(
        name="nan", array=np.asarray([np.nan], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="already exists with different data"):
        builder.add_initializer_from_array(
            name="nan", array=np.asarray([np.nan], dtype=np.float32)
        )


def _validate_relu6_test_extension(jaxpr: Any, opset: int) -> None:
    """Audit only relu6, then delegate every scope check to the source validator.

    Relu6 has the same unary dtype/shape contract as neg. Replacing its name
    solely for validation reuses those finite checks without admitting another
    primitive, dtype, shape, effect, or parameter in this test extension.
    """
    equations = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "jax.nn.relu6":
            assert not eqn.params
            equations.append(
                SimpleNamespace(
                    primitive=SimpleNamespace(name="neg"),
                    invars=eqn.invars,
                    outvars=eqn.outvars,
                    params=eqn.params,
                    effects=eqn.effects,
                )
            )
        else:
            equations.append(eqn)
    validate_source_jaxpr(
        SimpleNamespace(
            invars=jaxpr.invars,
            outvars=jaxpr.outvars,
            constvars=jaxpr.constvars,
            effects=jaxpr.effects,
            eqns=equations,
        ),
        opset,
    )


@onnx_function
def _relu6_inner(x: jax.Array) -> jax.Array:
    return jax.nn.relu6(x)


@onnx_function
def _relu6_outer(x: jax.Array) -> jax.Array:
    return _relu6_inner(x)


def _relu6_inline(x: jax.Array) -> jax.Array:
    return jax.nn.relu6(x)


def _relu6_nested(x: jax.Array) -> jax.Array:
    return _relu6_outer(x)


@onnx_function
def _relu6_initializer_owning_inner(x: jax.Array) -> jax.Array:
    return Relu6Plugin._PRIM.bind(x)


@onnx_function
def _relu6_initializer_owning_outer(x: jax.Array) -> jax.Array:
    return _relu6_initializer_owning_inner(x)


def _relu6_initializer_owning_nested(x: jax.Array) -> jax.Array:
    return _relu6_initializer_owning_outer(x)


def _relu6_initializer_owning_inline(x: jax.Array) -> jax.Array:
    return Relu6Plugin._PRIM.bind(x)


def _relu6_export(fn: Any, dtype: Any, *, owning_diagnostic: bool = False) -> ir.Model:
    # The initializer regression explicitly exercises the existing owning
    # plugin. Ordinary native SOURCE exports retain their unchanged default.
    descriptor = vars(Relu6Plugin)["binding_specs"]
    missing = object()
    bindings = []
    observed_owning: list[Any] = []
    validator = _validate_relu6_test_extension
    if owning_diagnostic:
        if fn is _relu6_nested:
            # A distinct target identity cannot reuse an earlier native body.
            fn = _relu6_initializer_owning_nested
        else:
            fn = _relu6_initializer_owning_inline

        def validate_owning(jaxpr: Any, opset: int) -> None:
            observed_owning.extend(
                eqn for eqn in jaxpr.eqns if eqn.primitive.name == "jax.nn.relu6"
            )
            _validate_relu6_test_extension(jaxpr, opset)

        validator = validate_owning
        specs = Relu6Plugin.binding_specs()
        for spec in specs:
            module = import_module(spec.target)
            bindings.append((module, spec.attr, getattr(module, spec.attr, missing)))
    try:
        model: ir.Model = conversion_api._convert_to_ir(
            fn=fn,
            inputs=[jax.ShapeDtypeStruct((5,), dtype)],
            input_params=None,
            model_name="source_initializer_relu6",
            opset=23,
            enable_double_precision=False,
            record_primitive_calls_file=None,
            strict_optimizer_failures=True,
            precision_policy=PrecisionPolicy.SOURCE,
            jaxpr_validator=validator,
        )
        if owning_diagnostic:
            assert (
                observed_owning
            ), "The initializer diagnostic must reach the owning primitive"
        return model
    finally:
        assert vars(Relu6Plugin)["binding_specs"] is descriptor
        for module, attr, original in bindings:
            assert getattr(module, attr, missing) is original


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
@pytest.mark.parametrize("fn", [_relu6_inline, _relu6_nested], ids=["inline", "nested"])
def test_source_relu6_typed_initializers_survive_plugin_and_function_export(
    dtype: Any, fn: Any, tmp_path: Path
) -> None:
    model = _relu6_export(fn, dtype, owning_diagnostic=True)
    if fn is _relu6_nested:
        assert len(model.functions) == 2
    proto = ir.to_proto(model)
    path = tmp_path / "relu6.onnx"
    onnx.save_model(proto, path)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    if fn is _relu6_nested:
        functions = {function.name: function for function in reloaded.functions}
        assert len(functions) == 2
        outer = functions["_relu6_initializer_owning_outer"]
        inner = functions["_relu6_initializer_owning_inner"]
        (outer_call,) = [
            node
            for node in reloaded.graph.node
            if (node.domain, node.op_type) == (outer.domain, outer.name)
        ]
        (inner_call,) = [
            node
            for node in outer.node
            if (node.domain, node.op_type) == (inner.domain, inner.name)
        ]
        assert list(outer_call.input) == [reloaded.graph.input[0].name]
        assert list(outer_call.output) == [reloaded.graph.output[0].name]
        assert list(inner_call.input) == list(outer.input)
        assert list(inner_call.output) == list(outer.output)
    expected_dtype = ir.DataType.from_numpy(np.dtype(dtype))
    assert reloaded.graph.input[0].type.tensor_type.elem_type == int(expected_dtype)
    assert reloaded.graph.output[0].type.tensor_type.elem_type == int(expected_dtype)
    payloads = [
        onnx.numpy_helper.to_array(value) for value in reloaded.graph.initializer
    ]
    for function in reloaded.functions:
        for node in function.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    payloads.append(onnx.numpy_helper.to_array(attr.t))
    owning_consumers = []
    for nodes, inputs, outputs, initializers in (
        (
            reloaded.graph.node,
            [value.name for value in reloaded.graph.input],
            [value.name for value in reloaded.graph.output],
            reloaded.graph.initializer,
        ),
        *(
            (function.node, list(function.input), list(function.output), ())
            for function in reloaded.functions
        ),
    ):
        constants = {value.name: value for value in initializers}
        for node in nodes:
            if node.op_type == "Constant":
                constants[node.output[0]] = next(
                    attr.t for attr in node.attribute if attr.name == "value"
                )
        needed = {outputs[0]}
        for node in reversed(nodes):
            if needed.intersection(node.output):
                needed.update(node.input)
        for maximum in nodes:
            if maximum.op_type != "Max":
                continue
            minimum = next(
                node
                for node in nodes
                if node.op_type == "Min" and node.input[0] == maximum.output[0]
            )
            assert maximum.input[0] == inputs[0] and minimum.output[0] in needed
            for node, coefficient in ((maximum, 0), (minimum, 6)):
                _assert_payload(
                    onnx.numpy_helper.to_array(constants[node.input[1]]),
                    np.asarray(coefficient, dtype=dtype),
                )
            owning_consumers.append((maximum.name, minimum.name))
    assert len(owning_consumers) == 1
    for coefficient in (0, 6):
        expected = np.asarray(coefficient, dtype=dtype)
        assert any(
            payload.dtype == expected.dtype
            and payload.shape == expected.shape
            and payload.tobytes() == expected.tobytes()
            for payload in payloads
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("fn", [_relu6_inline, _relu6_nested], ids=["inline", "nested"])
@pytest.mark.parametrize("optimize", [False, True], ids=["ort-off", "ort-on"])
def test_source_relu6_serialized_cpu_output_matches_typed_jax_reference(
    dtype: Any, fn: Any, optimize: bool, tmp_path: Path
) -> None:
    model = _relu6_export(fn, dtype)
    path = tmp_path / "relu6_cpu.onnx"
    onnx.save_model(ir.to_proto(model), path)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    payload = np.asarray([-2.5, 1 + 2**-30, 5.5, 6, 8.25], dtype=dtype)
    reference = np.asarray(fn(jnp.asarray(payload)))
    compiled_reference = np.asarray(jax.jit(fn)(jnp.asarray(payload)))
    assert reference.dtype == np.dtype(dtype)
    _assert_payload(compiled_reference, reference)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        reloaded.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    assert session.get_providers() == ["CPUExecutionProvider"]
    (actual,) = session.run(None, {session.get_inputs()[0].name: payload})
    _assert_payload(actual, reference)


def test_product_source_profile_still_excludes_relu6() -> None:
    def validate_actual(jaxpr: Any, opset: int) -> None:
        validate_source_jaxpr(jaxpr, opset)

    # Retain the historical ID and exact original F32 export as a positive.
    model = conversion_api._convert_to_ir(
        fn=_relu6_inline,
        inputs=[jax.ShapeDtypeStruct((5,), np.float32)],
        input_params=None,
        model_name="not_admitted_relu6",
        opset=23,
        enable_double_precision=False,
        record_primitive_calls_file=None,
        precision_policy=PrecisionPolicy.SOURCE,
        jaxpr_validator=validate_actual,
    )
    assert isinstance(model, ir.Model)
    assert (
        model.graph.inputs[0].dtype == model.graph.outputs[0].dtype == ir.DataType.FLOAT
    )
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    # The owning diagnostic fixture does not admit native BF16 Min/Max.
    with pytest.raises(
        SourcePrecisionError, match="floating minimum/maximum"
    ) as caught:
        conversion_api._convert_to_ir(
            fn=_relu6_inline,
            inputs=[jax.ShapeDtypeStruct((5,), jnp.bfloat16)],
            input_params=None,
            model_name="not_admitted_relu6",
            opset=23,
            enable_double_precision=False,
            record_primitive_calls_file=None,
            precision_policy=PrecisionPolicy.SOURCE,
            jaxpr_validator=validate_actual,
        )
    assert caught.value.primitive == "max"
