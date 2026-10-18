# tests/extra_tests/converter/test_complex_constant_packing.py

"""Packing complex constants changes serialized storage together with metadata."""

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx.converter.function_scope import FunctionScope
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.ir_utils import tensor_attr
from jax2onnx.plugins._complex_utils import (
    ensure_packed_real_pair,
    pack_native_complex,
)


_POLICIES = [
    pytest.param(PrecisionPolicy.LEGACY, False, id="legacy-single"),
    pytest.param(PrecisionPolicy.LEGACY, True, id="legacy-double"),
    pytest.param(PrecisionPolicy.SOURCE, False, id="source"),
]


def _context(policy: PrecisionPolicy, double: bool) -> IRContext:
    return IRContext(
        opset=23,
        enable_double_precision=double,
        precision_policy=policy,
    )


def _payload(dtype: Any, layout: str) -> np.ndarray:
    if np.dtype(dtype) == np.dtype(np.complex64):
        bits = np.asarray(
            [
                0x80000000,
                0,
                0x7FC00001,
                0x80000000,
                0x3F800001,
                0xC0200000,
                0,
                0x80000000,
                0x7FC00002,
                0x40600000,
                0x40880000,
                0xC0C00000,
            ],
            dtype=np.uint32,
        )
    else:
        bits = np.asarray(
            [
                0x8000000000000000,
                0,
                0x7FF8000000000001,
                0x8000000000000000,
                0x3FF0000000400000,
                0xC004000000000000,
                0,
                0x8000000000000000,
                0x7FF8000000000002,
                0x400C000000000000,
                0x4011000000000000,
                0xC018000000000000,
            ],
            dtype=np.uint64,
        )
    payload = bits.view(dtype)
    if layout == "scalar":
        return payload[:1].reshape(())
    if layout == "noncontiguous":
        return payload.reshape(2, 3).T
    return payload


def _expected(payload: np.ndarray, double: bool) -> np.ndarray:
    base = np.float32 if payload.dtype == np.dtype(np.complex64) else np.float64
    channels = (
        np.ascontiguousarray(payload)
        .reshape(-1)
        .view(base)
        .reshape((*payload.shape, 2))
    )
    return channels.astype(np.float64) if double else channels


def _assert_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


@pytest.mark.parametrize(("policy", "double"), _POLICIES)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("layout", ["scalar", "vector", "noncontiguous"])
@pytest.mark.parametrize("scope", ["graph", "function", "control_flow"])
def test_packing_complex_constant_preserves_actual_tensor_payload_in_every_scope(
    policy: PrecisionPolicy, double: bool, dtype: Any, layout: str, scope: str
) -> None:
    ctx = _context(policy, double)
    if scope != "graph":
        ctx = ctx.create_child_context(scope=scope)
    payload = _payload(dtype, layout)
    original_bits = payload.tobytes(order="C")
    value = ctx.builder.add_initializer_from_array(name="complex", array=payload)
    packed = pack_native_complex(ctx, value)
    expected = _expected(payload, double)
    assert packed is value
    assert packed.dtype == ir.DataType.from_numpy(expected.dtype)
    assert tuple(packed.shape) == expected.shape
    assert packed.const_value is not None
    _assert_exact(packed.const_value.numpy(), expected)
    assert payload.tobytes(order="C") == original_bits
    if scope != "graph":
        producer = packed.producer()
        assert producer.op_type == "Constant"
        assert producer.attributes["value"].as_tensor() is packed.const_value
        _assert_exact(producer.attributes["value"].as_tensor().numpy(), expected)
    before = len(ctx.builder.nodes)
    reused, base = ensure_packed_real_pair(ctx, packed, name_hint="already_packed")
    assert reused is packed and base == packed.dtype
    assert len(ctx.builder.nodes) == before
    assert tuple(reused.shape) == expected.shape


def _constant_model(
    policy: PrecisionPolicy, double: bool, dtype: Any, function: bool
) -> tuple[ir.Model, np.ndarray]:
    ctx = _context(policy, double)
    payload = _payload(dtype, "noncontiguous")
    expected = _expected(payload, double)
    if function:
        scope = FunctionScope(ctx, name="packed_constant", domain="test.complex")
        scope.begin([])
        value = scope.ctx.builder.add_initializer_from_array(
            name="complex", array=payload
        )
        packed = pack_native_complex(scope.ctx, value)
        definition = scope.end([packed])
        ctx.builder.graph.opset_imports[definition.domain] = 1
        output = ctx.builder.op(
            definition.name,
            [],
            domain=definition.domain,
            output=ir.val("packed", dtype=packed.dtype, shape=packed.shape),
        )
        ctx.builder.outputs.append(output)
        return (
            ir.Model(ctx.builder.graph, ir_version=10, functions=[definition]),
            expected,
        )
    value = ctx.builder.add_initializer_from_array(name="complex", array=payload)
    ctx.builder.outputs.append(pack_native_complex(ctx, value))
    return ctx.builder.to_ir_model(name="packed_constant", ir_version=10), expected


@pytest.mark.parametrize(("policy", "double"), _POLICIES)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("function", [False, True], ids=["initializer", "function"])
@pytest.mark.parametrize("optimize", [False, True], ids=["ort-off", "ort-on"])
def test_packed_complex_constant_serializes_checks_and_executes_exactly(
    policy: PrecisionPolicy,
    double: bool,
    dtype: Any,
    function: bool,
    optimize: bool,
    tmp_path: Path,
) -> None:
    model, expected = _constant_model(policy, double, dtype, function)
    path = tmp_path / "packed_complex_constant.onnx"
    onnx.save_model(ir.to_proto(model), path)
    proto = onnx.load_model(path)
    onnx.checker.check_model(proto, full_check=True)
    if function:
        value_attr = next(
            attr
            for node in proto.functions[0].node
            for attr in node.attribute
            if attr.name == "value"
        )
        actual_tensor = value_attr.t
    else:
        actual_tensor = proto.graph.initializer[0]
    _assert_exact(onnx.numpy_helper.to_array(actual_tensor), expected)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    assert session.get_providers() == ["CPUExecutionProvider"]
    (actual,) = session.run(None, {})
    _assert_exact(actual, expected)


@pytest.mark.parametrize(("policy", "double"), _POLICIES)
@pytest.mark.parametrize("mutation", ["attribute", "cache", "dtype", "shape"])
def test_packing_rejects_inconsistent_constant_storage_before_mutation(
    policy: PrecisionPolicy, double: bool, mutation: str
) -> None:
    ctx = _context(policy, double).create_child_context(scope="function")
    payload = _payload(np.complex64, "vector")
    value = ctx.builder.add_initializer_from_array(name="complex", array=payload)
    if mutation == "attribute":
        changed = payload.copy()
        changed.real[0] = np.float32(1)
        value.producer().attributes["value"] = tensor_attr("value", ir.tensor(changed))
    elif mutation == "cache":
        value.const_value = ir.tensor(payload.reshape(2, 3))
    elif mutation == "dtype":
        value.dtype = ir.DataType.COMPLEX128
    else:
        value.shape = ir.Shape([2, 3])
    before_dtype, before_shape = value.dtype, tuple(value.shape)
    with pytest.raises(ValueError, match="constant payload|attribute and cache"):
        pack_native_complex(ctx, value)
    assert value.dtype == before_dtype
    assert tuple(value.shape) == before_shape


@pytest.mark.parametrize(("policy", "double"), _POLICIES)
def test_packing_constant_uses_authoritative_attribute_without_cache(
    policy: PrecisionPolicy, double: bool
) -> None:
    ctx = _context(policy, double).create_child_context(scope="function")
    payload = _payload(np.complex64, "vector")
    value = ctx.builder.add_initializer_from_array(name="complex", array=payload)
    value.const_value = None
    pack_native_complex(ctx, value)
    _assert_exact(value.const_value.numpy(), _expected(payload, double))
