# tests/extra_tests/test_complex_constant_precision.py

"""Complex constants keep source values and store exact packed IR payloads."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins._complex_utils import (
    cast_real_tensor,
    ensure_packed_real_pair,
)
from jax2onnx.plugins.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
)
from tests.extra_tests.test_precision_control_flow import _assert_exact


class _SourceVar:
    def __init__(self, array):
        self.aval = SimpleNamespace(dtype=array.dtype, shape=array.shape)


def _packed(value):
    value = np.asarray(value)
    return np.stack((value.real, value.imag), axis=-1)


_CONSTANT64: npt.NDArray[np.complex64] = np.asarray(
    [complex(2**24 + 1, 2.25), complex(-3.25, -4.5)], np.complex64
)
_CONSTANT128: npt.NDArray[np.complex128] = np.asarray(
    [complex(2**24 + 1, 2.25), complex(-3.25, -4.5)], np.complex128
)
_UNIT64: npt.NDArray[np.complex64] = np.asarray(1j, np.complex64)
_UNIT128: npt.NDArray[np.complex128] = np.asarray(1j, np.complex128)


def _complex_constant64(value):
    shared = jnp.asarray(_CONSTANT64)
    unit = jnp.asarray(_UNIT64)
    return value * unit + shared


def _complex_constant128(value):
    shared = jnp.asarray(_CONSTANT128)
    unit = jnp.asarray(_UNIT128)
    return value * unit + shared


def _call_complex_constant64(value):
    shared = jnp.asarray(_CONSTANT64)
    unit = jnp.asarray(_UNIT64)
    return _complex_constant64(value), shared, shared.astype(jnp.complex128), unit


def _call_complex_constant128(value):
    shared = jnp.asarray(_CONSTANT128)
    unit = jnp.asarray(_UNIT128)
    return _complex_constant128(value), shared, shared.astype(jnp.complex128), unit


@pytest.fixture
def registered_complex_constant_target(dtype, inside_function):
    if not inside_function:
        yield None
        return
    function, caller = (
        (_complex_constant64, _call_complex_constant64)
        if dtype is np.complex64
        else (_complex_constant128, _call_complex_constant128)
    )
    qualified = f"{function.__module__}.{function.__name__}"
    try:
        onnx_function(unique=True)(function)
        yield caller
    finally:
        PLUGIN_REGISTRY.pop(f"onnx_fn::{qualified}", None)
        ONNX_FUNCTION_PLUGIN_REGISTRY.pop(qualified, None)


@pytest.mark.parametrize(
    "dtype",
    [np.dtype([("unsupported", np.float32)])]
    + [
        dtype
        for dtype in (np.dtype(np.clongdouble),)
        if dtype not in (np.dtype(np.complex64), np.dtype(np.complex128))
    ],
)
def test_unknown_constant_dtype_is_not_reinterpreted_as_supported_complex(dtype):
    array = np.asarray(1, dtype=dtype)
    ctx = IRContext(opset=23)
    with pytest.raises(TypeError, match="Unsupported source dtype"):
        ctx.bind_const_for_var(object(), array)
    assert not ctx.builder.initializers
    assert not tuple(ctx.builder.graph)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("layout", ["scalar", "strided"])
@pytest.mark.parametrize("function_mode", [False, True])
def test_complex_constant_binding_preserves_component_bits(
    dtype, layout, function_mode
):
    word_dtype = np.uint32 if dtype is np.complex64 else np.uint64
    sign = 1 << (np.dtype(word_dtype).itemsize * 8 - 1)
    nan = 0x7FC12345 if dtype is np.complex64 else 0x7FF8000000001234
    words = np.asarray([0, sign, sign, 0, nan, sign | nan, 0, nan], word_dtype)
    array = words.view(dtype).reshape(2, 2)
    array = array[:1, :1].reshape(()) if layout == "scalar" else array.T[:, ::-1]
    var = _SourceVar(array)
    ctx = IRContext(opset=23)
    if function_mode:
        ctx = ctx.create_child_context(scope="function")

    value = ctx.bind_const_for_var(var, array)
    base_dtype = ir.DataType.FLOAT if dtype is np.complex64 else ir.DataType.DOUBLE
    payload = value.const_value.numpy()
    assert value.dtype == base_dtype
    assert value.shape == ir.Shape(array.shape + (2,))
    assert payload.dtype == array.real.dtype
    assert payload.shape == array.shape + (2,)
    assert payload.tobytes() == array.tobytes(order="C")
    assert not np.shares_memory(payload, array)
    assert ctx.try_evaluate_const(var) is array
    assert var.aval.dtype == np.dtype(dtype)
    assert ctx.require_value_for_var(var) is value

    # A shared source constant remains canonical when a consumer needs DOUBLE.
    packed, packed_dtype = ensure_packed_real_pair(ctx, value, name_hint="shared")
    assert packed is value and packed_dtype == base_dtype
    work = cast_real_tensor(ctx, value, ir.DataType.DOUBLE, name_hint="work")
    if base_dtype == ir.DataType.FLOAT:
        assert work is not value
        assert work.producer().inputs[0] is value
    else:
        assert work is value
    assert value.dtype == base_dtype
    assert value.const_value.numpy().tobytes() == array.tobytes(order="C")
    assert ctx.try_evaluate_const(var).tobytes(order="C") == array.tobytes(order="C")

    if function_mode:
        assert not ctx.builder.initializers
        constant = value.producer()
        assert constant.op_type == "Constant"
        tensor = constant.attributes["value"].as_tensor()
        assert tensor.dtype == base_dtype
        assert tensor.numpy().tobytes() == payload.tobytes()
    else:
        assert value in ctx.builder.initializers
        assert value.producer() is None


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("inside_function", [False, True])
def test_complex_constants_shared_by_native_width_and_widened_consumers(
    dtype, inside_function, registered_complex_constant_target
):
    constant = np.asarray([complex(2**24 + 1, 2.25), complex(-3.25, -4.5)], dtype)
    scalar = np.asarray(1j, dtype)
    x = np.asarray([2**24 + 3, -8.5], dtype=constant.real.dtype)

    def source(value):
        shared = jnp.asarray(constant)
        unit = jnp.asarray(scalar)
        return value * unit + shared, shared, shared.astype(jnp.complex128), unit

    target = registered_complex_constant_target if inside_function else source
    expected = (
        _packed(x * scalar + constant),
        _packed(constant),
        _packed(constant.astype(np.complex128)),
        _packed(scalar),
    )
    with jax.enable_x64():
        native_input = jnp.asarray(x)
        reference = tuple(_packed(value) for value in target(native_input))
        _assert_exact(reference, expected)
        compiled = jax.jit(target)
        _assert_exact(
            tuple(_packed(value) for value in compiled(native_input)), reference
        )
        _assert_exact(
            tuple(_packed(value) for value in compiled(native_input)), reference
        )
        model = to_onnx(target, [jax.ShapeDtypeStruct(x.shape, x.dtype)])

    onnx.checker.check_model(model, full_check=True)
    base_type = (
        onnx.TensorProto.FLOAT if dtype is np.complex64 else onnx.TensorProto.DOUBLE
    )
    assert [value.type.tensor_type.elem_type for value in model.graph.output] == [
        base_type,
        base_type,
        onnx.TensorProto.DOUBLE,
        base_type,
    ]
    tensors = list(model.graph.initializer)
    nodes = list(model.graph.node)
    for function in model.functions:
        nodes.extend(function.node)
    tensors.extend(
        attribute.t
        for node in nodes
        if node.op_type == "Constant"
        for attribute in node.attribute
        if attribute.type == onnx.AttributeProto.TENSOR
    )
    assert all(
        tensor.data_type
        not in (onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128)
        for tensor in tensors
    )
    payloads = [onnx.numpy_helper.to_array(tensor) for tensor in tensors]
    for original in (constant, scalar):
        packed = _packed(original)
        assert any(
            payload.dtype == packed.dtype
            and payload.shape == packed.shape
            and payload.tobytes() == packed.tobytes()
            for payload in payloads
        )
    if inside_function:
        assert model.functions
        assert any(
            node.domain == function.domain and node.op_type == function.name
            for function in model.functions
            for node in model.graph.node
        )
        assert any(
            node.op_type == "Constant"
            for function in model.functions
            for node in function.node
        )

    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {session.get_inputs()[0].name: x})
        _assert_exact(tuple(actual), reference)
