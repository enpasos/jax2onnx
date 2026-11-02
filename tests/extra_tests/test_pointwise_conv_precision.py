# tests/extra_tests/test_pointwise_conv_precision.py

"""Pointwise convolutions retain DOUBLE products and complex components."""

import jax
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx
from tests.extra_tests.test_precision_control_flow import _assert_exact


def _runtime_array(value):
    value = np.asarray(value)
    if np.issubdtype(value.dtype, np.complexfloating):
        return np.stack((value.real, value.imag), axis=-1)
    return value


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize(
    "rank,channels_last",
    [(1, False), (1, True), (2, False), (2, True)],
    ids=["ncw", "nwc", "nchw", "nhwc"],
)
def test_pointwise_conv_double_products_dynamic_batch(
    dtype, groups, rank, channels_last
):
    spatial_shape = (2,) * rank
    lhs = np.empty((2, 2 * groups, *spatial_shape), dtype=dtype)
    rhs = np.empty((2 * groups, 2, *((1,) * rank)), dtype=dtype)
    expected = np.empty_like(lhs)
    offset = np.arange(2 * 2**rank, dtype=np.float64).reshape((2, *spatial_shape)) / 4
    is_complex = dtype is np.complex128
    for group in range(groups):
        residual = 1 + group + offset
        lhs[:, 2 * group] = 2**24 + residual + (0.5j if is_complex else 0)
        lhs[:, 2 * group + 1] = -(2**24) + (0.25j if is_complex else 0)
        for output in range(2 * group, 2 * group + 2):
            scale = output + 1
            rhs[output] = scale * (1 + (0.5j if is_complex else 0))
            expected[:, output] = scale * (
                residual - 0.375 + 1j * (residual / 2 + 0.75)
                if is_complex
                else residual
            )
    if channels_last:
        lhs = np.moveaxis(lhs, 1, -1)
        rhs = np.transpose(rhs, [*range(2, rank + 2), 1, 0])
        expected = np.moveaxis(expected, 1, -1)
    spatial_layout = "W" if rank == 1 else "HW"
    lhs_layout = "N" + spatial_layout + "C" if channels_last else "NC" + spatial_layout
    rhs_layout = spatial_layout + "IO" if channels_last else "OI" + spatial_layout

    def source(value, weight):
        return jax.lax.conv_general_dilated(
            value,
            weight,
            window_strides=(1,) * rank,
            padding="VALID",
            feature_group_count=groups,
            dimension_numbers=(lhs_layout, rhs_layout, lhs_layout),
        )

    with jax.enable_x64():
        native_inputs = [jax.numpy.asarray(value) for value in (lhs, rhs)]
        _assert_exact(_runtime_array(source(*native_inputs)), _runtime_array(expected))
        compiled = jax.jit(source)
        _assert_exact(
            _runtime_array(compiled(*native_inputs)), _runtime_array(expected)
        )
        _assert_exact(
            _runtime_array(compiled(*native_inputs)), _runtime_array(expected)
        )
        model = to_onnx(
            source,
            [
                jax.ShapeDtypeStruct(("B", *lhs.shape[1:]), dtype),
                jax.ShapeDtypeStruct(rhs.shape, dtype),
            ],
        )
    onnx.checker.check_model(model, full_check=True)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.value_info, *model.graph.output)
    }
    contractions = [node for node in model.graph.node if node.op_type == "Einsum"]
    assert len(contractions) == groups * (4 if is_complex else 1)
    equation = "ncw,oc->now" if rank == 1 else "nchw,oc->nohw"
    assert all(
        next(attr.s.decode() for attr in node.attribute if attr.name == "equation")
        == equation
        for node in contractions
    )
    assert not any(
        node.op_type in {"Conv", "MatMul", "Cast", "CastLike"}
        for node in model.graph.node
    )
    assert all(
        types[node.output[0]] == onnx.TensorProto.DOUBLE
        for node in model.graph.node
        if node.op_type in {"Einsum", "Sub", "Add"}
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
        feeds = {
            meta.name: _runtime_array(value)
            for meta, value in zip(session.get_inputs(), (lhs, rhs))
        }
        (actual,) = session.run(None, feeds)
        _assert_exact(actual, _runtime_array(expected))


def test_nonpointwise_complex128_retains_double_conv_artifact():
    """Schema/type coverage only; DOUBLE Conv CPU numerics remain unverified."""

    def source(value, weight):
        return jax.lax.conv(value, weight, window_strides=(1, 1), padding="VALID")

    with jax.enable_x64():
        model = to_onnx(
            source,
            [
                jax.ShapeDtypeStruct((1, 1, 3, 3), np.complex128),
                jax.ShapeDtypeStruct((1, 1, 2, 2), np.complex128),
            ],
        )
    onnx.checker.check_model(model, full_check=True)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.value_info, *model.graph.output)
    }
    contractions = [node for node in model.graph.node if node.op_type == "Conv"]
    assert len(contractions) == 4
    assert all(
        types[node.output[0]] == onnx.TensorProto.DOUBLE for node in contractions
    )
    assert not any(node.op_type == "Einsum" for node in model.graph.node)
    assert not any(node.op_type in {"Cast", "CastLike"} for node in model.graph.node)
