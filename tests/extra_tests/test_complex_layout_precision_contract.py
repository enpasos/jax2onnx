# tests/extra_tests/test_complex_layout_precision_contract.py

"""Complex broadcasts retain the physical component channel."""

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx
from tests.extra_tests.test_precision_contract import check


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scalar_complex_broadcast_preserves_components_and_shared_original(dtype):
    def source(real, imag):
        original = jax.lax.complex(real, imag)
        expanded = jax.lax.broadcast_in_dim(original, (2, 3), ())
        return (
            jnp.real(expanded),
            jnp.imag(expanded),
            jnp.real(original),
            jnp.imag(original),
        )

    inputs = [np.asarray(2.0, dtype), np.asarray(-3.0, dtype)]
    with jax.enable_x64():
        _, reference = check(source, inputs)
    for index, original in enumerate(inputs):
        assert reference[index].dtype == dtype
        np.testing.assert_array_equal(reference[index], np.full((2, 3), original))
        np.testing.assert_array_equal(reference[index + 2], original)


def test_complex_formal_broadcast_retains_existing_packed_input_interface():
    def source(value):
        expanded = jax.lax.broadcast_in_dim(value, (2, 3), ())
        return (
            jnp.real(expanded),
            jnp.imag(expanded),
            jnp.real(value),
            jnp.imag(value),
        )

    def assert_outputs(actual, expected):
        assert len(actual) == len(expected)
        for value, wanted in zip(actual, expected, strict=True):
            value, wanted = np.asarray(value), np.asarray(wanted)
            assert value.dtype == wanted.dtype
            assert value.shape == wanted.shape
            np.testing.assert_array_equal(value, wanted)

    value = np.asarray(2 - 3j, np.complex128)
    with jax.enable_x64():
        native = jnp.asarray(value)
        reference = source(native)
        compiled = jax.jit(source)
        assert_outputs(compiled(native), reference)
        assert_outputs(compiled(native), reference)
        model = to_onnx(source, [jax.ShapeDtypeStruct((), np.complex128)])
    onnx.checker.check_model(model, full_check=True)
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
    assert [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] == [2]
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
        packed = np.stack([value.real, value.imag], axis=-1)
        actual = session.run(None, {session.get_inputs()[0].name: packed})
        assert_outputs(actual, reference)
    assert_outputs(
        reference,
        (np.full((2, 3), 2.0), np.full((2, 3), -3.0), value.real, value.imag),
    )
