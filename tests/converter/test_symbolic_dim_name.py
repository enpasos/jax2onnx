import jax.numpy as jnp
from jax2onnx import to_onnx 
import pytest

@pytest.mark.order(-1)  # run *after* the models have been produced
def test_symbolic_batch_dim_is_preserved():
    # Use abstracted axes with a symbolic name "B"

    model = to_onnx(fn=lambda x: jnp.squeeze(x, axis=(-1, -3)), inputs=[(1, "B", 1)])

    # extract the input tensor from the ONNX model
    # and check the symbolic dimension

    input_tensor = model.graph.input[0]
    input_shape = input_tensor.type.tensor_type.shape
    dim_param = input_shape.dim[1].dim_param

    # Assert that the symbolic dimension is preserved
    assert dim_param == "B", f"Expected symbolic dim 'B', got: {dim_param}"
