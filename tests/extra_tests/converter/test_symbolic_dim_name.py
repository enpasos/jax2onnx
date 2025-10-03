import jax.numpy as jnp
from jax2onnx.user_interface import to_onnx


def test_symbolic_batch_dim_is_preserved_ir():
    model = to_onnx(
        fn=lambda x: jnp.squeeze(x, axis=(-1, -3)),
        inputs=[(1, "B", 1)],
        model_name="symbolic_dim_test_ir",
    )
    input_tensor = model.graph.input[0]
    dim_param = input_tensor.type.tensor_type.shape.dim[1].dim_param
    assert dim_param == "B"
