import onnx
import jax.numpy as jnp
from flax import nnx
from jax2onnx import onnx_function, to_onnx


def test_onnx_function_deterministic_param_is_input():
    """
    Test that if input_params contains 'deterministic',
    then the ONNX function has 'deterministic' as an input of type BOOL.
    """

    @onnx_function
    class MLPBlock(nnx.Module):
        def __init__(self, num_hiddens, mlp_dim, rngs: nnx.Rngs):
            self.layers = [
                nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
                lambda x: nnx.gelu(x, approximate=False),
                nnx.Dropout(rate=0.1, rngs=rngs),
                nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
                nnx.Dropout(rate=0.1, rngs=rngs),
            ]

        def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
            for layer in self.layers:
                if isinstance(layer, nnx.Dropout):
                    x = layer(x, deterministic=deterministic)
                else:
                    x = layer(x)
            return x

    @onnx_function
    class SuperBlock(nnx.Module):
        def __init__(self):
            rngs = nnx.Rngs(0)
            self.layer_norm2 = nnx.LayerNorm(3, rngs=rngs)
            self.mlp = MLPBlock(num_hiddens=3, mlp_dim=6, rngs=rngs)

        def __call__(self, x, deterministic: bool = True):
            x_normalized = self.layer_norm2(x)
            return self.mlp(x_normalized, deterministic=deterministic)

    # Export with input_params containing deterministic
    model = to_onnx(
        SuperBlock(),
        inputs=[(5, 10, 3)],
        input_params={"deterministic": True},
        model_name="test_deterministic_param",
    )

    # Check that 'deterministic' is not an initializer (should be a real input)
    for init in model.graph.initializer:
        assert (
            init.name != "deterministic"
        ), "deterministic should not be an initializer!"

    # Check that 'deterministic' is a graph input and type is BOOL
    found_graph_input = False
    for inp in model.graph.input:
        if inp.name == "deterministic":
            found_graph_input = True
            assert (
                inp.type.tensor_type.elem_type == onnx.TensorProto.BOOL
            ), f"Graph input 'deterministic' should be BOOL, got {inp.type.tensor_type.elem_type}"
    assert found_graph_input, "Graph input 'deterministic' must be present!"

    # Check function input and type as before
    found = False
    wrong_type = None
    for func in model.functions:
        if "deterministic" in func.input:
            list(func.input).index("deterministic")
            # Check type in value_info
            for vi in func.value_info:
                if vi.name == "deterministic":
                    print(
                        f"DEBUG: deterministic elem_type is {vi.type.tensor_type.elem_type}"
                    )
                    if vi.type.tensor_type.elem_type != onnx.TensorProto.BOOL:
                        wrong_type = vi.type.tensor_type.elem_type
                    else:
                        found = True
    if not found:
        raise AssertionError(
            "deterministic must be a function input of type BOOL if input_params is set! (Present but wrong type: %s)"
            % wrong_type
        )
    if wrong_type is not None:
        raise AssertionError(f"deterministic input should be BOOL, got {wrong_type}")
