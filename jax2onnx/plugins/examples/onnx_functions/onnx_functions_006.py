# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_006.py


from flax import nnx

from jax2onnx.plugin_system import onnx_function, register_example, construct_and_call

  
@onnx_function
class SuperBlock(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(0)
        self.layer_norm2 = nnx.LayerNorm(256, rngs=rngs)

    # self.mlp = MLPBlock(num_hiddens=256, mlp_dim=512, rngs=rngs)

    def __call__(self, x):
        # return self.mlp(self.layer_norm2(x))
        return self.layer_norm2(x)


register_example(
    component="onnx_functions_006",
    description="one function on an outer layer.",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["flax.nnx.LayerNorm"],
    testcases=[
        {
            "testcase": "006_one_function_outer",
            "callable": construct_and_call(SuperBlock),
            "input_shapes": [("B", 10, 256)],
            "expected_number_of_function_instances": 1,
            "run_only_f32_variant": True,
        },
    ],
)
