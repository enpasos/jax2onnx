# file: jax2onnx/plugins/linear_general.py

import onnx.helper as oh
import onnx
import numpy as np
import jax.numpy as jnp
from flax import nnx
from jax2onnx.plugins.matmul import build_onnx_matmul  # Import MatMul plugin

def build_onnx_node(self, input_shapes, input_names, onnx_graph, parameters=None):
    """
    Constructs an ONNX node for `LinearGeneral`, ensuring proper handling of input reshaping,
    transformation, and weight application using the existing MatMul ONNX builder.
    """
    if parameters is None:
        parameters = {}

    if len(input_shapes) < 1:
        raise ValueError("Expected at least one input for LinearGeneral.")

    input_shape = list(map(int, input_shapes[0]))  # Convert to Python int
    input_name = input_names[0]

    in_features = tuple(map(int, self.in_features))  # Convert to tuple of Python ints
    out_features = tuple(map(int, self.out_features))  # Convert to tuple of Python ints
    axis = tuple(map(int, self.axis))  # Convert to tuple of Python ints
    use_bias = self.use_bias

    # Compute batch dimensions
    batch_dims = list(input_shape[:-len(axis)])
    output_shape = batch_dims + list(out_features)

    node_name = f"node{onnx_graph.counter_plusplus()}"

    # Reshape input if necessary
    reshaped_input_name = input_name
    reshaped_input_shape = input_shape
    if len(axis) > 1:
        reshaped_input_name = f"{node_name}_reshaped_input"
        feature_dim_prod = np.prod(in_features).item()
        reshaped_input_shape = batch_dims + [feature_dim_prod]

        shape_name = f"{node_name}_reshape_shape"
        shape_tensor = oh.make_tensor(
            name=shape_name,
            data_type=onnx.TensorProto.INT64,
            dims=[len(reshaped_input_shape)],
            vals=[int(dim) for dim in reshaped_input_shape],
        )
        onnx_graph.add_initializer(shape_tensor)
        onnx_graph.add_node(
            oh.make_node(
                "Reshape",
                inputs=[input_name, shape_name],
                outputs=[reshaped_input_name],
                name=f"{node_name}_reshape_input"
            )
        )
        onnx_graph.add_local_outputs([reshaped_input_shape], [reshaped_input_name])

    # Define weight matrix for MatMul
    weight_name = f"{node_name}_weight"

    # Reshape the kernel tensor to a 2D matrix for MatMul
    transposed_kernel = self.kernel.value
    kernel_shape = (self.kernel.value.shape[0], -1)  # Flatten to (256, 8*32)

    onnx_graph.add_initializer(
        oh.make_tensor(
            weight_name,
            onnx.TensorProto.FLOAT,
            list(kernel_shape),  # Ensure correct shape
            transposed_kernel.reshape(kernel_shape).astype(np.float32),
        )
    )

    # Call MatMul plugin
    _, matmul_out_names = build_onnx_matmul(
        function=lambda a, b: jnp.matmul(a, b),
        input_shapes=[reshaped_input_shape, kernel_shape],
        input_names=[reshaped_input_name, weight_name],
        onnx_graph=onnx_graph,
        parameters=None
    )

    final_out_name = matmul_out_names[0]

    # Bias addition if enabled
    if use_bias:
        bias_name = f"{node_name}_bias"
        bias_shape = out_features

        onnx_graph.add_initializer(
            oh.make_tensor(
                bias_name,
                onnx.TensorProto.FLOAT,
                list(bias_shape),
                self.bias.value.flatten().astype(np.float32).tolist(),
            )
        )
        final_out_name = f"{node_name}_output"
        onnx_graph.add_node(
            oh.make_node(
                "Add",
                inputs=[matmul_out_names[0], bias_name],
                outputs=[final_out_name],
                name=f"{node_name}_add_bias",
            )
        )
        onnx_graph.add_local_outputs([output_shape], [final_out_name])

    return [output_shape], [final_out_name]


# Attach the ONNX conversion function to the nnx.LinearGeneral class
nnx.LinearGeneral.build_onnx_node = build_onnx_node


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of `nnx.LinearGeneral`.
    """
    return [
        {
            "model_name": "linear_general",
            "model": lambda: nnx.LinearGeneral(
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 8, 32)],
            "build_onnx_node": nnx.LinearGeneral.build_onnx_node
        },
        {
            "model_name": "linear_general_2",
            "model": lambda: nnx.LinearGeneral(
                in_features=(256,),
                out_features=(8, 32),
                axis=(-1,),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 256)],
            "build_onnx_node": nnx.LinearGeneral.build_onnx_node
        }
    ]
