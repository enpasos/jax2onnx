# file: jax2onnx/plugins/linear_general.py

import numpy as np
import onnx
import onnx.helper as oh
from flax import nnx
from jax2onnx.to_onnx import Z
from jax2onnx.plugins.matmul import to_onnx_matmul  # Import MatMul plugin


def to_onnx(self, z, parameters=None):
    """
    Converts an `nnx.LinearGeneral` layer into an ONNX `MatMul` node with optional reshaping.

    Args:
        self: The `nnx.LinearGeneral` instance.
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict, optional): Additional parameters (currently unused).

    Returns:
        Z: Updated instance with new shapes and names.
    """

    if parameters is None:
        parameters = {}

    onnx_graph = z.onnx_graph
    input_shape = list(map(int, z.shapes[0]))  # Convert to Python int
    input_name = z.names[0]

    in_features = tuple(map(int, self.in_features))  # Tuple of Python ints
    out_features = tuple(map(int, self.out_features))  # Tuple of Python ints
    axis = tuple(map(int, self.axis))  # Tuple of Python ints
    use_bias = self.use_bias

    # Compute batch dimensions
    batch_dims = list(input_shape[: -len(axis)])
    output_shape = batch_dims + list(out_features)

    node_name = f"node{onnx_graph.next_id()}"

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
                name=f"{node_name}_reshape_input",
            )
        )
        onnx_graph.add_local_outputs([reshaped_input_shape], [reshaped_input_name])

    # Define weight matrix for MatMul
    weight_name = f"{node_name}_weight"
    kernel_shape = (
        np.prod(in_features).item(),
        np.prod(out_features).item(),
    )  # Compute kernel shape dynamically
    transposed_kernel = self.kernel.value.reshape(kernel_shape)

    # Store in ONNX format
    onnx_graph.add_initializer(
        oh.make_tensor(
            name=weight_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=list(kernel_shape),
            vals=transposed_kernel.flatten().astype(np.float32).tolist(),
        )
    )

    # Call MatMul plugin
    z = to_onnx_matmul(
        Z(
            [reshaped_input_shape, kernel_shape],
            [reshaped_input_name, weight_name],
            onnx_graph,
        )
    )

    # Reshape MatMul output to the expected shape
    final_out_name = f"{node_name}_reshaped_output"
    shape_out_name = f"{node_name}_reshape_output_shape"

    shape_out_tensor = oh.make_tensor(
        name=shape_out_name,
        data_type=onnx.TensorProto.INT64,
        dims=[len(output_shape)],
        vals=[int(dim) for dim in output_shape],
    )
    onnx_graph.add_initializer(shape_out_tensor)
    onnx_graph.add_node(
        oh.make_node(
            "Reshape",
            inputs=[z.names[0], shape_out_name],
            outputs=[final_out_name],
            name=f"{node_name}_reshape_output",
        )
    )
    onnx_graph.add_local_outputs([output_shape], [final_out_name])

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
        bias_out_name = f"{node_name}_output"
        onnx_graph.add_node(
            oh.make_node(
                "Add",
                inputs=[final_out_name, bias_name],
                outputs=[bias_out_name],
                name=f"{node_name}_add_bias",
            )
        )
        onnx_graph.add_local_outputs([output_shape], [bias_out_name])
        return Z([output_shape], [bias_out_name], onnx_graph)

    return Z([output_shape], [final_out_name], onnx_graph)


# Attach the ONNX conversion function to `nnx.LinearGeneral`
nnx.LinearGeneral.to_onnx = to_onnx


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of `nnx.LinearGeneral`.
    """
    return [
        {
            "model_name": "linear_general",
            "model": nnx.LinearGeneral(
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 8, 32)],
            "to_onnx": nnx.LinearGeneral.to_onnx,
        },
        {
            "model_name": "linear_general_2",
            "model": nnx.LinearGeneral(
                in_features=(256,),
                out_features=(8, 32),
                axis=(-1,),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 256)],
            "to_onnx": nnx.LinearGeneral.to_onnx,
        },
        {
            "model_name": "linear_general_mha_projection",
            "model": nnx.LinearGeneral(
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 8, 32)],
            "to_onnx": nnx.LinearGeneral.to_onnx,
        },
    ]
