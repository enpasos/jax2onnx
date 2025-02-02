# file: jax2onnx/plugins/linear_general.py

import onnx.helper as oh
import onnx
import numpy as np
from flax import nnx

def build_onnx_node(self, input_shapes, input_names, onnx_graph, parameters=None):
    """
    Constructs an ONNX node for `LinearGeneral`, ensuring proper handling of input reshaping
    and transformation, especially for cases like multi-head attention.
    """
    if parameters is None:
        parameters = {}

    if len(input_shapes) < 1:
        raise ValueError("Expected at least one input for LinearGeneral.")

    input_shape = input_shapes[0]
    input_name = input_names[0]

    in_features = tuple(self.in_features)  # Ensure tuple format
    out_features = tuple(self.out_features)  # Ensure tuple format
    axis = tuple(self.axis)  # Ensure tuple format
    use_bias = self.use_bias

    # Compute batch dimensions
    batch_dims = list(input_shape[:-len(axis)])
    output_shape = batch_dims + list(out_features)

    node_name = f"node{onnx_graph.counter_plusplus()}"

    # Flatten input if necessary for ONNX compatibility
    reshaped_input_name = input_name
    reshaped_input_shape = input_shape
    if len(axis) > 1:
        reshaped_input_name = f"{node_name}_reshaped_input"
        feature_dim_prod = np.prod(in_features)
        reshaped_input_shape = batch_dims + [feature_dim_prod]

        shape_name = f"{node_name}_reshape_shape"
        shape_tensor = oh.make_tensor(
            name=shape_name,
            data_type=onnx.TensorProto.INT64,
            dims=[len(reshaped_input_shape)],
            vals=np.array(reshaped_input_shape, dtype=np.int64).tolist(),
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

    # Define weight matrix for MatMul
    weight_name = f"{node_name}_weight"
    transposed_kernel = self.kernel.value.reshape((np.prod(in_features), np.prod(out_features)))
    onnx_graph.add_initializer(
        oh.make_tensor(
            weight_name,
            onnx.TensorProto.FLOAT,
            list(transposed_kernel.shape),
            transposed_kernel.flatten().astype(np.float32).tolist(),
        )
    )

    # MatMul operation
    matmul_out_name = f"{node_name}_matmul"
    onnx_graph.add_node(
        oh.make_node(
            "MatMul",
            inputs=[reshaped_input_name, weight_name],
            outputs=[matmul_out_name],
            name=f"{node_name}_matmul",
        )
    )



    # Bias addition if enabled
    final_out_name = matmul_out_name
    if use_bias:
        bias_name = f"{node_name}_bias"
        onnx_graph.add_initializer(
            oh.make_tensor(
                bias_name,
                onnx.TensorProto.FLOAT,
                list(self.bias.shape),
                self.bias.value.flatten().astype(np.float32).tolist(),
            )
        )
        final_out_name = f"{node_name}_output"
        onnx_graph.add_node(
            oh.make_node(
                "Add",
                inputs=[matmul_out_name, bias_name],
                outputs=[final_out_name],
                name=f"{node_name}_add_bias",
            )
        )

    # Reshape output to match expected dimensions
    final_output_name = final_out_name
    if len(axis) > 1:
        final_output_name = f"{node_name}_reshaped_output"
        shape_out_name = f"{node_name}_reshape_output_shape"

        shape_out_tensor = oh.make_tensor(
            name=shape_out_name,
            data_type=onnx.TensorProto.INT64,
            dims=[len(output_shape)],
            vals=np.array(output_shape, dtype=np.int64).tolist(),
        )
        onnx_graph.add_initializer(shape_out_tensor)
        onnx_graph.add_node(
            oh.make_node(
                "Reshape",
                inputs=[final_out_name, shape_out_name],
                outputs=[final_output_name],
                name=f"{node_name}_reshape_output"
            )
        )

    onnx_graph.add_local_outputs([output_shape], [final_output_name])
    return [output_shape], [final_output_name]


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
        }
    ]
