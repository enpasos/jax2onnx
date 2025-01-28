# file: jax2onnx/plugins/reshape.py
import onnx.helper as oh
import jax
import jax.numpy as jnp
import onnx
from jax2onnx.transpose_utils import transpose_to_onnx, transpose_to_jax, jax_shape_to_onnx_shape, onnx_shape_to_jax_shape

def build_reshape_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    if not isinstance(parameters, dict) or "shape" not in parameters:
        raise ValueError("Parameters for reshape must include 'shape'.")

    new_shape = tuple(parameters["shape"])
    input_name = input_names[0]

    node_name = f"node{onnx_graph.get_counter()}"
    onnx_graph.increment_counter()
    output_names = [f"{node_name}_output"]

    # Determine if transpositions are needed
    apply_pre_transpose = parameters.get("apply_pre_transpose", False)
    apply_post_transpose = parameters.get("apply_post_transpose", False)
    pre_transpose_perm = parameters.get("pre_transpose_perm", [0, 2, 3, 1])  # Default NCHW → NHWC
    post_transpose_perm = parameters.get("post_transpose_perm", [0, 3, 1, 2])  # Default NHWC → NCHW

    transposed_input_name = input_name
    pre_transposed_tensor = jax_inputs[0]  # Track modified tensor

    # Step 1: Optional Transpose from ONNX (NCHW) to JAX (NHWC)
    if apply_pre_transpose:
        transposed_input_name = f"{node_name}_transposed"
        onnx_graph.add_node(
            oh.make_node(
                "Transpose",
                inputs=[input_name],
                outputs=[transposed_input_name],
                perm=pre_transpose_perm,
                name=f"{node_name}_transpose1"
            )
        )
        pre_transposed_tensor = jnp.transpose(transpose_to_onnx(jax_inputs[0] ), pre_transpose_perm)
        onnx_graph.add_local_outputs([pre_transposed_tensor], [transposed_input_name])  # Track intermediate output
        onnx_graph.value_info.append(oh.make_tensor_value_info(transposed_input_name, onnx.TensorProto.FLOAT, jax_shape_to_onnx_shape(jax_inputs[0].shape)))

    # Step 2: Reshape using JAX-like shape
    shape_tensor_name = f"{node_name}_shape"
    onnx_graph.add_initializer(
        oh.make_tensor(
            name=shape_tensor_name,
            data_type=onnx.TensorProto.INT64,
            dims=[len(new_shape)],
            vals=list(new_shape),
        )
    )

    reshaped_output_name = f"{node_name}_reshaped"
    onnx_graph.add_node(
        oh.make_node(
            "Reshape",
            inputs=[transposed_input_name, shape_tensor_name],
            outputs=[reshaped_output_name],
            name=f"{node_name}_reshape"
        )
    )
    reshaped_tensor = pre_transposed_tensor.reshape(new_shape)
    onnx_graph.add_local_outputs([reshaped_tensor], [reshaped_output_name])  # Track reshape output

    final_output_name = reshaped_output_name
    final_output_tensor = reshaped_tensor  # Track modified tensor

    # Step 3: Optional Transpose back from NHWC to ONNX format (NCHW)
    if apply_post_transpose:
        final_output_name = f"{node_name}_final_output"
        onnx_graph.add_node(
            oh.make_node(
                "Transpose",
                inputs=[reshaped_output_name],
                outputs=[final_output_name],
                perm=post_transpose_perm,
                name=f"{node_name}_transpose2"
            )
        )
        final_output_tensor = jnp.transpose(reshaped_tensor, post_transpose_perm)
        onnx_graph.add_local_outputs([final_output_tensor], [final_output_name])  # Track final output

    # Register final output in ONNX graph
    onnx_graph.add_local_outputs([final_output_tensor], [final_output_name])
    return [jax_inputs[0].reshape(new_shape)], [final_output_name]


# Assign the reshape node builder
jax.numpy.reshape.build_onnx_node = build_reshape_onnx_node


# Example test parameters
def get_test_params():
    return [
        {
            "model_name": "reshape",
            "model": lambda: lambda x: jnp.reshape(x, shape=(10, 3)),
            "input_shapes": [(30,)],
            "build_onnx_node": jnp.reshape.build_onnx_node,
            "parameters": {"shape": (10, 3)},  # Simple reshape with no transpositions
        },

        {
            "model_name": "reshape2",
            "model": lambda: lambda x: jnp.reshape(x, (3, 3136)),
            "input_shapes": [(3, 7, 7, 64)],
            "build_onnx_node": jnp.reshape.build_onnx_node,
            "parameters": {
                "shape": (3, 3136),
                "apply_pre_transpose": True,  # Enable pre-transposition
                "pre_transpose_perm": [0, 2, 3, 1],  # Custom NCHW → NHWC transposition
                "apply_post_transpose": False,  # Disable post-transposition
            },
        },
    ]
