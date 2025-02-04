# file: jax2onnx/plugins/dot_product_attention.py

import onnx
import onnx.helper as oh
import jax.numpy as jnp
import flax.nnx as nnx

def build_dot_product_attention_onnx_node(function, input_shapes, input_names, onnx_graph, parameters=None):
    if not isinstance(parameters, dict):
        raise ValueError("dot_product_attention parameters must be a dict.")

    softmax_axis = parameters.get("softmax_axis", -1)
    if len(input_shapes) < 3:
        raise ValueError("dot_product_attention requires at least 3 inputs: [query, key, value].")

    q_shape, k_shape, v_shape = input_shapes[:3]
    q_onnx_name, k_onnx_name, v_onnx_name = input_names[:3]

    node_prefix = f"node{onnx_graph.counter_plusplus()}"

    # Compute scaling factor d = sqrt(E)
    depth = int(k_shape[-1])  # Ensure depth is an integer scalar
    scale_value = 1.0 / jnp.sqrt(depth)


    # Create ONNX constant for the scaling factor
    scale_const_name = f"{node_prefix}_scale_const"
    onnx_graph.add_node(
        oh.make_node(
            "Constant",
            inputs=[],
            outputs=[scale_const_name],
            name=f"{node_prefix}_const",
            value=oh.make_tensor(
                name=scale_const_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=[],
                vals=[scale_value]
            )
        )
    )

    # Compute attention scores using Einsum: (BNHE, BMHE) -> (BNHM)
    attn_scores_name = f"{node_prefix}_attn_scores"
    attn_scores_shape = [q_shape[0], q_shape[1], q_shape[2], k_shape[2]]  # B, N, H, M
    onnx_graph.add_node(
        oh.make_node(
            "Einsum",
            inputs=[q_onnx_name, k_onnx_name],
            outputs=[attn_scores_name],
            name=f"{node_prefix}_einsum_qk",
            equation="BNHE,BMHE->BNHM"
        )
    )
    onnx_graph.add_local_outputs([attn_scores_shape], [attn_scores_name])

    # Scale attention scores
    scaled_attn_scores_name = f"{node_prefix}_scaled_attn_scores"
    onnx_graph.add_node(
        oh.make_node(
            "Mul",
            inputs=[attn_scores_name, scale_const_name],
            outputs=[scaled_attn_scores_name],
            name=f"{node_prefix}_scale"
        )
    )
    onnx_graph.add_local_outputs([attn_scores_shape], [scaled_attn_scores_name])

    # Apply softmax on correct axis
    ndims = len(q_shape)
    softmax_axis = softmax_axis if softmax_axis >= 0 else (softmax_axis % ndims)
    attn_weights_name = f"{node_prefix}_attn_weights"
    onnx_graph.add_node(
        oh.make_node(
            "Softmax",
            inputs=[scaled_attn_scores_name],
            outputs=[attn_weights_name],
            name=f"{node_prefix}_softmax",
            axis=softmax_axis
        )
    )
    onnx_graph.add_local_outputs([attn_scores_shape], [attn_weights_name])

    # Compute final attention output using Einsum: (BNHM, BMHE) -> (BNHE)
    attn_output_name = f"{node_prefix}_attn_output"
    attn_output_shape = [q_shape[0], q_shape[1], q_shape[2], v_shape[-1]]  # B, N, H, E
    onnx_graph.add_node(
        oh.make_node(
            "Einsum",
            inputs=[attn_weights_name, v_onnx_name],
            outputs=[attn_output_name],
            name=f"{node_prefix}_einsum_attn_v",
            equation="BNHM,BMHE->BNHE"
        )
    )
    onnx_graph.add_local_outputs([attn_output_shape], [attn_output_name])

    return [attn_output_shape], [attn_output_name]


# Assign ONNX node builder to dot_product_attention function
nnx.dot_product_attention.build_onnx_node = build_dot_product_attention_onnx_node

# Example test parameters
def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of dot-product attention.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "dot_product_attention",
            "model": lambda: lambda q, k, v: nnx.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 16)],
            "build_onnx_node": nnx.dot_product_attention.build_onnx_node,

        },
    ]
