# file: jax2onnx/plugins/dot_product_attention.py

import flax.nnx as nnx
import jax.numpy as jnp
import onnx
import onnx.helper as oh


def build_dot_product_attention_onnx_node(z, **params):
    """
    Converts `nnx.dot_product_attention` into an ONNX node.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        **params: Additional parameters such as `softmax_axis`.

    Returns:
        Z: Updated instance with new shapes and names.
    """

    input_shapes = z.shapes
    input_names = z.names
    onnx_graph = z.onnx_graph

    # ✅ Provide a default value if softmax_axis is missing
    softmax_axis = params.get("softmax_axis", -1)

    if len(input_shapes) < 3:
        raise ValueError(
            "dot_product_attention requires at least 3 inputs: [query, key, value]."
        )

    q_shape, k_shape, v_shape = input_shapes[:3]
    q_onnx_name, k_onnx_name, v_onnx_name = input_names[:3]

    node_prefix = f"node{onnx_graph.next_id()}"

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
                vals=[scale_value],
            ),
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
            equation="BNHE,BMHE->BNHM",
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
            name=f"{node_prefix}_scale",
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
            axis=softmax_axis,
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
            equation="BNHM,BMHE->BNHE",
        )
    )
    onnx_graph.add_local_outputs([attn_output_shape], [attn_output_name])

    z.shapes = [attn_output_shape]
    z.names = [attn_output_name]
    z.jax_function = nnx.dot_product_attention
    return z


# Assign ONNX node builder to dot_product_attention function
nnx.dot_product_attention.to_onnx = build_dot_product_attention_onnx_node


# Example test parameters
def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of dot-product attention.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "testcase": "dot_product_attention",
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "to_onnx": nnx.dot_product_attention.to_onnx,
        },
        {
            "testcase": "dot_product_attention_shape_check",
            "input_shapes": [
                (2, 4, 8, 16),
                (2, 6, 8, 16),
                (2, 6, 8, 16),
            ],  # K has different sequence length
            "to_onnx": nnx.dot_product_attention.to_onnx,
        },
        {
            "testcase": "dot_product_attention_softmax_axis",
            "input_shapes": [(2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 16)],
            "to_onnx": nnx.dot_product_attention.to_onnx,
            "params": {"softmax_axis": -1},  # Explicit test for softmax axis
        },
    ]
