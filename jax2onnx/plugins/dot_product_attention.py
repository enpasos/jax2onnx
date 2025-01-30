# file: jax2onnx/plugins/dot_product_attention.py

import onnx
import onnx.helper as oh
import numpy as np
import jax
import jax.numpy as jnp
from jax2onnx.transpose_utils import (
    transpose_to_onnx,
    transpose_to_jax,
    jax_shape_to_onnx_shape,
    onnx_shape_to_jax_shape
)


# here we are going for the flax.nnx function
# https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/attention.html#flax.nnx.dot_product_attention
# its main parameters are query, key, value (no bias for now)
# lets loot at the dimensions in the order JAX uses (as we do not have a corresponding ONNX component we simply follow the JAX order)
# query q: (B, N, H, E)
# key k: (B, M, H, E)
# value v: (B, M, H, D)
# where B is the batch size,
# H is the number of heads,
# N is the number of queries,
# M is the number of keys,
# E is the embedding dimension,
# and D is the output dimension
# split model dimension D into H heads of size
# d=D/H


def build_dot_product_attention_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    if not isinstance(parameters, dict):
        raise ValueError("dot_product_attention parameters must be a dict.")

    # 1) Parse parameters
    softmax_axis = parameters.get("softmax_axis", -1)
    # apply_pre_transpose = parameters.get("apply_pre_transpose", False)
    # pre_transpose_perm = parameters.get("pre_transpose_perm", [0, 1, 2, 3])
    # apply_post_transpose = parameters.get("apply_post_transpose", False)
    # post_transpose_perm = parameters.get("post_transpose_perm", [0, 1, 2, 3])

    if len(jax_inputs) < 3:
        raise ValueError("dot_product_attention requires at least 3 inputs: [query, key, value].")

    # has_bias = (len(jax_inputs) == 4)
    q_jax, k_jax, v_jax = jax_inputs[:3]
    # bias = jax_inputs[3] if has_bias else None

    q_onnx_name, k_onnx_name, v_onnx_name = input_names[:3]
    # bias_name = input_names[3] if has_bias else None

    # 2) Reference JAX call
    jax_attention_output = jax.nn.dot_product_attention(q_jax, k_jax, v_jax)  #, bias=bias)

    node_prefix = f"node{onnx_graph.counter_plusplus()}"


    # jax_b = bias


    # for scaling calculate d=D/H
    d = q_jax.shape[-1] // q_jax.shape[-2]
    # multiply over the last dimension EE (queries · keys) and keep the other dimensions (B,N,H,M) .
    attn_logits = jnp.einsum('BNHE,BMHE->BNHM', q_jax, k_jax) / np.sqrt(d)


    # optional masking we take care of later


    # 4) Build the ONNX subgraph for Q*K^T => Softmax => *V
    k_t_name = f"{node_prefix}_k_t"
    onnx_graph.add_node(
        oh.make_node(
            "Transpose",
            inputs=[trans_k_name],
            outputs=[k_t_name],
            name=f"{node_prefix}_transpose_kT",
            perm=[0, 1, 3, 2]
        )
    )
    jax_k_t = jnp.transpose(jax_k, [0, 2, 1, 3])
    onnx_graph.add_local_outputs([jax_k_t], [k_t_name])

    scores_name = f"{node_prefix}_scores"
    onnx_graph.add_node(
        oh.make_node(
            "MatMul",
            inputs=[trans_q_name, k_t_name],
            outputs=[scores_name],
            name=f"{node_prefix}_matmul_qk"
        )
    )
    jax_q_k_t = jnp.matmul(jax_q, jax_k_t)
    onnx_graph.add_local_outputs([jax_q_k_t], [scores_name])

    final_scores_name = scores_name
    # if has_bias:
    #     final_scores_name = f"{node_prefix}_scores_bias"
    #     onnx_graph.add_node(
    #         oh.make_node(
    #             "Add",
    #             inputs=[scores_name, trans_bias_name],
    #             outputs=[final_scores_name],
    #             name=f"{node_prefix}_add_bias"
    #         )
    #     )

    # Softmax
    ndims = len(jax_v.shape)
    if softmax_axis < 0:
        softmax_axis = softmax_axis % ndims
    attn_weights_name = f"{node_prefix}_attn_weights"
    onnx_graph.add_node(
        oh.make_node(
            "Softmax",
            inputs=[final_scores_name],
            outputs=[attn_weights_name],
            name=f"{node_prefix}_softmax",
            axis=softmax_axis
        )
    )

    # output = attn_weights * V
    attn_output_name = f"{node_prefix}_attn_output"
    onnx_graph.add_node(
        oh.make_node(
            "MatMul",
            inputs=[attn_weights_name, trans_v_name],
            outputs=[attn_output_name],
            name=f"{node_prefix}_matmul_attn_v"
        )
    )

    # 5) Optional post-transpose
    final_output_name = attn_output_name
    final_output_jax = jax_attention_output

    # if apply_post_transpose:
    #     final_output_name = f"{node_prefix}_output_transposed"
    #     onnx_graph.add_node(
    #         oh.make_node(
    #             "Transpose",
    #             inputs=[attn_output_name],
    #             outputs=[final_output_name],
    #             name=f"{node_prefix}_transpose_output",
    #             perm=post_transpose_perm
    #         )
    #     )
    #     final_output_jax = jnp.transpose(jax_attention_output, post_transpose_perm)
    #
    #     # Register shape info for final output
    #     onnx_graph.value_info.append(
    #         oh.make_tensor_value_info(
    #             final_output_name,
    #             onnx.TensorProto.FLOAT,
    #             list(final_output_jax.shape),
    #         )
    #     )

    # 6) Register final outputs
    onnx_graph.add_local_outputs([final_output_jax], [final_output_name])
    return [jax_attention_output], [final_output_name]


jax.nn.dot_product_attention.build_onnx_node = build_dot_product_attention_onnx_node


def get_test_params():
    return [
        {
            "model_name": "dot_product_attention_basic",
            "model": lambda: lambda q, k, v: jax.nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 16)],  # Q, K, V
            "build_onnx_node": jax.nn.dot_product_attention.build_onnx_node,
            "parameters": {
                "softmax_axis": 1,
                "apply_pre_transpose": False,
                "pre_transpose_perm": [0, 1, 3, 2],
                "apply_post_transpose": False,
            },
        },
    ]
