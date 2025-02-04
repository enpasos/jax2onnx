# file: jax2onnx/plugins/multiheadattention.py


import jax.numpy as jnp
import flax.nnx as nnx
from jax2onnx.plugins.reshape import build_reshape_onnx_node

def build_multihead_attention_onnx_node(
        self,  # Instance of nnx.MultiHeadAttention
        input_shapes,
        input_names,
        onnx_graph,
        parameters=None
):
    """
    Converts MultiHeadAttention to ONNX format.

    Steps:
      1) Use `LinearGeneral` for query, key, value projections.
      2) Reshape Q/K/V -> (B, L, num_heads, head_dim)
      3) Apply dot_product_attention.
      4) Merge heads back to (B, L, num_heads * head_dim).
      5) Apply final projection using `LinearGeneral`.
      6) Return output shape and output name.
    """

    if parameters is None:
        parameters = {}

    # 1) Validate input
    if len(input_shapes) < 1:
        raise ValueError("MultiHeadAttention expects at least one input (Q=K=V).")

    input_shape = input_shapes[0]  # (B, L, in_features)
    input_name = input_names[0]
    B, L, in_features = input_shape

    # 2) Retrieve MHA config
    num_heads = self.num_heads
    qkv_features = self.qkv_features or in_features
    head_dim = qkv_features // num_heads


    q_out_shape, [q_name] = self.query.build_onnx_node([input_shape], [input_name], onnx_graph)
    k_out_shape, [k_name] = self.key.build_onnx_node([input_shape], [input_name], onnx_graph)
    v_out_shape, [v_name] = self.value.build_onnx_node([input_shape], [input_name], onnx_graph)

    # 4) Reshape to (B, L, num_heads, head_dim)
    q_4d_shape, [q_4d] = build_reshape_onnx_node(jnp.reshape, q_out_shape, [q_name], onnx_graph, {"shape": (B, L, num_heads, head_dim)})
    k_4d_shape, [k_4d] = build_reshape_onnx_node(jnp.reshape, k_out_shape, [k_name], onnx_graph, {"shape": (B, L, num_heads, head_dim)})
    v_4d_shape, [v_4d] = build_reshape_onnx_node(jnp.reshape, v_out_shape, [v_name], onnx_graph, {"shape": (B, L, num_heads, head_dim)})


    # 5) Apply dot_product_attention
    dpa_params = {"softmax_axis": -1}
    attn_out_shape, [attn_out_name] = nnx.dot_product_attention.build_onnx_node(
        function=lambda q, k, v: nnx.dot_product_attention(q, k, v),
        input_shapes=[q_4d_shape[0], k_4d_shape[0], v_4d_shape[0]],
        input_names=[q_4d, k_4d, v_4d],
        onnx_graph=onnx_graph,
        parameters=dpa_params
    )

    print(f"DEBUG: Shape after attention: {attn_out_shape}")



    final_out_shape, [final_name] = self.out.build_onnx_node(
        attn_out_shape, [attn_out_name], onnx_graph
    )

    print(f"DEBUG: Final projection shape: {final_out_shape}")

    return final_out_shape, [final_name]


# Attach ONNX conversion function to nnx.MultiHeadAttention
nnx.MultiHeadAttention.build_onnx_node = build_multihead_attention_onnx_node

def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of MultiHeadAttention.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "multihead_attention",
            "model": lambda:
            nnx.MultiHeadAttention(
                num_heads=8,
                in_features=256,
                qkv_features=256,
                out_features=256,
                rngs=nnx.Rngs(0),
                decode=False  # ✅ Explicitly set decode=False
            ),
            "input_shapes": [(2, 4, 256)],  # (B=2, L=4, in_features=256)
        }
    ]
