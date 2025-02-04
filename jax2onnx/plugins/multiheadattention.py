# file: jax2onnx/plugins/multiheadattention.py

import onnx
import onnx.helper as oh
import jax.numpy as jnp
import flax.nnx as nnx

def build_multihead_attention_onnx_node(
        self,  # The instance of nnx.MultiHeadAttention
        input_shapes,
        input_names,
        onnx_graph,
        parameters=None
):
    """
    Converts MultiHeadAttention to ONNX format.

    Steps:
      1) Use `LinearGeneral` for query, key, value projections.
      2) Reshape Q/K/V -> (B, L, heads, head_dim)
      3) Use dot_product_attention
      4) Flatten heads back to (B, L, num_heads * head_dim)
      5) Use `LinearGeneral` for final projection.
      6) Return [output_shape], [output_name]
    """

    if parameters is None:
        parameters = {}

    # 1) Validate input
    if len(input_shapes) < 1:
        raise ValueError("MultiHeadAttention expects at least one input (Q=K=V).")

    input_shape = input_shapes[0]  # (B, L, in_features)
    input_name  = input_names[0]
    B, L, in_features = input_shape

    # 2) Retrieve MHA config from `self`
    num_heads     = self.num_heads
    qkv_features  = self.qkv_features or in_features
    out_features  = self.out_features
    head_dim      = qkv_features // num_heads

    node_prefix = f"node{onnx_graph.counter_plusplus()}"

    # Use LinearGeneral for Q, K, V
    q_out_shape, [q_name] = self.query.build_onnx_node([input_shape], [input_name], onnx_graph)
    k_out_shape, [k_name] = self.key.build_onnx_node([input_shape], [input_name], onnx_graph)
    v_out_shape, [v_name] = self.value.build_onnx_node([input_shape], [input_name], onnx_graph)

    # Reshape to (B, L, num_heads, head_dim)
    def reshape_blnh(in_name, prefix):
        out_name = f"{in_name}_4d"
        shape_name = f"{prefix}_4d_shape"
        shape_4d = [B, L, num_heads, head_dim]

        onnx_graph.add_node(
            oh.make_node(
                "Constant",
                inputs=[],
                outputs=[shape_name],
                name=f"{prefix}_shape_const",
                value=oh.make_tensor(
                    name=shape_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(shape_4d)],
                    vals=shape_4d
                )
            )
        )
        onnx_graph.add_node(
            oh.make_node(
                "Reshape",
                inputs=[in_name, shape_name],
                outputs=[out_name],
                name=f"{prefix}_reshape"
            )
        )
        onnx_graph.add_local_outputs([shape_4d], [out_name])

        return out_name

    q_4d = reshape_blnh(q_name, f"{node_prefix}_q")
    k_4d = reshape_blnh(k_name, f"{node_prefix}_k")
    v_4d = reshape_blnh(v_name, f"{node_prefix}_v")

    # 3) Reuse dot_product_attention
    dpa_params = {"softmax_axis": -1}
    attn_out_shape, [attn_out_name] = nnx.dot_product_attention.build_onnx_node(
        [q_out_shape[0], k_out_shape[0], v_out_shape[0]], [q_4d, k_4d, v_4d], onnx_graph, dpa_params
    )

    # 4) Merge heads -> (B, L, num_heads * head_dim)
    merged_name = f"{attn_out_name}_merged"
    merged_shape_val = [B, L, num_heads * head_dim]
    shape_name = f"{merged_name}_shape_const"
    onnx_graph.add_node(
        oh.make_node(
            "Constant",
            inputs=[],
            outputs=[shape_name],
            name=f"{node_prefix}_merged_shape_init",
            value=oh.make_tensor(
                name=shape_name,
                data_type=onnx.TensorProto.INT64,
                dims=[len(merged_shape_val)],
                vals=merged_shape_val
            )
        )
    )
    onnx_graph.add_node(
        oh.make_node(
            "Reshape",
            inputs=[attn_out_name, shape_name],
            outputs=[merged_name],
            name=f"{node_prefix}_reshape_merge"
        )
    )
    merged_shape = [B, L, num_heads * head_dim]
    onnx_graph.add_local_outputs([merged_shape], [merged_name])

    # 5) Final projection using LinearGeneral
    final_out_shape, [final_name] = self.out.build_onnx_node([merged_shape], [merged_name], onnx_graph)

    return final_out_shape, [final_name]


# Attach as instance method
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
                rngs=nnx.Rngs(0)
            ),

            # Input shape => (B, L, in_features)
            "input_shapes": [(2, 4, 256)],
            # We rely on our instance-based plugin
            "build_onnx_node": nnx.MultiHeadAttention.build_onnx_node,
        }
    ]
