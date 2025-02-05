# file: jax2onnx/plugins/multiheadattention.py
import flax.nnx as nnx


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
      1) Apply `LinearGeneral` for query, key, and value projections.
      2) Process Q/K/V without additional reshaping (aligned with internal structure).
      3) Apply dot-product attention.
      4) Apply the final output projection using `LinearGeneral`.
      5) Return the final output shape and corresponding ONNX node name.
    """

    if parameters is None:
        parameters = {}

    # Validate input
    if len(input_shapes) < 1:
        raise ValueError("MultiHeadAttention expects at least one input (Q=K=V).")

    input_shape = input_shapes[0]  # (B, L, in_features)
    input_name = input_names[0]

    # Retrieve MHA configuration
    q_out_shape, [q_name] = self.query.to_onnx([input_shape], [input_name], onnx_graph)
    k_out_shape, [k_name] = self.key.to_onnx([input_shape], [input_name], onnx_graph)
    v_out_shape, [v_name] = self.value.to_onnx([input_shape], [input_name], onnx_graph)

    # Apply dot-product attention
    dpa_params = {"softmax_axis": -1}
    attn_out_shape, [attn_out_name] = nnx.dot_product_attention.to_onnx(
        function=lambda q, k, v: nnx.dot_product_attention(q, k, v),
        input_shapes=[q_out_shape[0], k_out_shape[0], v_out_shape[0]],
        input_names=[q_name, k_name, v_name],
        onnx_graph=onnx_graph,
        parameters=dpa_params
    )

    # Apply final projection
    final_out_shape, [final_name] = self.out.to_onnx(
        attn_out_shape, [attn_out_name], onnx_graph
    )

    return final_out_shape, [final_name]


# Attach ONNX conversion function to nnx.MultiHeadAttention
nnx.MultiHeadAttention.to_onnx = build_multihead_attention_onnx_node

def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of MultiHeadAttention.

    The test case verifies the correct transformation of query, key, and value inputs,
    followed by the attention computation and the final projection step.
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
                decode=False  # Explicitly set decode=False
            ),
            "input_shapes": [(2, 4, 256)],  # Input shape: (Batch=2, SeqLength=4, Features=256)
        }
    ]
