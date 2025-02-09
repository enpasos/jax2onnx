# file: jax2onnx/plugins/multiheadattention.py

import flax.nnx as nnx
from jax2onnx.to_onnx import Z


def to_onnx_multihead_attention(self, z, parameters=None):
    """
    Converts `nnx.MultiHeadAttention` into an ONNX equivalent.

    Steps:
      1) Apply `LinearGeneral` for query, key, and value projections.
      2) Compute dot-product attention.
      3) Apply the final output projection using `LinearGeneral`.
      4) Return the final output shape and corresponding ONNX node name.

    Args:
        self: The `nnx.MultiHeadAttention` instance.
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict, optional): Additional parameters (e.g., `softmax_axis` for attention).

    Returns:
        Z: Updated instance with new shapes and names.
    """

    if parameters is None:
        parameters = {}

    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]  # (B, L, in_features)
    input_name = z.names[0]

    # Query, Key, and Value projections
    z_q = self.query.to_onnx(Z([input_shape], [input_name], onnx_graph))
    z_k = self.key.to_onnx(Z([input_shape], [input_name], onnx_graph))
    z_v = self.value.to_onnx(Z([input_shape], [input_name], onnx_graph))

    # Apply dot-product attention
    dpa_params = {"softmax_axis": -1}
    z_attn = nnx.dot_product_attention.to_onnx(
        Z(
            [z_q.shapes[0], z_k.shapes[0], z_v.shapes[0]],
            [z_q.names[0], z_k.names[0], z_v.names[0]],
            onnx_graph,
        ),
        parameters=dpa_params,
    )

    # Apply final projection
    z_out = self.out.to_onnx(z_attn)

    return z_out


# Attach ONNX conversion function to `nnx.MultiHeadAttention`
nnx.MultiHeadAttention.to_onnx = to_onnx_multihead_attention


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of MultiHeadAttention.

    The test case verifies the correct transformation of query, key, and value inputs,
    followed by the attention computation and the final projection step.
    """
    return [
        {
            "testcase": "multihead_attention",
            "model": nnx.MultiHeadAttention(
                num_heads=8,
                in_features=256,
                qkv_features=256,
                out_features=256,
                rngs=nnx.Rngs(0),
                decode=False,  # Explicitly set decode=False
            ),
            "input_shapes": [
                (2, 4, 256)
            ],  # Input shape: (Batch=2, SeqLength=4, Features=256)
        }
    ]
