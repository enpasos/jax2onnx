# file: jax2onnx/plugins/multiheadattention.py

import onnx
import onnx.helper as oh
import jax.numpy as jnp
import jax.random
import flax.nnx as nnx

def build_multihead_attention_onnx_node(self, input_shapes, input_names, onnx_graph, parameters = None):
    """
    Because we do `nnx.MultiHeadAttention.build_onnx_node = build_multihead_attention_onnx_node`,
    Python expects signature: (self, input_shapes, input_names, onnx_graph, parameters).
    """
    # 1) Extract relevant config from 'self' (actual MHA instance)
    in_features = self.in_features
    num_heads = self.num_heads
    qkv_features = self.qkv_features or in_features
    # etc.

    # 2) For demonstration, let's do a minimal placeholder
    node_prefix = f"node{onnx_graph.counter_plusplus()}"
    rng = jax.random.PRNGKey(0)

    # Suppose input_shapes[0] => (B, L, in_features)
    q_shape = input_shapes[0]
    q_onnx_name = input_names[0]

    # Create random QKV param
    qkv_weight_shape = (in_features, 3 * qkv_features)
    qkv_data = jax.random.normal(rng, qkv_weight_shape).astype(jnp.float32).ravel()

    qkv_weight_name = f"{node_prefix}_qkv_weight"
    onnx_graph.add_node(
        oh.make_node(
            "Constant",
            inputs=[],
            outputs=[qkv_weight_name],
            name=f"{node_prefix}_const_qkv_weight",
            value=oh.make_tensor(
                name=qkv_weight_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=qkv_weight_shape,
                vals=qkv_data,
            )
        )
    )

    # Output minimal shape
    output_shape = [q_shape[0], q_shape[1], in_features]
    output_name = f"{node_prefix}_output"

    # "Identity" node for demonstration
    onnx_graph.add_node(
        oh.make_node(
            "Identity",
            inputs=[q_onnx_name],
            outputs=[output_name],
            name=f"{node_prefix}_identity"
        )
    )
    onnx_graph.add_local_outputs([output_shape], [output_name])

    return [output_shape], [output_name]

# Attach method-based plugin
nnx.MultiHeadAttention.build_onnx_node = build_multihead_attention_onnx_node

def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of multi-head attention.
    """
    return [
        {
            "model_name": "multihead_attention",
            # Provide required args to MHA plus decode in the forward pass
            "model": lambda: (lambda q: nnx.MultiHeadAttention(
                num_heads=8,
                in_features=256,
                qkv_features=256,
                out_features=256,
                rngs=nnx.Rngs(0),
            )(q, decode=False)),
            "input_shapes": [(1, 197, 256)],
            "build_onnx_node": nnx.MultiHeadAttention.build_onnx_node,
            "parameters": {},
        },
    ]

