# file: jax2onnx/plugins/multiheadattention.py
# file: jax2onnx/plugins/multiheadattention.py

import onnx.helper as oh
import onnx
import numpy as np
from flax import nnx
from jax2onnx.onnx_export import jax_shape_to_onnx_shape


def build_onnx_node(self, jax_inputs, input_names, onnx_graph, parameters=None):
    """
    Builds the ONNX subgraph for a MultiHeadAttention operation.

    Args:
        self: The nnx.MultiHeadAttention instance.
        jax_inputs: A list/tuple of input tensors in JAX format (e.g., [x]).
                    Typically x has shape [batch_size, seq_len, in_features].
        input_names: A list of corresponding input names in the ONNX graph.
        onnx_graph: The ONNX graph being constructed.
        parameters: Additional parameters (e.g., {"decode": False/True}).

    Returns:
        jax_outputs: The output tensors in JAX format.
        output_names: The corresponding output names in the ONNX graph.
    """
    # --------------------------------------------------------
    # 1) Handle decode parameter from 'parameters' (flexible approach).
    # --------------------------------------------------------
    decode = False
    if parameters is not None and "decode" in parameters:
        decode = parameters["decode"]

    # --------------------------------------------------------
    # 2) Extract the actual JAX input tensor(s)
    #    If you only have a single input (self-attention), we grab jax_inputs[0].
    #    If you have Q, K, and V separately, you'd adapt accordingly.
    # --------------------------------------------------------
    if not isinstance(jax_inputs, (list, tuple)):
        raise ValueError("Expected jax_inputs to be a list or tuple of arrays.")
    if len(jax_inputs) < 1:
        raise ValueError("Expected at least one input for MultiHeadAttention.")

    # In a self-attention scenario, just use the first array as Q
    x = jax_inputs[0]

    # --------------------------------------------------------
    # 3) Compute the reference JAX output using decode=...
    # --------------------------------------------------------
    jax_output = self(x, decode=decode)

    # Convert its shape to ONNX format (used later for final output)
    onnx_output_shape = jax_shape_to_onnx_shape(jax_output.shape)

    # Make a unique name for nodes
    node_name = f"node{onnx_graph.counter_plusplus()}"


    # --------------------------------------------------------
    # 4) Extract parameters (weights/bias) from the MHA module
    # --------------------------------------------------------
    num_heads = self.num_heads
    qkv_features = self.qkv_features  # typically num_heads * head_dim
    out_features = self.out_features
    in_features = self.in_features

    # Combine query/key/value weights (each is [in_features, qkv_features]) into one array:
    qkv_weights = np.concatenate(
        [
            self.query.kernel.value,
            self.key.kernel.value,
            self.value.kernel.value
        ],
        axis=-1  # shape = [in_features, 3 * qkv_features]
    ).astype(np.float32)

    # Gemm in ONNX is (M, K) * (K, N) => (M, N). We'll flatten input to (M, K).
    # So we transpose to have shape [3*qkv_features, in_features].
    qkv_weights = qkv_weights.T  # => [3*qkv_features, in_features]
    qkv_weights_name = f"{node_name}_qkv_weights"
    onnx_graph.add_initializer(
        oh.make_tensor(
            qkv_weights_name,
            onnx.TensorProto.FLOAT,
            qkv_weights.shape,
            qkv_weights.flatten()
        )
    )

    # QKV bias if self.use_bias
    bias_names = []
    if self.use_bias:
        qkv_bias = np.concatenate(
            [
                self.query.bias.value,
                self.key.bias.value,
                self.value.bias.value
            ],
            axis=0
        ).astype(np.float32)  # => [3*qkv_features]
        qkv_bias_name = f"{node_name}_qkv_bias"
        onnx_graph.add_initializer(
            oh.make_tensor(
                qkv_bias_name,
                onnx.TensorProto.FLOAT,
                qkv_bias.shape,
                qkv_bias.flatten()
            )
        )
        bias_names.append(qkv_bias_name)

    # Output projection
    out_proj_weight = self.out.kernel.value.astype(np.float32)  # [qkv_features, out_features]
    out_proj_weight = out_proj_weight.T  # => [out_features, qkv_features]
    out_proj_weight_name = f"{node_name}_out_proj_weight"
    onnx_graph.add_initializer(
        oh.make_tensor(
            out_proj_weight_name,
            onnx.TensorProto.FLOAT,
            out_proj_weight.shape,
            out_proj_weight.flatten()
        )
    )

    out_proj_bias_names = []
    if self.out.use_bias:
        out_proj_bias = self.out.bias.value.astype(np.float32)  # shape [out_features]
        out_proj_bias_name = f"{node_name}_out_proj_bias"
        onnx_graph.add_initializer(
            oh.make_tensor(
                out_proj_bias_name,
                onnx.TensorProto.FLOAT,
                out_proj_bias.shape,
                out_proj_bias.flatten()
            )
        )
        out_proj_bias_names.append(out_proj_bias_name)

    # --------------------------------------------------------
    # 5) QKV projection via Gemm
    #    Flatten -> Gemm -> Reshape -> Split => Q/K/V
    # --------------------------------------------------------
    x_name = input_names[0]  # The ONNX input name for x

    # (a) Flatten x => [batch_size*seq_len, in_features]
    flattened_x_name = f"{node_name}_flattened_x"
    onnx_graph.add_node(
        oh.make_node(
            "Flatten",
            inputs=[x_name],
            outputs=[flattened_x_name],
            name=f"{node_name}_flatten_x",
            axis=1
        )
    )

    # (b) Gemm => QKV
    qkv_output_name = f"{node_name}_qkv_output"
    gemm_inputs = [flattened_x_name, qkv_weights_name] + bias_names
    onnx_graph.add_node(
        oh.make_node(
            "Gemm",
            inputs=gemm_inputs,
            outputs=[qkv_output_name],
            name=f"{node_name}_gemm_qkv",
            alpha=1.0,
            beta=1.0,
            transA=False,
            transB=False
        )
    )

    # (c) Reshape QKV => [batch_size, seq_len, 3*qkv_features]
    qkv_3d_name = f"{node_name}_qkv_3d"
    shape_qkv_3d_name = f"{node_name}_shape_qkv_3d"

    # For demonstration, assume we know the shape of x
    B, L, _ = x.shape
    target_shape_qkv = np.array([B, L, 3 * qkv_features], dtype=np.int64)
    onnx_graph.add_initializer(
        oh.make_tensor(
            shape_qkv_3d_name,
            onnx.TensorProto.INT64,
            target_shape_qkv.shape,
            target_shape_qkv
        )
    )
    onnx_graph.add_node(
        oh.make_node(
            "Reshape",
            inputs=[qkv_output_name, shape_qkv_3d_name],
            outputs=[qkv_3d_name],
            name=f"{node_name}_reshape_qkv"
        )
    )

    # Split => Q, K, V
    q_name = f"{node_name}_Q"
    k_name = f"{node_name}_K"
    v_name = f"{node_name}_V"
    onnx_graph.add_node(
        oh.make_node(
            "Split",
            inputs=[qkv_3d_name],
            outputs=[q_name, k_name, v_name],
            name=f"{node_name}_split_qkv",
            axis=2,
            split=[qkv_features, qkv_features, qkv_features]
        )
    )

    # --------------------------------------------------------
    # 6) Reshape + Transpose each of Q, K, V to [B, num_heads, seq_len, head_dim]
    # --------------------------------------------------------
    head_dim = qkv_features // num_heads

    def reshape_transpose_4d(name_in, prefix):
        reshaped_name = f"{prefix}_reshaped"
        transposed_name = f"{prefix}_transposed"
        shape_4d_name = f"{prefix}_shape4d"
        shape_4d = np.array([B, L, num_heads, head_dim], dtype=np.int64)

        onnx_graph.add_initializer(
            oh.make_tensor(
                shape_4d_name,
                onnx.TensorProto.INT64,
                shape_4d.shape,
                shape_4d
            )
        )
        onnx_graph.add_node(
            oh.make_node(
                "Reshape",
                inputs=[name_in, shape_4d_name],
                outputs=[reshaped_name],
                name=f"{prefix}_reshape"
            )
        )
        onnx_graph.add_node(
            oh.make_node(
                "Transpose",
                inputs=[reshaped_name],
                outputs=[transposed_name],
                name=f"{prefix}_transpose",
                perm=[0, 2, 1, 3]  # [B, num_heads, seq_len, head_dim]
            )
        )
        return transposed_name

    q_trans = reshape_transpose_4d(q_name, f"{node_name}_Q")
    k_trans = reshape_transpose_4d(k_name, f"{node_name}_K")
    v_trans = reshape_transpose_4d(v_name, f"{node_name}_V")

    # --------------------------------------------------------
    # 7) Scaled Dot-Product Attention: Q*K^T => Softmax => * V
    # --------------------------------------------------------
    # K^T
    k_t_name = f"{node_name}_K_t"
    onnx_graph.add_node(
        oh.make_node(
            "Transpose",
            inputs=[k_trans],
            outputs=[k_t_name],
            name=f"{node_name}_transpose_KT",
            perm=[0, 1, 3, 2]  # [B, num_heads, head_dim, seq_len]
        )
    )
    # MatMul => scores
    scores_name = f"{node_name}_scores"
    onnx_graph.add_node(
        oh.make_node(
            "MatMul",
            inputs=[q_trans, k_t_name],
            outputs=[scores_name],
            name=f"{node_name}_matmul_qk"
        )
    )
    # scale
    scale_value = 1.0 / np.sqrt(head_dim)
    scale_const_name = f"{node_name}_scale_const"
    onnx_graph.add_initializer(
        oh.make_tensor(
            scale_const_name,
            onnx.TensorProto.FLOAT,
            [],  # scalar
            [scale_value]
        )
    )
    scaled_scores_name = f"{node_name}_scaled_scores"
    onnx_graph.add_node(
        oh.make_node(
            "Mul",
            inputs=[scores_name, scale_const_name],
            outputs=[scaled_scores_name],
            name=f"{node_name}_mul_scale"
        )
    )
    # (Optional) Mask goes here if needed
    # Softmax
    attn_weights_name = f"{node_name}_attn_weights"
    onnx_graph.add_node(
        oh.make_node(
            "Softmax",
            inputs=[scaled_scores_name],
            outputs=[attn_weights_name],
            name=f"{node_name}_softmax_attn",
            axis=3  # softmax along the last dimension (seq_len)
        )
    )
    # Attention output = MatMul(attn_weights, V)
    attn_output_name = f"{node_name}_attn_output"
    onnx_graph.add_node(
        oh.make_node(
            "MatMul",
            inputs=[attn_weights_name, v_trans],
            outputs=[attn_output_name],
            name=f"{node_name}_matmul_attn_v"
        )
    )

    # --------------------------------------------------------
    # 8) Merge heads => [B, seq_len, num_heads * head_dim]
    # --------------------------------------------------------
    attn_output_t_name = f"{node_name}_attn_output_t"
    onnx_graph.add_node(
        oh.make_node(
            "Transpose",
            inputs=[attn_output_name],
            outputs=[attn_output_t_name],
            name=f"{node_name}_transpose_back",
            perm=[0, 2, 1, 3]  # B, seq_len, num_heads, head_dim
        )
    )
    final_attn_3d_name = f"{node_name}_attn_3d"
    shape_attn_3d_name = f"{node_name}_shape_attn_3d"
    target_shape_attn = np.array([B, L, qkv_features], dtype=np.int64)
    onnx_graph.add_initializer(
        oh.make_tensor(
            shape_attn_3d_name,
            onnx.TensorProto.INT64,
            target_shape_attn.shape,
            target_shape_attn
        )
    )
    onnx_graph.add_node(
        oh.make_node(
            "Reshape",
            inputs=[attn_output_t_name, shape_attn_3d_name],
            outputs=[final_attn_3d_name],
            name=f"{node_name}_reshape_final"
        )
    )

    # --------------------------------------------------------
    # 9) Output projection => Gemm => [B, L, out_features]
    # --------------------------------------------------------
    flattened_attn_name = f"{node_name}_flattened_attn"
    onnx_graph.add_node(
        oh.make_node(
            "Flatten",
            inputs=[final_attn_3d_name],
            outputs=[flattened_attn_name],
            name=f"{node_name}_flatten_attn",
            axis=1
        )
    )
    out_proj_name = f"{node_name}_out_proj"
    gemm_inputs = [flattened_attn_name, out_proj_weight_name] + out_proj_bias_names
    onnx_graph.add_node(
        oh.make_node(
            "Gemm",
            inputs=gemm_inputs,
            outputs=[out_proj_name],
            name=f"{node_name}_gemm_out",
            alpha=1.0,
            beta=1.0,
            transA=False,
            transB=False
        )
    )
    final_output_name = f"{node_name}_final_output"
    shape_final_name = f"{node_name}_shape_final"
    target_shape_final = np.array([B, L, out_features], dtype=np.int64)
    onnx_graph.add_initializer(
        oh.make_tensor(
            shape_final_name,
            onnx.TensorProto.INT64,
            target_shape_final.shape,
            target_shape_final
        )
    )
    onnx_graph.add_node(
        oh.make_node(
            "Reshape",
            inputs=[out_proj_name, shape_final_name],
            outputs=[final_output_name],
            name=f"{node_name}_reshape_out"
        )
    )

    # --------------------------------------------------------
    # 10) Return the JAX output + ONNX names
    # --------------------------------------------------------
    output_names = [final_output_name]
    jax_outputs = [jax_output]  # for reference
    onnx_graph.add_local_outputs(jax_outputs, output_names)
    return jax_outputs, output_names


# Attach our custom build_onnx_node to nnx.MultiHeadAttention
nnx.MultiHeadAttention.build_onnx_node = build_onnx_node


def get_test_params():
    """
    Provides test parameters for MultiHeadAttention, including a
    'parameters' dict with decode=False for maximum flexibility.
    """
    return [
        {
            "model_name": "multihead_attention",
            "model": lambda: nnx.MultiHeadAttention(
                num_heads=8,
                qkv_features=256,
                out_features=256,
                in_features=256,
                use_bias=True,
                rngs=nnx.Rngs(0),
                # We won't set decode here; we'll let parameters handle it
            ),
            "input_shapes": [(1, 197, 256)],
            "build_onnx_node": nnx.MultiHeadAttention.build_onnx_node,
            "export": {"decode": False}  # flexible approach
        }
    ]
