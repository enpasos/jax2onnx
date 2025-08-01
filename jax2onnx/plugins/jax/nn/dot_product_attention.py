# file: jax2onnx/plugins/jax/nn/dot_product_attention.py
# --------------------------------------------------------------------------- #
#   Dot-Product-Attention primitive → ONNX                                    #
# --------------------------------------------------------------------------- #
from typing import TYPE_CHECKING

import numpy as np
from jax import numpy as jnp
from jax import core, nn
from jax.extend.core import Primitive
from onnx import TensorProto, helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# --------------------------------------------------------------------------- #
#   Ensure jnp.einsum has a batching rule (needed by reference implementation) #
# --------------------------------------------------------------------------- #
from jax.interpreters import batching

   

def _safe_masked_dpa(q, k, v, q_lens, kv_lens):
    """
    q, k, v: (B, H, Sq, D), (B, H, Sk, D), (B, H, Sk, D)
    q_lens, kv_lens: (B,)
    """
    # Ensure ONNX graph is consistently typed in f64 runs:
    # if the abstract dtype is float64, insert explicit casts up front so
    # MatMul inputs are DOUBLE even if model inputs were materialized as FLOAT.
    if q.dtype == jnp.float64:
        q = q.astype(jnp.float64)
        k = k.astype(jnp.float64)
        v = v.astype(jnp.float64)
    dtype = q.dtype
    B, H, Sq, D = q.shape
    Sk = k.shape[-2]

    # Scaled scores
    scale = jnp.asarray(1.0 / np.sqrt(D), dtype)
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale      # (B,H,Sq,Sk)

    # Build padding mask with positive axes for CumSum
    ones_q = jnp.ones_like(q[..., :1], dtype=jnp.int32)          # (B,H,Sq,1)
    axis_q = ones_q.ndim - 2                                     # == 2
    pos_q  = jnp.cumsum(ones_q, axis=axis_q) - 1                 # (B,H,Sq,1)

    ones_k = jnp.ones_like(k[..., :1], dtype=jnp.int32)          # (B,H,Sk,1)
    axis_k = ones_k.ndim - 2                                     # == 2
    pos_k  = jnp.cumsum(ones_k, axis=axis_k) - 1                 # (B,H,Sk,1)
    pos_k  = jnp.swapaxes(pos_k, -2, -1)                         # (B,H,1,Sk)

    ql = jnp.asarray(q_lens, dtype=jnp.int32)[:, None, None, None]   # (B,1,1,1)
    kl = jnp.asarray(kv_lens, dtype=jnp.int32)[:, None, None, None]  # (B,1,1,1)

    mask_q = pos_q < ql                 # (B,H,Sq,1)
    mask_k = pos_k < kl                 # (B,H,1,Sk)
    mask   = mask_q & mask_k            # (B,H,Sq,Sk)

    large_neg = jnp.asarray(-1e30, dtype)
    masked_logits = jnp.where(mask, scores, large_neg)

    # Safe softmax
    m = jnp.max(masked_logits, axis=-1, keepdims=True)
    exp = jnp.exp(masked_logits - m)
    denom = jnp.sum(exp, axis=-1, keepdims=True)
    attn = jnp.where(denom > 0, exp / denom, jnp.zeros_like(exp))

    out = jnp.matmul(attn, v)           # (B,H,Sq,D)
    return out


# Callable definitions for test cases
def dpa_with_mask(q, k, v, mask):
    """Wrapper for dot_product_attention with a boolean mask."""
    return nn.dot_product_attention(q, k, v, mask=mask)


def dpa_with_causal_mask(q, k, v):
    """Wrapper for dot_product_attention with causal masking."""
    return nn.dot_product_attention(q, k, v, is_causal=True)


def dpa_with_padding_mask(q, k, v, q_len, kv_len):
    """Wrapper for dpa with padding masks."""
    return nn.dot_product_attention(
        q, k, v, query_seq_lengths=q_len, key_value_seq_lengths=kv_len
    )


def dpa_with_local_window_mask(q, k, v):
    """Wrapper for dpa with a local window mask."""
    return nn.dot_product_attention(q, k, v, local_window_size=(1, 1))


# --------------------------------------------------------------------------- #
#   JAX primitive stub                                                        #
# --------------------------------------------------------------------------- #
nn.dot_product_attention_p = Primitive("nn.dot_product_attention")
nn.dot_product_attention_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=nn.dot_product_attention_p.name,
    jax_doc=(
        "https://flax.readthedocs.io/en/latest/api_reference/"
        "flax.nnx/nn/attention.html#flax.nn.dot_product_attention"
    ),
    onnx=[
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Not", "doc": "https://onnx.ai/onnx/operators/onnx__Not.html"},
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        },
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nn",
    component="dot_product_attention",
    testcases=[
        {
            "testcase": "dpa_basic",
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_positional_bias_mask",
            # passes bias=None, mask=None *positionally*
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v, None, None),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_diff_heads_embed",
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v),
            "input_shapes": [(1, 2, 4, 16), (1, 2, 4, 16), (1, 2, 4, 16)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_batch4_seq16",
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v),
            "input_shapes": [(4, 2, 16, 8), (4, 2, 16, 8), (4, 2, 16, 8)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_float64",
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "input_dtype": np.float64,
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_heads1_embed4",
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 1, 8, 4), (2, 1, 8, 4), (2, 1, 8, 4)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_heads8_embed8",
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 8, 8, 8), (2, 8, 8, 8), (2, 8, 8, 8)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_batch1_seq2",
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v),
            "input_shapes": [(1, 2, 2, 8), (1, 2, 2, 8), (1, 2, 2, 8)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_batch8_seq4",
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v),
            "input_shapes": [(8, 2, 4, 16), (8, 2, 4, 16), (8, 2, 4, 16)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_axis1",
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_with_tensor_mask",
            "callable": dpa_with_mask,
            "input_shapes": [
                (2, 8, 4, 16),
                (2, 16, 4, 16),
                (2, 16, 4, 16),
                (2, 4, 8, 16),
            ],
            "input_dtypes": [np.float32, np.float32, np.float32, np.bool_],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_tiny_mask_all_valid",
            "callable": dpa_with_mask,
            "input_values": [
                np.arange(1 * 2 * 1 * 4).reshape((1, 2, 1, 4)).astype(np.float32),
                np.arange(1 * 3 * 1 * 4).reshape((1, 3, 1, 4)).astype(np.float32),
                np.arange(1 * 3 * 1 * 4).reshape((1, 3, 1, 4)).astype(np.float32),
                np.ones((1, 1, 2, 3), dtype=bool),
            ],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_tiny_mask_mixed",
            "callable": dpa_with_mask,
            "input_values": [
                np.arange(1 * 2 * 1 * 4).reshape((1, 2, 1, 4)).astype(np.float32),
                np.arange(1 * 3 * 1 * 4).reshape((1, 3, 1, 4)).astype(np.float32),
                np.arange(1 * 3 * 1 * 4).reshape((1, 3, 1, 4)).astype(np.float32),
                np.array([[[[True, False, True], [False, True, False]]]], dtype=bool),
            ],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_one_false",
            "callable": dpa_with_mask,
            "input_values": [
                np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32),
                np.array(
                    [[[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]]], dtype=np.float32
                ),
                np.array(
                    [[[[10.0, 20.0, 30.0, 40.0]], [[50.0, 60.0, 70.0, 80.0]]]],
                    dtype=np.float32,
                ),
                np.array([[[[True, False]]]], dtype=bool),
            ],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_mostly_false",
            "callable": dpa_with_mask,
            "input_values": [
                np.ones((1, 1, 1, 4), np.float32),
                np.ones((1, 2, 1, 4), np.float32),
                np.ones((1, 2, 1, 4), np.float32) * 7,
                np.array([[[[False, True]]]], dtype=bool),  # Not all entries are masked
            ],
            "expected_output_numpy": [np.zeros((1, 1, 1, 4), np.float32)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_with_causal_mask",
            "callable": dpa_with_causal_mask,
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_with_padding_mask",
            "callable": dpa_with_padding_mask,
            "input_values": [
                np.random.randn(2, 8, 4, 16).astype(np.float32),
                np.random.randn(2, 8, 4, 16).astype(np.float32),
                np.random.randn(2, 8, 4, 16).astype(np.float32),
                np.array([8, 4], dtype=np.int32),
                np.array([8, 7], dtype=np.int32),
            ],
            "atol_f64": 1e-6,
            "rtol_f64": 1e-6,
            "run_only_f32_variant": True,
        },
        
        {
            "testcase": "dpa_with_local_window_mask",
            "callable": dpa_with_local_window_mask,
            "input_shapes": [(1, 16, 1, 4), (1, 16, 1, 4), (1, 16, 1, 4)],
            "atol_f64": 1e-6,  # Absolute tolerance for float64
            "rtol_f64": 1e-6,  # Relative tolerance for float64
        },
        {
            "testcase": "dpa_mask_none",
            # explicit None mask should not get bound as an operand
            "callable": lambda q, k, v: nn.dot_product_attention(q, k, v, mask=None),
            "input_shapes": [
                (2, 4, 8, 32),  # q
                (2, 4, 8, 32),  # k
                (2, 4, 8, 32),  # v
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class DotProductAttentionPlugin(PrimitiveLeafPlugin):
    # stash the real JAX impl so we can call it at runtime
    _ORIG_CALL = nn.dot_product_attention

    @staticmethod
    def abstract_eval(q, k, v, *args, **kwargs):
        return core.ShapedArray(q.shape, q.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        q, k, v, *optional_inputs = node_inputs
        out_var = node_outputs[0]

        q_name, k_name, v_name = map(s.get_name, (q, k, v))
        out_name = s.get_name(out_var)
        B, T, N, H = q.aval.shape
        _, S, _, _ = k.aval.shape
        np_dtype = q.aval.dtype
        builder = s.builder
        onnx_dtype = builder._numpy_dtype_to_onnx(np_dtype)

        # Register both in metadata and legacy value_info to satisfy all builder paths
        def _reg(name: str, shape):
            shp = tuple(int(d) for d in shape)
            builder.register_value_info_metadata(name, shp, onnx_dtype)
            # legacy path (expects numpy dtype)
            s.add_shape_info(name, shp, np_dtype)

        q_t = s.get_unique_name("q_T")
        k_t = s.get_unique_name("k_T")
        s.add_node(helper.make_node("Transpose", [q_name], [q_t], perm=[0, 2, 1, 3]))
        s.add_node(helper.make_node("Transpose", [k_name], [k_t], perm=[0, 2, 3, 1]))
        _reg(q_t, (B, N, T, H))
        _reg(k_t, (B, N, H, S))

        logits = s.get_unique_name("attn_scores")
        s.add_node(helper.make_node("MatMul", [q_t, k_t], [logits]))
        _reg(logits, (B, N, T, S))

        scale_const = s.get_constant_name(np.array(1.0 / np.sqrt(H), dtype=np_dtype))
        # 1) scale the raw attention scores
        scaled_scores = s.get_unique_name("scaled_scores")
        s.add_node(helper.make_node("Mul", [logits, scale_const], [scaled_scores]))
        _reg(scaled_scores, (B, N, T, S))

        final_logits = scaled_scores

        # 2) if causal masking was requested, blank out j > i entries by injecting a
        #    lower-triangular mask + a "-infinity" fill
        if params.get("is_causal", False):
            # extract static dims from the input abstract (batch, seq_q, heads, head_dim)
            b, seq_q, n_heads, head_dim = node_inputs[0].aval.shape
            # for keys the sequence length is at axis=1
            seq_k = node_inputs[1].aval.shape[1]
            # build a [seq_q, seq_k] lower-triangular boolean mask
            mask_np = np.tril(np.ones((seq_q, seq_k), dtype=bool))
            mask_name = s.get_constant_name(mask_np)

            # build a scalar "-∞" of the correct dtype
            neg_inf_np = np.array(-np.inf, dtype=node_inputs[0].aval.dtype)
            neg_inf_name = s.get_constant_name(neg_inf_np)

            # apply:   masked = Where(mask, scaled_scores, -inf)
            masked_name = s.get_unique_name("masked_scores")
            s.add_node(
                helper.make_node(
                    "Where",
                    inputs=[mask_name, scaled_scores, neg_inf_name],
                    outputs=[masked_name],
                    name=s.get_unique_name("where"),
                )
            )
            _reg(masked_name, (b, n_heads, seq_q, seq_k))
            final_logits = masked_name

        # Handle optional mask input
        if optional_inputs:
            mask_var = optional_inputs[0]
            mask_name = s.get_name(mask_var)
            mask_bool_name = s.get_unique_name("mask_bool")
            s.add_node(
                helper.make_node(
                    "Cast", [mask_name], [mask_bool_name], to=TensorProto.BOOL
                )
            )
            # BOOL needs direct registration in legacy path as well
            shp_bool = tuple(int(d) for d in mask_var.aval.shape)
            builder.register_value_info_metadata(mask_bool_name, shp_bool, TensorProto.BOOL)
            s.add_shape_info(mask_bool_name, shp_bool, bool)
            # JAX fills masked positions with the minimum finite value
            very_neg = np.array(np.finfo(np_dtype).min, dtype=np_dtype)
            large_negative_number_const = s.get_constant_name(very_neg)
            masked_logits = s.get_unique_name("masked_logits")
            s.add_node(
                helper.make_node(
                    "Where",
                    inputs=[mask_bool_name, scaled_scores, large_negative_number_const],
                    outputs=[masked_logits],
                )
            )
            _reg(masked_logits, (B, N, T, S))
            final_logits = masked_logits

        # 3) finally softmax over the last (key) axis
        weights = s.get_unique_name("attn_weights")
        s.add_node(helper.make_node("Softmax", [final_logits], [weights], axis=-1))
        _reg(weights, (B, N, T, S))

        # 3.a If a boolean mask was provided, zero-out masked positions *after* softmax.
        #     This ensures rows with all entries masked become all-zeros (not uniform),
        #     matching JAX's lengths behavior.
        if optional_inputs:
            # Reuse the earlier mask_bool_name; cast it to data dtype and multiply.
            mask_float = s.get_unique_name("mask_float")
            s.add_node(
                helper.make_node("Cast", [mask_bool_name], [mask_float], to=onnx_dtype)
            )
            # Register shape info for the casted mask using the mask boolean shape.
            shp_bool = tuple(int(d) for d in optional_inputs[0].aval.shape)
            builder.register_value_info_metadata(mask_float, shp_bool, onnx_dtype)
            s.add_shape_info(mask_float, shp_bool, np_dtype)

            weights_masked = s.get_unique_name("attn_weights_masked")
            s.add_node(helper.make_node("Mul", [weights, mask_float], [weights_masked]))
            _reg(weights_masked, (B, N, T, S))
            weights = weights_masked

        v_t = s.get_unique_name("v_T")
        out_t = s.get_unique_name("out_T")
        s.add_node(helper.make_node("Transpose", [v_name], [v_t], perm=[0, 2, 1, 3]))
        _reg(v_t, (B, N, S, H))
        s.add_node(helper.make_node("MatMul", [weights, v_t], [out_t]))
        _reg(out_t, (B, N, T, H))
        s.add_node(
            helper.make_node("Transpose", [out_t], [out_name], perm=[0, 2, 1, 3])
        )
        _reg(out_name, (B, T, N, H))

    @staticmethod
    def _dot_product_attention(q, k, v, *args, **kwargs):
        """
        Pull out any `mask` or `is_causal` flag, and bind the primitive so that
        both get propagated into eqn.params and/or eqn.inputs.
        """
        # --- 1) extract positional mask (args[1] if bias placeholder at args[0]) ---
        mask = None
        if len(args) >= 2:
            mask = args[1]

        # --- 2) keyword‐mask overrides ---
        if "mask" in kwargs:
            mask = kwargs.pop("mask")

        # --- 3) lengths → synthesize boolean padding mask (broadcast over heads) ---
        q_lens = kwargs.pop("query_seq_lengths", None)
        kv_lens = kwargs.pop("key_value_seq_lengths", None)
        if (q_lens is not None) or (kv_lens is not None):
            if (q_lens is None) or (kv_lens is None):
                raise TypeError(
                    "Both query_seq_lengths and key_value_seq_lengths must be provided together."
                )
            # Shapes: q:(B,T,N,H), k:(B,S,N,H)
            # Build (B,1,T,S) boolean mask, broadcast across heads N
            # Use static dims from aval during tracing; fall back to array .shape when eager.
            tq = (getattr(q, "aval", q).shape)[1]
            sk = (getattr(k, "aval", k).shape)[1]
            # build index grids without advanced indexing (works with dynamic dims)
            t_idx = jnp.arange(tq, dtype=jnp.int32).reshape(1, 1, tq, 1)  # (1,1,T,1)
            s_idx = jnp.arange(sk, dtype=jnp.int32).reshape(1, 1, 1, sk)  # (1,1,1,S)
            ql = jnp.asarray(q_lens, dtype=jnp.int32)[:, None, None, None]  # (B,1,1,1)
            kl = jnp.asarray(kv_lens, dtype=jnp.int32)[:, None, None, None] # (B,1,1,1)
            mask = (t_idx < ql) & (s_idx < kl)  # (B,1,T,S) → broadcasts to (B,N,T,S)

        # --- 4) extract is_causal (default False) ---
        is_causal = bool(kwargs.pop("is_causal", False))

        # --- 5) bind with the right signature ---
        if mask is not None:
            # explicit mask → goes in inputs[3]
            return nn.dot_product_attention_p.bind(q, k, v, mask)
        if is_causal:
            # no mask tensor but causal flag → goes in params
            return nn.dot_product_attention_p.bind(q, k, v, is_causal=True)
        # neither mask nor causal → plain attention
        return nn.dot_product_attention_p.bind(q, k, v)

    @staticmethod
    def get_monkey_patch():
        """
        Install a wrapper that:
         - If we're *exporting* (i.e. in a JAX tracer / ShapeDtypeStruct), bind to the ONNX primitive;
         - Otherwise (eager numpy/jax.Array inputs), call the real JAX dot_product_attention.
        """
        from jax.core import Tracer

        def patched(q, k, v, *args, **kwargs):
            # Detect ONNX‐tracing: ShapeDtypeStruct or JAX Tracer
            if hasattr(q, "aval") or isinstance(q, Tracer):
                # export path → primitive
                return DotProductAttentionPlugin._dot_product_attention(
                    q, k, v, *args, **kwargs
                )
            # runtime path → fall back to real JAX implementation
            return DotProductAttentionPlugin._ORIG_CALL(q, k, v, *args, **kwargs)

        return patched

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nn],
            "patch_function": lambda _: DotProductAttentionPlugin.get_monkey_patch(),
            "target_attribute": "dot_product_attention",
        }


# attach abstract-eval
nn.dot_product_attention_p.def_abstract_eval(DotProductAttentionPlugin.abstract_eval)


# --------------------------- batching rule -----------------------------------
def dpa_batch(xs, dims, *, axis=-1):
    assert len(set(d for d in dims if d is not None)) <= 1
    q, k, v, *rest = xs
    bdim = next((d for d in dims if d is not None), None)

    if bdim is not None and bdim != 0:
        q = jnp.moveaxis(q, bdim, 0)
        k = jnp.moveaxis(k, bdim, 0)
        v = jnp.moveaxis(v, bdim, 0)
        if rest:
            rest = [
                jnp.moveaxis(r, d, 0) if d is not None else r
                for r, d in zip(rest, dims[3:])
            ]

    out = nn.dot_product_attention(q, k, v, *rest, axis=axis)
    return out, 0


batching.primitive_batchers[nn.dot_product_attention_p] = dpa_batch
