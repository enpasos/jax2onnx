# repro_dynamic.py

import jax
import jax.numpy as jnp
import equinox as eqx
from jax2onnx.plugins.examples.eqx.gpt_oss import TransformerBlock, GPTOSSConfig

def main():
    config = GPTOSSConfig(
        num_hidden_layers=1,
        num_experts=4,
        experts_per_token=2,
        vocab_size=32,
        hidden_size=64,
        intermediate_size=64,
        swiglu_limit=7.0,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=16,
        initial_context_length=32,
    )
    key = jax.random.PRNGKey(0)
    block = TransformerBlock(config, layer_idx=0, key=key, param_dtype=jnp.float32)

    # Create dynamic shape input
    # We simulate what jax2onnx does during export/test
    # But here we just want to trigger the error.
    # The error happens when tracing with dynamic shapes.
    # We can use jax.make_jaxpr with symbolic shapes?
    
    # Or we can try to run it with a tracer that behaves like the one in jax2onnx?
    # Actually, let's just use jax2onnx's test utility if possible, or just run it with jax.jit and see if we can trigger something similar?
    # But standard jax.jit handles dynamic shapes fine.
    # The error comes from `_DynamicDimSentinel` which is specific to jax2onnx (or jax's internal dynamic shape handling?).
    # Actually `_DynamicDimSentinel` is likely from `jax.experimental.jax2dx` or similar, or `jax2onnx`'s own mechanism.
    
    # Let's try to use `jax2onnx.to_onnx` which is what the test likely uses (or `construct_and_call` wrapper).
    
    from jax2onnx import to_onnx
    
    # Define a function that calls the block
    def forward(x):
        return block.attn(x) # Just test attention first
        
    # Input with dynamic sequence length
    input_shape = (1, "seq", 64) 
    
    # We need to construct dummy input with concrete shape for tracing, but specify dynamic axes?
    # jax2onnx.to_onnx(func, *args, input_shapes=...)
    
    dummy_input = jnp.zeros((1, 8, 64), dtype=jnp.float32)
    
    try:
        # Pass input spec directly. "seq" indicates dynamic dimension.
        onnx_model = to_onnx(
            forward,
            inputs=[(1, "seq", 64)]
        )
        print("Export successful!")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
