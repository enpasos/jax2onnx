
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax2onnx.plugins.examples.eqx.dino import VisionTransformer
from jax2onnx import to_onnx
import onnxruntime as ort

def main():
    img_size = 224
    patch_size = 16
    embed_dim = 384
    depth = 12
    num_heads = 6
    num_storage_tokens = 0 # Test S14 case
    
    key = jax.random.PRNGKey(1) # Matches S14 test case index
    
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_storage_tokens=num_storage_tokens,
        key=key,
    )
    
    # Input
    dummy_input = jnp.zeros((1, 3, img_size, img_size), dtype=jnp.float32)
    
    # JAX run
    jax_out = model(dummy_input)
    print(f"JAX output shape: {jax_out.shape}")
    
    # ONNX export
    onnx_model = to_onnx(
        model,
        inputs=[dummy_input],
        model_name="dino_s16"
    )
    
    # ONNX run
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    onnx_out = sess.run(None, {"in_0": np.array(dummy_input)})[0]
    print(f"ONNX output shape: {onnx_out.shape}")
    
    # Compare
    diff = np.abs(jax_out - onnx_out)
    max_diff = np.max(diff)
    print(f"Max diff: {max_diff}")
    
    if max_diff > 1e-3:
        print("FAILURE: Max diff > 1e-3")
        # Debug: check where divergence happens?
        # Maybe we can export with intermediate outputs if needed.
    else:
        print("SUCCESS")

if __name__ == "__main__":
    main()
