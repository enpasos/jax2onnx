#!/usr/bin/env python
"""
Test script for validating parameter handling in nested ONNX functions
"""

import jax
import jax.numpy as jnp
import onnx
import onnxruntime as ort
from flax import nnx

from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.parameter_validation import validate_onnx_model_parameters


# Define test classes that reproduce the issue in onnx_functions_016.py
@onnx_function
class NestedBlock(nnx.Module):
    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Dropout(rate=0.1, rngs=rngs),
            nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            nnx.Dropout(rate=0.1, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


@onnx_function
class SuperBlock(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(0)
        num_hiddens = 256
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp = NestedBlock(num_hiddens, mlp_dim=512, rngs=rngs)

    def __call__(self, x, deterministic: bool = True):
        return self.mlp(self.layer_norm2(x), deterministic=deterministic)


def test_parameter_passing():
    """Test parameter passing between nested ONNX functions"""
    print("Testing parameter passing between nested ONNX functions...")

    # Create instance
    super_block = SuperBlock()

    # Define input shapes
    input_shapes = [(5, 10, 256)]

    # Convert to ONNX with our improved parameter handling
    print("Converting model to ONNX...")
    model = to_onnx(super_block, input_shapes, model_name="nested_param_test")

    # Save the ONNX model
    output_path = "nested_param_test.onnx"
    onnx.save_model(model, output_path)
    print(f"Model saved to {output_path}")

    # Validate parameter connections
    print("\nValidating parameter connections...")
    validation_errors = validate_onnx_model_parameters(model)
    if validation_errors:
        print("⚠️ Validation errors found:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("✅ No validation errors - parameters are properly connected!")

    # Test running inference with the model
    print("\nTesting model with ONNX Runtime...")
    try:
        # Create random input data
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (5, 10, 256))

        # Run with JAX
        jax_result = super_block(x, deterministic=True)

        # Run with ONNX Runtime
        session = ort.InferenceSession(output_path)
        input_name = session.get_inputs()[0].name
        onnx_input = {input_name: x.astype("float32")}
        onnx_result = session.run(None, onnx_input)[0]

        # Compare results
        is_close = jnp.allclose(jax_result, onnx_result, rtol=1e-5, atol=1e-5)
        print(f"✅ JAX and ONNX results match: {is_close}")

        return True
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        return False


if __name__ == "__main__":
    test_parameter_passing()
