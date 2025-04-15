#!/usr/bin/env python
"""
Test script for detecting duplicate parameters in ONNX functions
"""

import jax.numpy as jnp
import onnx
from flax import nnx

from jax2onnx import onnx_function, to_onnx


# Define test classes that reproduce the issue with duplicate parameters
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


def check_duplicate_function_inputs(model):
    """Check if any function in the model has duplicate parameter inputs in its definition.

    Args:
        model: An ONNX model

    Returns:
        List of tuples (function_name, duplicate_inputs) or empty list if no duplicates
    """
    duplicates = []

    for function in model.functions:
        # Check for direct duplicates in the function input definition
        seen_inputs = set()
        duplicate_inputs = []

        for input_name in function.input:
            if input_name in seen_inputs:
                duplicate_inputs.append(input_name)
            else:
                seen_inputs.add(input_name)

        # Also check function call nodes for duplicate inputs
        for node in function.node:
            if node.domain == "custom" and node.op_type in [
                f.name for f in model.functions
            ]:
                # This is a function call node
                seen_call_inputs = set()
                for input_name in node.input:
                    if input_name in seen_call_inputs:
                        duplicate_inputs.append(
                            f"Duplicate input '{input_name}' in function call {node.name}"
                        )
                    else:
                        seen_call_inputs.add(input_name)

        if duplicate_inputs:
            duplicates.append((function.name, duplicate_inputs))

    return duplicates


def test_duplicate_parameters():
    """Test that functions don't have duplicate parameters"""
    print("Testing for duplicate parameters in ONNX functions...")

    # Create instance
    super_block = SuperBlock()

    # Define input shapes
    input_shapes = [(5, 10, 256)]

    # Convert to ONNX
    print("Converting model to ONNX...")
    model = to_onnx(super_block, input_shapes, model_name="duplicate_param_test")

    # Save the ONNX model (for inspection if needed)
    output_path = "docs/onnx/duplicate_param_test.onnx"
    onnx.save_model(model, output_path)

    # Check for duplicate parameters
    print("\nChecking for duplicate parameters...")
    duplicates = check_duplicate_function_inputs(model)

    # Report duplicates
    if duplicates:
        print("\nDuplicate parameters found:")
        for function_name, duplicate_inputs in duplicates:
            print(
                f"  - Function '{function_name}' has duplicate inputs: {', '.join(duplicate_inputs)}"
            )
    else:
        print("✅ No duplicate parameters found in any function")

    # Assert no duplicates
    assert not duplicates, "Functions should not have duplicate parameter inputs"


if __name__ == "__main__":
    test_duplicate_parameters()
