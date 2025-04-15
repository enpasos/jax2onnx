#!/usr/bin/env python
"""
Test script for validating parameter handling in nested ONNX functions
"""

import jax
import jax.numpy as jnp
import numpy
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
    output_path = "docs/onnx/nested_param_test.onnx"
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

    # Check for correct parameter naming
    print("\nValidating parameter naming...")
    graph = model.graph
    incorrect_names = []

    # Check input parameters in the main graph
    for input_node in graph.input:
        if input_node.name.startswith("var_") and not input_node.name.startswith(
            "var_0"
        ):  # Skip the primary input
            print(f"Found generic input name in main graph: {input_node.name}")
            incorrect_names.append(input_node.name)

    # Check node names for parameters in the main graph
    for node in graph.node:
        if node.op_type == "Constant" and any(
            output.startswith("var_") for output in node.output
        ):
            for output in node.output:
                if output.startswith("var_"):
                    print(f"Found generic constant name in main graph: {output}")
                    incorrect_names.append(output)

    # Check inside functions for generic parameter names that should be descriptive
    # Look specifically for nodes that use the boolean parameter
    deterministic_nodes_with_generic_inputs = []

    for function in model.functions:
        print(f"Checking function '{function.name}' for generic parameter names...")
        # First, collect all node inputs that should be the deterministic parameter
        deterministic_inputs = []
        for node in function.node:
            if node.name.startswith("not_deterministic"):
                # These nodes take the deterministic parameter as input
                # The node input should be named "deterministic" not "var_X"
                for input_name in node.input:
                    deterministic_inputs.append((node.name, input_name))
                    if input_name.startswith("var_"):
                        print(
                            f"Found generic parameter name in function node '{node.name}': {input_name}"
                        )
                        incorrect_names.append(input_name)
                        deterministic_nodes_with_generic_inputs.append(
                            (node.name, input_name)
                        )

        # Now check function inputs - there should be one matching the deterministic parameter
        for input_name in function.input:
            if input_name == "deterministic":
                print(
                    f"✅ Found properly named parameter in function inputs: {input_name}"
                )
            elif input_name.startswith("var_") and input_name in [
                inp for _, inp in deterministic_inputs
            ]:
                # This is a generic name for what should be the deterministic parameter
                print(
                    f"❌ Found generic parameter name in function inputs: {input_name}"
                )
                incorrect_names.append(input_name)

    # Create a flag to check if complex names were found in the logs
    import re

    # Write logs to a temp file we can analyze
    print("\nExamining parameter names from logs...")

    # Define our expectation
    complex_name_pattern = re.compile(r"deterministic_const__\w+")
    complex_names_found = False

    # Check ONNX model initializers for complex names
    for initializer in model.graph.initializer:
        if complex_name_pattern.match(initializer.name):
            print(
                f"❌ Found complex parameter name in model initializers: {initializer.name}"
            )
            complex_names_found = True

    # Also check for complex names in the model graph nodes
    for node in graph.node:
        for output in node.output:
            if complex_name_pattern.match(output):
                print(f"❌ Found complex parameter name in graph nodes: {output}")
                complex_names_found = True

    # Assert proper naming
    if incorrect_names:
        print(
            "❌ Found generic parameter names. Expected descriptive names like 'deterministic'"
        )
        for name in incorrect_names:
            print(f"  - {name}")
        assert not incorrect_names, "Parameter names should be descriptive, not generic"
    elif complex_names_found:
        print("❌ Found complex parameter names. Expected simple name 'deterministic'")
        assert (
            not complex_names_found
        ), "Parameter names should be simple (just 'deterministic'), not complex like 'deterministic_const___0'"
    else:
        print("✅ All parameters have proper descriptive names")

    # Test running inference with the model
    print("\nTesting model with ONNX Runtime...")
    try:
        # Create random input data
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (5, 10, 256))

        # Run with JAX
        jax_result = super_block(x, deterministic=True)

        # Run with ONNX Runtime
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        # Convert JAX array to numpy array
        x_np = numpy.array(x, dtype=numpy.float32)
        onnx_input = {input_name: x_np}
        onnx_result = session.run(None, onnx_input)[0]

        # Compare results
        is_close = jnp.allclose(jax_result, onnx_result, rtol=1e-5, atol=1e-5)
        print(f"✅ JAX and ONNX results match: {is_close}")
        assert is_close, "JAX and ONNX results should match"

    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        assert False, f"Test failed with error: {str(e)}"


if __name__ == "__main__":
    test_parameter_passing()
