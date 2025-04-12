import os
import pytest
import onnx
from flax import nnx
from jax import numpy as jnp

# Import from the main package
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TestQuickstart:
    """Tests for quickstart.py example"""

    def test_quickstart_execution(self):
        """Test that quickstart.py runs without errors and produces expected output"""

        # Make sure the output directory exists
        os.makedirs("docs/onnx", exist_ok=True)

        # Import and run the quickstart module

        # Now check that the output file was created
        output_path = "docs/onnx/my_callable.onnx"
        assert os.path.exists(output_path), f"Output file {output_path} was not created"

        # Load and validate the ONNX model
        model = onnx.load(output_path)
        onnx.checker.check_model(model)

        # Check model properties
        graph = model.graph
        assert len(graph.input) == 1
        assert graph.input[0].name == "var_0"  # Updated to match actual name convention
        assert len(graph.output) == 1


class TestQuickstartFunctions:
    """Tests for quickstart_functions.py example"""

    def test_quickstart_functions_execution(self):
        """Test that quickstart_functions.py runs without errors and produces expected output"""

        # Make sure the output directory exists
        os.makedirs("docs/onnx", exist_ok=True)

        # Import and run the module

        # Check that the output file was created
        output_path = "docs/onnx/model_with_function.onnx"
        assert os.path.exists(output_path), f"Output file {output_path} was not created"

        # Load and validate the ONNX model
        model = onnx.load(output_path)
        onnx.checker.check_model(model)

        # Verify that the model contains functions (it should preserve function hierarchy)
        assert len(model.functions) > 0, "Model should contain ONNX functions"

    def test_mlpblock_function(self):
        """Test MLPBlock class with onnx_function decorator"""
        from jax2onnx.quickstart_functions import MLPBlock

        # Create an instance
        block = MLPBlock(256, rngs=nnx.Rngs(0))

        # Test inference
        x = jnp.ones((10, 256))
        output = block(x)

        # Check shape
        assert output.shape == (10, 256)

    def test_mymodel_function(self):
        """Test MyModel class that uses MLPBlock"""
        from jax2onnx.quickstart_functions import MyModel

        # Create an instance
        model = MyModel(256, rngs=nnx.Rngs(0))

        # Test inference
        x = jnp.ones((10, 256))
        output = model(x)

        # Check shape
        assert output.shape == (10, 256)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
