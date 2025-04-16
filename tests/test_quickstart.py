import os
import pytest
import onnx
import subprocess


class TestQuickstart:
    """Tests for quickstart.py example"""

    def test_quickstart_execution(self):
        """Test that quickstart.py runs without errors and produces expected output"""

        # Make sure the output directory exists
        os.makedirs("docs/onnx", exist_ok=True)

        # Run the quickstart.py script as a subprocess
        subprocess.run(["python", "jax2onnx/quickstart.py"], check=True)

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

        # Run the quickstart_functions.py script as a subprocess
        subprocess.run(["python", "jax2onnx/quickstart_functions.py"], check=True)

        # Check that the output file was created
        output_path = "docs/onnx/model_with_function.onnx"
        assert os.path.exists(output_path), f"Output file {output_path} was not created"

        # Load and validate the ONNX model
        model = onnx.load(output_path)
        onnx.checker.check_model(model)

        # Verify that the model contains functions (it should preserve function hierarchy)
        assert len(model.functions) > 0, "Model should contain ONNX functions"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
