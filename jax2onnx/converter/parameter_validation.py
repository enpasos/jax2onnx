"""
Parameter validation module for JAX2ONNX

This module provides validation utilities to ensure parameters are properly
connected in the ONNX graph, particularly for nested ONNX functions.
"""

import onnx
from typing import List


def validate_parameter_connections(model: onnx.ModelProto) -> List[str]:
    """
    Validates that all parameters in the ONNX model are properly connected.

    Args:
        model: The ONNX ModelProto to validate

    Returns:
        List of error messages, empty if no errors
    """
    errors = []

    # Track all node inputs and outputs
    all_inputs = set()
    all_outputs = set()
    all_initializers = {init.name for init in model.graph.initializer}
    all_graph_inputs = {input.name for input in model.graph.input}

    # First, gather all node inputs and outputs in the main graph
    for node in model.graph.node:
        for input_name in node.input:
            all_inputs.add(input_name)
        for output_name in node.output:
            all_outputs.add(output_name)

    # Check if any node inputs have no source
    for input_name in all_inputs:
        if (
            input_name not in all_outputs
            and input_name not in all_initializers
            and input_name not in all_graph_inputs
        ):
            errors.append(f"Node input '{input_name}' has no source in the main graph")

    # Validate functions if any
    if model.functions:
        for function in model.functions:
            function_inputs = set(function.input)
            set(function.output)
            function_internal_outputs = set()

            # Track all node outputs within this function
            for node in function.node:
                for output_name in node.output:
                    function_internal_outputs.add(output_name)

            # Check each node in the function
            for node in function.node:
                for input_name in node.input:
                    if (
                        input_name not in function_inputs
                        and input_name not in function_internal_outputs
                        and input_name not in all_initializers
                    ):
                        errors.append(
                            f"Node input '{input_name}' in function '{function.name}' "
                            f"has no source (not a function input, other node output, or initializer)"
                        )

    return errors


def validate_deterministic_parameter(model: onnx.ModelProto) -> List[str]:
    """
    Specifically validates that 'deterministic' parameters are properly connected.

    Args:
        model: The ONNX ModelProto to validate

    Returns:
        List of error messages, empty if no errors
    """
    errors = []
    param_name = "deterministic"

    # Check if deterministic is used in function nodes
    deterministic_nodes = []
    for node in model.graph.node:
        for input_name in node.input:
            if param_name in input_name:
                deterministic_nodes.append(node)

    # Check if deterministic parameter exists in model inputs or initializers
    has_deterministic_input = any(
        param_name in input.name for input in model.graph.input
    )
    has_deterministic_init = any(
        param_name in init.name for init in model.graph.initializer
    )

    if deterministic_nodes and not (has_deterministic_input or has_deterministic_init):
        errors.append(
            f"'{param_name}' parameter is used in nodes but not provided as "
            f"model input or initializer"
        )

    # Check functions for deterministic parameters
    if model.functions:
        for function in model.functions:
            function_uses_deterministic = False
            function_has_deterministic_input = False

            # Check if function nodes use deterministic
            for node in function.node:
                for input_name in node.input:
                    if param_name in input_name:
                        function_uses_deterministic = True
                        break
                if function_uses_deterministic:
                    break

            # Check if deterministic is a function input
            for input_name in function.input:
                if param_name in input_name:
                    function_has_deterministic_input = True
                    break

            if function_uses_deterministic and not function_has_deterministic_input:
                errors.append(
                    f"Function '{function.name}' uses '{param_name}' parameter internally "
                    f"but doesn't declare it as an input"
                )

    return errors


def validate_onnx_model_parameters(model: onnx.ModelProto) -> List[str]:
    """
    Full validation of parameter connections in an ONNX model.

    Args:
        model: The ONNX ModelProto to validate

    Returns:
        List of error messages, empty if no errors
    """
    errors = []
    errors.extend(validate_parameter_connections(model))
    errors.extend(validate_deterministic_parameter(model))
    return errors
