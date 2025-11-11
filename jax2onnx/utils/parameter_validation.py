# jax2onnx/utils/parameter_validation.py

"""Utilities for validating parameter wiring on serialized ONNX models.

These helpers operate on ``onnx.ModelProto`` objects produced by the IR-only
converter. They stay lightweight so policy tests can run them without pulling
in additional converter dependencies.
"""

from __future__ import annotations

from typing import List

import onnx


def validate_parameter_connections(model: onnx.ModelProto) -> List[str]:
    """Check that every node input is produced by something in scope.

    The scan walks the top-level graph first, then any functions that hang off
    the model. A node input is considered valid when it is produced by another
    node, backed by an initializer, or declared as a graph input.
    """

    errors: List[str] = []

    graph = model.graph
    all_inputs = {node_input for node in graph.node for node_input in node.input}
    all_outputs = {node_output for node in graph.node for node_output in node.output}
    all_initializers = {initializer.name for initializer in graph.initializer}
    all_graph_inputs = {graph_input.name for graph_input in graph.input}

    for input_name in all_inputs:
        if (
            input_name
            and input_name not in all_outputs
            and input_name not in all_initializers
            and input_name not in all_graph_inputs
        ):
            errors.append(f"Node input '{input_name}' has no source in the main graph")

    for function in model.functions:
        function_inputs = set(function.input)
        function_internal_outputs = {
            node_output for node in function.node for node_output in node.output
        }

        for node in function.node:
            for input_name in node.input:
                if (
                    input_name
                    and input_name not in function_inputs
                    and input_name not in function_internal_outputs
                    and input_name not in all_initializers
                ):
                    errors.append(
                        f"Node input '{input_name}' in function '{function.name}' has no source"
                    )

    return errors


def validate_deterministic_parameter(model: onnx.ModelProto) -> List[str]:
    """Ensure ``deterministic`` parameters are either inputs or initializers."""

    errors: List[str] = []
    param_name = "deterministic"

    deterministic_consumers = []
    for node in model.graph.node:
        if any(param_name in input_name for input_name in node.input):
            deterministic_consumers.append(node)

    has_deterministic_input = any(param_name in i.name for i in model.graph.input)
    has_deterministic_initializer = any(
        param_name in initializer.name for initializer in model.graph.initializer
    )

    if deterministic_consumers and not (
        has_deterministic_input or has_deterministic_initializer
    ):
        errors.append(
            "'deterministic' parameter is used in nodes but not provided as an input "
            "or initializer"
        )

    for function in model.functions:
        function_uses_deterministic = any(
            param_name in input_name
            for node in function.node
            for input_name in node.input
        )
        function_has_deterministic_input = any(
            param_name in input_name for input_name in function.input
        )

        if function_uses_deterministic and not function_has_deterministic_input:
            errors.append(
                f"Function '{function.name}' uses 'deterministic' parameter internally "
                "but does not declare it as an input"
            )

    return errors


def validate_onnx_model_parameters(model: onnx.ModelProto) -> List[str]:
    """Run the standard ONNX parameter validation suite."""

    errors: List[str] = []
    errors.extend(validate_parameter_connections(model))
    errors.extend(validate_deterministic_parameter(model))
    return errors
