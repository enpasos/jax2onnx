"""
ONNX Function Builder Module

This module provides specialized functionality for building ONNX functions.
It works with OnnxBuilder but focuses specifically on the function-related aspects
of ONNX model construction.
"""

from typing import Any, Dict, List, Optional

from onnx import FunctionProto, NodeProto, ValueInfoProto, helper

from jax2onnx.converter.name_generator import UniqueNameGenerator
from jax2onnx.converter.onnx_builder import CUSTOM_DOMAIN


class OnnxFunctionBuilder:
    """
    A specialized builder for ONNX functions, handling the creation,
    registration, and management of ONNX function definitions.

    This class separates function-specific building logic from the general
    ONNX model building functionality to improve code organization.
    """

    def __init__(self, onnx_builder: Optional[Any] = None):
        """
        Initialize an ONNX function builder.

        Args:
            onnx_builder: Optional OnnxBuilder instance to integrate with.
                          If provided, this function builder will use the same
                          name generator and can access the OnnxBuilder's context.
        """
        self.functions: Dict[str, FunctionProto] = {}
        self.onnx_builder = onnx_builder

        # Use shared name generator if onnx_builder is provided, otherwise create new one
        if onnx_builder is not None:
            self.name_generator = onnx_builder.name_generator
        else:
            self.name_generator = UniqueNameGenerator()

    def register_function(self, function_proto: FunctionProto) -> None:
        """
        Register an ONNX function with the builder.

        Args:
            function_proto: The FunctionProto object to register.
        """
        self.functions[function_proto.name] = function_proto

    def create_function(
        self,
        domain: str,
        name: str,
        inputs: List[str],
        outputs: List[str],
        nodes: List[NodeProto],
        opset: int,
        value_info: Optional[List[ValueInfoProto]] = None,
    ) -> FunctionProto:
        """
        Create an ONNX function with the specified parameters.

        Args:
            domain: The domain of the function.
            name: The name of the function.
            inputs: The input names of the function.
            outputs: The output names of the function.
            nodes: The nodes contained in the function.
            opset: The opset version to use.
            value_info: Optional value info for intermediate tensors.

        Returns:
            An ONNX FunctionProto object.
        """
        function_proto = helper.make_function(
            domain=domain,
            fname=name,
            inputs=inputs,
            outputs=outputs,
            nodes=nodes,
            opset_imports=[
                helper.make_opsetid("", opset),
                helper.make_opsetid(domain, 1),
            ],
            value_info=value_info or [],
        )
        self.register_function(function_proto)
        return function_proto

    def get_unique_function_name(self, base_name: str) -> str:
        """
        Get a unique function name based on the provided base name.

        Args:
            base_name: The base name for the function.

        Returns:
            A unique function name.
        """
        return self.name_generator.get(base_name)

    def get_all_functions(self) -> List[FunctionProto]:
        """
        Get all registered functions, ensuring no duplicates.

        Returns:
            A list of unique FunctionProto objects.
        """
        # Use a dictionary to filter duplicates by name
        unique_functions = {f.name: f for f in self.functions.values()}
        return list(unique_functions.values())

    def create_function_call_node(
        self,
        function_name: str,
        input_names: List[str],
        output_names: List[str],
        domain: str = CUSTOM_DOMAIN,
        node_name: Optional[str] = None,
        op_type: Optional[str] = None,
    ) -> NodeProto:
        """
        Create a node that calls an ONNX function.

        Args:
            function_name: The name of the function to call.
            input_names: The input tensor names.
            output_names: The output tensor names.
            domain: The function domain.
            node_name: Optional name for the node.
            op_type: Optional operation type for the node.

        Returns:
            A NodeProto representing a function call.
        """
        if node_name is None:
            node_name = self.name_generator.get(function_name.split(".")[-1])

        return helper.make_node(
            op_type=op_type or function_name,
            inputs=input_names,
            outputs=output_names,
            name=node_name,
            domain=domain,
        )

    def merge_functions(self, other_functions: Dict[str, FunctionProto]) -> None:
        """
        Merge functions from another source into this builder.

        Args:
            other_functions: Dictionary of functions to merge.
        """
        import warnings

        for name, func in other_functions.items():
            if name not in self.functions:
                self.functions[name] = func
            else:
                warnings.warn(
                    f"[Duplicate function] Skipping already-registered function '{name}'",
                    RuntimeWarning,
                )
