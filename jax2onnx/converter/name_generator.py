"""
Name Generator Module

This module provides utilities for generating unique names and obtaining
fully qualified names for objects. These utilities are used throughout the
JAX2ONNX converter to ensure unique identifiers for nodes, tensors, and other
components in the generated ONNX models.
"""

from collections import defaultdict


class UniqueNameGenerator:
    """
    Generates unique names based on a base name and context.
    """

    def __init__(self):
        # Initialize counters for each context and base name combination.
        self._counters = defaultdict(int)

    def get(self, base_name: str = "node", context: str = "default") -> str:
        """
        Generate a unique name by appending a counter to the base name.

        The counter is specific to each combination of base_name and context,
        allowing independent counters for different contexts (e.g., different
        functions, modules, or scopes).

        Args:
            base_name: The base name for the generated name.
            context: An optional context to differentiate name scopes.
                     Different contexts maintain separate counters.

        Returns:
            A unique name string in the format "{base_name}_{counter}".

        Examples:
            >>> generator = UniqueNameGenerator()
            >>> generator.get("node")
            'node_0'
            >>> generator.get("node")
            'node_1'
            >>> generator.get("node", context="function1")
            'node_0'  # Different context starts its own counter
        """
        context_and_base_name = context + "_" + base_name
        count = self._counters[context_and_base_name]
        name = f"{base_name}_{count}"
        self._counters[context_and_base_name] += 1
        return name


def get_qualified_name(obj) -> str:
    """
    Get the fully qualified name of an object, including its module and class.

    Args:
        obj: The object to get the qualified name for.

    Returns:
        A string representing the fully qualified name of the object.
        Falls back to str(obj) if module or qualname attributes are missing.
    """
    try:
        module = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)

        if module is not None and qualname is not None:
            return f"{module}.{qualname}"
        elif hasattr(obj, "__name__"):
            return obj.__name__
        else:
            return str(obj)
    except Exception:
        # Ultimate fallback in case of any errors
        return str(obj)
