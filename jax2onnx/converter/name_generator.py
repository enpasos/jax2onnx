# file: jax2onnx/converter/name_generator.py
from collections import defaultdict


class UniqueNameGenerator:
    """
    Generates unique names based on a base name and context.
    """

    def __init__(self):
        # Initialize counters for each context and base name combination.
        self._counters = defaultdict(int)

    def get(self, base_name: str = "node", context="default") -> str:
        """
        Generate a unique name by appending a counter to the base name.

        Args:
            base_name: The base name for the generated name.
            context: An optional context to differentiate name scopes.

        Returns:
            A unique name string.
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
    """
    return f"{obj.__module__}.{obj.__qualname__}"
