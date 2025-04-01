# file: jax2onnx/converter/name_generator.py
from collections import defaultdict


class UniqueNameGenerator:
    """Generates unique names within a scope, tracking counts per base name."""

    def __init__(self):
        # Use defaultdict to automatically handle new base names
        self._counters = defaultdict(int)

    def get(self, base_name: str = "node") -> str:
        """
        Generates a unique name by appending an incrementing counter to the base_name.

        Args:
            base_name: The prefix or type name to make unique.

        Returns:
            A unique name string (e.g., "base_name_0", "base_name_1").
        """
        count = self._counters[base_name]
        name = f"{base_name}_{count}"
        self._counters[base_name] += 1
        print(f"Generated name: {name}")
        return name


class GlobalNameCounter:
    def __init__(self):
        self._counter = 0

    def get(self, prefix: str = "node") -> str:
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        # print(f"Generated name: {name}")
        return name


def get_qualified_name(obj) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"
