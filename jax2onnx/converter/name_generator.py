# file: jax2onnx/converter/name_generator.py
from collections import defaultdict


class UniqueNameGenerator:

    def __init__(self):
        self._counters = defaultdict(int)

    def get(self, base_name: str = "node", context="default") -> str:
        context_and_base_name = context + "_" + base_name
        count = self._counters[context_and_base_name]
        name = f"{base_name}_{count}"
        self._counters[context_and_base_name] += 1
        print(f"Generated name: {name}")
        return name


def get_qualified_name(obj) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"
