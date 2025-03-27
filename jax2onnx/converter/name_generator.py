class GlobalNameCounter:
    def __init__(self):
        self._counter = 0

    def get(self, prefix: str = "node") -> str:
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        print(f"Generated name: {name}")
        return name
