# jax2onnx/quickstart_functions.py

from __future__ import annotations

from pathlib import Path

from flax import nnx
from jax2onnx import onnx_function, to_onnx


@onnx_function
class MLPBlock(nnx.Module):
    def __init__(self, dim, *, rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs)
        self.batchnorm = nnx.BatchNorm(dim, rngs=rngs)

    def __call__(self, x):
        return nnx.gelu(self.linear2(self.batchnorm(nnx.gelu(self.linear1(x)))))


class MyModel(nnx.Module):
    def __init__(self, dim, *, rngs):
        self.block1 = MLPBlock(dim, rngs=rngs)
        self.block2 = MLPBlock(dim, rngs=rngs)

    def __call__(self, x):
        return self.block2(self.block1(x))


def _default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "onnx"
        / "model_with_function.onnx"
    )


def export_quickstart_functions_model(output_path: str | Path | None = None) -> Path:
    target = Path(output_path) if output_path is not None else _default_output_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    callable_model = MyModel(256, rngs=nnx.Rngs(0))
    to_onnx(callable_model, [(100, 256)], return_mode="file", output_path=target)
    return target


if __name__ == "__main__":
    export_quickstart_functions_model()
