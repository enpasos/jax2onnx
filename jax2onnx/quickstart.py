# jax2onnx/quickstart.py

from __future__ import annotations

from pathlib import Path

from flax import nnx
from jax2onnx import to_onnx


class MLP(nnx.Module):
    def __init__(self, din, dmid, dout, *, rngs):
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
        return self.linear2(x)


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "onnx" / "my_callable.onnx"


def export_quickstart_model(output_path: str | Path | None = None) -> Path:
    target = Path(output_path) if output_path is not None else _default_output_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    model = MLP(din=30, dmid=20, dout=10, rngs=nnx.Rngs(0))
    to_onnx(model, [("B", 30)], return_mode="file", output_path=target)
    return target


if __name__ == "__main__":
    export_quickstart_model()
