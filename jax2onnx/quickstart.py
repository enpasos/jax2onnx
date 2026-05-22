# jax2onnx/quickstart.py

from __future__ import annotations

from pathlib import Path

from flax import nnx
import jax
from jax2onnx import to_onnx
from jax2onnx import onnx_function


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
        return self.linear2(x)


class WebMLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear2(nnx.gelu(self.linear1(x)))


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[1] / "onnx" / "my_callable.onnx"


def export_quickstart_model(output_path: str | Path | None = None) -> Path:
    target = Path(output_path) if output_path is not None else _default_output_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    model = MLP(din=30, dmid=20, dout=10, rngs=nnx.Rngs(0))
    to_onnx(model, [("B", 30)], return_mode="file", output_path=target)
    return target


def build_quickstart_web_model() -> WebMLP:
    return WebMLP(din=8, dmid=6, dout=3, rngs=nnx.Rngs(0))


def export_quickstart_web_model(output_path: str | Path | None = None) -> Path:
    target = (
        Path(output_path)
        if output_path is not None
        else Path(__file__).resolve().parents[1] / "onnx" / "web_mlp.onnx"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    model = build_quickstart_web_model()
    to_onnx(
        model,
        [("B", 8)],
        return_mode="file",
        output_path=target,
        export_mode="web",
    )
    return target


@onnx_function
class MLPBlock(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs)
        self.bn = nnx.BatchNorm(dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.gelu(self.linear2(self.bn(nnx.gelu(self.linear1(x)))))


class FnModel(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs) -> None:
        self.block1 = MLPBlock(dim, rngs=rngs)
        self.block2 = MLPBlock(dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.block2(self.block1(x))


def export_quickstart_functions(output_path: str | Path | None = None) -> Path:
    target = (
        Path(output_path)
        if output_path is not None
        else Path(__file__).resolve().parents[1] / "onnx" / "model_with_function.onnx"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    model = FnModel(256, rngs=nnx.Rngs(0))
    to_onnx(model, [(100, 256)], return_mode="file", output_path=target)
    return target


def main() -> None:
    export_quickstart_model()
    export_quickstart_web_model()
    export_quickstart_functions()


if __name__ == "__main__":
    main()
