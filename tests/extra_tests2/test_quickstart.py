from __future__ import annotations

import onnx

from flax import nnx

from jax2onnx import onnx_function
from jax2onnx.user_interface import to_onnx


class _QuickstartMLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
        return self.linear2(x)


@onnx_function
class _MLPBlock(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs)
        self.batchnorm = nnx.BatchNorm(dim, rngs=rngs)

    def __call__(self, x):
        return nnx.gelu(self.linear2(self.batchnorm(nnx.gelu(self.linear1(x)))))


class _QuickstartModel(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.block1 = _MLPBlock(dim, rngs=rngs)
        self.block2 = _MLPBlock(dim, rngs=rngs)

    def __call__(self, x):
        return self.block2(self.block1(x))


class TestQuickstart:
    def test_quickstart_execution_ir(self, tmp_path):
        nnx_rng = nnx.Rngs(0)
        model = _QuickstartMLP(din=30, dmid=20, dout=10, rngs=nnx_rng)
        onnx_model = to_onnx(
            model,
            [("B", 30)],
            model_name="quickstart_ir",
        )
        out_file = tmp_path / "my_callable.onnx"
        onnx.save_model(onnx_model, out_file)
        loaded = onnx.load(out_file)
        onnx.checker.check_model(loaded)
        graph = loaded.graph
        assert len(graph.input) == 1
        assert len(graph.output) == 1


class TestQuickstartFunctions:
    def test_quickstart_functions_execution_ir(self, tmp_path):
        nnx_rng = nnx.Rngs(0)
        model = _QuickstartModel(256, rngs=nnx_rng)
        onnx_model = to_onnx(
            model,
            [(100, 256)],
            model_name="quickstart_functions_ir",
        )
        out_file = tmp_path / "model_with_function.onnx"
        onnx.save_model(onnx_model, out_file)
        loaded = onnx.load(out_file)
        onnx.checker.check_model(loaded)
        op_types = {node.op_type for node in loaded.graph.node}
        for fn in getattr(loaded, "functions", []) or []:
            op_types.update(node.op_type for node in fn.node)
        assert "BatchNormalization" in op_types
        assert "Gemm" in op_types
