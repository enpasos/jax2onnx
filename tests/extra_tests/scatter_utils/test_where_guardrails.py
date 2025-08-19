# file: tests/extra_tests/scatter_utils/test_where_guardrails.py


import numpy as np
from types import SimpleNamespace
from onnx import TensorProto

from jax2onnx.plugins.jax.lax.scatter_utils import (
    ShapeDtypeStruct, emit_where_with_guardrails, _ensure_np_dtype
)


class FakeBuilder:
    def __init__(self):
        self._nodes = []
        self._consts = {}
        self._name_counter = 0
        self.opset = 18

    def _numpy_dtype_to_onnx(self, dt):
        dt = _ensure_np_dtype(dt)
        if dt == np.float32: return TensorProto.FLOAT
        if dt == np.float64: return TensorProto.DOUBLE
        if dt == np.int64:   return TensorProto.INT64
        if dt == np.int32:   return TensorProto.INT32
        if dt == np.bool_:   return TensorProto.BOOL
        raise KeyError(str(dt))

    def make_name(self, base="n"):  # not used, kept for parity
        self._name_counter += 1
        return f"{base}_{self._name_counter}"


class FakeConverter:
    def __init__(self):
        self.builder = FakeBuilder()
        self.shape_env = {}
        self._uniq = 0

    def get_unique_name(self, base: str) -> str:
        self._uniq += 1
        return f"{base}_{self._uniq}"

    def get_constant_name(self, arr: np.ndarray) -> str:
        key = (arr.dtype.str, tuple(arr.reshape(-1).tolist()))
        name = f"const_{hash(key) & 0xFFFFFFFF:x}"
        self.builder._consts[name] = arr
        # minimal info for tests (rank & dtype)
        self.add_shape_info(name, (int(arr.size),) if arr.ndim == 1 else tuple(arr.shape), arr.dtype)
        return name

    def add_node(self, node):
        self.builder._nodes.append(node)

    def add_shape_info(self, name, shape, dtype):
        self.shape_env[name] = ShapeDtypeStruct(tuple(shape), dtype)


def test_where_guardrails_broadcasts_cond_and_casts_bool():
    s = FakeConverter()
    B, L = 5, 4
    # inputs already registered in shape_env
    cond = "cond"
    x = "x"
    y = "y"
    s.shape_env[cond] = ShapeDtypeStruct((B, L), np.int64)          # not bool
    s.shape_env[x]    = ShapeDtypeStruct((B, L, 1, 1), np.float32)
    s.shape_env[y]    = ShapeDtypeStruct((B, L, 1, 1), np.float32)

    out = emit_where_with_guardrails(s, cond, x, y, context="UT")
    assert out in s.shape_env
    # target shape should be fully concrete (B,L,1,1)
    assert s.shape_env[out].shape == (B, L, 1, 1)
    assert _ensure_np_dtype(s.shape_env[out].dtype) == np.float32

    # Ensure a Cast→BOOL node was inserted for cond
    casted = [n for n in s.builder._nodes if n.op_type == "Cast"]
    assert casted, "cond→BOOL Cast not emitted"