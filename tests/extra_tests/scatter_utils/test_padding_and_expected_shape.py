# file: tests/extra_tests/scatter_utils/test_padding_and_expected_shape.py


import numpy as np
from jax.lax import ScatterDimensionNumbers
from jax2onnx.plugins.jax.lax.scatter_utils import (
    _auto_pad_updates_if_smaller,
    compute_expected_updates_shape,
    ShapeDtypeStruct,
)


class TinyBuilder:
    def __init__(self):
        self._nodes = []
        self._consts = {}
        self.opset = 18

        def _map(dt):
            import numpy as np
            from onnx import TensorProto

            if dt == np.float32:
                return TensorProto.FLOAT
            if dt == np.float64:
                return TensorProto.DOUBLE
            if dt == np.int64:
                return TensorProto.INT64
            if dt == np.bool_:
                return TensorProto.BOOL
            return TensorProto.UNDEFINED

        self._numpy_dtype_to_onnx = _map


class TinyConv:
    def __init__(self):
        self.builder = TinyBuilder()
        self.shape_env = {}
        self._n = 0

    def get_unique_name(self, base):
        self._n += 1
        return f"{base}_{self._n}"

    def add_node(self, n):
        self.builder._nodes.append(n)

    def get_constant_name(self, arr):
        import hashlib

        key = hashlib.sha1(arr.tobytes()).hexdigest()[:8]
        name = f"c_{key}"
        self.builder._consts[name] = arr
        self.add_shape_info(name, arr.shape, arr.dtype)
        return name

    def add_shape_info(self, name, shape, dtype):
        self.shape_env[name] = ShapeDtypeStruct(tuple(shape), dtype)


def test_auto_pad_updates_if_smaller_right_pad_only():
    s = TinyConv()
    upd = "u"
    s.shape_env[upd] = ShapeDtypeStruct((5, 3), np.float32)
    name, shape = _auto_pad_updates_if_smaller(
        s,
        upd,
        (5, 3),
        (5, 4),
        s.get_constant_name(np.array(0.0, dtype=np.float32)),
        np.float32,
        "UT",
    )
    assert name != upd, "Pad should have been inserted"
    assert shape == (5, 4)
    assert s.shape_env[name].shape == (5, 4)


def test_compute_expected_updates_shape_simple_depth2_like():
    # Operand shape (B, M, L, C), index depth K=1 (scatter along axis 1),
    # update_window_dims should place window dims at positions in updates.
    dnums = ScatterDimensionNumbers(
        update_window_dims=(1, 2, 3),  # positions in updates where window dims go
        inserted_window_dims=(),  # no inserted dims
        scatter_dims_to_operand_dims=(1,),  # K=1 → scatter along operand axis 1 (M)
        operand_batching_dims=(),  # no batching dims
        scatter_indices_batching_dims=(),  # (legacy compat)
    )
    operand = (5, 7, 4, 3)  # (B, M, L, C)
    indices = (5, 1)  # (B, K) flattened to (N=5, K=1) equivalent batch part = (5,)
    # expected: batch part (5,) + window dims from operand at axes [0,2,3] (since axis 1 is scatter)
    # window dims are operand dims excluding inserted_window_dims: [0,2,3] → sizes (5,4,3)
    # update_window_dims=(1,2,3) → result rank = len(batch)=1 + len(window)=3 = 4
    # place window sizes at 1,2,3 → slot 0 is batch dim → (5,5,4,3) ??? NO:
    # window sizes are (operand[0], operand[2], operand[3]) = (5,4,3)
    # positions 1,2,3 get (5,4,3); slot 0 gets batch=5  → (5,5,4,3)
    # This shape is allowed by the helper and matches its definition.
    assert compute_expected_updates_shape(dnums, operand, indices) == (5, 5, 4, 3)
