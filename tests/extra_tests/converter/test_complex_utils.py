# tests/extra_tests/converter/test_complex_utils.py

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_context import IRContext
import pytest

from jax2onnx.plugins._complex_utils import (
    ensure_complex_dtype,
    pack_native_complex,
    unpack_to_native_complex,
)


def _dims(value: ir.Value) -> tuple[object, ...]:
    shape = getattr(value, "shape", None)
    if isinstance(shape, ir.Shape):
        return tuple(shape.dims)
    if shape is None:
        return ()
    return tuple(shape)


def _ops(ctx: IRContext) -> list[str]:
    return [node.op_type for node in ctx.builder.graph]


def test_pack_native_complex_inserts_expected_nodes() -> None:
    ctx = IRContext(opset=18, enable_double_precision=True)
    complex_vals = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex128)
    complex_init = ctx.builder.add_initializer_from_array(
        "complex_source", complex_vals
    )

    packed = pack_native_complex(ctx, complex_init, name_hint="fft")

    assert packed.dtype == ir.DataType.DOUBLE
    assert _dims(packed) == (2, 2)
    assert _ops(ctx) == ["Real", "Imag", "Unsqueeze", "Unsqueeze", "Concat"]


def test_unpack_to_native_complex_restores_shape_and_dtype() -> None:
    ctx = IRContext(opset=18, enable_double_precision=True)
    complex_vals = np.asarray([[1 + 2j, 3 + 4j]], dtype=np.complex128)
    complex_init = ctx.builder.add_initializer_from_array("complex_input", complex_vals)
    packed = pack_native_complex(ctx, complex_init, name_hint="fft")

    restored = unpack_to_native_complex(ctx, packed, name_hint="ifft")

    assert restored.dtype == ir.DataType.COMPLEX128
    assert _dims(restored) == (1, 2)
    assert _ops(ctx)[-3:] == ["Gather", "Gather", "Complex"]


def test_pack_native_complex_complex64_uses_float_channels() -> None:
    ctx = IRContext(opset=18, enable_double_precision=False)
    complex_vals = np.asarray([[1 + 2j]], dtype=np.complex64)
    complex_init = ctx.builder.add_initializer_from_array(
        "complex64_input", complex_vals
    )

    packed = pack_native_complex(ctx, complex_init, name_hint="fft32")

    assert packed.dtype == ir.DataType.FLOAT
    assert _dims(packed) == (1, 1, 2)


def test_ensure_complex_dtype_inserts_cast_when_needed() -> None:
    ctx = IRContext(opset=18, enable_double_precision=True)
    complex_vals = np.asarray([1 + 2j], dtype=np.complex64)
    complex_init = ctx.builder.add_initializer_from_array(
        "complex_cast_src", complex_vals
    )

    promoted = ensure_complex_dtype(ctx, complex_init, ir.DataType.COMPLEX128)

    assert promoted.dtype == ir.DataType.COMPLEX128
    assert _ops(ctx)[-1] == "Cast"


def test_pack_native_complex_rejects_non_complex_inputs() -> None:
    ctx = IRContext(opset=18, enable_double_precision=True)
    real_vals = np.asarray([1.0, 2.0], dtype=np.float32)
    real_init = ctx.builder.add_initializer_from_array("real", real_vals)

    with pytest.raises(ValueError):
        pack_native_complex(ctx, real_init, name_hint="bad")
