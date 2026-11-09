# tests/extra_tests/framework/test_reshape_opset_semantics.py

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
from numpy.typing import NDArray

from jax2onnx import to_onnx


ReshapeFn = Callable[[jax.Array], jax.Array]
_OPSETS = [11, 12, 13, 14, None]


def _reshape(api: str, x: jax.Array, shape: tuple[object, ...]) -> jax.Array:
    if api == "lax":
        return jax.lax.reshape(x, shape)
    if api == "method":
        return x.reshape(shape)
    return jnp.reshape(x, shape)


def _export(
    fn: ReshapeFn,
    inputs: NDArray[np.float32] | jax.ShapeDtypeStruct,
    opset: int | None,
) -> onnx.ModelProto:
    model = (
        to_onnx(fn, [inputs]) if opset is None else to_onnx(fn, [inputs], opset=opset)
    )
    onnx.checker.check_model(model, full_check=True)
    if opset is not None and opset < 14:
        assert all(
            attr.name != "allowzero"
            for node in model.graph.node
            if node.op_type == "Reshape"
            for attr in node.attribute
        )
    return model


def _assert_runtime(
    model: onnx.ModelProto,
    fn: ReshapeFn,
    *inputs: NDArray[np.float32],
) -> None:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for x in inputs:
        actual = session.run(None, {session.get_inputs()[0].name: x})[0]
        expected = np.asarray(fn(jnp.asarray(x)))
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("opset", _OPSETS)
@pytest.mark.parametrize("api", ["lax", "method", "jnp"])
def test_ordinary_reshape_uses_target_opset(api: str, opset: int | None) -> None:
    for input_shape, target in [((2, 3), (3, 2)), ((1,), ()), ((), (1, 1))]:
        x = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)

        def fn(x: jax.Array) -> jax.Array:
            return _reshape(api, x, target)

        model = _export(fn, x, opset)
        _assert_runtime(model, fn, x)


@pytest.mark.parametrize("opset", [14, None])
@pytest.mark.parametrize("api", ["lax", "method", "jnp"])
def test_literal_zero_dimensions_are_preserved(api: str, opset: int | None) -> None:
    for input_shape, target in [
        ((3, 0), (0, 3)),
        ((0, 3), (3, 0, 1)),
        ((2, 0, 0), (0, 0, 2)),
    ]:
        x = np.empty(input_shape, dtype=np.float32)

        def fn(x: jax.Array) -> jax.Array:
            return _reshape(api, x, target)

        model = _export(fn, x, opset)
        _assert_runtime(model, fn, x)


@pytest.mark.parametrize("opset", [11, 12, 13])
@pytest.mark.parametrize("api", ["lax", "method", "jnp"])
def test_old_opsets_keep_copy_zeros_and_reject_literal_zeros(
    api: str, opset: int
) -> None:
    x = np.empty((0, 3), dtype=np.float32)

    def aligned(x: jax.Array) -> jax.Array:
        return _reshape(api, x, (0, 1, 3))

    _assert_runtime(_export(aligned, x, opset), aligned, x)

    def moved(x: jax.Array) -> jax.Array:
        return _reshape(api, x, (3, 0))

    with pytest.raises(NotImplementedError, match="zero.*opset.*14"):
        to_onnx(moved, [x], opset=opset)


@pytest.mark.parametrize("opset", _OPSETS)
def test_inferred_zero_dimension_does_not_become_copy_zero(
    opset: int | None,
) -> None:
    x = np.empty((3, 0), dtype=np.float32)

    for target in [(-1,), (-1, 3), (2, -1, 3)]:

        def fn(x: jax.Array) -> jax.Array:
            return jnp.reshape(x, target)

        _assert_runtime(_export(fn, x, opset), fn, x)


@pytest.mark.parametrize("opset", _OPSETS)
@pytest.mark.parametrize("api", ["lax", "method", "jnp"])
def test_symbolic_reshape_keeps_dimensions_in_their_target_positions(
    api: str, opset: int | None
) -> None:
    spec = jax.ShapeDtypeStruct(jax.export.symbolic_shape("B,2,3"), jnp.float32)

    def fn(x: jax.Array) -> jax.Array:
        return _reshape(api, x, (2, x.shape[0], 3))

    model = _export(fn, spec, opset)
    sizes = [1, 4] if opset is not None and opset < 14 else [0, 1, 4]
    _assert_runtime(
        model,
        fn,
        *(np.arange(size * 6, dtype=np.float32).reshape(size, 2, 3) for size in sizes),
    )


@pytest.mark.parametrize("opset", [14, None])
@pytest.mark.parametrize("api", ["method", "jnp"])
def test_symbolic_reshape_with_inferred_dimension(api: str, opset: int | None) -> None:
    spec = jax.ShapeDtypeStruct(jax.export.symbolic_shape("B,2,3"), jnp.float32)

    def fn(x: jax.Array) -> jax.Array:
        return _reshape(api, x, (2, x.shape[0], -1))

    model = _export(fn, spec, opset)
    _assert_runtime(
        model,
        fn,
        np.arange(6, dtype=np.float32).reshape(1, 2, 3),
        np.arange(24, dtype=np.float32).reshape(4, 2, 3),
    )


@pytest.mark.parametrize("opset", [14, None])
@pytest.mark.parametrize("api", ["lax", "method", "jnp"])
def test_literal_and_symbolic_zero_dimensions(api: str, opset: int | None) -> None:
    spec = jax.ShapeDtypeStruct(jax.export.symbolic_shape("B,0"), jnp.float32)

    def fn(x: jax.Array) -> jax.Array:
        return _reshape(api, x, (0, x.shape[0]))

    model = _export(fn, spec, opset)
    _assert_runtime(
        model,
        fn,
        np.empty((0, 0), dtype=np.float32),
        np.empty((4, 0), dtype=np.float32),
    )
