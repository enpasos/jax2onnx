# tests/extra_tests/test_precision_config_restoration.py

"""Source precision mutations take effect and leave caller globals intact."""

from __future__ import annotations

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import allclose, to_onnx
from jax2onnx._compat.jax import preserve_precision_globals


@pytest.fixture(autouse=True)
def _restore_test_precision_globals() -> Iterator[None]:
    with preserve_precision_globals():
        yield


def _precision_values() -> tuple[object, ...]:
    return (
        jax.config.jax_enable_x64,
        jax.config.jax_default_matmul_precision,
        jax.config.jax_explicit_x64_dtypes,
        jax.config.jax_numpy_dtype_promotion,
    )


def _prepare_globals() -> tuple[object, ...]:
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "high")
    jax.config.update("jax_explicit_x64_dtypes", "warn")
    jax.config.update("jax_numpy_dtype_promotion", "standard")
    return _precision_values()


def _mutate_globals() -> None:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_explicit_x64_dtypes", "allow")
    jax.config.update("jax_numpy_dtype_promotion", "strict")


def _source(x: jax.Array) -> jax.Array:
    _mutate_globals()
    wide = x.astype(jnp.float64)
    return (wide + jnp.asarray(1, dtype=jnp.float64)) - wide


def _raising_source(x: jax.Array) -> jax.Array:
    del x
    _mutate_globals()
    raise RuntimeError("precision source error")


def _assert_fp64_one(model: onnx.ModelProto, values: np.ndarray) -> None:
    onnx.checker.check_model(model, full_check=True)
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(None, {session.get_inputs()[0].name: values})
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.ones((1,), dtype=np.float64))


def test_export_preserves_source_changes_and_restores_all_precision_globals() -> None:
    before = _prepare_globals()
    values = np.asarray([2**24], dtype=np.float32)
    with preserve_precision_globals():
        reference = np.asarray(_source(jnp.asarray(values)))
    assert reference.dtype == np.float64
    np.testing.assert_array_equal(reference, np.ones((1,), dtype=np.float64))
    assert _precision_values() == before

    model = to_onnx(_source, [values])

    assert _precision_values() == before
    _assert_fp64_one(model, values)


def test_export_error_restores_all_precision_globals(tmp_path) -> None:
    before = _prepare_globals()
    output = tmp_path / "failed.onnx"
    with pytest.raises(RuntimeError, match="precision source error"):
        to_onnx(
            _raising_source,
            [np.asarray([2**24], dtype=np.float32)],
            return_mode="file",
            output_path=output,
        )
    assert _precision_values() == before
    assert not output.exists()


def test_outer_thread_local_x64_does_not_leak_into_global_state() -> None:
    before = _prepare_globals()
    values = np.asarray([2**24], dtype=np.float32)
    with jax.enable_x64():
        model = to_onnx(_source, [values])
        assert jax.config.jax_enable_x64
        assert _precision_values()[1:] == before[1:]
    assert _precision_values() == before
    _assert_fp64_one(model, values)


@pytest.mark.parametrize("source_raises", [False, True])
def test_allclose_restores_globals_after_source_success_or_error(
    tmp_path, source_raises: bool
) -> None:
    before = _prepare_globals()
    values = np.asarray([2**24], dtype=np.float32)
    path = to_onnx(
        _source,
        [values],
        return_mode="file",
        output_path=tmp_path / "source.onnx",
    )
    assert _precision_values() == before

    valid, message = allclose(
        _raising_source if source_raises else _source,
        path,
        [values],
        rtol=0,
        atol=0,
    )

    assert _precision_values() == before
    if source_raises:
        assert not valid
        assert "precision source error" in message
    else:
        assert valid, message
