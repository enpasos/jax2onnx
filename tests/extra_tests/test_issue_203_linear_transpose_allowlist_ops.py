# tests/extra_tests/test_issue_203_linear_transpose_allowlist_ops.py

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax2onnx.plugins.jax._autodiff_utils import (
    get_linear_transpose_fallback_allowlist,
)
from jax2onnx.user_interface import to_onnx


_EXPECTED_LINEAR_TRANSPOSE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "jax.numpy.add",
        "jax.numpy.concatenate",
        "jax.numpy.moveaxis",
        "jax.numpy.reshape",
        "jax.numpy.select",
        "jax.numpy.split",
        "jax.numpy.squeeze",
        "jax.numpy.stack",
        "jax.numpy.take",
        "jax.numpy.tile",
        "jax.numpy.transpose",
        "jax.numpy.where",
    }
)


def test_linear_transpose_allowlist_matches_conversion_regression_matrix() -> None:
    assert (
        get_linear_transpose_fallback_allowlist()
        == _EXPECTED_LINEAR_TRANSPOSE_ALLOWLIST
    )


@jax.jit
def _issue_203_linear_transpose_reshape(
    x: jnp.ndarray, cotangent: jnp.ndarray
) -> jnp.ndarray:
    def fn(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.reshape(y, (3, 2))

    (x_bar,) = jax.linear_transpose(fn, x)(cotangent)
    return x_bar


@jax.jit
def _issue_203_linear_transpose_squeeze(
    x: jnp.ndarray, cotangent: jnp.ndarray
) -> jnp.ndarray:
    def fn(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.squeeze(y, axis=2)

    (x_bar,) = jax.linear_transpose(fn, x)(cotangent)
    return x_bar


@jax.jit
def _issue_203_linear_transpose_moveaxis(
    x: jnp.ndarray, cotangent: jnp.ndarray
) -> jnp.ndarray:
    def fn(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.moveaxis(y, 2, 0)

    (x_bar,) = jax.linear_transpose(fn, x)(cotangent)
    return x_bar


@jax.jit
def _issue_203_linear_transpose_stack(
    x: jnp.ndarray, cotangent: jnp.ndarray
) -> jnp.ndarray:
    def fn(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([y, y], axis=0)

    (x_bar,) = jax.linear_transpose(fn, x)(cotangent)
    return x_bar


@jax.jit
def _issue_203_linear_transpose_concatenate(
    x: jnp.ndarray, cotangent: jnp.ndarray
) -> jnp.ndarray:
    def fn(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([y, y], axis=1)

    (x_bar,) = jax.linear_transpose(fn, x)(cotangent)
    return x_bar


@jax.jit
def _issue_203_linear_transpose_take(
    x: jnp.ndarray, cotangent: jnp.ndarray
) -> jnp.ndarray:
    idx = jnp.asarray([0, 2], dtype=jnp.int32)

    def fn(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(y, idx, axis=1)

    (x_bar,) = jax.linear_transpose(fn, x)(cotangent)
    return x_bar


@jax.jit
def _issue_203_linear_transpose_tile(
    x: jnp.ndarray, cotangent: jnp.ndarray
) -> jnp.ndarray:
    def fn(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.tile(y, (2, 1))

    (x_bar,) = jax.linear_transpose(fn, x)(cotangent)
    return x_bar


@jax.jit
def _issue_203_linear_transpose_where(
    x: jnp.ndarray, cotangent: jnp.ndarray
) -> jnp.ndarray:
    cond = jnp.asarray(
        [[True, False, True], [False, True, False]],
        dtype=jnp.bool_,
    )

    def fn(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(cond, y, jnp.zeros_like(y))

    (x_bar,) = jax.linear_transpose(fn, x)(cotangent)
    return x_bar


@jax.jit
def _issue_203_linear_transpose_select(
    x: jnp.ndarray, cotangent: jnp.ndarray
) -> jnp.ndarray:
    cond_a = jnp.asarray(
        [[True, False, False], [False, True, False]],
        dtype=jnp.bool_,
    )
    cond_b = jnp.asarray(
        [[False, True, False], [False, False, True]],
        dtype=jnp.bool_,
    )

    def fn(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.select([cond_a, cond_b], [y, 2.0 * y], default=jnp.zeros_like(y))

    (x_bar,) = jax.linear_transpose(fn, x)(cotangent)
    return x_bar


@jax.jit
def _issue_203_linear_transpose_split(
    x: jnp.ndarray,
    cotangent_0: jnp.ndarray,
    cotangent_1: jnp.ndarray,
) -> jnp.ndarray:
    def fn(y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        parts = jnp.split(y, 2, axis=1)
        return parts[0], parts[1]

    (x_bar,) = jax.linear_transpose(fn, x)((cotangent_0, cotangent_1))
    return x_bar


def test_issue_203_linear_transpose_reshape_exports_with_ir_pipeline():
    to_onnx(
        _issue_203_linear_transpose_reshape,
        inputs=[(2, 3), (3, 2)],
        model_name="issue_203_linear_transpose_reshape",
    )


def test_issue_203_linear_transpose_squeeze_exports_with_ir_pipeline():
    to_onnx(
        _issue_203_linear_transpose_squeeze,
        inputs=[(2, 3, 1), (2, 3)],
        model_name="issue_203_linear_transpose_squeeze",
    )


def test_issue_203_linear_transpose_moveaxis_exports_with_ir_pipeline():
    to_onnx(
        _issue_203_linear_transpose_moveaxis,
        inputs=[(2, 3, 4), (4, 2, 3)],
        model_name="issue_203_linear_transpose_moveaxis",
    )


def test_issue_203_linear_transpose_stack_exports_with_ir_pipeline():
    to_onnx(
        _issue_203_linear_transpose_stack,
        inputs=[(2, 3), (2, 2, 3)],
        model_name="issue_203_linear_transpose_stack",
    )


def test_issue_203_linear_transpose_concatenate_exports_with_ir_pipeline():
    to_onnx(
        _issue_203_linear_transpose_concatenate,
        inputs=[(2, 3), (2, 6)],
        model_name="issue_203_linear_transpose_concatenate",
    )


def test_issue_203_linear_transpose_take_exports_with_ir_pipeline():
    to_onnx(
        _issue_203_linear_transpose_take,
        inputs=[(2, 3), (2, 2)],
        model_name="issue_203_linear_transpose_take",
    )


def test_issue_203_linear_transpose_tile_exports_with_ir_pipeline():
    to_onnx(
        _issue_203_linear_transpose_tile,
        inputs=[(2, 3), (4, 3)],
        model_name="issue_203_linear_transpose_tile",
    )


def test_issue_203_linear_transpose_where_exports_with_ir_pipeline():
    to_onnx(
        _issue_203_linear_transpose_where,
        inputs=[(2, 3), (2, 3)],
        model_name="issue_203_linear_transpose_where",
    )


def test_issue_203_linear_transpose_select_exports_with_ir_pipeline():
    to_onnx(
        _issue_203_linear_transpose_select,
        inputs=[(2, 3), (2, 3)],
        model_name="issue_203_linear_transpose_select",
    )


def test_issue_203_linear_transpose_split_exports_with_ir_pipeline():
    to_onnx(
        _issue_203_linear_transpose_split,
        inputs=[(2, 4), (2, 2), (2, 2)],
        model_name="issue_203_linear_transpose_split",
    )
