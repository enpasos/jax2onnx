# tests/extra_tests/framework/test_reduce_ops_coverage.py

import jax
import jax.numpy as jnp
import numpy as np
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.user_interface import to_onnx


def _assert_graph_has(
    fn,
    x: np.ndarray,
    expected_path: str,
    *,
    model_name: str,
) -> None:
    model = to_onnx(
        fn,
        inputs=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
        model_name=model_name,
        enable_double_precision=False,
    )
    check = EG([expected_path], no_unused_inputs=True)
    assert check(model)


def test_reduce_l1():
    # jax.numpy.linalg.norm(ord=1) -> ReduceL1
    def f(x):
        return jnp.linalg.norm(x, ord=1, axis=1, keepdims=True)

    x = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]], dtype=np.float32)

    # Expectation: ReduceL1 should be used directly or fused
    _assert_graph_has(f, x, "ReduceL1", model_name="reduce_l1_coverage")


def test_reduce_l2():
    # jax.numpy.linalg.norm(ord=2) -> ReduceL2
    def f(x):
        return jnp.linalg.norm(x, ord=2, axis=1)

    x = np.array([[3.0, 4.0], [6.0, 8.0]], dtype=np.float32)

    _assert_graph_has(f, x, "ReduceL2", model_name="reduce_l2_coverage")


def test_reduce_log_sum():
    # log(sum(x)) -> ReduceLogSum
    def f(x):
        return jnp.log(jnp.sum(x, axis=1))

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    _assert_graph_has(f, x, "ReduceLogSum", model_name="reduce_log_sum_coverage")


def test_reduce_sum_square_mul():
    # sum(x * x) -> ReduceSumSquare
    def f(x):
        return jnp.sum(x * x, axis=1)

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    _assert_graph_has(
        f,
        x,
        "ReduceSumSquare",
        model_name="reduce_sum_square_mul_coverage",
    )


def test_reduce_sum_square_pow():
    # sum(x ** 2) -> ReduceSumSquare
    # This specifically tests the new fusion logic in reduce_sum.py
    def f(x):
        return jnp.sum(x**2, axis=1)

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    _assert_graph_has(
        f,
        x,
        "ReduceSumSquare",
        model_name="reduce_sum_square_pow_coverage",
    )


def test_reduce_sum_square_square_api():
    # sum(jnp.square(x)) -> ReduceSumSquare
    def f(x):
        return jnp.sum(jnp.square(x), axis=1)

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    _assert_graph_has(
        f,
        x,
        "ReduceSumSquare",
        model_name="reduce_sum_square_api_coverage",
    )
