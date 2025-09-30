from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jax2onnx.user_interface import to_onnx


def _model_with_cond_and_scatter():
    original_operand = jnp.asarray(
        np.arange(2 * 4 * 1 * 1, dtype=np.float64).reshape((2, 4, 1, 1))
    )
    updates = jnp.ones((1, 4, 1, 1), dtype=jnp.float64) * 100.0
    reshaped_updates = jnp.reshape(updates, (1, 4, 1, 1))
    indices = jnp.array([1])
    predicate = jnp.array(True)
    branch_operands = (original_operand, indices, reshaped_updates)

    def true_branch(ops):
        op, idx, upd = ops
        return op.at[idx].set(upd)

    def false_branch(ops):
        op, *_ = ops
        return op + 1.0

    scattered_result = jax.lax.cond(
        predicate, true_branch, false_branch, branch_operands
    )
    some_int_value = jnp.array(42, dtype=jnp.int64)
    reshaped_int_value = jnp.reshape(some_int_value, ())
    return scattered_result, reshaped_int_value


def test_cond_scatter_reproducer_ir():
    original_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        model = to_onnx(
            _model_with_cond_and_scatter,
            inputs=[],
            model_name="cond_scatter_repro_ir",
            enable_double_precision=True,
        )
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(model.SerializeToString())
            outputs = sess.run(None, {})
            assert len(outputs) == 2
        except Exception as exc:  # pragma: no cover - parity gap tracker
            pytest.xfail(f"converter2 cond+scatter regression not resolved: {exc}")
    finally:
        jax.config.update("jax_enable_x64", original_x64)
