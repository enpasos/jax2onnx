from .fori_loop import model_fn

# test_fori_loop.py
import jax.numpy as jnp


def reference_fori_loop(x_init, steps=5):
    x = x_init
    for _ in range(steps):
        x = x + 0.1 * x**2
    return x


def test_model_fn_counter_and_value():
    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    out_x, out_counter = model_fn(x)
    # Check counter
    assert out_counter == 5
    # Check value
    expected_x = reference_fori_loop(x)
    assert jnp.allclose(out_x, expected_x, rtol=1e-5, atol=1e-6)
