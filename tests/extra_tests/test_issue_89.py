# file: tests/extra_tests/test_issue_89.py


from collections.abc import Callable
from functools import partial
from jax2onnx import to_onnx
import pytest
import jax
from jax import numpy as jnp
import optax


def basic_integral_fn(dx_dt: jax.Array, init: jax.Array, dt: jax.Array) -> jax.Array:
    """A basic integral function to test gradient descent."""
    return init + jnp.cumsum(dx_dt * dt, axis=0)


def basic_goal_fn(agent_trajectory: jax.Array, goal: jax.Array) -> jax.Array:
    """A basic loss function to test gradient descent."""
    return jnp.sum((agent_trajectory - goal) ** 2)


def basic_col_avoidance_fn(
    agent_trajectory: jax.Array, obstacles: jax.Array
) -> jax.Array:
    """A basic collision avoidance function to test gradient descent."""
    dists = jnp.linalg.norm(
        agent_trajectory[:, None, :] - obstacles[None, :, :], axis=-1
    )
    min_dist = jnp.min(dists, axis=1)
    return jnp.sum(jnp.exp(-min_dist) / 10)  # Penalize close distances


def basic_loss_fn(
    dx_dt: jax.Array,
    init: jax.Array,
    goal: jax.Array | None = None,
    obst: jax.Array | None = None,
    dt: jax.Array | None = None,
):
    dt = jnp.array(0.1) if dt is None else dt
    path = basic_integral_fn(dx_dt, init, dt)
    goal_loss = jnp.array(0.0) if goal is None else basic_goal_fn(path, goal)
    obst_loss = jnp.array(0.0) if obst is None else basic_col_avoidance_fn(path, obst)
    return goal_loss + obst_loss


@partial(jax.jit, static_argnames=("optimizer", "unroll"))
def optax_test_fn(
    loss_fn: Callable[[jax.Array], jax.Array],
    init: jax.Array,
    lr: jax.Array,
    optimizer: optax.GradientTransformation = optax.adam(1e-2),
    unroll: bool = False,
) -> jax.Array:
    """A basic optimization function using Optax to test optimization."""
    opt_state = optimizer.init(init)

    def step_fn(
        state: tuple[jax.Array, optax.OptState], lr: jax.Array
    ) -> tuple[tuple[jax.Array, optax.OptState], jax.Array]:
        params, opt_state = state
        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates * lr)
        loss = loss_fn(params)
        return (params, opt_state), loss

    (optimized_params, _), _ = jax.lax.scan(
        step_fn,
        (init, opt_state),
        lr,
        length=lr.shape[0],
        unroll=unroll,
    )

    return optimized_params


@pytest.mark.parametrize("unroll", [True, False])
@pytest.mark.parametrize("num_steps", [8, 32, 128])
def test_basic_grad_descent_onnx(num_steps: int, unroll: bool):
    """Test the basic gradient descent function."""
    key = jax.random.PRNGKey(0)
    pred = jax.random.normal(key, (num_steps, 2)) * 0.1
    init = jnp.array([0.0, 0.0])
    goal = jnp.array([1.0, 1.0])
    obstacles = jnp.array([[0.5, 0.5], [0.6, 0.6]])
    lr = jnp.linspace(0.1, 1.0, num_steps).reshape(1, num_steps, 1)
    loss_fn = jax.tree_util.Partial(
        basic_loss_fn, init=init, goal=goal, obst=obstacles, dt=jnp.array(0.1)
    )
    jitted_optax_fn = jax.jit(partial(optax_test_fn, loss_fn, unroll=unroll))

    to_onnx(
        jitted_optax_fn,
        [pred.shape, lr.shape],
    )
