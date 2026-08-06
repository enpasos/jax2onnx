from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import pytest

import onnx
from jax2onnx import to_onnx


def _convert_and_run(
    fn: Callable[..., object],
    inputs: Sequence[jax.Array],
    *,
    opset: int = 18,
) -> tuple[object, list[np.ndarray]]:
    model = to_onnx(fn, list(inputs), opset=opset)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds = {
        value_info.name: np.asarray(value)
        for value_info, value in zip(model.graph.input, inputs)
    }
    return model, session.run(None, feeds)


def _assert_ort_parity(
    fn: Callable[..., object],
    inputs: Sequence[jax.Array],
    *,
    opset: int = 18,
) -> object:
    model, actual = _convert_and_run(fn, inputs, opset=opset)
    expected = jax.tree_util.tree_leaves(fn(*inputs))
    assert len(actual) == len(expected)
    for actual_leaf, expected_leaf in zip(actual, expected):
        np.testing.assert_allclose(
            actual_leaf, np.asarray(expected_leaf), rtol=1e-5, atol=1e-5
        )
    return model


def _assert_top_level_loop_interface(model: onnx.ModelProto) -> None:
    graph = model.graph
    available = {value.name for value in graph.input}
    available.update(value.name for value in graph.initializer)

    for node in graph.node:
        if node.op_type == "Loop":
            unresolved = [name for name in node.input if name and name not in available]
            assert not unresolved, f"Loop has unresolved parent inputs: {unresolved}"

            body = next(
                attribute.g for attribute in node.attribute if attribute.name == "body"
            )
            assert len(body.input) == len(node.input)
            assert len(body.output) == len(node.output) + 1
        available.update(name for name in node.output if name)


@pytest.mark.parametrize("opset", [18, 21])
def test_vmapped_group_norm_uses_valid_opset_lowering(opset: int) -> None:
    norm = eqx.nn.GroupNorm(groups=2, channels=4)
    fn = jax.vmap(norm)
    x = jnp.arange(2 * 4 * 3 * 3, dtype=jnp.float32).reshape(2, 4, 3, 3)

    model = _assert_ort_parity(fn, [x], opset=opset)
    group_norm_nodes = [
        node for node in model.graph.node if node.op_type == "GroupNormalization"
    ]
    assert len(group_norm_nodes) == int(opset >= 21)


@pytest.mark.parametrize("opset", [18, 21])
@pytest.mark.parametrize(
    "perturbation",
    [0.0, 1e-4],
    ids=["constant", "near_constant"],
)
def test_vmapped_group_norm_zero_variance_policy(
    opset: int, perturbation: float
) -> None:
    norm = eqx.nn.GroupNorm(groups=16, channels=16)
    fn = jax.vmap(norm)

    values = jax.random.normal(
        jax.random.PRNGKey(7),
        (1, 16, 1, 1),
        dtype=jnp.float32,
    )
    x = jnp.broadcast_to(values, (1, 16, 32, 32))
    if perturbation:
        noise = jax.random.normal(
            jax.random.PRNGKey(8),
            x.shape,
            dtype=x.dtype,
        )
        x = x + perturbation * noise

    _, actual = _convert_and_run(fn, [x], opset=opset)
    expected = np.asarray(fn(x))
    assert np.isfinite(expected).all()
    assert np.isfinite(actual[0]).all()
    np.testing.assert_allclose(
        actual[0],
        expected,
        rtol=1e-4,
        atol=1e-2,
    )


def test_adaptive_average_pool_supports_nondivisible_target() -> None:
    pool = eqx.nn.AdaptiveAvgPool1d(9)
    fn = jax.vmap(pool)
    x = jnp.arange(2 * 4 * 64, dtype=jnp.float32).reshape(2, 4, 64)

    _assert_ort_parity(fn, [x])


def test_adaptive_average_pool_2d_supports_nondivisible_target_unbatched() -> None:
    pool = eqx.nn.AdaptiveAvgPool2d((3, 4))
    x = jnp.arange(3 * 7 * 11, dtype=jnp.float32).reshape(3, 7, 11)

    _assert_ort_parity(pool, [x])


def test_adaptive_max_pool_2d_supports_nondivisible_target_batched() -> None:
    pool = eqx.nn.AdaptiveMaxPool2d((3, 4))
    fn = jax.vmap(pool)
    x = jnp.arange(2 * 3 * 7 * 11, dtype=jnp.float32).reshape(2, 3, 7, 11)

    _assert_ort_parity(fn, [x])


def test_linear_accepts_key_inside_sequential() -> None:
    key = jax.random.PRNGKey(0)
    sequential = eqx.nn.Sequential([eqx.nn.Linear(4, 3, key=key)])
    fn = jax.vmap(lambda x: sequential(x, key=key))
    x = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)

    _assert_ort_parity(fn, [x])


def test_reshape_folding_preserves_required_patch_reshape() -> None:
    conv = eqx.nn.Conv2d(
        3,
        8,
        kernel_size=8,
        stride=8,
        dtype=jnp.float32,
        key=jax.random.PRNGKey(0),
    )
    identity = eqx.nn.Identity()

    def one(image: jax.Array) -> jax.Array:
        features = conv(image)
        channels, height, width = features.shape
        tokens = jnp.transpose(features, (1, 2, 0)).reshape(height * width, channels)
        tokens = jax.vmap(identity)(tokens)
        features = jnp.transpose(tokens.reshape(height, width, channels), (2, 0, 1))
        tokens = jnp.transpose(features, (1, 2, 0)).reshape(-1, channels)
        prefix = jnp.zeros((5, channels), dtype=tokens.dtype)
        return jnp.concatenate([prefix, tokens], axis=0)

    fn = jax.vmap(one)
    x = jnp.arange(3 * 32 * 32, dtype=jnp.float32).reshape(1, 3, 32, 32) / 100
    _assert_ort_parity(fn, [x])


@pytest.mark.parametrize(
    ("fn", "x"),
    [
        (
            eqx.internal.unvmap_any,
            jnp.asarray([[False, True], [False, False]]),
        ),
        (
            eqx.internal.unvmap_all,
            jnp.asarray([[True, True], [True, False]]),
        ),
        (
            eqx.internal.unvmap_max,
            jnp.asarray([[1, 7], [3, 2]], dtype=jnp.int32),
        ),
        (
            lambda value: eqx.internal.nonbatchable(value),
            jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        ),
    ],
    ids=["unvmap_any", "unvmap_all", "unvmap_max", "nonbatchable"],
)
def test_equinox_internal_control_primitives(
    fn: Callable[[jax.Array], jax.Array], x: jax.Array
) -> None:
    _assert_ort_parity(fn, [x])


def test_select_if_vmap_unbatched_is_identity() -> None:
    def fn(pred: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
        return eqx.internal.select_if_vmap_p.bind(pred, x, y)

    pred = jnp.asarray(False)
    x = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    y = jnp.asarray([10.0, 20.0], dtype=jnp.float32)
    model, actual = _convert_and_run(fn, [pred, x, y])

    node_types = [node.op_type for node in model.graph.node]
    assert "Identity" in node_types
    assert "Where" not in node_types
    np.testing.assert_array_equal(actual[0], np.asarray(x))


def test_equinox_control_primitives_inside_loop_condition() -> None:
    def fn(x: jax.Array) -> jax.Array:
        def cond(state: tuple[jax.Array, jax.Array]) -> jax.Array:
            value, count = state
            predicate = eqx.internal.unvmap_any(value < 3)
            predicate = eqx.internal.nonbatchable(
                predicate, allow_constant_across_batch=True
            )
            return predicate & (count < 5)

        def body(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            return state[0] + 1, state[1] + 1

        return jax.lax.while_loop(cond, body, (x, jnp.int32(0)))[0]

    x = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    _assert_ort_parity(fn, [x])


def test_vmapped_equinox_internal_while_loop() -> None:
    def loop(x: jax.Array) -> jax.Array:
        return eqx.internal.while_loop(
            lambda value: jnp.mean(value) < 3.0,
            lambda value: value + 1.0,
            x,
            kind="lax",
        )

    fn = jax.vmap(loop)
    x = jnp.asarray([[1.0, 1.0], [2.5, 2.5]], dtype=jnp.float32)

    _assert_ort_parity(fn, [x])


def test_unbatched_equinox_internal_while_loop() -> None:
    def fn(x: jax.Array) -> jax.Array:
        return eqx.internal.while_loop(
            lambda value: jnp.mean(value) < 3.0,
            lambda value: value + 1.0,
            x,
            kind="lax",
        )

    x = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    model, actual = _convert_and_run(fn, [x])

    np.testing.assert_allclose(
        actual[0],
        np.asarray([3.0, 4.0], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    _assert_top_level_loop_interface(model)


def test_equinox_conv_parameter_remains_available_to_loop() -> None:
    conv = eqx.nn.Conv2d(
        1,
        1,
        kernel_size=1,
        dtype=jnp.float32,
        key=jax.random.PRNGKey(0),
    )

    def fn(x: jax.Array) -> jax.Array:
        initial = conv(x)
        return eqx.internal.while_loop(
            lambda _: jnp.asarray(True),
            lambda value: conv(value),
            initial,
            max_steps=1,
            kind="lax",
        )

    x = jnp.ones((1, 2, 2), dtype=jnp.float32)
    model = _assert_ort_parity(fn, [x])
    _assert_top_level_loop_interface(model)
