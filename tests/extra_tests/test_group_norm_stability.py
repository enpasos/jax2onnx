from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx


_CHANNEL_VALUES: np.ndarray = np.asarray(
    [0.09006345, 0.00062171, -0.14996266, -0.04716587],
    dtype=np.float32,
)


def _constant_nchw(*, channels_per_group: int = 1) -> jax.Array:
    channel_values: np.ndarray = np.repeat(_CHANNEL_VALUES, channels_per_group)
    values = np.broadcast_to(
        channel_values[:, None, None], (len(channel_values), 32, 32)
    )
    return jnp.asarray(values.copy())


def _convert_and_run(
    fn: Callable[[jax.Array], object],
    x: jax.Array,
    *,
    opset: int,
) -> tuple[onnx.ModelProto, np.ndarray]:
    model = to_onnx(fn, [x], opset=opset)
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(
        None,
        {session.get_inputs()[0].name: np.asarray(x)},
    )
    return model, actual


def _uses_anchor_centering(model: onnx.ModelProto) -> bool:
    producers = {
        output: node for node in model.graph.node for output in node.output if output
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    for shifted_node in model.graph.node:
        if shifted_node.op_type != "Sub" or len(shifted_node.input) < 2:
            continue
        anchor_node = producers.get(shifted_node.input[1])
        if anchor_node is None or anchor_node.op_type != "Slice":
            continue
        shifted = shifted_node.output[0]
        for mean_node in consumers.get(shifted, []):
            if mean_node.op_type != "ReduceMean":
                continue
            mean = mean_node.output[0]
            if any(
                node.op_type == "Sub"
                and len(node.input) >= 2
                and node.input[0] == shifted
                and node.input[1] == mean
                for node in consumers.get(shifted, [])
            ):
                return True
    return False


def _assert_explicit_stable_result(
    model: onnx.ModelProto,
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    assert not any(
        node.op_type in {"GroupNormalization", "InstanceNormalization"}
        for node in model.graph.node
    )
    assert _uses_anchor_centering(model)
    np.testing.assert_array_equal(expected, np.zeros_like(expected))
    np.testing.assert_array_equal(actual, np.zeros_like(actual))


@pytest.mark.parametrize("opset", [18, 21, 23])
@pytest.mark.parametrize("channelwise_affine", [False, True])
def test_equinox_constant_group_norm_is_exactly_centered(
    opset: int,
    channelwise_affine: bool,
) -> None:
    channels_per_group = 1 if channelwise_affine else 2
    channels = 4 * channels_per_group
    norm = eqx.nn.GroupNorm(
        groups=4,
        channels=channels if channelwise_affine else None,
        channelwise_affine=channelwise_affine,
    )
    x = _constant_nchw(channels_per_group=channels_per_group)

    model, actual = _convert_and_run(norm, x, opset=opset)
    _assert_explicit_stable_result(model, actual, np.asarray(norm(x)))


@pytest.mark.parametrize("opset", [18, 21, 23])
@pytest.mark.parametrize("framework", ["nnx", "linen"])
def test_flax_slow_group_norm_is_exactly_centered(
    opset: int,
    framework: str,
) -> None:
    x = jnp.moveaxis(_constant_nchw(), 0, -1)[None, ...]
    if framework == "nnx":
        norm = nnx.GroupNorm(
            num_features=4,
            num_groups=4,
            use_fast_variance=False,
            rngs=nnx.Rngs(0),
        )
        fn = norm
    else:
        module = nn.GroupNorm(
            num_groups=4,
            use_fast_variance=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )
        variables = module.init(jax.random.PRNGKey(0), x)

        def fn(value: jax.Array) -> object:
            return module.apply(variables, value)

    model, actual = _convert_and_run(fn, x, opset=opset)
    _assert_explicit_stable_result(model, actual, np.asarray(fn(x)))


def test_vmapped_equinox_group_norm_keeps_batches_independent() -> None:
    norm = eqx.nn.GroupNorm(groups=4, channels=4)
    fn = jax.vmap(norm)
    x = jnp.stack([_constant_nchw(), _constant_nchw() + jnp.float32(0.25)])

    model, actual = _convert_and_run(fn, x, opset=23)
    _assert_explicit_stable_result(model, actual, np.asarray(fn(x)))


@pytest.mark.parametrize("opset", [18, 23])
@pytest.mark.parametrize("distribution", ["near_constant", "random"])
def test_equinox_stable_group_norm_retains_general_parity(
    opset: int,
    distribution: str,
) -> None:
    rng = np.random.default_rng(12)
    if distribution == "near_constant":
        x_np = np.asarray(_constant_nchw()) + rng.normal(
            scale=1e-5, size=(4, 32, 32)
        ).astype(np.float32)
    else:
        x_np = rng.normal(size=(4, 9, 7)).astype(np.float32)
    x = jnp.asarray(x_np)
    norm = eqx.nn.GroupNorm(groups=2, channels=4)

    _, actual = _convert_and_run(norm, x, opset=opset)
    np.testing.assert_allclose(actual, np.asarray(norm(x)), rtol=1e-5, atol=1e-5)


def test_nnx_fast_variance_keeps_direct_mean_path() -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        use_fast_variance=True,
        rngs=nnx.Rngs(0),
    )
    x = jnp.arange(1 * 4 * 5 * 4, dtype=jnp.float32).reshape(1, 4, 5, 4) / 13

    model, actual = _convert_and_run(norm, x, opset=23)
    assert not _uses_anchor_centering(model)
    np.testing.assert_allclose(actual, np.asarray(norm(x)), rtol=1e-5, atol=1e-5)
