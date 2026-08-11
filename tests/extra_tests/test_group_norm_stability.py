# tests/extra_tests/test_group_norm_stability.py

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
    normalization_mode: str = "auto",
) -> tuple[onnx.ModelProto, np.ndarray]:
    model = to_onnx(
        fn,
        [x],
        opset=opset,
        normalization_mode=normalization_mode,
    )
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
    assert not _uses_anchor_centering(model)
    assert sum(node.op_type == "Equal" for node in model.graph.node) == 2
    assert sum(node.op_type == "Where" for node in model.graph.node) == 1
    np.testing.assert_array_equal(expected, np.zeros_like(expected))
    np.testing.assert_array_equal(actual, np.zeros_like(actual))


@pytest.mark.parametrize("opset", [17, 18, 21, 23])
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


@pytest.mark.parametrize("opset", [17, 18, 21, 23])
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


@pytest.mark.parametrize("opset", [21, 23])
def test_nnx_fast_variance_semantic_mode_uses_native_group_norm(opset: int) -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        use_fast_variance=True,
        rngs=nnx.Rngs(0),
    )
    x = jnp.arange(1 * 4 * 5 * 4, dtype=jnp.float32).reshape(1, 4, 5, 4) / 13

    model, actual = _convert_and_run(
        norm,
        x,
        opset=opset,
        normalization_mode="semantic",
    )
    group_node = next(
        node for node in model.graph.node if node.op_type == "GroupNormalization"
    )
    assert sum(node.op_type == "GroupNormalization" for node in model.graph.node) == 1
    attrs = {
        attr.name: onnx.helper.get_attribute_value(attr)
        for attr in group_node.attribute
    }
    assert attrs["num_groups"] == 2
    assert attrs["stash_type"] == onnx.TensorProto.FLOAT
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    assert tuple(initializers[group_node.input[1]].dims) == (4,)
    assert tuple(initializers[group_node.input[2]].dims) == (4,)
    assert not _uses_anchor_centering(model)
    np.testing.assert_allclose(actual, np.asarray(norm(x)), rtol=1e-5, atol=1e-5)


def test_rank2_native_group_norm_uses_tensorrt_compatible_rank4_input() -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        use_fast_variance=True,
        rngs=nnx.Rngs(0),
    )
    x = jnp.arange(8, dtype=jnp.float32).reshape(2, 4) / 7

    model, actual = _convert_and_run(
        norm,
        x,
        opset=21,
        normalization_mode="semantic",
    )
    group_node = next(
        node for node in model.graph.node if node.op_type == "GroupNormalization"
    )
    unsqueeze = next(
        node
        for node in model.graph.node
        if group_node.input[0] in node.output and node.op_type == "Unsqueeze"
    )
    squeeze = next(
        node
        for node in model.graph.node
        if group_node.output[0] in node.input and node.op_type == "Squeeze"
    )
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    axes = onnx.numpy_helper.to_array(initializers[unsqueeze.input[1]])
    np.testing.assert_array_equal(axes, np.asarray([2, 3], dtype=np.int64))
    squeeze_axes = onnx.numpy_helper.to_array(initializers[squeeze.input[1]])
    np.testing.assert_array_equal(squeeze_axes, axes)

    value_info = {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    input_dims = value_info[model.graph.input[0].name].type.tensor_type.shape.dim
    output_dims = value_info[model.graph.output[0].name].type.tensor_type.shape.dim
    assert tuple(dim.dim_value for dim in input_dims) == (2, 4)
    assert tuple(dim.dim_value for dim in output_dims) == (2, 4)
    dims = value_info[group_node.input[0]].type.tensor_type.shape.dim
    assert tuple(dim.dim_value for dim in dims) == (2, 4, 1, 1)
    np.testing.assert_allclose(actual, np.asarray(norm(x)), rtol=1e-5, atol=1e-5)


def test_rank3_native_group_norm_uses_tensorrt_compatible_rank4_input() -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        use_fast_variance=True,
        rngs=nnx.Rngs(0),
    )
    x = (jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) - 12) / 5

    model, actual = _convert_and_run(
        norm,
        x,
        opset=21,
        normalization_mode="semantic",
    )
    group_node = next(
        node for node in model.graph.node if node.op_type == "GroupNormalization"
    )
    unsqueeze = next(
        node
        for node in model.graph.node
        if group_node.input[0] in node.output and node.op_type == "Unsqueeze"
    )
    squeeze = next(
        node
        for node in model.graph.node
        if group_node.output[0] in node.input and node.op_type == "Squeeze"
    )
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    axes = onnx.numpy_helper.to_array(initializers[unsqueeze.input[1]])
    np.testing.assert_array_equal(axes, np.asarray([3], dtype=np.int64))
    squeeze_axes = onnx.numpy_helper.to_array(initializers[squeeze.input[1]])
    np.testing.assert_array_equal(squeeze_axes, axes)

    value_info = {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    dims = value_info[group_node.input[0]].type.tensor_type.shape.dim
    assert tuple(dim.dim_value for dim in dims) == (2, 4, 3, 1)
    np.testing.assert_allclose(actual, np.asarray(norm(x)), rtol=1e-5, atol=1e-5)


def test_rank5_native_group_norm_flattens_only_spatial_dimensions() -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        use_fast_variance=True,
        rngs=nnx.Rngs(0),
    )
    x = (
        jnp.arange(2 * 2 * 3 * 5 * 4, dtype=jnp.float32).reshape(2, 2, 3, 5, 4) - 120
    ) / 17

    model, actual = _convert_and_run(
        norm,
        x,
        opset=21,
        normalization_mode="semantic",
    )
    group_node = next(
        node for node in model.graph.node if node.op_type == "GroupNormalization"
    )
    producer = next(
        node for node in model.graph.node if group_node.input[0] in node.output
    )
    consumer = next(
        node for node in model.graph.node if group_node.output[0] in node.input
    )
    assert producer.op_type == "Reshape"
    assert consumer.op_type == "Reshape"
    assert sum(node.op_type == "ReduceProd" for node in model.graph.node) == 1

    value_info = {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    core_dims = value_info[group_node.input[0]].type.tensor_type.shape.dim
    assert tuple(dim.dim_value for dim in core_dims) == (2, 4, 30, 1)
    output_dims = value_info[model.graph.output[0].name].type.tensor_type.shape.dim
    assert tuple(dim.dim_value for dim in output_dims) == (2, 2, 3, 5, 4)
    np.testing.assert_allclose(actual, np.asarray(norm(x)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(0, 3, 4), (2, 0, 4)])
def test_semantic_group_norm_uses_explicit_path_for_static_empty_shapes(
    shape: tuple[int, ...],
) -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        use_fast_variance=True,
        rngs=nnx.Rngs(0),
    )
    x = jnp.empty(shape, dtype=jnp.float32)

    model, actual = _convert_and_run(
        norm,
        x,
        opset=21,
        normalization_mode="semantic",
    )
    assert not any(node.op_type == "GroupNormalization" for node in model.graph.node)
    assert sum(node.op_type == "ReduceMean" for node in model.graph.node) == 2
    assert actual.shape == shape
    np.testing.assert_array_equal(actual, np.asarray(norm(x)))


@pytest.mark.parametrize(
    ("opset", "normalization_mode"),
    [
        (17, "auto"),
        (18, "auto"),
        (18, "semantic"),
        (21, "auto"),
        (23, "auto"),
        (23, "decomposed"),
    ],
)
def test_nnx_fast_variance_uses_explicit_path_when_requested_or_required(
    opset: int,
    normalization_mode: str,
) -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        use_fast_variance=True,
        rngs=nnx.Rngs(0),
    )
    x = jnp.arange(1 * 4 * 5 * 4, dtype=jnp.float32).reshape(1, 4, 5, 4) / 13

    model, actual = _convert_and_run(
        norm,
        x,
        opset=opset,
        normalization_mode=normalization_mode,
    )
    assert not any(
        node.op_type in {"GroupNormalization", "InstanceNormalization"}
        for node in model.graph.node
    )
    assert sum(node.op_type == "ReduceMean" for node in model.graph.node) == 2
    assert not _uses_anchor_centering(model)
    np.testing.assert_allclose(actual, np.asarray(norm(x)), rtol=1e-5, atol=1e-5)


def test_slow_variance_semantic_mode_remains_framework_explicit() -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        use_fast_variance=False,
        rngs=nnx.Rngs(0),
    )
    x = jnp.arange(1 * 4 * 5 * 4, dtype=jnp.float32).reshape(1, 4, 5, 4) / 13

    model, actual = _convert_and_run(
        norm,
        x,
        opset=23,
        normalization_mode="semantic",
    )
    assert not any(node.op_type == "GroupNormalization" for node in model.graph.node)
    assert sum(node.op_type == "Where" for node in model.graph.node) == 1
    np.testing.assert_allclose(actual, np.asarray(norm(x)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("opset", [17, 18, 21, 23])
def test_slow_group_norm_preserves_high_offset_framework_rounding(opset: int) -> None:
    norm = nnx.GroupNorm(
        num_features=2,
        num_groups=1,
        use_fast_variance=False,
        use_scale=False,
        use_bias=False,
        rngs=nnx.Rngs(0),
    )
    x = jnp.asarray([[[[1_000_000.0, 1_000_000.0625]]]], dtype=jnp.float32)

    model, actual = _convert_and_run(norm, x, opset=opset)
    assert not _uses_anchor_centering(model)
    np.testing.assert_allclose(actual, np.asarray(norm(x)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("opset", [17, 18, 21, 23])
def test_slow_group_norm_does_not_overflow_during_centering(opset: int) -> None:
    norm = nnx.GroupNorm(
        num_features=2,
        num_groups=1,
        use_fast_variance=False,
        use_scale=False,
        use_bias=False,
        rngs=nnx.Rngs(0),
    )
    x = jnp.asarray([[[[3e38, -3e38]]]], dtype=jnp.float32)

    model, actual = _convert_and_run(norm, x, opset=opset)
    expected = np.asarray(norm(x))
    assert not _uses_anchor_centering(model)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_slow_group_norm_does_not_zero_nonfinite_constant_groups(value: float) -> None:
    norm = nnx.GroupNorm(
        num_features=2,
        num_groups=1,
        use_fast_variance=False,
        use_scale=False,
        use_bias=False,
        rngs=nnx.Rngs(0),
    )
    x = jnp.asarray([[[[value, value]]]], dtype=jnp.float32)

    _, actual = _convert_and_run(norm, x, opset=23)
    expected = np.asarray(norm(x))
    assert np.isnan(expected).all()
    assert np.isnan(actual).all()


@pytest.mark.parametrize("opset", [17, 18, 21, 23])
def test_slow_group_norm_supports_symbolic_empty_spatial_dimensions(
    opset: int,
) -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        use_fast_variance=False,
        rngs=nnx.Rngs(0),
    )
    symbolic_input = jax.ShapeDtypeStruct(
        jax.export.symbolic_shape("B,H,W,4"),
        jnp.float32,
    )
    model = to_onnx(norm, [symbolic_input], opset=opset)
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    for shape in ((2, 0, 5, 4), (2, 3, 0, 4), (0, 3, 5, 4), (2, 3, 5, 4)):
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        (actual,) = session.run(None, {input_name: values})
        assert actual.shape == shape
        assert np.isfinite(actual).all()
