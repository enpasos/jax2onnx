# tests/extra_tests/test_conv_transpose_geometry.py

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import equinox as eqx
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx


@pytest.fixture(autouse=True)
def _float32_defaults() -> Iterator[None]:
    # Some suite helpers enable x64 at import time; Equinox would then create
    # float64 weights. These exports mirror the report's float32 configuration.
    original_x64 = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", original_x64)


def _random(shape: tuple[int, ...], seed: int = 0) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=jnp.float32)


def _assert_ort_parity(
    fn: Callable[..., jax.Array],
    inputs: Sequence[jax.Array],
    *,
    opset: int = 20,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> onnx.ModelProto:
    model = to_onnx(fn, list(inputs), opset=opset)
    # full_check runs strict shape inference, which rejects a ConvTranspose whose
    # attributes disagree with the output shape stamped from JAX.
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds = {
        graph_input.name: np.asarray(value)
        for graph_input, value in zip(session.get_inputs(), inputs, strict=True)
    }
    (actual,) = session.run(None, feeds)
    expected = np.asarray(fn(*inputs))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return model


def _op_types(model: onnx.ModelProto) -> list[str]:
    return [node.op_type for node in model.graph.node]


def _conv_transpose_attrs(model: onnx.ModelProto) -> dict[str, Any]:
    (node,) = [node for node in model.graph.node if node.op_type == "ConvTranspose"]
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def test_equinox_conv_transpose_preserves_stride() -> None:
    layer = eqx.nn.ConvTranspose2d(4, 4, 2, stride=2, key=jax.random.PRNGKey(3))
    for x in (jnp.ones((4, 2, 3), dtype=jnp.float32), _random((4, 2, 3))):
        model = _assert_ort_parity(lambda image: layer(image), [x])
        assert np.asarray(layer(x)).shape == (4, 4, 6)
        assert _conv_transpose_attrs(model)["strides"] == [2, 2]


_EQX_CONV_TRANSPOSE_CASES: dict[str, dict[str, Any]] = {
    "stride1": {"kernel_size": 3},
    "stride1_padded": {"kernel_size": 3, "padding": 1},
    "asymmetric_stride": {"kernel_size": (3, 2), "stride": (2, 3)},
    "asymmetric_padding": {
        "kernel_size": 3,
        "stride": 2,
        "padding": ((1, 0), (0, 2)),
    },
    "output_padding": {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
    },
    "output_padding_beyond_padding": {
        "kernel_size": 3,
        "stride": 3,
        "output_padding": 2,
    },
    "same": {"kernel_size": 3, "stride": 2, "padding": "SAME"},
    "same_lower": {"kernel_size": 4, "stride": 3, "padding": "SAME_LOWER"},
    "dilation": {"kernel_size": 3, "stride": 2, "padding": 1, "dilation": 2},
    "groups": {
        "in_channels": 4,
        "out_channels": 6,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "groups": 2,
    },
    "depthwise": {
        "in_channels": 4,
        "out_channels": 4,
        "kernel_size": 2,
        "stride": 2,
        "groups": 4,
    },
    "one_dimensional": {
        "num_spatial_dims": 1,
        "kernel_size": 3,
        "stride": 3,
        "padding": 1,
    },
}


@pytest.mark.parametrize(
    "config",
    list(_EQX_CONV_TRANSPOSE_CASES.values()),
    ids=list(_EQX_CONV_TRANSPOSE_CASES),
)
def test_equinox_conv_transpose_geometry(config: dict[str, Any]) -> None:
    kwargs: dict[str, Any] = {
        "num_spatial_dims": 2,
        "in_channels": 3,
        "out_channels": 5,
        **config,
    }
    layer = eqx.nn.ConvTranspose(**kwargs, key=jax.random.PRNGKey(1))
    spatial = (5, 7)[: kwargs["num_spatial_dims"]]
    x = _random((kwargs["in_channels"], *spatial))

    model = _assert_ort_parity(lambda image: layer(image), [x])

    if any(s != 1 for s in layer.stride):
        assert _conv_transpose_attrs(model)["strides"] == list(layer.stride)


def test_vmapped_equinox_conv_transpose() -> None:
    layer = eqx.nn.ConvTranspose2d(
        3, 5, 3, stride=2, padding=1, output_padding=1, key=jax.random.PRNGKey(2)
    )

    model = _assert_ort_parity(jax.vmap(layer), [_random((2, 3, 4, 5))])

    assert _conv_transpose_attrs(model)["strides"] == [2, 2]


@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("transpose_kernel", [False, True])
def test_nnx_conv_transpose(padding: str, transpose_kernel: bool) -> None:
    layer = nnx.ConvTranspose(
        3,
        5,
        (3, 2),
        strides=(2, 3),
        padding=padding,
        transpose_kernel=transpose_kernel,
        rngs=nnx.Rngs(0),
    )

    model = _assert_ort_parity(lambda x: layer(x), [_random((1, 4, 5, 3))])

    assert _conv_transpose_attrs(model)["strides"] == [2, 3]


def test_nnx_grouped_input_dilated_conv() -> None:
    layer = nnx.Conv(
        4,
        6,
        (3, 3),
        input_dilation=2,
        feature_group_count=2,
        padding=((1, 1), (1, 1)),
        rngs=nnx.Rngs(0),
    )

    model = _assert_ort_parity(lambda x: layer(x), [_random((1, 3, 5, 4))])

    attrs = _conv_transpose_attrs(model)
    assert attrs["strides"] == [2, 2]
    assert attrs["group"] == 2


def test_lax_conv_transpose_nhwc_with_constant_kernel() -> None:
    kernel = _random((3, 2, 4, 6), seed=1)

    def fn(x: jax.Array) -> jax.Array:
        return jax.lax.conv_transpose(
            x,
            kernel,
            strides=(2, 3),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    model = _assert_ort_parity(fn, [_random((2, 3, 5, 4))])

    assert _conv_transpose_attrs(model)["strides"] == [2, 3]


_KERNEL: jax.Array = _random((4, 2, 3, 3), seed=1)


@pytest.mark.parametrize(
    ("fn", "x", "message"),
    [
        (
            lambda x: jax.lax.conv_general_dilated(
                x, _KERNEL, (1, 1), "VALID", batch_group_count=2
            ),
            _random((4, 2, 5, 5)),
            "batch_group_count",
        ),
        (
            lambda x: jax.lax.conv_general_dilated(
                x, _KERNEL, (2, 2), ((1, 1), (1, 1)), lhs_dilation=(2, 2)
            ),
            _random((1, 2, 5, 5)),
            "window_strides",
        ),
        (
            lambda x: jax.lax.conv_general_dilated(
                x,
                _KERNEL.astype(jnp.complex64),
                (1, 1),
                ((1, 1), (1, 1)),
                lhs_dilation=(2, 2),
            ),
            _random((1, 2, 5, 5)).astype(jnp.complex64),
            "Complex transposed convolution",
        ),
    ],
    ids=["batch_group_count", "strided_input_dilation", "complex_transpose"],
)
def test_unsupported_conv_configurations_fail_explicitly(
    fn: Callable[[jax.Array], jax.Array], x: jax.Array, message: str
) -> None:
    with pytest.raises(NotImplementedError, match=message):
        to_onnx(fn, [x], opset=20)


_GELU_FUNCTIONS: dict[str, Callable[..., jax.Array]] = {
    # Resolve the attribute at call time so the export-time patch applies.
    "jax_nn": lambda x, approximate: jax.nn.gelu(x, approximate=approximate),
    "nnx": lambda x, approximate: nnx.gelu(x, approximate=approximate),
}


@pytest.mark.parametrize("opset", [18, 19, 20, 23])
@pytest.mark.parametrize("approximate", [False, True], ids=["exact", "tanh"])
@pytest.mark.parametrize("api", list(_GELU_FUNCTIONS))
def test_gelu_export_respects_opset(api: str, approximate: bool, opset: int) -> None:
    gelu = _GELU_FUNCTIONS[api]
    x = jnp.linspace(-6.0, 6.0, 30, dtype=jnp.float32).reshape(5, 6)

    model = _assert_ort_parity(
        lambda value: gelu(value, approximate), [x], opset=opset, atol=1e-6, rtol=1e-5
    )

    assert {imp.domain: imp.version for imp in model.opset_import}[""] == opset
    assert ("Gelu" in _op_types(model)) == (opset >= 20)
