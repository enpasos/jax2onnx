# tests/extra_tests/test_equinox_interop_regressions.py

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

import onnx
from jax2onnx import to_onnx
from jax2onnx.plugins.equinox.eqx.nn.sequential import (
    SequentialPlugin,
    _layer_ignores_key,
)


def _convert_and_run(
    fn: Callable[..., object],
    inputs: Sequence[jax.Array],
    *,
    opset: int = 18,
    enable_double_precision: bool = False,
) -> tuple[onnx.ModelProto, list[np.ndarray]]:
    model = to_onnx(
        fn,
        list(inputs),
        opset=opset,
        enable_double_precision=enable_double_precision,
    )
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
    enable_double_precision: bool = False,
) -> onnx.ModelProto:
    model, actual = _convert_and_run(
        fn,
        inputs,
        opset=opset,
        enable_double_precision=enable_double_precision,
    )
    expected = jax.tree_util.tree_leaves(fn(*inputs))
    assert len(actual) == len(expected)
    for actual_leaf, expected_leaf in zip(actual, expected):
        np.testing.assert_allclose(
            actual_leaf, np.asarray(expected_leaf), rtol=1e-5, atol=1e-5
        )
    return model


def _linen_apply(module: nn.Module, sample: jax.Array) -> Callable[[jax.Array], object]:
    variables = module.init(jax.random.PRNGKey(0), sample)
    return lambda value: module.apply(variables, value)


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


@pytest.mark.parametrize("opset", [17, 18, 21])
def test_vmapped_group_norm_uses_valid_opset_lowering(opset: int) -> None:
    norm = eqx.nn.GroupNorm(groups=2, channels=4)
    fn = jax.vmap(norm)
    x = jnp.arange(2 * 4 * 3 * 3, dtype=jnp.float32).reshape(2, 4, 3, 3)

    model = _assert_ort_parity(fn, [x], opset=opset)
    group_norm_nodes = [
        node for node in model.graph.node if node.op_type == "GroupNormalization"
    ]
    assert not group_norm_nodes


@pytest.mark.parametrize("opset", [17, 18, 21])
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


@pytest.mark.parametrize("opset", [17, 18, 21, 23])
def test_nested_vmapped_group_norm_preserves_each_batch_axis(opset: int) -> None:
    norm = eqx.nn.GroupNorm(groups=2, channels=4)
    fn = jax.vmap(jax.vmap(norm))
    x = jnp.arange(2 * 3 * 4 * 2 * 2, dtype=jnp.float32).reshape(2, 3, 4, 2, 2)

    _assert_ort_parity(fn, [x], opset=opset)


@pytest.mark.parametrize("opset", [17, 18, 21, 23])
def test_nested_vmapped_group_norm_supports_symbolic_batch_axes(opset: int) -> None:
    norm = eqx.nn.GroupNorm(groups=2, channels=4)
    fn = jax.vmap(jax.vmap(norm))
    symbolic_input = jax.ShapeDtypeStruct(
        jax.export.symbolic_shape("B1,B2,4,2,2"),
        jnp.float32,
    )
    model = to_onnx(fn, [symbolic_input], opset=opset)
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    for batch_shape in ((2, 3), (1, 4), (0, 3)):
        shape = (*batch_shape, 4, 2, 2)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 10
        (actual,) = session.run(None, {input_name: values})
        expected = np.asarray(fn(jnp.asarray(values)))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("opset", [18, 21])
def test_equinox_group_norm_float16_statistics(opset: int) -> None:
    norm = eqx.nn.GroupNorm(groups=2, channels=4)
    fn = jax.vmap(norm)
    x = jnp.full((2, 4, 3, 3), 1000.0, dtype=jnp.float16)
    x = x.at[0, 0, 0, 0].set(jnp.float16(1000.5))

    _, actual = _convert_and_run(fn, [x], opset=opset)
    expected = np.asarray(fn(x))
    assert actual[0].dtype == expected.dtype
    np.testing.assert_allclose(actual[0], expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("opset", [18, 21, 23])
@pytest.mark.parametrize("base", [1e3, 1e4])
def test_equinox_group_norm_high_offset_float32_statistics(
    opset: int, base: float
) -> None:
    norm = eqx.nn.GroupNorm(groups=2, channels=4)
    fn = jax.vmap(norm)
    x = base + jnp.arange(2 * 4 * 2 * 2, dtype=jnp.float32).reshape(2, 4, 2, 2) / 4

    _, actual = _convert_and_run(fn, [x], opset=opset)
    expected = np.asarray(fn(x))
    assert np.isfinite(actual[0]).all()
    np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("opset", [18, 21])
@pytest.mark.parametrize("use_fast_variance", [False, True])
@pytest.mark.parametrize(
    ("module_dtype", "param_dtype"),
    [(jnp.float16, jnp.float16), (None, jnp.float32)],
    ids=["explicit_f16", "inferred_f32_result"],
)
def test_nnx_group_norm_float16_statistics(
    opset: int,
    use_fast_variance: bool,
    module_dtype: object,
    param_dtype: object,
) -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        dtype=module_dtype,
        param_dtype=param_dtype,
        use_fast_variance=use_fast_variance,
        rngs=nnx.Rngs(0),
    )
    x = jnp.full((2, 3, 3, 4), 1000.0, dtype=jnp.float16)
    x = x.at[0, 0, 0, 0].set(jnp.float16(1000.5))

    model, actual = _convert_and_run(norm, [x], opset=opset)
    expected = np.asarray(norm(x))
    assert actual[0].dtype == expected.dtype
    np.testing.assert_allclose(actual[0], expected, rtol=2e-3, atol=5e-2)
    group_norm_count = sum(
        node.op_type == "GroupNormalization" for node in model.graph.node
    )
    assert group_norm_count == 0


@pytest.mark.parametrize("opset", [18, 21])
@pytest.mark.parametrize("use_fast_variance", [False, True])
def test_nnx_group_norm_explicit_float16_quantizes_float32_input(
    opset: int, use_fast_variance: bool
) -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        dtype=jnp.float16,
        param_dtype=jnp.float32,
        use_fast_variance=use_fast_variance,
        rngs=nnx.Rngs(0),
    )
    # Keep the signal well-conditioned for the deliberately less stable
    # E[x**2] - E[x]**2 variance path while retaining a clear float16
    # quantization effect (float16 spacing is 0.03125 around 32).
    x = 32.0 + jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4) / 100

    _, actual = _convert_and_run(norm, [x], opset=opset)
    expected = np.asarray(norm(x))
    assert actual[0].dtype == expected.dtype == np.dtype(np.float16)
    np.testing.assert_allclose(actual[0], expected, rtol=2e-3, atol=5e-2)


@pytest.mark.parametrize("opset", [17, 18, 21, 23])
def test_nnx_group_norm_fast_variance_preserves_framework_reduction_layout(
    opset: int,
) -> None:
    norm = nnx.GroupNorm(
        num_features=4,
        num_groups=2,
        dtype=jnp.float16,
        param_dtype=jnp.float32,
        use_fast_variance=True,
        rngs=nnx.Rngs(0),
    )
    x = 1000.0 + jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4) / 10

    _, actual = _convert_and_run(norm, [x], opset=opset)
    expected = np.asarray(norm(x))
    assert actual[0].dtype == expected.dtype == np.dtype(np.float16)
    # Transposing NHWC to NCHW before the fast E[x**2] - E[x]**2 reduction
    # changes its floating-point reduction order and misses this tolerance.
    np.testing.assert_allclose(actual[0], expected, rtol=3e-3, atol=6e-2)


@pytest.mark.parametrize("kind", ["group", "instance"])
def test_linen_norm_preserves_module_result_dtype(kind: str) -> None:
    module: nn.Module
    if kind == "group":
        module = nn.GroupNorm(
            num_groups=2,
            dtype=jnp.float16,
            param_dtype=jnp.float32,
            use_fast_variance=True,
        )
    else:
        module = nn.InstanceNorm(
            dtype=jnp.float16,
            param_dtype=jnp.float32,
            use_fast_variance=True,
        )
    x = 32.0 + jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4) / 100
    fn = _linen_apply(module, x)

    model, actual = _convert_and_run(fn, [x], opset=18)
    expected = np.asarray(fn(x))
    assert actual[0].dtype == expected.dtype == np.dtype(np.float16)
    np.testing.assert_allclose(actual[0], expected, rtol=3e-3, atol=2e-2)
    assert all(
        node.op_type not in {"GroupNormalization", "InstanceNormalization"}
        for node in model.graph.node
    )


@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize("kind", ["group", "instance"])
@pytest.mark.parametrize("use_fast_variance", [False, True])
def test_linen_norm_float16_reductions_without_float32_promotion(
    opset: int, kind: str, use_fast_variance: bool
) -> None:
    module: nn.Module
    if kind == "group":
        module = nn.GroupNorm(
            num_groups=2,
            dtype=jnp.float16,
            param_dtype=jnp.float16,
            use_fast_variance=use_fast_variance,
            force_float32_reductions=False,
        )
    else:
        module = nn.InstanceNorm(
            dtype=jnp.float16,
            param_dtype=jnp.float16,
            use_fast_variance=use_fast_variance,
            force_float32_reductions=False,
        )
    x = (
        jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float16).reshape(2, 3, 3, 4)
        - jnp.float16(35.5)
    ) / 64
    fn = _linen_apply(module, x)

    model, actual = _convert_and_run(fn, [x], opset=opset)
    onnx.checker.check_model(model, full_check=True)
    expected = np.asarray(fn(x))
    assert actual[0].dtype == expected.dtype == np.dtype(np.float16)
    np.testing.assert_allclose(actual[0], expected, rtol=4e-3, atol=4e-3)
    assert all(
        node.op_type not in {"GroupNormalization", "InstanceNormalization"}
        for node in model.graph.node
    )


@pytest.mark.parametrize("kind", ["group", "instance"])
def test_linen_norm_rejects_unrepresentable_raw_input_staging(kind: str) -> None:
    original_x64 = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        module: nn.Module
        if kind == "group":
            module = nn.GroupNorm(
                num_groups=2,
                dtype=jnp.float16,
                param_dtype=jnp.float16,
                use_fast_variance=False,
                force_float32_reductions=False,
            )
        else:
            module = nn.InstanceNorm(
                dtype=jnp.float16,
                param_dtype=jnp.float16,
                use_fast_variance=False,
                force_float32_reductions=False,
            )
        x = 1.0 + jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4) / 64
        fn = _linen_apply(module, x)

        with pytest.raises(NotImplementedError, match="dtype staging"):
            to_onnx(fn, [x], opset=21, enable_double_precision=True)
    finally:
        jax.config.update("jax_enable_x64", original_x64)


@pytest.mark.parametrize("opset", [17, 18, 21])
@pytest.mark.parametrize("kind", ["group", "instance"])
def test_linen_norm_slow_variance_preserves_wider_affine_dtype(
    opset: int, kind: str
) -> None:
    original_x64 = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        module: nn.Module
        if kind == "group":
            module = nn.GroupNorm(
                num_groups=2,
                dtype=jnp.float32,
                param_dtype=jnp.float64,
                use_fast_variance=False,
                force_float32_reductions=True,
            )
        else:
            module = nn.InstanceNorm(
                dtype=jnp.float32,
                param_dtype=jnp.float64,
                use_fast_variance=False,
                force_float32_reductions=True,
            )
        x = 1.0 + jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4) / 64
        fn = _linen_apply(module, x)

        model = to_onnx(
            fn,
            [x],
            opset=opset,
            enable_double_precision=True,
        )
        onnx.checker.check_model(model, full_check=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        input_meta = session.get_inputs()[0]
        assert input_meta.type == "tensor(double)"
        actual = session.run(
            None,
            {input_meta.name: np.asarray(x, dtype=np.float64)},
        )
        expected = np.asarray(fn(x))
        np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=1e-6)
    finally:
        jax.config.update("jax_enable_x64", original_x64)


@pytest.mark.parametrize("kind", ["group", "instance"])
def test_linen_norm_slow_variance_preserves_float16_dtype(kind: str) -> None:
    dtype = jnp.float16
    module: nn.Module
    if kind == "group":
        module = nn.GroupNorm(
            num_groups=2,
            dtype=dtype,
            param_dtype=dtype,
            use_fast_variance=False,
            force_float32_reductions=True,
        )
    else:
        module = nn.InstanceNorm(
            dtype=dtype,
            param_dtype=dtype,
            use_fast_variance=False,
            force_float32_reductions=True,
        )
    x = jnp.arange(2 * 3 * 3 * 4, dtype=dtype).reshape(2, 3, 3, 4) / 64
    fn = _linen_apply(module, x)

    model, actual = _convert_and_run(fn, [x], opset=18)
    expected = np.asarray(fn(x))
    assert actual[0].dtype == expected.dtype
    np.testing.assert_allclose(
        np.asarray(actual[0], dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    assert not any(
        node.op_type in {"GroupNormalization", "InstanceNormalization"}
        for node in model.graph.node
    )


@pytest.mark.parametrize("kind", ["group", "instance"])
def test_linen_norm_slow_variance_keeps_narrow_affine_path_explicit(
    kind: str,
) -> None:
    module: nn.Module
    if kind == "group":
        module = nn.GroupNorm(
            num_groups=2,
            param_dtype=jnp.float16,
            use_fast_variance=False,
            force_float32_reductions=True,
        )
    else:
        module = nn.InstanceNorm(
            param_dtype=jnp.float16,
            use_fast_variance=False,
            force_float32_reductions=True,
        )
    x = 1.0 + jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4) / 64
    fn = _linen_apply(module, x)

    model, actual = _convert_and_run(fn, [x], opset=18)
    expected = np.asarray(fn(x))
    np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=1e-5)
    assert not any(
        node.op_type in {"GroupNormalization", "InstanceNormalization"}
        for node in model.graph.node
    )


@pytest.mark.parametrize("opset", [17, 18, 21])
@pytest.mark.parametrize("kind", ["group", "instance"])
def test_linen_norm_slow_variance_preserves_integer_input_support(
    opset: int, kind: str
) -> None:
    original_x64 = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        module: nn.Module
        if kind == "group":
            module = nn.GroupNorm(
                num_groups=2,
                dtype=jnp.float64,
                param_dtype=jnp.float64,
                use_fast_variance=False,
                force_float32_reductions=True,
            )
        else:
            module = nn.InstanceNorm(
                dtype=jnp.float64,
                param_dtype=jnp.float64,
                use_fast_variance=False,
                force_float32_reductions=True,
            )
        x = jnp.arange(2 * 3 * 3 * 4, dtype=jnp.int32).reshape(2, 3, 3, 4)
        fn = _linen_apply(module, x)

        model, actual = _convert_and_run(
            fn,
            [x],
            opset=opset,
            enable_double_precision=True,
        )
        expected = np.asarray(fn(x))
        assert actual[0].dtype == expected.dtype == np.dtype(np.float64)
        np.testing.assert_allclose(actual[0], expected, rtol=1e-12, atol=1e-12)
        assert not any(
            node.op_type in {"GroupNormalization", "InstanceNormalization"}
            for node in model.graph.node
        )
    finally:
        jax.config.update("jax_enable_x64", original_x64)


@pytest.mark.parametrize("opset", [17, 18, 21])
def test_linen_instance_norm_rank2_slow_variance_preserves_empty_axes(
    opset: int,
) -> None:
    module = nn.InstanceNorm(
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        use_fast_variance=False,
        force_float32_reductions=True,
    )
    x = jnp.arange(2 * 4, dtype=jnp.float32).reshape(2, 4) / 8
    fn = _linen_apply(module, x)

    model, actual = _convert_and_run(fn, [x], opset=opset)
    onnx.checker.check_model(model, full_check=True)
    expected = np.asarray(fn(x))
    np.testing.assert_allclose(actual[0], expected, rtol=1e-6, atol=1e-6)
    assert actual[0].shape == expected.shape == (2, 4)


@pytest.mark.parametrize("opset", [17, 18, 21, 23])
@pytest.mark.parametrize("kind", ["group", "instance"])
def test_linen_norm_high_offset_fast_variance_uses_explicit_lowering(
    opset: int, kind: str
) -> None:
    module: nn.Module
    if kind == "group":
        module = nn.GroupNorm(
            num_groups=2,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            use_fast_variance=True,
        )
    else:
        module = nn.InstanceNorm(
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            use_fast_variance=True,
        )
    x = 10000.0 + jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4) / 4
    fn = _linen_apply(module, x)

    model, actual = _convert_and_run(fn, [x], opset=opset)
    expected = np.asarray(fn(x))
    # Fast variance is reduction-order-sensitive at this offset. The explicit
    # layout-preserving path stays close to Linen; the native ORT kernels miss
    # by thousands for this input.
    np.testing.assert_allclose(actual[0], expected, rtol=2e-4, atol=5e-1)
    assert all(
        node.op_type not in {"GroupNormalization", "InstanceNormalization"}
        for node in model.graph.node
    )


@pytest.mark.parametrize("kind", ["group", "instance"])
def test_linen_norm_slow_variance_is_valid_at_opset17(kind: str) -> None:
    module: nn.Module
    if kind == "group":
        module = nn.GroupNorm(
            num_groups=2,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            use_fast_variance=False,
        )
    else:
        module = nn.InstanceNorm(
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            use_fast_variance=False,
        )
    x = 32.0 + jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4) / 100
    fn = _linen_apply(module, x)

    model, actual = _convert_and_run(fn, [x], opset=17)
    expected = np.asarray(fn(x))
    np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=5e-5)
    sum_squares = [
        node for node in model.graph.node if node.op_type == "ReduceSumSquare"
    ]
    if kind == "group":
        assert not sum_squares
        assert sum(node.op_type == "ReduceMean" for node in model.graph.node) == 2
        return
    assert sum_squares
    for node in sum_squares:
        assert len(node.input) == 1
        assert any(attr.name == "axes" for attr in node.attribute)


@pytest.mark.parametrize("use_fast_variance", [False, True])
def test_nnx_group_norm_infers_float64_from_parameters(
    use_fast_variance: bool,
) -> None:
    original_x64 = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        norm = nnx.GroupNorm(
            num_features=4,
            num_groups=2,
            dtype=None,
            param_dtype=jnp.float64,
            use_fast_variance=use_fast_variance,
            rngs=nnx.Rngs(0),
        )
        x = (
            1000.0
            + jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4) / 10
        )
        model = to_onnx(
            norm,
            [x],
            opset=21,
            enable_double_precision=True,
        )
        onnx.checker.check_model(model, full_check=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        actual = session.run(
            None,
            {session.get_inputs()[0].name: np.asarray(x, dtype=np.float64)},
        )
        expected = np.asarray(norm(x))
        assert actual[0].dtype == expected.dtype == np.dtype(np.float64)
        np.testing.assert_allclose(actual[0], expected, rtol=1e-9, atol=1e-9)
    finally:
        jax.config.update("jax_enable_x64", original_x64)


def test_group_norm_high_offset_float64_statistics() -> None:
    original_x64 = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        eqx_norm = eqx.nn.GroupNorm(groups=2, channels=4)
        nnx_norm = nnx.GroupNorm(
            num_features=4,
            num_groups=2,
            dtype=jnp.float64,
            param_dtype=jnp.float64,
            use_fast_variance=False,
            rngs=nnx.Rngs(0),
        )
        eqx_fn = jax.vmap(eqx_norm)
        eqx_x = (
            1e8 + jnp.arange(2 * 4 * 2 * 2, dtype=jnp.float64).reshape(2, 4, 2, 2) / 4
        )
        nnx_x = jnp.moveaxis(eqx_x, 1, -1)

        _assert_ort_parity(
            eqx_fn,
            [eqx_x],
            opset=21,
            enable_double_precision=True,
        )
        _assert_ort_parity(
            nnx_norm,
            [nnx_x],
            opset=21,
            enable_double_precision=True,
        )
    finally:
        jax.config.update("jax_enable_x64", original_x64)


@pytest.mark.parametrize("opset", [17, 18])
def test_adaptive_average_pool_supports_nondivisible_target(opset: int) -> None:
    pool = eqx.nn.AdaptiveAvgPool1d(9)
    fn = jax.vmap(pool)
    x = jnp.arange(2 * 4 * 64, dtype=jnp.float32).reshape(2, 4, 64)

    model = _assert_ort_parity(fn, [x], opset=opset)
    reductions = [node for node in model.graph.node if node.op_type == "ReduceMean"]
    assert len(reductions) == 9
    if opset < 18:
        assert all(len(node.input) == 1 for node in reductions)
        for node in reductions:
            axes = next(attr for attr in node.attribute if attr.name == "axes")
            assert tuple(axes.ints) == (2,)
    else:
        axes_inputs = [node.input[1] for node in reductions]
        assert len(set(axes_inputs)) == 1


def test_adaptive_average_pool_2d_supports_nondivisible_target_unbatched() -> None:
    pool = eqx.nn.AdaptiveAvgPool2d((3, 4))
    x = jnp.arange(3 * 7 * 11, dtype=jnp.float32).reshape(3, 7, 11)

    _assert_ort_parity(pool, [x])


@pytest.mark.parametrize("opset", [17, 18])
def test_adaptive_max_pool_2d_supports_nondivisible_target_batched(opset: int) -> None:
    pool = eqx.nn.AdaptiveMaxPool2d((3, 4))
    fn = jax.vmap(pool)
    x = jnp.arange(2 * 3 * 7 * 11, dtype=jnp.float32).reshape(2, 3, 7, 11)

    _assert_ort_parity(fn, [x], opset=opset)


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_deterministic_mixed_sequential_accepts_key(nested: bool) -> None:
    key = jax.random.PRNGKey(0)
    activation = eqx.nn.Sequential([eqx.nn.Identity(), eqx.nn.Lambda(jax.nn.relu)])
    layers = [
        eqx.nn.Linear(4, 3, key=key),
        activation if nested else eqx.nn.Lambda(jax.nn.relu),
        eqx.nn.Identity(),
    ]
    sequential = eqx.nn.Sequential(layers)
    fn = jax.vmap(lambda x: sequential(x, key=key))
    x = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)

    _assert_ort_parity(fn, [x])


def test_sequential_keeps_key_for_dropout() -> None:
    key = jax.random.PRNGKey(0)
    sequential = eqx.nn.Sequential(
        [eqx.nn.Linear(4, 4, key=key), eqx.nn.Dropout(p=0.5, inference=False)]
    )
    assert not all(_layer_ignores_key(layer) for layer in sequential.layers)

    seen_keys: list[jax.Array | None] = []

    def original(
        model: eqx.nn.Sequential,
        value: jax.Array,
        state: object,
        *,
        key: jax.Array | None,
    ) -> jax.Array:
        del model, state
        seen_keys.append(key)
        return value

    wrapped = SequentialPlugin._patch_call(original)
    wrapped(sequential, jnp.ones((4,), dtype=jnp.float32), key=key)
    assert seen_keys and seen_keys[0] is key


@pytest.mark.parametrize(
    "dropout",
    [
        eqx.nn.Dropout(p=0.5, inference=True),
        eqx.nn.Dropout(p=0.0, inference=False),
    ],
    ids=["inference", "zero_probability"],
)
def test_deterministic_dropout_sequential_accepts_key(
    dropout: eqx.nn.Dropout,
) -> None:
    key = jax.random.PRNGKey(0)
    sequential = eqx.nn.Sequential(
        [
            eqx.nn.Linear(4, 4, key=key),
            dropout,
            eqx.nn.Lambda(jax.nn.relu),
        ]
    )
    assert all(_layer_ignores_key(layer) for layer in sequential.layers)

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
