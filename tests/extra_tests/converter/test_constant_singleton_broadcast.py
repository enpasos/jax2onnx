# tests/extra_tests/converter/test_constant_singleton_broadcast.py

"""Singleton broadcast views retain the stored source parameter."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx


def _session(model: onnx.ModelProto) -> ort.InferenceSession:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _run(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    return _session(model).run(None, feeds)


def _replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    source = next(item for item in model.graph.initializer if item.name == name)
    source.CopyFrom(numpy_helper.from_array(value, name=name))


@pytest.mark.parametrize("views_first", [False, True])
def test_shared_weight_views_follow_source_edits(
    tmp_path: Path, views_first: bool
) -> None:
    weight = np.arange(1, 9, dtype=np.float32)

    def target(x):
        if views_first:
            row = jax.lax.broadcast_in_dim(weight, (1, 8), (1,))
            column = jax.lax.broadcast_in_dim(weight, (8, 1), (0,))
            direct = x * weight
        else:
            direct = x * weight
            row = jax.lax.broadcast_in_dim(weight, (1, 8), (1,))
            column = jax.lax.broadcast_in_dim(weight, (8, 1), (0,))
        return direct, x[None, :] * row, x[:, None] * column

    path = tmp_path / "shared_weight.onnx"
    model = to_onnx(
        target,
        inputs=[jax.ShapeDtypeStruct((8,), jnp.float32)],
        input_names=["x"],
    )
    onnx.save_model(model, path)
    loaded = onnx.load_model(path)
    stored = [
        value
        for value in loaded.graph.initializer
        if value.data_type == onnx.TensorProto.FLOAT
    ]
    assert len(stored) == 1
    source = stored[0]
    assert list(source.dims) == [8]
    assert numpy_helper.to_array(source).tobytes() == weight.tobytes()
    views = [node for node in loaded.graph.node if node.op_type == "Unsqueeze"]
    assert len(views) == 2
    assert all(node.input[0] == source.name for node in views)
    initializers = {value.name: value for value in loaded.graph.initializer}
    assert [
        numpy_helper.to_array(initializers[node.input[1]]).tolist() for node in views
    ] == [[0], [1]]
    assert all(
        initializers[node.input[1]].data_type == onnx.TensorProto.INT64
        for node in views
    )

    x = np.arange(2, 10, dtype=np.float32)
    for new_weight in (weight, weight * 3 + 2):
        _replace_initializer(loaded, source.name, new_weight)
        onnx.save_model(loaded, path)
        reloaded = onnx.load_model(path)
        actual = _run(reloaded, {"x": x})
        expected = (
            x * new_weight,
            x[None, :] * new_weight[None, :],
            x[:, None] * new_weight[:, None],
        )
        for value, wanted in zip(actual, expected):
            np.testing.assert_array_equal(value, wanted)


def test_equal_independent_sources_remain_independent() -> None:
    first = np.arange(8, dtype=np.float32)
    second = first.copy()

    def target():
        return (
            first,
            jax.lax.broadcast_in_dim(first, (1, 8), (1,)),
            second,
            jax.lax.broadcast_in_dim(second, (8, 1), (0,)),
        )

    model = to_onnx(target, inputs=[])
    stored = [
        item
        for item in model.graph.initializer
        if item.data_type == onnx.TensorProto.FLOAT
    ]
    assert len(stored) == 2
    assert all(list(item.dims) == [8] for item in stored)
    views = [node for node in model.graph.node if node.op_type == "Unsqueeze"]
    assert {node.input[0] for node in views} == {item.name for item in stored}
    changed = first + 20
    _replace_initializer(model, views[0].input[0], changed)
    actual = _run(model, {})
    for value, wanted in zip(
        actual, (changed, changed[None, :], second, second[:, None])
    ):
        np.testing.assert_array_equal(value, wanted)


@pytest.mark.parametrize("opset", [11, 12, 13, 14, 23])
def test_opset_fallback_keeps_constant_broadcasts(opset: int) -> None:
    weight = np.arange(8, dtype=np.float32)
    model = to_onnx(
        lambda: (weight, jax.lax.broadcast_in_dim(weight, (1, 8), (1,))),
        inputs=[],
        opset=opset,
        output_names=["weight", "view"],
    )
    assert [item.name for item in model.graph.output] == ["weight", "view"]
    views = [node for node in model.graph.node if node.op_type == "Unsqueeze"]
    assert len(views) == int(opset >= 13)
    stored = [
        item
        for item in model.graph.initializer
        if item.data_type == onnx.TensorProto.FLOAT
    ]
    assert sorted(tuple(item.dims) for item in stored) == (
        [(8,)] if opset >= 13 else [(1, 8), (8,)]
    )
    for actual, expected in zip(_run(model, {}), (weight, weight[None, :])):
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "source_shape,target_shape,bdims",
    [
        ((0,), (1, 0, 1), (1,)),
        ((0, 0), (0, 1, 0, 1), (0, 2)),
        ((), (1, 1), ()),
        ((), (), ()),
        ((8,), (8,), (0,)),
    ],
)
def test_static_shape_edges(source_shape, target_shape, bdims) -> None:
    weight = np.zeros(source_shape, dtype=np.float32)
    model = to_onnx(
        lambda: jax.lax.broadcast_in_dim(weight, target_shape, bdims),
        inputs=[],
        output_names=["view"],
    )
    assert [item.name for item in model.graph.output] == ["view"]
    if len(source_shape) < len(target_shape):
        assert [node.op_type for node in model.graph.node] == ["Unsqueeze"]
    (actual,) = _run(model, {})
    assert actual.shape == target_shape
    assert actual.dtype == weight.dtype
    assert actual.tobytes() == weight.tobytes()


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, np.float64, jnp.bfloat16],
)
def test_constant_dtype_and_bytes(dtype, x64: bool) -> None:
    if not x64 and np.dtype(dtype) == np.dtype(np.float64):
        pytest.skip("JAX requires x64 for the prepared dtype")
    with jax.enable_x64(x64):
        weight = np.asarray([-0.0, 0.0, 1.0, -2.0], dtype=dtype)
        model = to_onnx(
            lambda: jax.lax.broadcast_in_dim(weight, (1, 4), (1,)),
            inputs=[],
        )
    views = [node for node in model.graph.node if node.op_type == "Unsqueeze"]
    assert len(views) == 1
    source = next(
        item for item in model.graph.initializer if item.name == views[0].input[0]
    )
    stored = numpy_helper.to_array(source)
    assert stored.dtype == weight.dtype
    assert stored.tobytes() == weight.tobytes()
    onnx.checker.check_model(model, full_check=True)
    if np.dtype(dtype) == np.dtype(jnp.bfloat16):
        # CPU supports the view; Python run() cannot return BF16 arrays.
        _session(model)
    else:
        (actual,) = _run(model, {})
        assert actual.dtype == weight.dtype
        assert actual.tobytes() == weight.tobytes()


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint8, np.bool_])
def test_integer_and_boolean_constant_views_keep_folding(dtype, x64: bool) -> None:
    if not x64 and np.dtype(dtype) == np.dtype(np.int64):
        pytest.skip("JAX requires x64 for the prepared dtype")
    with jax.enable_x64(x64):
        weight = np.asarray([0, 1, 2, 3], dtype=dtype)
        model = to_onnx(
            lambda: (
                weight,
                jax.lax.broadcast_in_dim(weight, (1, 4), (1,)),
                jax.lax.broadcast_in_dim(weight, (4, 1), (0,)),
            ),
            inputs=[],
        )
    assert not model.graph.node
    assert sorted(tuple(item.dims) for item in model.graph.initializer) == [
        (1, 4),
        (4,),
        (4, 1),
    ]
    for stored in model.graph.initializer:
        payload = numpy_helper.to_array(stored)
        assert payload.dtype == weight.dtype
        assert payload.tobytes() == weight.tobytes()
    for actual, expected in zip(
        _run(model, {}), (weight, weight[None, :], weight[:, None])
    ):
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape,index,payload,add",
    [
        ((1, 1), 0, -100.0, False),
        ((3,), 1, 42.0, False),
        ((4,), 0, -1.0, False),
        ((6,), 3, 5.0, True),
    ],
)
def test_scatter_constant_indices_remain_known(
    shape, index, payload, add, dtype
) -> None:
    def target(x):
        update = jnp.asarray(payload, dtype=x.dtype)
        return x.at[index].add(update) if add else x.at[index].set(update)

    with jax.enable_x64(np.dtype(dtype) == np.dtype(np.float64)):
        model = to_onnx(
            target,
            inputs=[jax.ShapeDtypeStruct(shape, dtype)],
            input_names=["x"],
        )
    assert not any(node.op_type in ("NonZero", "Gather") for node in model.graph.node)
    (scatter,) = [node for node in model.graph.node if node.op_type == "ScatterND"]
    producers = {output: node for node in model.graph.node for output in node.output}
    reshape = producers[scatter.input[1]]
    assert reshape.op_type == "Reshape"
    cast = producers[reshape.input[0]]
    assert cast.op_type == "Cast"
    assert (
        next(attr.i for attr in cast.attribute if attr.name == "to")
        == onnx.TensorProto.INT64
    )
    indices = next(
        item for item in model.graph.initializer if item.name == cast.input[0]
    )
    assert indices.data_type == onnx.TensorProto.INT32
    np.testing.assert_array_equal(numpy_helper.to_array(indices), [index])
    assert (
        sum(node.op_type == "Unsqueeze" for node in model.graph.node)
        == int(len(shape) == 2) + 2
    )
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    expected = x.copy()
    if add:
        expected[index] += payload
    else:
        expected[index] = payload
    (actual,) = _run(model, {"x": x})
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("opset", [11, 12])
def test_bfloat16_constant_views_keep_old_opset_fallback(opset: int) -> None:
    weight = np.asarray([-0.0, 1.0, -2.0], dtype=jnp.bfloat16)
    model = to_onnx(
        lambda: jax.lax.broadcast_in_dim(weight, (1, 3), (1,)), inputs=[], opset=opset
    )
    assert not any(node.op_type == "Unsqueeze" for node in model.graph.node)
    (source,) = model.graph.initializer
    assert source.data_type == onnx.TensorProto.BFLOAT16
    assert list(source.dims) == [1, 3]
    assert numpy_helper.to_array(source).tobytes() == weight.tobytes()
    _session(model)


@pytest.mark.parametrize("opset", [13, 23])
def test_float8_constant_views_keep_folding(opset: int) -> None:
    weight = np.asarray([1, 2], dtype=jnp.float8_e4m3fn)
    model = to_onnx(
        lambda: jax.lax.broadcast_in_dim(weight, (1, 2), (1,)),
        inputs=[],
        opset=opset,
    )
    assert not model.graph.node
    (source,) = model.graph.initializer
    assert source.data_type == onnx.TensorProto.FLOAT8E4M3FN
    assert list(source.dims) == [1, 2]
    assert numpy_helper.to_array(source).tobytes() == weight.tobytes()
    onnx.checker.check_model(model, full_check=True)


def test_genuinely_expanding_constant_broadcast_keeps_folding() -> None:
    weight = np.arange(8, dtype=np.float32)
    model = to_onnx(lambda: jax.lax.broadcast_in_dim(weight, (2, 8), (1,)), inputs=[])
    assert not any(node.op_type == "Unsqueeze" for node in model.graph.node)
    assert any(list(item.dims) == [2, 8] for item in model.graph.initializer)
    (actual,) = _run(model, {})
    np.testing.assert_array_equal(actual, np.broadcast_to(weight, (2, 8)))


def test_nonfinite_payload_and_signed_zero_are_not_transformed() -> None:
    weight = np.asarray(
        [0x00000000, 0x80000000, 0x7FC12345, 0xFFC54321, 0x7F800000],
        dtype=np.uint32,
    ).view(np.float32)
    model = to_onnx(lambda: jax.lax.broadcast_in_dim(weight, (1, 5), (1,)), inputs=[])
    (view,) = model.graph.node
    assert view.op_type == "Unsqueeze"
    source = next(
        item for item in model.graph.initializer if item.name == view.input[0]
    )
    assert numpy_helper.to_array(source).tobytes() == weight.tobytes()
    (actual,) = _run(model, {})
    assert actual.tobytes() == weight.tobytes()


@onnx_function
def _function_constant_view(x):
    weight = np.asarray([1, 2, 3, 4], dtype=np.float32)
    return jax.lax.broadcast_in_dim(weight, (1, 4), (1,))


@onnx_function
def _function_shared_constant_views(x):
    weight = np.asarray([1, 2, 3, 4], dtype=np.float32)
    direct = x * weight
    row = x[None, :] * jax.lax.broadcast_in_dim(weight, (1, 4), (1,))
    column = x[:, None] * jax.lax.broadcast_in_dim(weight, (4, 1), (0,))
    return jnp.concatenate((direct, row.reshape(4), column.reshape(4)))


def test_function_local_constant_view_preserves_output_boundary() -> None:
    model = to_onnx(
        lambda x: _function_constant_view(x),
        inputs=[jax.ShapeDtypeStruct((), jnp.float32)],
        input_names=["x"],
        output_names=["view"],
    )
    (function,) = model.functions
    views = [node for node in function.node if node.op_type == "Unsqueeze"]
    assert len(views) == 1
    (view,) = views
    assert list(function.output) == list(view.output)
    constants = {
        node.output[0]: node for node in function.node if node.op_type == "Constant"
    }
    assert set(view.input) <= constants.keys()
    assert not model.graph.initializer
    assert [item.name for item in model.graph.output] == ["view"]
    (actual,) = _run(model, {"x": np.asarray(0, dtype=np.float32)})
    np.testing.assert_array_equal(actual, np.asarray([[1, 2, 3, 4]], dtype=np.float32))


def test_function_local_views_share_one_constant_with_direct_consumer() -> None:
    model = to_onnx(
        lambda x: _function_shared_constant_views(x),
        inputs=[jax.ShapeDtypeStruct((4,), jnp.float32)],
        input_names=["x"],
        output_names=["products"],
    )
    (function,) = model.functions
    constants = [
        node
        for node in function.node
        if node.op_type == "Constant"
        and next(
            attribute.t.data_type
            for attribute in node.attribute
            if attribute.name == "value"
        )
        == onnx.TensorProto.FLOAT
    ]
    assert len(constants) == 1
    source = constants[0].output[0]
    views = [node for node in function.node if node.op_type == "Unsqueeze"]
    assert len(views) == 2
    assert all(node.input[0] == source for node in views)
    assert any(node.op_type == "Mul" and source in node.input for node in function.node)
    x = np.asarray([2, 3, 4, 5], dtype=np.float32)
    (actual,) = _run(model, {"x": x})
    np.testing.assert_array_equal(
        actual, np.tile(x * np.arange(1, 5, dtype=np.float32), 3)
    )
