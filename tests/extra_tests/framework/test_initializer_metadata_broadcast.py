# tests/extra_tests/framework/test_initializer_metadata_broadcast.py

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.plugins.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
)


@pytest.fixture(autouse=True)
def _clear_local_functions() -> Iterator[None]:
    yield
    for qualified in tuple(ONNX_FUNCTION_PLUGIN_REGISTRY):
        if qualified.startswith(f"{__name__}."):
            PLUGIN_REGISTRY.pop(f"onnx_fn::{qualified}", None)
            ONNX_FUNCTION_PLUGIN_REGISTRY.pop(qualified, None)


def _nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    return [
        *model.graph.node,
        *(node for function in model.functions for node in function.node),
    ]


def _tensors(model: onnx.ModelProto) -> list[onnx.TensorProto]:
    return [
        *model.graph.initializer,
        *(
            attribute.t
            for node in _nodes(model)
            for attribute in node.attribute
            if attribute.type == onnx.AttributeProto.TENSOR
        ),
    ]


def _props(tensor: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.metadata_props}


def _assert_storage(tensor: onnx.TensorProto, source: np.ndarray) -> None:
    array = onnx.numpy_helper.to_array(tensor)
    assert array.dtype == source.dtype
    assert array.size == source.size
    # Numeric equality would hide signed zeros and noncanonical NaN payloads.
    assert array.tobytes(order="C") == source.tobytes(order="C")


def _run(model: onnx.ModelProto, x: np.ndarray) -> list[np.ndarray]:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if x.size == 0:
        # ORT 1.24.1's RemoveNode optimizer rejects some valid empty graphs.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, {session.get_inputs()[0].name: x})


@pytest.mark.parametrize(
    "source_shape,target_shape,bdims,return_mode,x64",
    [
        ((), (1, 1), (), "proto", False),
        ((3,), (1, 3), (1,), "ir", True),
        ((3,), (3, 1, 1), (0,), "file", False),
        ((2, 3), (1, 2, 1, 3), (1, 3), "proto", False),
        ((0, 3), (1, 0, 1, 3), (1, 3), "proto", False),
    ],
    ids=["scalar", "fp64-ir", "trailing-file", "interleaved", "empty"],
)
def test_singleton_insertion_stores_one_exact_annotated_view(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    bdims: tuple[int, ...],
    return_mode: Literal["proto", "ir", "file"],
    x64: bool,
    tmp_path: Path,
) -> None:
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        array = np.arange(np.prod(source_shape, dtype=int), dtype=dtype).reshape(
            source_shape
        )
        weight = jnp.asarray(array)
        tags = {"source": "broadcast.weight", "opaque/empty": ""}

        def apply(x: jax.Array) -> jax.Array:
            return x + jax.lax.broadcast_in_dim(weight, target_shape, bdims)

        export_options: dict[str, Any] = {}
        if return_mode == "file":
            export_options["output_path"] = tmp_path / "broadcast.onnx"
        exported = to_onnx(
            apply,
            [jax.ShapeDtypeStruct(target_shape, dtype)],
            initializer_metadata=lambda source: tags if source is weight else None,
            return_mode=return_mode,
            **export_options,
        )
        if isinstance(exported, ir.Model):
            # Check the returned artifact before exercising protobuf/runtime too.
            annotated_ir = [
                value
                for value in exported.graph.initializers.values()
                if value.const_value is not None and value.const_value.metadata_props
            ]
            assert len(annotated_ir) == 1
            assert annotated_ir[0].const_value.metadata_props == tags
            assert tuple(annotated_ir[0].shape) == target_shape
            model = ir.to_proto(exported)
        elif return_mode == "file":
            model = onnx.load_model(exported)
        else:
            model = exported
        tagged = [tensor for tensor in _tensors(model) if _props(tensor)]
        assert len(tagged) == 1
        assert _props(tagged[0]) == tags
        assert tuple(tagged[0].dims) == target_shape
        _assert_storage(tagged[0], array)
        assert [node.op_type for node in model.graph.node] == ["Add"]
        x = np.arange(array.size, dtype=dtype).reshape(target_shape)
        actual = _run(model, x)[0]
        assert actual.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(actual, np.asarray(apply(jnp.asarray(x))))


def test_nested_function_singleton_view_preserves_zero_and_nan_payload_bits() -> None:
    bits = np.asarray([0x00000000, 0x80000000, 0x7FC01234, 0x3F800000], dtype=np.uint32)
    weight = jnp.asarray(bits.view(np.float32))
    tags = {"source": "nested.singleton", "application/utf8": "café ❄"}

    @onnx_function
    class Inner:
        def __call__(self, x: jax.Array) -> jax.Array:
            return x + jax.lax.broadcast_in_dim(weight, (1, 4), (1,))

    inner = Inner()

    @onnx_function
    class Outer:
        def __call__(self, x: jax.Array) -> jax.Array:
            return inner(x)

    outer = Outer()
    model = to_onnx(
        lambda x: outer(x),
        [(1, 4)],
        initializer_metadata=lambda source: tags if source is weight else None,
    )
    assert len(model.functions) == 2
    tagged = [tensor for tensor in _tensors(model) if _props(tensor)]
    assert len(tagged) == 1
    assert _props(tagged[0]) == tags
    assert tuple(tagged[0].dims) == (1, 4)
    _assert_storage(tagged[0], bits.view(np.float32))
    assert not {"Concat", "Reshape", "Expand"} & {
        node.op_type for node in _nodes(model)
    }
    x = np.arange(4, dtype=np.float32).reshape(1, 4)
    np.testing.assert_allclose(
        _run(model, x)[0], np.asarray(outer(jnp.asarray(x))), equal_nan=True
    )
    tagged[0].raw_data = np.full((1, 4), 7, dtype=np.float32).tobytes()
    np.testing.assert_array_equal(_run(model, x)[0], x + 7)


def test_repeated_singleton_broadcast_reuses_the_same_source_view() -> None:
    weight = jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float32)
    tags = {"source": "shared"}

    def apply(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        first = jax.lax.broadcast_in_dim(weight, (1, 3), (1,))
        second = jax.lax.broadcast_in_dim(weight, (1, 3), (1,))
        return x + first, x * second

    model = to_onnx(
        apply,
        [(1, 3)],
        initializer_metadata=lambda source: tags if source is weight else None,
    )
    tagged = [tensor for tensor in _tensors(model) if _props(tensor)]
    assert len(tagged) == 1
    assert _props(tagged[0]) == tags
    assert tuple(tagged[0].dims) == (1, 3)
    consumers = [node for node in model.graph.node if tagged[0].name in node.input]
    assert {node.op_type for node in consumers} == {"Add", "Mul"}
    assert len(model.graph.node) == 2
    tagged[0].raw_data = np.full((1, 3), 5, dtype=np.float32).tobytes()
    x = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    actual = _run(model, x)
    np.testing.assert_array_equal(actual[0], x + 5)
    np.testing.assert_array_equal(actual[1], x * 5)


@pytest.mark.parametrize("annotate_neighbor", [False, True])
def test_equal_parameter_views_retain_independently_editable_uses(
    annotate_neighbor: bool,
) -> None:
    left = jnp.asarray(np.ones((3,), dtype=np.float32))
    right = jnp.asarray(np.ones((3,), dtype=np.float32))
    assert left is not right
    tags = {"source": "identical opaque annotation"}

    def metadata(source: object) -> dict[str, str] | None:
        return (
            tags if source is left or (annotate_neighbor and source is right) else None
        )

    def apply(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            x + jax.lax.broadcast_in_dim(left, (1, 3), (1,)),
            x + jax.lax.broadcast_in_dim(right, (1, 3), (1,)),
        )

    model = to_onnx(apply, [(1, 3)], initializer_metadata=metadata)
    assert [node.op_type for node in model.graph.node] == ["Add", "Add"]
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    # Select the stored source by each output's actual data dependencies.
    carriers = [
        initializers[
            next(name for name in producers[output.name].input if name in initializers)
        ]
        for output in model.graph.output
    ]
    assert carriers[0] is not carriers[1]
    assert _props(carriers[0]) == tags
    assert _props(carriers[1]) == (tags if annotate_neighbor else {})
    x = np.arange(3, dtype=np.float32).reshape(1, 3)
    for index, tensor in enumerate(carriers):
        original = tensor.SerializeToString()
        tensor.raw_data = np.full((1, 3), 5, dtype=np.float32).tobytes()
        for output_index, actual in enumerate(_run(model, x)):
            np.testing.assert_array_equal(
                actual, x + (5 if output_index == index else 1)
            )
        tensor.ParseFromString(original)


@pytest.mark.parametrize(
    "source_shape,target_shape,bdims",
    [
        ((), (2, 3), ()),
        ((1, 3), (2, 3), (0, 1)),
        ((), (0,), ()),
        ((1, 3), (0, 3), (0, 1)),
        ((0, 1), (0, 4), (0, 1)),
    ],
    ids=[
        "scalar-expansion",
        "axis-expansion",
        "scalar-to-empty",
        "axis-to-empty",
        "empty-with-expansion",
    ],
)
def test_genuine_broadcast_retains_source_storage_and_runtime_expand(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    bdims: tuple[int, ...],
) -> None:
    array = np.full(source_shape, 2, dtype=np.float32)
    weight = jnp.asarray(array)
    tags = {"source": "genuine.broadcast"}
    model = to_onnx(
        lambda x: x + jax.lax.broadcast_in_dim(weight, target_shape, bdims),
        [target_shape],
        initializer_metadata=lambda source: tags if source is weight else None,
    )
    tagged = [tensor for tensor in _tensors(model) if _props(tensor)]
    assert len(tagged) == 1
    assert _props(tagged[0]) == tags
    assert tuple(tagged[0].dims) == source_shape
    _assert_storage(tagged[0], array)
    assert "Expand" in {node.op_type for node in model.graph.node}
    tagged[0].raw_data = np.full(source_shape, 5, dtype=np.float32).tobytes()
    x = np.zeros(target_shape, dtype=np.float32)
    np.testing.assert_array_equal(_run(model, x)[0], x + 5)


def test_singleton_broadcast_of_cast_retains_original_dtype_and_computation() -> None:
    weight = jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float16)
    tags = {"source": "cast.weight"}

    def apply(x: jax.Array) -> jax.Array:
        return x + jax.lax.broadcast_in_dim(weight.astype(jnp.float32), (1, 3), (1,))

    model = to_onnx(
        apply,
        [(1, 3)],
        initializer_metadata=lambda source: tags if source is weight else None,
    )
    tagged = [tensor for tensor in _tensors(model) if _props(tensor)]
    assert len(tagged) == 1
    assert tuple(tagged[0].dims) == (3,)
    assert tagged[0].data_type == onnx.TensorProto.FLOAT16
    _assert_storage(tagged[0], np.asarray(weight))
    assert {"Cast", "Expand"} <= {node.op_type for node in model.graph.node}
    tagged[0].raw_data = np.full((3,), 5, dtype=np.float16).tobytes()
    x = np.arange(3, dtype=np.float32).reshape(1, 3)
    actual = _run(model, x)[0]
    assert actual.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(actual, x + 5)


def test_symbolic_target_keeps_runtime_expansion_for_different_batch_sizes() -> None:
    weight = jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float32)
    tags = {"source": "dynamic.weight"}

    def apply(x: jax.Array) -> jax.Array:
        return x + jax.lax.broadcast_in_dim(weight, (x.shape[0], 3), (1,))

    model = to_onnx(
        apply,
        [("B", 3)],
        initializer_metadata=lambda source: tags if source is weight else None,
    )
    tagged = [tensor for tensor in _tensors(model) if _props(tensor)]
    assert len(tagged) == 1
    assert tuple(tagged[0].dims) == (3,)
    _assert_storage(tagged[0], np.asarray(weight))
    assert "Expand" in {node.op_type for node in model.graph.node}
    tagged[0].raw_data = np.full((3,), 5, dtype=np.float32).tobytes()
    for batch in (1, 2):
        x = np.zeros((batch, 3), dtype=np.float32)
        np.testing.assert_array_equal(_run(model, x)[0], x + 5)


def test_layer_scale_and_vera_vectors_do_not_add_shape_subgraphs() -> None:
    # A small deterministic layer-scale and VeRA-style low-rank adapter uses
    # several captured vectors exactly as larger optional framework models do.
    width, rank = 32, 4
    sources = {
        "layer_scale": jnp.linspace(0.5, 1.0, width, dtype=jnp.float32),
        "rank_scale": jnp.linspace(0.5, 1.0, rank, dtype=jnp.float32),
        "output_scale": jnp.linspace(0.25, 0.5, width, dtype=jnp.float32),
        "down": jnp.asarray(
            np.linspace(-0.1, 0.1, width * rank, dtype=np.float32).reshape(width, rank)
        ),
        "up": jnp.asarray(
            np.linspace(-0.2, 0.2, rank * width, dtype=np.float32).reshape(rank, width)
        ),
    }

    def apply(x: jax.Array) -> jax.Array:
        scaled = x * sources["layer_scale"][None, :]
        hidden = (x @ sources["down"]) * sources["rank_scale"][None, :]
        residual = (hidden @ sources["up"]) * sources["output_scale"][None, :]
        return scaled + residual

    def metadata(source: object) -> dict[str, str] | None:
        return next(
            (
                {"source_path": name}
                for name, value in sources.items()
                if source is value
            ),
            None,
        )

    ordinary = to_onnx(apply, [(2, width)])
    model = to_onnx(apply, [(2, width)], initializer_metadata=metadata)
    tagged = {
        _props(tensor)["source_path"]: tensor
        for tensor in _tensors(model)
        if _props(tensor)
    }
    assert tagged.keys() == sources.keys()
    for name, value in sources.items():
        _assert_storage(tagged[name], np.asarray(value))
    assert tuple(tagged["layer_scale"].dims) == (1, width)
    assert tuple(tagged["rank_scale"].dims) == (1, rank)
    assert tuple(tagged["output_scale"].dims) == (1, width)
    assert not {"Concat", "Reshape", "Expand"} & {
        node.op_type for node in _nodes(model)
    }
    assert len(model.graph.node) <= 8
    assert len(model.SerializeToString()) <= len(ordinary.SerializeToString()) + 2048
    x = np.linspace(-1, 1, 2 * width, dtype=np.float32).reshape(2, width)
    np.testing.assert_allclose(
        _run(model, x)[0], np.asarray(apply(jnp.asarray(x))), rtol=1e-5, atol=1e-6
    )


def test_all_skip_callbacks_keep_ordinary_singleton_export_behavior() -> None:
    weight = jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float32)

    def apply(x: jax.Array) -> jax.Array:
        return x + jax.lax.broadcast_in_dim(weight, (1, 3), (1,))

    ordinary = to_onnx(apply, [(1, 3)])
    for callback in (lambda source: None, lambda source: {}):
        skipped = to_onnx(apply, [(1, 3)], initializer_metadata=callback)
        assert [node.SerializeToString() for node in skipped.graph.node] == [
            node.SerializeToString() for node in ordinary.graph.node
        ]
        assert [tensor.SerializeToString() for tensor in skipped.graph.initializer] == [
            tensor.SerializeToString() for tensor in ordinary.graph.initializer
        ]
        assert not any(_props(tensor) for tensor in _tensors(skipped))
        x = np.arange(3, dtype=np.float32).reshape(1, 3)
        np.testing.assert_array_equal(_run(skipped, x)[0], x + np.asarray(weight))
