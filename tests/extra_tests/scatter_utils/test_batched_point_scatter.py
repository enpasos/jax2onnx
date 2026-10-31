# tests/extra_tests/scatter_utils/test_batched_point_scatter.py

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
from jax import lax
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx
from jax2onnx.plugins.jax.lax.scatter_utils import (
    ScatterSpec,
    _expand_batched_elementwise_indices,
    _normalize_dimension_numbers,
)


def _session(model: Any) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


@pytest.mark.parametrize(
    "layout", ["leading", "operand-nonleading", "indices-nonleading"]
)
def test_batched_scatter_add_preserves_collisions_drops_and_batch_isolation(
    layout: str,
) -> None:
    operand = np.asarray([[10, 20, 30, 40], [100, 200, 300, 400]], np.int32)
    indices = np.asarray(
        [[[0], [0], [-1], [4], [2]], [[0], [1], [1], [3], [4]]], np.int32
    )
    updates = np.asarray([[1, 2, 9, 7, 4], [1000, 3, 5, 11, 13]], np.int32)
    expected = np.asarray([[13, 20, 34, 40], [1100, 208, 300, 411]], np.int32)
    operand_batch, indices_batch, scatter_axis = 0, 0, 1
    if layout == "operand-nonleading":
        operand, expected = operand.T.copy(), expected.T.copy()
        operand_batch, scatter_axis = 1, 0
    elif layout == "indices-nonleading":
        indices, updates = indices.transpose(1, 0, 2).copy(), updates.T.copy()
        indices_batch = 1
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(scatter_axis,),
        scatter_dims_to_operand_dims=(scatter_axis,),
        operand_batching_dims=(operand_batch,),
        scatter_indices_batching_dims=(indices_batch,),
    )

    def fn(data: jax.Array, coordinates: jax.Array, values: jax.Array) -> jax.Array:
        return lax.scatter_add(data, coordinates, values, dimensions, mode="drop")

    np.testing.assert_array_equal(np.asarray(fn(operand, indices, updates)), expected)
    model = to_onnx(fn, inputs=[operand, indices, updates], return_mode="ir")
    scatter = [node for node in model.graph if node.op_type == "ScatterND"]
    assert len(scatter) == 1
    assert scatter[0].attributes["reduction"].as_string() == "add"
    assert scatter[0].inputs[1].dtype == ir.DataType.INT64
    assert scatter[0].inputs[0].dtype == scatter[0].inputs[2].dtype == ir.DataType.INT32
    session = _session(ir.to_proto(model))
    feeds = {
        entry.name: value
        for entry, value in zip(
            session.get_inputs(), [operand, indices, updates], strict=True
        )
    }
    actual = session.run(None, feeds)
    assert len(actual) == 1
    np.testing.assert_array_equal(actual[0], expected)


@pytest.mark.parametrize("indices_pairing", [(0, 1), (1, 0)])
def test_batched_scatter_supports_multiple_batch_axes_with_promise_in_bounds(
    indices_pairing: tuple[int, int],
) -> None:
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(2,),
        scatter_dims_to_operand_dims=(2,),
        operand_batching_dims=(0, 1),
        scatter_indices_batching_dims=indices_pairing,
    )

    def fn(data: jax.Array, coordinates: jax.Array, values: jax.Array) -> jax.Array:
        return lax.scatter_add(
            data, coordinates, values, dimensions, mode="promise_in_bounds"
        )

    data = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    coordinates = np.broadcast_to(
        np.asarray([0, 2, 2], np.int32).reshape(1, 1, 3, 1), (2, 3, 3, 1)
    ).copy()
    updates = np.arange(1, 19, dtype=np.int32).reshape(2, 3, 3)
    if indices_pairing == (1, 0):
        coordinates = coordinates.transpose(1, 0, 2, 3).copy()
        updates = updates.transpose(1, 0, 2).copy()
    expected = np.asarray(fn(data, coordinates, updates))
    model = to_onnx(fn, inputs=[data, coordinates, updates])
    session = _session(model)
    feeds = {
        entry.name: value
        for entry, value in zip(
            session.get_inputs(), [data, coordinates, updates], strict=True
        )
    }
    np.testing.assert_array_equal(session.run(None, feeds)[0], expected)


def test_batched_scatter_empty_updates_preserve_operand() -> None:
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,),
        operand_batching_dims=(0,),
        scatter_indices_batching_dims=(0,),
    )

    def fn(data: jax.Array, coordinates: jax.Array, values: jax.Array) -> jax.Array:
        return lax.scatter_add(data, coordinates, values, dimensions, mode="drop")

    inputs = [
        np.arange(8, dtype=np.int32).reshape(2, 4),
        np.empty((2, 0, 1), np.int32),
        np.empty((2, 0), np.int32),
    ]
    np.testing.assert_array_equal(np.asarray(fn(*inputs)), inputs[0])
    model = to_onnx(fn, inputs=inputs)
    session = _session(model)
    feeds = {
        entry.name: value
        for entry, value in zip(session.get_inputs(), inputs, strict=True)
    }
    np.testing.assert_array_equal(session.run(None, feeds)[0], inputs[0])


@pytest.mark.parametrize("opset", [11, 13, 14])
def test_point_scatter_reshape_attributes_respect_requested_opset(opset: int) -> None:
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )

    def fn(data: jax.Array, coordinates: jax.Array, values: jax.Array) -> jax.Array:
        return lax.scatter(
            data,
            coordinates,
            values,
            dimensions,
            mode="promise_in_bounds",
            unique_indices=True,
        )

    inputs = [
        np.arange(4, dtype=np.int32).reshape(2, 2),
        np.asarray([[[0, 1]], [[1, 0]]], np.int32),
        np.asarray([[9], [7]], np.int32),
    ]
    model = to_onnx(fn, inputs=inputs, opset=opset)
    reshapes = [node for node in model.graph.node if node.op_type == "Reshape"]
    assert len(reshapes) >= 2
    for node in reshapes:
        allowzero = [
            attribute.i for attribute in node.attribute if attribute.name == "allowzero"
        ]
        assert allowzero == ([1] if opset >= 14 else [])
    session = _session(model)
    feeds = {
        entry.name: value
        for entry, value in zip(session.get_inputs(), inputs, strict=True)
    }
    np.testing.assert_array_equal(
        session.run(None, feeds)[0], np.asarray([[0, 9], [7, 3]], np.int32)
    )


def test_scatter_spec_retains_native_batch_dimension_roles() -> None:
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,),
        operand_batching_dims=(0,),
        scatter_indices_batching_dims=(0,),
    )
    spec = _normalize_dimension_numbers(dimensions)
    assert spec.operand_batching_dims == spec.scatter_indices_batching_dims == (0,)


@pytest.mark.parametrize(
    "fault",
    ["window", "unpaired", "extent", "updates-shape", "axis", "duplicate", "clip"],
)
def test_unsupported_batched_scatter_rejects_before_ir_emission(fault: str) -> None:
    class NoEmission:
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError("unsupported batched scatter emitted IR")

    spec = ScatterSpec((), (1,), (1,), (0,), (0,))
    operand_shape, indices_shape, updates_shape = (2, 4), (2, 3, 1), (2, 3)
    mode = lax.GatherScatterMode.FILL_OR_DROP
    if fault == "window":
        spec = replace(spec, update_window_dims=(1,))
    elif fault == "unpaired":
        spec = replace(spec, scatter_indices_batching_dims=())
    elif fault == "extent":
        operand_shape = (3, 4)
    elif fault == "updates-shape":
        updates_shape = (2, 4)
    elif fault == "axis":
        spec = replace(spec, scatter_indices_batching_dims=(2,))
    elif fault == "duplicate":
        spec = replace(
            spec, operand_batching_dims=(0, 0), scatter_indices_batching_dims=(0, 1)
        )
    elif fault == "clip":
        mode = lax.GatherScatterMode.CLIP
    with pytest.raises(NotImplementedError, match="batched point scatter"):
        _expand_batched_elementwise_indices(
            NoEmission(),
            indices_val=ir.Value(
                name="indices",
                type=ir.TensorType(ir.DataType.INT32),
                shape=ir.Shape(indices_shape),
            ),
            operand_shape=operand_shape,
            indices_shape=indices_shape,
            updates_shape=updates_shape,
            spec=spec,
            mode=mode,
        )
