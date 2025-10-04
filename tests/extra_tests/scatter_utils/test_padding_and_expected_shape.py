# tests/extra_tests/scatter_utils/test_padding_and_expected_shape.py

from __future__ import annotations

from jax import lax

from jax2onnx.plugins.jax.lax.scatter_utils import (
    ScatterSpec,
    _compute_window_operand_dims,
    _normalize_dimension_numbers,
    _resolve_operand_to_update_map,
)


def test_slice_scatter_window_metadata_matches_expected_shape():
    operand_shape = (5, 7, 4, 3)
    indices_shape = (5, 1)
    expected_updates_shape = (5, 5, 4, 3)

    spec = _normalize_dimension_numbers(
        lax.ScatterDimensionNumbers(
            update_window_dims=(1, 2, 3),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(1,),
        )
    )

    assert isinstance(spec, ScatterSpec)

    window_axes = _compute_window_operand_dims(spec, len(operand_shape))
    assert window_axes == (0, 2, 3)

    full_map, window_map = _resolve_operand_to_update_map(spec, len(operand_shape))
    assert full_map == {0: 1, 2: 2, 3: 3}
    assert window_map == {0: 1, 2: 2, 3: 3}

    batch_dims = indices_shape[:-1]
    window_dims = tuple(operand_shape[axis] for axis in window_axes)
    computed_updates_shape = batch_dims + window_dims

    assert (
        computed_updates_shape == expected_updates_shape
    ), "plugins scatter metadata should preserve the legacy window layout"


def test_scatter_window_map_excludes_scatter_axes():
    spec = _normalize_dimension_numbers(
        lax.ScatterDimensionNumbers(
            update_window_dims=(0, 1, 2, 3),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(1,),
        )
    )

    full_map, window_map = _resolve_operand_to_update_map(spec, operand_rank=4)
    assert full_map == {0: 0, 1: 1, 2: 2, 3: 3}
    assert window_map == {0: 0, 2: 2, 3: 3}
