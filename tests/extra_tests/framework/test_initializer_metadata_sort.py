# tests/extra_tests/framework/test_initializer_metadata_sort.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import to_onnx
from tests.extra_tests.framework.test_initializer_metadata_contract import (
    _assert_storage,
    _properties,
    _run,
    _tensors,
)
from tests.extra_tests.framework.test_initializer_metadata_optimizer import (
    _computation_profile,
    _output_expressions_without_annotations,
    _storage_profile,
)


@pytest.mark.parametrize("frontend", ["numpy", "lax"])
def test_nan_last_sort_preserves_annotated_capture_and_edited_source_uses(
    frontend: str,
) -> None:
    bits = np.asarray(
        [0x7FC01234, 0x7F800000, 0, 0x80000000, 0xFF800000, 0x40400000],
        dtype=np.uint32,
    )
    array = bits.view(np.float32)
    weight = jnp.asarray(array)
    tags = {"source": "sort.weight", "opaque/empty": ""}

    def ordered(x: jax.Array) -> jax.Array:
        return jnp.sort(x) if frontend == "numpy" else jax.lax.sort(x, is_stable=True)

    def apply(x: jax.Array) -> jax.Array:
        return ordered(jnp.concatenate((x, weight)))

    def reference(x: np.ndarray, source: np.ndarray) -> np.ndarray:
        return np.asarray(
            ordered(jnp.concatenate((jnp.asarray(x), jnp.asarray(source))))
        )

    ordinary = to_onnx(apply, [(4,)], output_names=["sorted_values"])
    model = to_onnx(
        apply,
        [(4,)],
        output_names=["sorted_values"],
        initializer_metadata=lambda source: tags if source is weight else None,
    )
    assert _computation_profile(model) == _computation_profile(ordinary)
    assert _output_expressions_without_annotations(
        model
    ) == _output_expressions_without_annotations(ordinary)
    assert _storage_profile(model) == _storage_profile(ordinary)
    tagged = [tensor for tensor in _tensors(model) if _properties(tensor)]
    assert len(tagged) == 1
    assert _properties(tagged[0]) == tags
    assert tuple(tagged[0].dims) == array.shape
    _assert_storage(tagged[0], array)
    x = np.asarray([np.inf, -1, 0.0, -0.0], dtype=np.float32)
    expected = reference(x, array)
    for exported in (ordinary, model):
        onnx.checker.check_model(exported, full_check=True)
        assert exported.graph.output[0].name == "sorted_values"
        actual = _run(exported, x)[0]
        assert actual.dtype == np.dtype(np.float32)
        assert actual.shape == (10,)
        # Sorting must move untouched elements, retaining zero and NaN bits.
        assert actual.tobytes() == expected.tobytes()
    replacement = np.asarray(
        [0xFFC05678, 0x40800000, 0x80000000, 0, 0x7F800000, 0xC0800000],
        dtype=np.uint32,
    ).view(np.float32)
    tagged[0].raw_data = replacement.tobytes()
    _assert_storage(tagged[0], replacement)
    assert _properties(tagged[0]) == tags
    onnx.checker.check_model(model, full_check=True)
    changed = _run(model, x)[0]
    assert changed.tobytes() == reference(x, replacement).tobytes()
    assert changed.tobytes() != expected.tobytes()
