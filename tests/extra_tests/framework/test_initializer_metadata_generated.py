# tests/extra_tests/framework/test_initializer_metadata_generated.py

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import InitializerMetadataError, onnx_function, to_onnx
from jax2onnx.plugins.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
)
from tests.extra_tests.framework.test_initializer_metadata_contract import (
    _assert_storage,
    _properties,
    _run,
    _tensors,
)
from tests.extra_tests.framework.test_initializer_metadata_faults import _inject


@pytest.fixture(autouse=True)
def _clear_local_functions() -> Iterator[None]:
    yield
    for qualified in tuple(ONNX_FUNCTION_PLUGIN_REGISTRY):
        if qualified.startswith(f"{__name__}."):
            PLUGIN_REGISTRY.pop(f"onnx_fn::{qualified}", None)
            ONNX_FUNCTION_PLUGIN_REGISTRY.pop(qualified, None)


# Bounded seeds cover rank 0–3, singleton/empty dimensions, 1–3 sources,
# equal/distinct/shared sources, strided NumPy captures, and 0–2 Function scopes.
_CASES = [
    (0, (), 1, 0, "jax"),
    (1, (1,), 1, 1, "numpy_strided"),
    (2, (0,), 2, 2, "jax"),
    (3, (1, 3), 3, 0, "jax"),
    (4, (0, 1), 2, 1, "jax"),
    (5, (2, 1, 3), 1, 2, "numpy_contiguous"),
    (6, (1, 0, 2), 1, 0, "numpy_strided"),
    (7, (1, 1, 1), 3, 1, "jax"),
    (8, (2, 3), 1, 1, "numpy_strided"),
    (9, (1, 2, 3), 3, 2, "jax"),
]


def _ir_tensors(model: ir.Model) -> Iterator[ir.TensorProtocol]:
    for value in model.graph.initializers.values():
        if value.const_value is not None:
            yield value.const_value
    for graph in [
        model.graph,
        *(function.graph for function in model.functions.values()),
    ]:
        for node in graph:
            for attribute in node.attributes.values():
                if attribute.type == ir.AttributeType.TENSOR:
                    yield attribute.as_tensor()
                elif attribute.type == ir.AttributeType.TENSORS:
                    yield from attribute.as_tensors()


@pytest.mark.parametrize("seed,shape,count,depth,kind", _CASES)
@pytest.mark.parametrize("corrupt", [False, True], ids=["valid", "one_fault"])
def test_bounded_generated_capture_storage_and_use_contract(
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    shape: tuple[int, ...],
    count: int,
    depth: int,
    kind: str,
    corrupt: bool,
) -> None:
    random = np.random.default_rng(seed)
    size = int(np.prod(shape))
    sources: list[Any] = []
    references: dict[str, np.ndarray] = {}
    for index in range(count):
        array = random.normal(size=shape).astype(np.float32)
        if index == 1 and seed % 2:
            array = np.array(references["source-0"], copy=True)
        if kind == "numpy_strided":
            backing = np.empty((max(1, size) * 2,), dtype=np.float32)
            backing[::2][:size] = array.reshape(-1)
            source = backing[::2][:size].reshape(shape)
            if size > 1:
                assert not source.flags.c_contiguous
        elif kind == "numpy_contiguous":
            source = np.ascontiguousarray(array).reshape(shape)
        else:
            source = jnp.asarray(array)
        sources.append(source)
        references[f"source-{index}"] = np.asarray(source)
    seen: set[str] = set()

    def annotate(source: object) -> dict[str, str]:
        if kind == "jax":
            name = next(
                (f"source-{i}" for i, value in enumerate(sources) if source is value),
                None,
            )
            assert name is not None
        else:
            # JAX may canonicalize NumPy objects; test the actual capture boundary.
            name = "source-0"
            assert np.asarray(source).dtype == references[name].dtype
            assert np.asarray(source).shape == references[name].shape
            assert np.asarray(source).tobytes() == references[name].tobytes()
        seen.add(name)
        return {"source": name, "seed": str(seed), "opaque": ""}

    def body(x: jax.Array) -> jax.Array:
        # Each source is reused and only its storage reshape may be folded.
        result = x
        for source in sources:
            result = (result + jnp.reshape(source, (size,))) * 0.5 + jnp.reshape(
                source, (size,)
            )
        return result

    @onnx_function
    class Inner:
        def __call__(self, x: jax.Array) -> jax.Array:
            return body(x)

    inner = Inner()

    @onnx_function
    class Outer:
        def __call__(self, x: jax.Array) -> jax.Array:
            return inner(x)

    target = body if depth == 0 else inner if depth == 1 else Outer()
    calls: list[ir.Model] = []
    if corrupt:

        def mutate(model: ir.Model) -> None:
            carriers = [
                tensor for tensor in _ir_tensors(model) if tensor.metadata_props
            ]
            assert carriers
            del carriers[0].metadata_props["opaque"]

        calls = _inject(monkeypatch, mutate)
        with pytest.raises(InitializerMetadataError) as error:
            to_onnx(target, [(size,)], initializer_metadata=annotate)
        assert error.value.reason == "preservation_failed"
        assert len(calls) == 1
        return

    model = to_onnx(target, [(size,)], initializer_metadata=annotate)
    assert seen == set(references)
    carriers = [tensor for tensor in _tensors(model) if _properties(tensor)]
    assert {_properties(tensor)["source"] for tensor in carriers} == set(references)
    for tensor in carriers:
        name = _properties(tensor)["source"]
        assert _properties(tensor) == {"source": name, "seed": str(seed), "opaque": ""}
        _assert_storage(tensor, references[name])
    onnx.checker.check_model(model, full_check=True)
    x = np.linspace(0.25, 0.75, size, dtype=np.float32)
    expected = body(jnp.asarray(x))
    actual = _run(model, x)
    assert len(actual) == 1
    np.testing.assert_allclose(actual[0], expected, rtol=1e-6, atol=1e-6)
