# tests/extra_tests/framework/test_initializer_metadata_contract.py

from __future__ import annotations

from collections.abc import Iterator, Mapping
import gc
from types import MappingProxyType
from typing import Any, cast
import weakref

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import InitializerMetadataError, onnx_function, to_onnx
from jax2onnx import user_interface
from jax2onnx.converter.initializer_metadata import active_registry
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


def _graph_tensors(graph: onnx.GraphProto) -> Iterator[onnx.TensorProto]:
    yield from graph.initializer
    yield from _node_tensors(graph.node)


def _node_tensors(nodes: Any) -> Iterator[onnx.TensorProto]:
    for node in nodes:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from _graph_tensors(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph in attribute.graphs:
                    yield from _graph_tensors(graph)


def _tensors(model: onnx.ModelProto) -> list[onnx.TensorProto]:
    return list(_graph_tensors(model.graph)) + [
        tensor
        for function in model.functions
        for tensor in _node_tensors(function.node)
    ]


def _properties(tensor: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.metadata_props}


def _run(model: onnx.ModelProto, x: np.ndarray) -> list[np.ndarray]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if x.size == 0:
        # ORT 1.24.1's optimizer crashes on some legal empty reshape chains.
        # Keep checking their actual kernels; external optimizers are outside IM-07.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return cast(list[np.ndarray], session.run(None, {session.get_inputs()[0].name: x}))


def _assert_storage(tensor: onnx.TensorProto, expected: np.ndarray) -> None:
    actual = onnx.numpy_helper.to_array(tensor)
    assert actual.dtype == expected.dtype
    assert actual.size == expected.size
    # Equality/allclose cannot distinguish signed zeros or NaN payloads.
    assert (
        np.ascontiguousarray(actual).tobytes()
        == np.ascontiguousarray(expected).tobytes()
    )


def test_opaque_utf8_entries_are_copied_during_export() -> None:
    left = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    right = jnp.asarray([5.0, 7.0], dtype=jnp.float32)
    shared = {"应用/❄": "café\x00終", "trainable": "opaque", "source_path": ""}
    snapshots: dict[int, dict[str, str]] = {}

    def annotate(source: object) -> Mapping[str, str]:
        if id(source) in snapshots:
            return snapshots[id(source)]
        if snapshots:
            # Mutate the previously returned mapping while export is still running.
            shared["应用/❄"] = "later mutation"
        snapshots[id(source)] = dict(shared)
        return MappingProxyType(shared)

    model = to_onnx(
        lambda x: (x + left, x * right), [(2,)], initializer_metadata=annotate
    )
    assert set(snapshots) == {id(left), id(right)}
    carriers = model.graph.initializer
    assert len(carriers) == 2
    for tensor in carriers:
        array = onnx.numpy_helper.to_array(tensor)
        source = left if np.array_equal(array, left) else right
        assert _properties(tensor) == snapshots[id(source)]
    assert {tuple(sorted(_properties(t).items())) for t in carriers} == {
        tuple(sorted(snapshot.items())) for snapshot in snapshots.values()
    }
    onnx.checker.check_model(model, full_check=True)


def test_precapture_copy_does_not_acquire_original_objects_annotation() -> None:
    original = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    captured_copy = jnp.array(original, copy=True)
    assert captured_copy is not original
    observed: list[object] = []

    def annotate(source: object) -> dict[str, str] | None:
        observed.append(source)
        return {"source": "original"} if source is original else None

    def target(x):
        return x + captured_copy

    model = to_onnx(target, [(2,)], initializer_metadata=annotate)
    assert observed and all(source is captured_copy for source in observed)
    assert model == to_onnx(target, [(2,)])
    assert all(not _properties(tensor) for tensor in _tensors(model))


@pytest.mark.parametrize("result", [{"\ud800": "value"}, {"key": "\udfff"}])
def test_unencodable_callback_text_is_rejected(result: dict[str, str]) -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    with pytest.raises(ValueError):
        to_onnx(lambda x: x + weight, [(2,)], initializer_metadata=lambda _: result)


class _LyingText(str):
    def encode(self, *args: Any, **kwargs: Any) -> bytes:
        return b"pretend this is encodable"

    def __str__(self) -> str:
        return "spoofed text"


@pytest.mark.parametrize(
    "result", [{_LyingText("\ud800"): "value"}, {"key": _LyingText("\udfff")}]
)
def test_str_subclass_cannot_hide_unencodable_underlying_text(
    result: dict[str, str],
) -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    with pytest.raises(ValueError):
        to_onnx(lambda x: x + weight, [(2,)], initializer_metadata=lambda _: result)


def test_str_subclass_properties_are_copied_as_exact_plain_text() -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    model = to_onnx(
        lambda x: x + weight,
        [(2,)],
        return_mode="ir",
        initializer_metadata=lambda _: {_LyingText("应用/❄"): _LyingText("café\x00終")},
    )
    carriers = [
        value.const_value
        for value in model.graph.initializers.values()
        if value.const_value is not None
    ]
    assert len(carriers) == 1
    properties = carriers[0].metadata_props
    assert dict(properties) == {"应用/❄": "café\x00終"}
    assert all(
        type(key) is str and type(value) is str for key, value in properties.items()
    )


@pytest.mark.parametrize(
    "result", [False, 1, "metadata", [("key", "value")], {None: "v"}, {"k": b"v"}]
)
def test_invalid_callback_values_raise_type_error(result: Any) -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    with pytest.raises(TypeError):
        to_onnx(lambda x: x + weight, [(2,)], initializer_metadata=lambda _: result)


def _repeated_capture(weight: jax.Array) -> Any:
    @onnx_function
    class Inner:
        def __call__(self, x: jax.Array) -> jax.Array:
            return x + weight

    inner = Inner()
    return lambda x: (x + weight, inner(x))


@pytest.mark.parametrize("results", [(None, {}), ({}, None)])
def test_repeated_none_and_empty_snapshots_are_equivalent(
    results: tuple[Any, Any],
) -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    target = _repeated_capture(weight)
    baseline = to_onnx(target, [(2,)])
    calls = 0

    def annotate(source: object) -> Any:
        nonlocal calls
        assert source is weight
        result = results[calls % 2]
        calls += 1
        return result

    model = to_onnx(target, [(2,)], initializer_metadata=annotate)
    assert calls >= 2
    assert model == baseline
    assert all(not _properties(tensor) for tensor in _tensors(model))


@pytest.mark.parametrize(
    "results",
    [({"key": "one"}, {"key": "two"}), (None, {"key": "one"}), ({"key": "one"}, {})],
)
def test_repeated_capture_inconsistency_fails_and_does_not_poison_next_export(
    results: tuple[Any, Any],
) -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    target = _repeated_capture(weight)
    calls = 0

    def annotate(source: object) -> Any:
        nonlocal calls
        assert source is weight
        result = results[min(calls, 1)]
        calls += 1
        return result

    with pytest.raises(InitializerMetadataError) as error:
        to_onnx(target, [(2,)], initializer_metadata=annotate)
    assert isinstance(error.value, ValueError)
    assert error.value.reason == "inconsistent_callback"
    assert calls >= 2
    clean = to_onnx(target, [(2,)], initializer_metadata=lambda _: {"clean": "yes"})
    carriers = _tensors(clean)
    assert len(carriers) == 2
    assert all(_properties(tensor) == {"clean": "yes"} for tensor in carriers)
    plain = to_onnx(target, [(2,)])
    assert all(not _properties(tensor) for tensor in _tensors(plain))
    x = np.asarray([3.0, 4.0], dtype=np.float32)
    for result in _run(clean, x):
        np.testing.assert_array_equal(result, x + np.asarray(weight))


_DTYPES = [
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
]


def _source_array(dtype_name: str, shape: str) -> np.ndarray:
    dtype = np.dtype(ml_dtypes.bfloat16 if dtype_name == "bfloat16" else dtype_name)
    if shape == "empty":
        return np.empty((2, 0, 1), dtype=dtype)
    if dtype_name in {"float16", "bfloat16", "float32", "float64"}:
        if dtype_name == "float16":
            bits = np.asarray(
                [0, 0x8000, 0x7E01, 0xFE12, 0x3C00, 0xBC00], dtype=np.uint16
            )
        elif dtype_name == "bfloat16":
            bits = np.asarray(
                [0, 0x8000, 0x7FC1, 0xFFD2, 0x3F80, 0xBF80], dtype=np.uint16
            )
        elif dtype_name == "float32":
            bits = np.asarray(
                [0, 0x80000000, 0x7FC00001, 0xFFC01234, 0x3F800000, 0xBF800000],
                dtype=np.uint32,
            )
        else:
            bits = np.asarray(
                [
                    0,
                    0x8000000000000000,
                    0x7FF8000000000001,
                    0xFFF8000000001234,
                    0x3FF0000000000000,
                    0xBFF0000000000000,
                ],
                dtype=np.uint64,
            )
        values = bits.view(dtype)
    elif dtype_name == "bool":
        values = np.asarray([False, True, True, False, True, False], dtype=dtype)
    else:
        limits = np.iinfo(dtype)
        values = np.asarray([0, 1, limits.min, limits.max, 2, 3], dtype=dtype)
    return values[1].reshape(()) if shape == "scalar" else values.reshape(2, 3)


@pytest.mark.parametrize(
    "dtype_name,x64",
    [
        (dtype, x64)
        for dtype in _DTYPES
        for x64 in (False, True)
        if x64 or dtype not in {"int64", "uint64", "float64"}
    ],
)
@pytest.mark.parametrize("shape", ["scalar", "empty", "dense"])
def test_all_promised_dense_storage_is_exact(
    dtype_name: str, shape: str, x64: bool
) -> None:
    with jax.enable_x64(x64):
        original = _source_array(dtype_name, shape)
        weight = jnp.asarray(original)
        reference = np.asarray(weight)
        assert reference.dtype == original.dtype
        assert reference.tobytes() == original.tobytes()
        entries = {"dtype": dtype_name, "shape": shape, "empty-value": ""}
        seen: list[object] = []

        def annotate(source: object) -> dict[str, str]:
            seen.append(source)
            assert source is weight
            return entries

        model = to_onnx(lambda x: (x, weight), [(1,)], initializer_metadata=annotate)
        assert seen
        carriers = _tensors(model)
        assert len(carriers) == 1
        assert _properties(carriers[0]) == entries
        assert tuple(carriers[0].dims) == reference.shape
        _assert_storage(carriers[0], reference)
        onnx.checker.check_model(model, full_check=True)
        if dtype_name != "bfloat16":
            # ORT's Python API cannot expose BF16 outputs; storage is still required.
            actual = _run(model, np.asarray([1.0], dtype=np.float32))[1]
            assert actual.dtype == reference.dtype
            assert actual.shape == reference.shape
            assert actual.tobytes() == reference.tobytes()


def test_annotated_complex_storage_is_rejected_without_changing_plain_export() -> None:
    weight = jnp.asarray([1.0 + 2.0j, 3.0 - 4.0j], dtype=jnp.complex64)

    def target(x):
        return x + jnp.real(weight)

    plain = to_onnx(target, [(2,)])
    x = np.asarray([5.0, 6.0], dtype=np.float32)
    np.testing.assert_array_equal(_run(plain, x)[0], np.asarray(target(jnp.asarray(x))))
    with pytest.raises(InitializerMetadataError) as error:
        to_onnx(target, [(2,)], initializer_metadata=lambda _: {"source": "complex"})
    assert error.value.reason == "unsupported_representation"
    skipped = to_onnx(target, [(2,)], initializer_metadata=lambda _: {})
    assert skipped == plain


@pytest.mark.parametrize("annotate_right", [False, True])
def test_equal_sources_keep_independent_uses_with_identical_tags_or_untagged_neighbor(
    annotate_right: bool,
) -> None:
    left = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    right = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    assert left is not right
    entries = {"opaque": "identical source labels"}

    def annotate(source: object) -> dict[str, str] | None:
        return (
            entries if source is left or (annotate_right and source is right) else None
        )

    def apply(x: jax.Array, a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
        return x + a, x * b

    model = to_onnx(
        lambda x: apply(x, left, right), [(2,)], initializer_metadata=annotate
    )
    assert len(model.graph.initializer) == 2
    operations = {
        node.op_type: node
        for node in model.graph.node
        if node.op_type in {"Add", "Mul"}
    }
    left_name = operations["Add"].input[1]
    right_name = operations["Mul"].input[1]
    assert left_name != right_name
    carriers = {tensor.name: tensor for tensor in model.graph.initializer}
    assert _properties(carriers[left_name]) == entries
    assert _properties(carriers[right_name]) == (entries if annotate_right else {})
    x = np.asarray([5.0, 7.0], dtype=np.float32)
    for actual, expected in zip(_run(model, x), apply(jnp.asarray(x), left, right)):
        np.testing.assert_array_equal(actual, expected)
    replacement = np.asarray([11.0, 13.0], dtype=np.float32)
    changed = onnx.numpy_helper.from_array(replacement, name=left_name)
    changed.metadata_props.extend(carriers[left_name].metadata_props)
    carriers[left_name].CopyFrom(changed)
    onnx.checker.check_model(model, full_check=True)
    # This observes artifact dataflow, independently of tags or the private registry.
    for actual, expected in zip(
        _run(model, x), apply(jnp.asarray(x), jnp.asarray(replacement), right)
    ):
        np.testing.assert_array_equal(actual, expected)


def test_returned_ir_does_not_retain_callback_or_export_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    registry_references: list[weakref.ReferenceType[Any]] = []
    original = user_interface.postprocess_ir_model

    def inspect_registry(model: ir.Model) -> Any:
        registry = active_registry()
        assert registry is not None
        registry_references.append(weakref.ref(registry))
        return original(model)

    class Annotate:
        def __call__(self, source: object) -> dict[str, str]:
            assert source is weight
            return {"source": "weight"}

    monkeypatch.setattr(user_interface, "postprocess_ir_model", inspect_registry)
    callback = Annotate()
    callback_reference = weakref.ref(callback)
    model = to_onnx(
        lambda x: x + weight, [(2,)], return_mode="ir", initializer_metadata=callback
    )
    del callback
    gc.collect()
    assert callback_reference() is None
    assert registry_references and all(
        reference() is None for reference in registry_references
    )
    assert active_registry() is None
    carriers = _tensors(ir.to_proto(model))
    assert len(carriers) == 1
    assert _properties(carriers[0]) == {"source": "weight"}


def test_failed_export_releases_callback_and_registry_before_clean_export() -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    registry_references: list[weakref.ReferenceType[Any]] = []

    class FailingAnnotation:
        def __call__(self, source: object) -> None:
            assert source is weight
            registry = active_registry()
            assert registry is not None
            registry_references.append(weakref.ref(registry))
            raise RuntimeError("caller-owned failure text")

    callback = FailingAnnotation()
    callback_reference = weakref.ref(callback)
    with pytest.raises(RuntimeError, match="caller-owned failure text"):
        to_onnx(lambda x: x + weight, [(2,)], initializer_metadata=callback)
    del callback
    gc.collect()
    assert callback_reference() is None
    assert registry_references and all(
        reference() is None for reference in registry_references
    )
    assert active_registry() is None
    clean = to_onnx(
        lambda x: x + weight, [(2,)], initializer_metadata=lambda _: {"fresh": "yes"}
    )
    assert [_properties(tensor) for tensor in _tensors(clean)] == [{"fresh": "yes"}]


def test_nested_export_in_callback_restores_outer_session() -> None:
    outer_weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    inner_weight = jnp.asarray([3.0, 4.0], dtype=jnp.float32)
    inner_models: list[onnx.ModelProto] = []

    def annotate(source: object) -> dict[str, str]:
        assert source is outer_weight
        if not inner_models:
            outer_registry = active_registry()
            inner_models.append(
                to_onnx(
                    lambda x: x * inner_weight,
                    [(2,)],
                    initializer_metadata=lambda value: (
                        {"source": "inner"} if value is inner_weight else None
                    ),
                )
            )
            assert active_registry() is outer_registry
        return {"source": "outer"}

    outer = to_onnx(lambda x: x + outer_weight, [(2,)], initializer_metadata=annotate)
    assert active_registry() is None
    assert len(inner_models) == 1
    for model, label in ((outer, "outer"), (inner_models[0], "inner")):
        carriers = _tensors(model)
        assert len(carriers) == 1
        assert _properties(carriers[0]) == {"source": label}
    x = np.asarray([5.0, 6.0], dtype=np.float32)
    np.testing.assert_array_equal(_run(outer, x)[0], x + np.asarray(outer_weight))
    np.testing.assert_array_equal(
        _run(inner_models[0], x)[0], x * np.asarray(inner_weight)
    )
