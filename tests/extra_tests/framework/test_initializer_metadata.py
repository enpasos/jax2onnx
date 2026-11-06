# tests/extra_tests/framework/test_initializer_metadata.py

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
)


@pytest.fixture(autouse=True)
def _clear_local_function_registrations() -> Iterator[None]:
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
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                yield attr.t
            elif attr.type == onnx.AttributeProto.TENSORS:
                yield from attr.tensors
            elif attr.type == onnx.AttributeProto.GRAPH:
                yield from _graph_tensors(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for graph in attr.graphs:
                    yield from _graph_tensors(graph)


def _tensors(model: onnx.ModelProto) -> list[onnx.TensorProto]:
    return list(_graph_tensors(model.graph)) + [
        tensor for fn in model.functions for tensor in _node_tensors(fn.node)
    ]


def _props(tensor: onnx.TensorProto) -> dict[str, str]:
    return {prop.key: prop.value for prop in tensor.metadata_props}


def _run(model: onnx.ModelProto, x: np.ndarray) -> list[np.ndarray]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return cast(list[np.ndarray], session.run(None, {session.get_inputs()[0].name: x}))


@pytest.mark.parametrize("mode", ["proto", "ir", "file"])
@pytest.mark.parametrize("source_kind", ["jax", "numpy"])
def test_captured_initializer_metadata_preserves_identity_and_return_modes(
    mode: str, source_kind: str, tmp_path: Path
) -> None:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    weight = jnp.asarray(array) if source_kind == "jax" else array
    seen: list[object] = []
    metadata = {"application/source": "encoder.weight", "trainable": "false"}

    def annotate(source: object) -> Mapping[str, str] | None:
        seen.append(source)
        return MappingProxyType(metadata)

    def model(x: jax.Array) -> jax.Array:
        return x + weight

    inputs = [jax.ShapeDtypeStruct((2, 3), np.float32)]
    if mode == "ir":
        result = to_onnx(model, inputs, return_mode="ir", initializer_metadata=annotate)
        assert isinstance(result, ir.Model)
        assert any(
            value.const_value is not None
            and dict(value.const_value.metadata_props) == metadata
            for value in result.graph.initializers.values()
        )
        proto = ir.to_proto(result)
    elif mode == "file":
        destination = tmp_path / "annotated.onnx"
        path = to_onnx(
            model,
            inputs,
            return_mode="file",
            output_path=destination,
            initializer_metadata=annotate,
        )
        assert path == str(destination)
        proto = onnx.load(path)
    else:
        proto = to_onnx(model, inputs, initializer_metadata=annotate)

    assert seen
    if source_kind == "jax":
        assert all(source is weight for source in seen)
    else:
        # JAX can canonicalize NumPy closure values before exposing consts.
        for source in seen:
            np.testing.assert_array_equal(source, array)
    assert [_props(tensor) for tensor in proto.graph.initializer] == [metadata]
    assert not proto.metadata_props
    metadata["application/source"] = "changed after export"
    assert _props(proto.graph.initializer[0])["application/source"] == "encoder.weight"
    onnx.checker.check_model(proto, full_check=True)
    x = np.full((2, 3), 0.5, dtype=np.float32)
    np.testing.assert_allclose(_run(proto, x)[0], x + array)


def test_equal_distinct_captured_arrays_keep_separate_annotations() -> None:
    left = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    right = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    assert left is not right

    def annotate(source: object) -> Mapping[str, str] | None:
        if source is left:
            return {"source": "left"}
        if source is right:
            return {"source": "right"}
        return None

    model = to_onnx(
        lambda x: (x + left, x * right),
        [(2,)],
        initializer_metadata=annotate,
    )
    assert sorted(_props(tensor)["source"] for tensor in model.graph.initializer) == [
        "left",
        "right",
    ]
    x = np.asarray([3.0, 4.0], dtype=np.float32)
    actual = _run(model, x)
    np.testing.assert_allclose(actual[0], x + np.asarray(left))
    np.testing.assert_allclose(actual[1], x * np.asarray(right))


def test_jit_captured_initializer_metadata_receives_original_array() -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    seen: list[object] = []

    def annotate(source: object) -> Mapping[str, str] | None:
        seen.append(source)
        return {"source": "jit.weight"} if source is weight else None

    model = to_onnx(
        jax.jit(lambda x: x + weight), [(2,)], initializer_metadata=annotate
    )
    assert seen and all(source is weight for source in seen)
    assert any(_props(tensor) == {"source": "jit.weight"} for tensor in _tensors(model))


@pytest.mark.parametrize("wrapper", ["custom_jvp", "custom_vjp", "remat"])
def test_inlined_captured_initializer_metadata(wrapper: str) -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    seen: list[object] = []

    def fn(x: jax.Array) -> jax.Array:
        return x + weight

    if wrapper == "custom_jvp":
        target = jax.custom_jvp(fn)
        target.defjvp(lambda primals, tangents: (fn(primals[0]), tangents[0]))
    elif wrapper == "custom_vjp":
        target = jax.custom_vjp(fn)
        target.defvjp(lambda x: (fn(x), None), lambda _, gradient: (gradient,))
    else:
        target = jax.checkpoint(fn)

    def annotate(source: object) -> Mapping[str, str] | None:
        seen.append(source)
        return {"source": wrapper} if source is weight else None

    model = to_onnx(target, [(2,)], initializer_metadata=annotate)
    assert seen and all(source is weight for source in seen)
    assert any(_props(tensor) == {"source": wrapper} for tensor in _tensors(model))
    x = np.asarray([3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(_run(model, x)[0], x + np.asarray(weight))


def test_nested_onnx_functions_inherit_callback() -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    seen: list[object] = []

    def annotate(source: object) -> Mapping[str, str] | None:
        seen.append(source)
        return {"source": "nested.weight"} if source is weight else None

    @onnx_function
    class Inner:
        def __call__(self, x: jax.Array) -> jax.Array:
            return x + weight

    inner = Inner()

    @onnx_function
    class Outer:
        def __call__(self, x: jax.Array) -> jax.Array:
            return inner(x) * weight

    outer = Outer()

    model = to_onnx(lambda x: outer(x), [(2,)], initializer_metadata=annotate)
    assert len(model.functions) == 2
    assert len(seen) >= 2 and all(source is weight for source in seen)
    annotated = [tensor for tensor in _tensors(model) if _props(tensor)]
    assert len(annotated) == 2
    assert all(_props(tensor) == {"source": "nested.weight"} for tensor in annotated)
    onnx.checker.check_model(model, full_check=True)
    x = np.asarray([3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(_run(model, x)[0], (x + np.asarray(weight)) * weight)


@pytest.mark.parametrize("control_flow", ["cond", "scan", "scan_xs", "while", "fori"])
def test_control_flow_child_contexts_inherit_initializer_metadata(
    control_flow: str,
) -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    seen: list[object] = []

    def annotate(source: object) -> Mapping[str, str] | None:
        seen.append(source)
        return {"source": "body.weight"} if source is weight else None

    @onnx_function
    class Inner:
        def __call__(self, x: jax.Array) -> jax.Array:
            return x + weight

    inner = Inner()

    def fn(x: jax.Array) -> jax.Array:
        if control_flow == "cond":
            return jax.lax.cond(x[0] > 0, inner, lambda value: -inner(value), x)
        if control_flow == "scan":
            return jax.lax.scan(
                lambda carry, _: (inner(carry), None), x, None, length=2
            )[0]
        if control_flow == "scan_xs":
            return jax.lax.scan(
                lambda carry, value: (inner(carry) + value, None),
                x,
                jnp.zeros((2, 2), dtype=jnp.float32),
            )[0]
        if control_flow == "while":
            return jax.lax.while_loop(
                lambda state: state[1] < 2,
                lambda state: (inner(state[0]), state[1] + 1),
                (x, 0),
            )[0]
        return jax.lax.fori_loop(0, 2, lambda _, carry: inner(carry), x)

    model = to_onnx(fn, [(2,)], initializer_metadata=annotate)
    assert any(source is weight for source in seen)
    assert any(
        _props(tensor) == {"source": "body.weight"} for tensor in _tensors(model)
    )
    onnx.checker.check_model(model, full_check=True)
    x = np.asarray([3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(_run(model, x)[0], np.asarray(fn(jnp.asarray(x))))


@pytest.mark.parametrize(
    ("result", "error", "message"),
    [
        ([("source", "weight")], TypeError, "Mapping\\[str, str\\] or None"),
        ({1: "weight"}, TypeError, "keys must be nonempty strings"),
        ({"": "weight"}, ValueError, "keys must be nonempty strings"),
        ({"source": 1}, TypeError, "value for 'source' must be a string"),
        ({"source": None}, TypeError, "value for 'source' must be a string"),
    ],
)
def test_initializer_metadata_rejects_malformed_results(
    result: Any, error: type[Exception], message: str
) -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    with pytest.raises(error, match=message):
        to_onnx(lambda x: x + weight, [(2,)], initializer_metadata=lambda _: result)


def test_initializer_metadata_must_be_callable_before_tracing() -> None:
    def should_not_trace(x: jax.Array) -> jax.Array:
        raise AssertionError("validation must precede tracing")

    with pytest.raises(
        TypeError, match="initializer_metadata must be callable or None"
    ):
        to_onnx(should_not_trace, [(2,)], initializer_metadata=cast(Any, {}))


@pytest.mark.parametrize("nested", [False, True])
def test_initializer_metadata_callback_exceptions_fail_export(nested: bool) -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    class Target:
        def __call__(self, x: jax.Array) -> jax.Array:
            return x + weight

    target = onnx_function(Target)() if nested else Target()

    def annotate(source: object) -> None:
        raise RuntimeError("annotation callback exploded")

    with pytest.raises(RuntimeError, match="annotation callback exploded"):
        to_onnx(lambda x: target(x), [(2,)], initializer_metadata=annotate)


def test_initializer_metadata_is_optional_and_export_local() -> None:
    weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    def fn(x: jax.Array) -> jax.Array:
        return (x + weight).reshape((1, 2)) + 1.0

    baseline = to_onnx(fn, [(2,)])
    explicit_none = to_onnx(fn, [(2,)], initializer_metadata=None)
    returns_none = to_onnx(fn, [(2,)], initializer_metadata=lambda _: None)
    returns_empty = to_onnx(fn, [(2,)], initializer_metadata=lambda _: {})
    assert baseline == explicit_none == returns_none == returns_empty
    seen: list[object] = []

    def annotate(source: object) -> Mapping[str, str]:
        seen.append(source)
        return {"source": "only the captured array"}

    annotated = to_onnx(fn, [(2,)], initializer_metadata=annotate)
    assert seen and all(source is weight for source in seen)
    assert any(_props(tensor) for tensor in _tensors(annotated))
    assert all(not _props(tensor) for tensor in _tensors(to_onnx(fn, [(2,)])))


def test_captured_binding_invokes_callback_before_array_conversion() -> None:
    class Source:
        converted = False

        def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
            self.converted = True
            return np.asarray([1.0, 2.0], dtype=np.float32)

    source = Source()
    metadata = {"source": "original"}

    def annotate(value: object) -> Mapping[str, str]:
        assert value is source
        assert not source.converted
        return metadata

    ctx = IRContext(opset=23, initializer_metadata=annotate)
    value = ctx.bind_captured_const_for_var(object(), source)
    metadata["source"] = "mutated"
    assert value.const_value is not None
    assert dict(value.const_value.metadata_props) == {"source": "original"}
    assert source.converted
    ctx.bind_const_for_var(object(), np.asarray([3.0], dtype=np.float32))
