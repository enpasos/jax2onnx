# tests/extra_tests/framework/test_initializer_metadata_function_identity.py

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import InitializerMetadataError, onnx_function, to_onnx
from jax2onnx.plugins.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
)
from jax2onnx.plugins import _function_keys
from tests.extra_tests.framework.test_initializer_metadata_contract import (
    _assert_storage,
    _properties,
    _run,
    _tensors,
)


@pytest.fixture(autouse=True)
def _clear_local_functions() -> Iterator[None]:
    yield
    for qualified in tuple(ONNX_FUNCTION_PLUGIN_REGISTRY):
        if qualified.startswith(f"{__name__}."):
            PLUGIN_REGISTRY.pop(f"onnx_fn::{qualified}", None)
            ONNX_FUNCTION_PLUGIN_REGISTRY.pop(qualified, None)


def _blocks(left: jax.Array, right: jax.Array, *, nested: bool) -> tuple[Any, Any]:
    @onnx_function(unique=True)
    @jax.tree_util.register_pytree_node_class
    class Inner:
        def __init__(self, weight: jax.Array) -> None:
            self.weight = weight

        def __call__(self, x: jax.Array) -> jax.Array:
            return x + self.weight

        def tree_flatten(self) -> tuple[tuple[jax.Array], None]:
            return (self.weight,), None

        @classmethod
        def tree_unflatten(cls, auxiliary: None, children: tuple[jax.Array]) -> Any:
            return cls(children[0])

    first, second = Inner(left), Inner(right)
    if not nested:
        return first, second

    @onnx_function(unique=True)
    @jax.tree_util.register_pytree_node_class
    class Outer:
        def __init__(self, child: Any) -> None:
            self.child = child

        def __call__(self, x: jax.Array) -> jax.Array:
            return self.child(x)

        def tree_flatten(self) -> tuple[tuple[Any], None]:
            return (self.child,), None

        @classmethod
        def tree_unflatten(cls, auxiliary: None, children: tuple[Any]) -> Any:
            return cls(children[0])

    return Outer(first), Outer(second)


class _Resolved(NamedTuple):
    expression: tuple[Any, ...]
    tensors: tuple[onnx.TensorProto, ...]


def _tensor_expression(tensor: onnx.TensorProto) -> tuple[Any, ...]:
    array = onnx.numpy_helper.to_array(tensor)
    return (
        "tensor",
        tensor.data_type,
        tuple(tensor.dims),
        array.tobytes(order="C"),
        tuple(sorted(_properties(tensor).items())),
    )


def _resolve_outputs(model: onnx.ModelProto) -> list[_Resolved]:
    """Expand call dependencies without copying or editing the public artifact.

    The expression ignores generated names and storage sharing. The tensor
    references point to the original protobuf so mutation tests edit its actual
    storage, including a Constant several nested Function calls away.
    """
    functions = {
        (function.domain, function.name, function.overload): function
        for function in model.functions
    }

    def scope(
        nodes: Any, initializers: Any, bindings: dict[str, Callable[[], _Resolved]]
    ) -> Callable[[str], _Resolved]:
        producers = {
            name: (node, index)
            for node in nodes
            for index, name in enumerate(node.output)
        }
        constants = {tensor.name: tensor for tensor in initializers}

        def resolve(name: str) -> _Resolved:
            if not name:
                return _Resolved(("omitted",), ())
            if name in constants:
                tensor = constants[name]
                return _Resolved(_tensor_expression(tensor), (tensor,))
            if name in bindings:
                return bindings[name]()
            node, output_index = producers[name]
            function = functions.get((node.domain, node.op_type, node.overload))
            if function is not None:
                arguments = {
                    formal: (lambda actual=actual: resolve(actual))
                    for formal, actual in zip(function.input, node.input)
                }
                child = scope(function.node, (), arguments)
                return child(function.output[output_index])
            if node.op_type == "Constant":
                attributes = [
                    attribute
                    for attribute in node.attribute
                    if attribute.type == onnx.AttributeProto.TENSOR
                ]
                assert len(attributes) == 1
                tensor = attributes[0].t
                return _Resolved(_tensor_expression(tensor), (tensor,))
            arguments = [resolve(argument) for argument in node.input]
            attributes = tuple(
                sorted(
                    (attribute.name, attribute.SerializeToString(deterministic=True))
                    for attribute in node.attribute
                )
            )
            return _Resolved(
                (
                    node.domain,
                    node.op_type,
                    output_index,
                    tuple(argument.expression for argument in arguments),
                    attributes,
                ),
                tuple(tensor for argument in arguments for tensor in argument.tensors),
            )

        return resolve

    inputs = {
        value.name: (
            lambda index=index, value=value: _Resolved(
                ("input", index, value.type.SerializeToString(deterministic=True)), ()
            )
        )
        for index, value in enumerate(model.graph.input)
    }
    resolve = scope(model.graph.node, model.graph.initializer, inputs)
    return [resolve(output.name) for output in model.graph.output]


def _one_carrier(output: _Resolved) -> onnx.TensorProto:
    unique = {id(tensor): tensor for tensor in output.tensors}
    assert len(unique) == 1
    return next(iter(unique.values()))


@pytest.mark.parametrize("nested", [False, True], ids=["unique", "nested_unique"])
@pytest.mark.parametrize(
    "pattern", ["both", "annotate_first", "skip_first", "all_skip"]
)
def test_unique_pytree_equal_instances_preserve_each_capture_and_its_uses(
    nested: bool, pattern: str
) -> None:
    left = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    right = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    assert left is not right
    first, second = _blocks(left, right, nested=nested)

    def target(x):
        return (first(x), second(x))

    baseline = to_onnx(target, [(2,)])
    observed: list[object] = []
    left_entries = (
        {"source": "identical opaque tag"}
        if pattern in {"both", "annotate_first"}
        else {}
    )
    right_entries = (
        {"source": "identical opaque tag"} if pattern in {"both", "skip_first"} else {}
    )

    def annotate(source: object) -> dict[str, str] | None:
        observed.append(source)
        assert source is left or source is right
        if source is left:
            return left_entries or None
        return right_entries

    model = to_onnx(target, [(2,)], initializer_metadata=annotate)
    # Both distinct callees must expose their captures, including a skipped first
    # callee and constants only discovered while tracing a nested Function.
    assert {id(source) for source in observed} == {id(left), id(right)}
    outputs = _resolve_outputs(model)
    assert len(outputs) == 2
    carriers = [_one_carrier(output) for output in outputs]
    assert _properties(carriers[0]) == left_entries
    assert _properties(carriers[1]) == right_entries
    for tensor, source in zip(carriers, (left, right)):
        _assert_storage(tensor, np.asarray(source))
    x = np.asarray([5.0, 7.0], dtype=np.float32)
    onnx.checker.check_model(baseline, full_check=True)
    onnx.checker.check_model(model, full_check=True)
    for result in _run(model, x):
        np.testing.assert_array_equal(result, x + np.asarray(left))
    if pattern == "all_skip":
        assert all(not _properties(tensor) for tensor in _tensors(model))
        assert [output.expression for output in outputs] == [
            output.expression for output in _resolve_outputs(baseline)
        ]
        assert not model.metadata_props and not baseline.metadata_props
        return

    assert carriers[0] is not carriers[1]
    # Select by the output's actual Function dependencies, not by metadata or
    # exporter bookkeeping. Only the source of that output may change.
    changed_index = 1 if pattern == "skip_first" else 0
    victim = carriers[changed_index]
    replacement = np.asarray([11.0, 13.0], dtype=np.float32)
    changed = onnx.numpy_helper.from_array(replacement, name=victim.name)
    changed.metadata_props.extend(victim.metadata_props)
    victim.CopyFrom(changed)
    onnx.checker.check_model(model, full_check=True)
    expected_sources = (
        (replacement, np.asarray(right))
        if changed_index == 0
        else (np.asarray(left), replacement)
    )
    for result, expected_source in zip(_run(model, x), expected_sources):
        np.testing.assert_array_equal(result, x + expected_source)


@pytest.mark.parametrize("nested", [False, True], ids=["unique", "nested_unique"])
@pytest.mark.parametrize("reuse", ["same_instance", "same_source"])
def test_unique_pytree_reuse_preserves_stable_same_source_annotations(
    nested: bool, reuse: str
) -> None:
    weight = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    first, second = _blocks(weight, weight, nested=nested)
    if reuse == "same_instance":
        second = first
    observed: list[object] = []

    def annotate(source: object) -> dict[str, str]:
        observed.append(source)
        assert source is weight
        return {"source": "shared weight"}

    model = to_onnx(
        lambda x: (first(x), second(x)), [(2,)], initializer_metadata=annotate
    )
    assert observed
    outputs = _resolve_outputs(model)
    assert len(outputs) == 2
    for output in outputs:
        tensor = _one_carrier(output)
        assert _properties(tensor) == {"source": "shared weight"}
        _assert_storage(tensor, np.asarray(weight))
    x = np.asarray([5.0, 7.0], dtype=np.float32)
    onnx.checker.check_model(model, full_check=True)
    for result in _run(model, x):
        np.testing.assert_array_equal(result, x + np.asarray(weight))


@pytest.mark.parametrize("nested", [False, True], ids=["unique", "nested_unique"])
@pytest.mark.parametrize(
    "results",
    [
        ({"source": "first"}, {"source": "second"}),
        (None, {"source": "accepted"}),
        ({"source": "accepted"}, None),
    ],
)
def test_unique_pytree_observed_same_source_results_must_remain_consistent(
    nested: bool, results: tuple[Any, Any]
) -> None:
    weight = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    first, _ = _blocks(weight, weight, nested=nested)
    observed: list[object] = []

    def annotate(source: object) -> Any:
        assert source is weight
        result = results[min(len(observed), 1)]
        observed.append(source)
        return result

    # The root capture and the Function capture force observed repeated results;
    # reusing the same callee alone need not retrace an already built body.
    with pytest.raises(InitializerMetadataError) as error:
        to_onnx(
            lambda x: (first(x), first(x), x + weight),
            [(2,)],
            initializer_metadata=annotate,
        )
    assert len(observed) >= 2
    assert error.value.reason == "inconsistent_callback"


@pytest.mark.parametrize("accepted", [False, True])
def test_nested_capture_callback_precedes_function_key_numpy_reads(
    monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    weight = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    first, _ = _blocks(weight, weight, nested=True)
    observed: list[object] = []
    reads: list[object] = []

    def asarray(value: object, *args: Any, **kwargs: Any) -> Any:
        if value is weight:
            reads.append(value)
            assert (
                observed
            ), "Function content fingerprint read source before its capture callback"
        return np.asarray(value, *args, **kwargs)

    # Scope the spy to the exporter's actual function-key NumPy calls so JAX's
    # own tracing reads are outside this capture-boundary assertion.
    monkeypatch.setattr(
        _function_keys,
        "np",
        SimpleNamespace(
            asarray=asarray,
            ndarray=np.ndarray,
            generic=np.generic,
            dtype=np.dtype,
            ascontiguousarray=np.ascontiguousarray,
            uint8=np.uint8,
            frombuffer=np.frombuffer,
        ),
    )

    def annotate(source: object) -> dict[str, str] | None:
        assert source is weight
        observed.append(source)
        return {"source": "weight"} if accepted else None

    model = to_onnx(lambda x: first(x), [(2,)], initializer_metadata=annotate)
    assert observed
    if not accepted:
        assert reads, "The all-skip content-cache path must exercise the NumPy-read spy"
    tensor = _one_carrier(_resolve_outputs(model)[0])
    assert _properties(tensor) == ({"source": "weight"} if accepted else {})
    _assert_storage(tensor, np.asarray(weight))


def test_nonunique_function_static_array_configuration_keeps_distinct_captured_sources() -> (
    None
):
    left = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    right = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    observed: list[object] = []

    @onnx_function
    class Block:
        def __call__(
            self, x: jax.Array, *, configuration: dict[str, jax.Array]
        ) -> jax.Array:
            return x + configuration["weight"]

    block = Block()

    def target(x):
        return (
            block(x, configuration={"weight": left}),
            block(x, configuration={"weight": right}),
        )

    baseline = to_onnx(target, [(2,)])

    def annotate(source: object) -> dict[str, str]:
        observed.append(source)
        assert source is left or source is right
        return {"source": "same opaque tag"}

    model = to_onnx(target, [(2,)], initializer_metadata=annotate)
    assert {id(source) for source in observed} == {id(left), id(right)}
    carriers = [_one_carrier(output) for output in _resolve_outputs(model)]
    assert len(carriers) == 2 and carriers[0] is not carriers[1]
    assert all(
        _properties(tensor) == {"source": "same opaque tag"} for tensor in carriers
    )
    replacement = np.asarray([11.0, 13.0], dtype=np.float32)
    changed = onnx.numpy_helper.from_array(replacement, name=carriers[0].name)
    changed.metadata_props.extend(carriers[0].metadata_props)
    carriers[0].CopyFrom(changed)
    x = np.asarray([5.0, 7.0], dtype=np.float32)
    onnx.checker.check_model(baseline, full_check=True)
    onnx.checker.check_model(model, full_check=True)
    actual = _run(model, x)
    np.testing.assert_array_equal(actual[0], x + replacement)
    np.testing.assert_array_equal(actual[1], x + np.asarray(right))
