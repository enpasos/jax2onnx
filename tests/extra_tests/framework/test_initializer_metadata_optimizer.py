# tests/extra_tests/framework/test_initializer_metadata_optimizer.py

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import InitializerMetadataError, onnx_function, to_onnx
from jax2onnx.converter.initializer_metadata import InitializerMetadataRegistry
from jax2onnx.converter.verified_rewrites import (
    can_replace_node,
    simplify_graph,
)
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
from tests.extra_tests.framework.test_initializer_metadata_function_identity import (
    _resolve_outputs,
)


@pytest.fixture(autouse=True)
def _clear_local_functions() -> Iterator[None]:
    yield
    for qualified in tuple(ONNX_FUNCTION_PLUGIN_REGISTRY):
        if qualified.startswith(f"{__name__}."):
            PLUGIN_REGISTRY.pop(f"onnx_fn::{qualified}", None)
            ONNX_FUNCTION_PLUGIN_REGISTRY.pop(qualified, None)


def _nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    def children(nodes: Any) -> Iterator[onnx.NodeProto]:
        for node in nodes:
            yield node
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    yield from children(attribute.g.node)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph in attribute.graphs:
                        yield from children(graph.node)

    return [
        *children(model.graph.node),
        *(node for function in model.functions for node in children(function.node)),
    ]


@pytest.mark.parametrize(
    "scope,return_mode",
    [
        ("root", "proto"),
        ("root", "ir"),
        ("root", "file"),
        ("nested-function", "proto"),
        ("cond", "proto"),
    ],
)
def test_repeated_subexpressions_share_computation_and_preserve_source_uses(
    scope: str,
    return_mode: Literal["proto", "ir", "file"],
    tmp_path: Path,
) -> None:
    weight = jnp.asarray(np.arange(6, dtype=np.float32).reshape(2, 3) / 4)
    tags = {"source": "shared.sub", "opaque": ""}

    def repeated(x: jax.Array) -> jax.Array:
        return (x - weight) * (x - weight)

    @onnx_function
    class Inner:
        def __call__(self, x: jax.Array) -> jax.Array:
            return repeated(x)

    inner = Inner()

    @onnx_function
    class Outer:
        def __call__(self, x: jax.Array) -> jax.Array:
            return inner(x)

    outer = Outer()

    def apply(x: jax.Array) -> jax.Array:
        if scope == "nested-function":
            return outer(x)
        if scope == "cond":
            return jax.lax.cond(
                x[0, 0] > 0,
                repeated,
                lambda y: (y + weight) * (y + weight),
                x,
            )
        return repeated(x)

    export_options: dict[str, Any] = {}
    if return_mode == "file":
        export_options["output_path"] = tmp_path / "shared_sub.onnx"
    exported = to_onnx(
        apply,
        [(2, 3)],
        output_names=["squared"],
        initializer_metadata=lambda source: tags if source is weight else None,
        return_mode=return_mode,
        **export_options,
    )
    if isinstance(exported, ir.Model):
        model = ir.to_proto(exported)
    elif return_mode == "file":
        model = onnx.load_model(exported)
    else:
        model = exported
    onnx.checker.check_model(model, full_check=True)
    assert model.graph.output[0].name == "squared"
    assert sum(node.op_type == "Sub" for node in _nodes(model)) == 1
    tagged = [tensor for tensor in _tensors(model) if _properties(tensor)]
    assert tagged and all(_properties(tensor) == tags for tensor in tagged)
    for tensor in tagged:
        _assert_storage(tensor, np.asarray(weight))
    for sign in (1, -1):
        x = sign * np.linspace(0.5, 1.5, 6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_allclose(_run(model, x)[0], np.asarray(apply(jnp.asarray(x))))
    for tensor in tagged:
        tensor.raw_data = np.full(tuple(tensor.dims), 3, dtype=np.float32).tobytes()
    for sign in (1, -1):
        x = sign * np.ones((2, 3), dtype=np.float32)
        expected = (x + 3) ** 2 if scope == "cond" and sign < 0 else (x - 3) ** 2
        np.testing.assert_array_equal(_run(model, x)[0], expected)


class _FlattenRoundTrip(eqx.Module):
    weight: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        flat = jnp.reshape(x, (-1,))
        restored = jnp.reshape(flat, x.shape)
        return restored + self.weight


class _RepeatedSub(eqx.Module):
    weight: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return (x - self.weight) * (x - self.weight)


def _computation_profile(model: onnx.ModelProto) -> Counter[str]:
    return Counter(
        node.op_type
        for node in _nodes(model)
        if not node.domain and node.op_type != "Constant"
    )


def _output_expressions_without_annotations(model: onnx.ModelProto) -> list[object]:
    normalized = onnx.ModelProto()
    normalized.CopyFrom(model)
    for tensor in _tensors(normalized):
        tensor.ClearField("metadata_props")
    # This expands Function calls and follows ordered operand dependencies.
    # Generated names and initializer-versus-Constant storage do not affect it.
    return [output.expression for output in _resolve_outputs(normalized)]


def _storage_profile(
    model: onnx.ModelProto,
) -> Counter[tuple[int, tuple[int, ...], bytes]]:
    return Counter(
        (
            tensor.data_type,
            tuple(tensor.dims),
            onnx.numpy_helper.to_array(tensor).tobytes(),
        )
        for tensor in _tensors(model)
    )


@pytest.mark.parametrize(
    "kind", ["eqx-repeated-sub", "eqx-roundtrip", "nnx-identity", "nested-repeated-sub"]
)
def test_shared_cleanup_matches_plain_skip_and_annotated_framework_fixtures(
    kind: str,
) -> None:
    shape = (2, 3)
    if kind == "nnx-identity":
        linear = nnx.LinearGeneral(
            3,
            3,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=nnx.Rngs(0),
        )
        weight = linear.kernel.get_value()

        def apply(x: jax.Array) -> jax.Array:
            projected = linear(x)
            return jnp.reshape(projected, projected.shape) + x

        def edited_reference(x: np.ndarray) -> np.ndarray:
            return (
                x @ np.full((3, 3), 7, dtype=np.float32)
                + np.asarray(linear.bias.get_value())
                + x
            )

        expected_profile = Counter({"Gemm": 1, "Add": 1})
    else:
        weight = jnp.asarray(np.arange(6, dtype=np.float32).reshape(shape) / 4)
        module = (
            _FlattenRoundTrip(weight)
            if kind == "eqx-roundtrip"
            else _RepeatedSub(weight)
        )
        apply = module
        if kind == "nested-repeated-sub":

            @onnx_function
            class Inner:
                def __call__(self, x: jax.Array) -> jax.Array:
                    return module(x)

            inner = Inner()

            @onnx_function
            class Outer:
                def __call__(self, x: jax.Array) -> jax.Array:
                    return inner(x)

            apply = Outer()

        def edited_reference(x: np.ndarray) -> np.ndarray:
            return x + 7 if kind == "eqx-roundtrip" else (x - 7) ** 2

        expected_profile = (
            Counter({"Add": 1})
            if kind == "eqx-roundtrip"
            else Counter({"Sub": 1, "Mul": 1})
        )
    tags = {"source": "framework.weight", "opaque": ""}
    models: dict[str, onnx.ModelProto] = {}
    x = np.linspace(-0.5, 1, 6, dtype=np.float32).reshape(shape)
    for profile in ("plain", "none", "empty", "annotated"):
        options: dict[str, Any] = {}
        if profile == "none":
            options["initializer_metadata"] = lambda source: None
        elif profile == "empty":
            options["initializer_metadata"] = lambda source: {}
        elif profile == "annotated":
            options["initializer_metadata"] = lambda source: (
                tags if source is weight else None
            )
        model = to_onnx(apply, [shape], output_names=["result"], **options)
        models[profile] = model
        assert model.graph.output[0].name == "result"
        assert _computation_profile(model) == expected_profile
        if kind == "eqx-roundtrip":
            used_names = {name for node in model.graph.node for name in node.input} | {
                output.name for output in model.graph.output
            }
            assert all(tensor.name in used_names for tensor in model.graph.initializer)
        onnx.checker.check_model(model, full_check=True)
        np.testing.assert_allclose(
            _run(model, x)[0], np.asarray(apply(jnp.asarray(x))), rtol=1e-6, atol=1e-6
        )
        if profile != "annotated":
            assert not any(_properties(tensor) for tensor in _tensors(model))
    ordinary_expression = _output_expressions_without_annotations(models["plain"])
    ordinary_storage = _storage_profile(models["plain"])
    for model in models.values():
        assert _output_expressions_without_annotations(model) == ordinary_expression
        assert _storage_profile(model) == ordinary_storage
    annotated = models["annotated"]
    tagged = [tensor for tensor in _tensors(annotated) if _properties(tensor)]
    assert len(tagged) == 1
    assert _properties(tagged[0]) == tags
    _assert_storage(tagged[0], np.asarray(weight))
    tagged[0].raw_data = np.full(tuple(tagged[0].dims), 7, dtype=np.float32).tobytes()
    np.testing.assert_allclose(
        _run(annotated, x)[0], edited_reference(x), rtol=1e-6, atol=1e-6
    )


def test_equinox_flatten_roundtrip_removes_shape_nodes_and_keeps_editable_weight() -> (
    None
):
    module = _FlattenRoundTrip(
        jnp.asarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    )
    tags = {"source": "roundtrip.weight"}
    model = to_onnx(
        module,
        [(2, 3)],
        initializer_metadata=lambda source: tags if source is module.weight else None,
    )
    assert [node.op_type for node in model.graph.node] == ["Add"]
    tagged = [tensor for tensor in _tensors(model) if _properties(tensor)]
    assert len(tagged) == 1
    assert _properties(tagged[0]) == tags
    _assert_storage(tagged[0], np.asarray(module.weight))
    x = np.linspace(-1, 1, 6, dtype=np.float32).reshape(2, 3)
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_allclose(_run(model, x)[0], np.asarray(module(jnp.asarray(x))))
    tagged[0].raw_data = np.full((2, 3), 7, dtype=np.float32).tobytes()
    np.testing.assert_array_equal(_run(model, x)[0], x + 7)


@pytest.mark.parametrize("annotate_neighbor", [False, True])
def test_cse_keeps_equal_parameter_sources_independently_editable(
    annotate_neighbor: bool,
) -> None:
    left = jnp.asarray(np.ones((2, 3), dtype=np.float32))
    right = jnp.asarray(np.ones((2, 3), dtype=np.float32))
    assert left is not right
    tags = {"source": "identical opaque properties"}

    def metadata(source: object) -> dict[str, str] | None:
        return (
            tags if source is left or (annotate_neighbor and source is right) else None
        )

    def apply(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (x - left) * (x - left), (x - right) * (x - right)

    model = to_onnx(apply, [(2, 3)], initializer_metadata=metadata)
    assert sum(node.op_type == "Sub" for node in model.graph.node) == 2
    producers = {name: node for node in model.graph.node for name in node.output}
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    carriers = []
    for output in model.graph.output:
        product = producers[output.name]
        assert product.op_type == "Mul" and product.input[0] == product.input[1]
        subtraction = producers[product.input[0]]
        carriers.append(initializers[subtraction.input[1]])
    assert carriers[0] is not carriers[1]
    assert _properties(carriers[0]) == tags
    assert _properties(carriers[1]) == (tags if annotate_neighbor else {})
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    for index, tensor in enumerate(carriers):
        original = tensor.SerializeToString()
        tensor.raw_data = np.full((2, 3), 4, dtype=np.float32).tobytes()
        onnx.checker.check_model(model, full_check=True)
        for output_index, actual in enumerate(_run(model, x)):
            np.testing.assert_array_equal(
                actual, (x - (4 if output_index == index else 1)) ** 2
            )
        tensor.ParseFromString(original)


def _op(
    graph: ir.Graph,
    op_type: str,
    inputs: Sequence[ir.Value],
    name: str,
    *,
    shape: tuple[int, ...] = (2, 3),
    dtype: ir.DataType = ir.DataType.FLOAT,
    attributes: Sequence[ir.Attr] = (),
    domain: str = "",
) -> ir.Value:
    output = ir.val(name, dtype=dtype, shape=shape)
    graph.append(
        ir.Node(domain, op_type, inputs, outputs=[output], attributes=attributes)
    )
    return output


def _initializer(graph: ir.Graph, array: np.ndarray, name: str) -> ir.Value:
    tensor = ir.tensor(array)
    value = ir.val(name, dtype=tensor.dtype, shape=array.shape, const_value=tensor)
    graph.register_initializer(value)
    return value


def _captured_graph(
    shape: tuple[int, ...] = (2, 3),
) -> tuple[InitializerMetadataRegistry, ir.Graph, ir.Value, ir.Value]:
    graph = ir.Graph(
        [ir.val("x", dtype=ir.DataType.FLOAT, shape=shape)],
        [],
        nodes=[],
        opset_imports={"": 18},
        name="protected_optimizer",
    )
    array = np.arange(np.prod(shape, dtype=int), dtype=np.float32).reshape(shape)
    source = _initializer(graph, array, "weight")
    registry = InitializerMetadataRegistry(lambda value: {"source": "same opaque tag"})
    registry.capture(registry.observe(array), source, array)
    base = _op(graph, "Sub", [graph.inputs[0], source], "base", shape=shape)
    return registry, graph, source, base


def _finish(
    registry: InitializerMetadataRegistry, graph: ir.Graph, *outputs: ir.Value
) -> ir.Model:
    graph.outputs.extend(outputs)
    model = ir.Model(graph, ir_version=10)
    registry.seal_ir(model)
    return model


@pytest.mark.parametrize(
    "kind", ["identity", "same-dtype-cast", "identity-reshape", "inverse"]
)
def test_proved_alias_rewrites_preserve_exact_storage_and_surviving_uses(
    kind: str,
) -> None:
    registry, graph, source, base = _captured_graph()
    if kind == "identity":
        alias = _op(graph, "Identity", [base], "alias")
    elif kind == "same-dtype-cast":
        alias = _op(
            graph,
            "Cast",
            [base],
            "alias",
            attributes=[ir.AttrInt64("to", int(base.dtype))],
        )
    else:
        restored_shape = _initializer(
            graph, np.asarray([2, 3], dtype=np.int64), "shape"
        )
        if kind == "inverse":
            flat_shape = _initializer(
                graph, np.asarray([6], dtype=np.int64), "flat_shape"
            )
            flat = _op(graph, "Reshape", [base, flat_shape], "flat", shape=(6,))
            middle = _op(graph, "Identity", [flat], "middle", shape=(6,))
        else:
            middle = base
        alias = _op(graph, "Reshape", [middle, restored_shape], "alias")
    result = _op(graph, "Mul", [base, alias], "result")
    model = _finish(registry, graph, result)
    original_storage = source.const_value.numpy().tobytes()
    original_tensor = source.const_value
    simplify_graph(model, registry)
    registry.validate_ir(model)
    assert result.producer().inputs[0] is result.producer().inputs[1] is base
    assert source.const_value is original_tensor
    assert source.const_value.numpy().tobytes() == original_storage
    assert source.const_value.metadata_props == {"source": "same opaque tag"}
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    x = np.arange(6, dtype=np.float32).reshape(2, 3) + 2
    np.testing.assert_array_equal(
        _run(proto, x)[0], np.full((2, 3), 4, dtype=np.float32)
    )


@dataclass
class _CSEFixture:
    registry: InitializerMetadataRegistry
    model: ir.Model
    first: ir.Value
    duplicate: ir.Value
    other: ir.Value
    right_survivor: ir.Value


def _cse_fixture() -> _CSEFixture:
    registry, graph, left, first = _captured_graph()
    array = np.array(left.const_value.numpy(), copy=True)
    right = _initializer(graph, array, "independent_equal_weight")
    registry.capture(registry.observe(array), right, array)
    duplicate = _op(graph, "Sub", [graph.inputs[0], left], "duplicate")
    other = _op(graph, "Sub", [graph.inputs[0], right], "other")
    survivor = _op(graph, "Add", [graph.inputs[0], right], "right_survivor")
    product = _op(graph, "Mul", [first, duplicate], "product")
    separate = _op(graph, "Add", [other, survivor], "separate")
    model = _finish(registry, graph, product, separate)
    return _CSEFixture(registry, model, first, duplicate, other, survivor)


def test_registry_rejects_equal_source_replacement_without_changing_graph() -> None:
    fixture = _cse_fixture()
    duplicate = fixture.duplicate.producer()
    assert not can_replace_node(duplicate, fixture.other)
    assert not fixture.registry.replace_uses_with_verified_value(
        fixture.model, duplicate, fixture.other
    )
    assert duplicate.graph is fixture.model.graph
    fixture.registry.validate_ir(fixture.model)


def test_verified_rewrite_cannot_bless_preexisting_use_corruption() -> None:
    fixture = _cse_fixture()
    # Both stored sources remain live, with identical properties and bytes.
    # The unrelated branch alone is redirected to the other captured source.
    fixture.other.producer().replace_input_with(1, fixture.first.producer().inputs[1])
    with pytest.raises(InitializerMetadataError) as error:
        fixture.registry.replace_uses_with_verified_value(
            fixture.model, fixture.duplicate.producer(), fixture.first
        )
    assert error.value.reason == "preservation_failed"
    assert fixture.duplicate.producer().graph is fixture.model.graph


def test_verified_rewrite_rejects_an_extra_source_use_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _cse_fixture()
    original = ir.convenience.replace_all_uses_with
    calls: list[object] = []

    def corrupt(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        calls.append(args)
        fixture.other.producer().replace_input_with(
            1, fixture.first.producer().inputs[1]
        )

    monkeypatch.setattr(ir.convenience, "replace_all_uses_with", corrupt)
    with pytest.raises(InitializerMetadataError) as error:
        fixture.registry.replace_uses_with_verified_value(
            fixture.model, fixture.duplicate.producer(), fixture.first
        )
    assert len(calls) == 1
    assert error.value.reason == "preservation_failed"
    # A rejected mutation must not establish a new accepted graph seal.
    with pytest.raises(InitializerMetadataError):
        fixture.registry.validate_ir(fixture.model)


def test_lossy_cast_roundtrip_is_retained() -> None:
    registry, graph, _, base = _captured_graph()
    half = _op(
        graph,
        "Cast",
        [base],
        "half",
        dtype=ir.DataType.FLOAT16,
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT16))],
    )
    restored = _op(
        graph,
        "Cast",
        [half],
        "restored",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    result = _op(graph, "Add", [base, restored], "result")
    model = _finish(registry, graph, result)
    assert not can_replace_node(restored.producer(), base)
    simplify_graph(model, registry)
    registry.validate_ir(model)
    assert sum(node.op_type == "Cast" for node in graph) == 2
    assert result.producer().inputs[1] is restored


@pytest.mark.parametrize("inverse", [False, True])
def test_annotated_shape_storage_cannot_be_removed_by_a_reshape_proof(
    inverse: bool,
) -> None:
    registry, graph, _, base = _captured_graph()
    array = np.asarray([6] if inverse else [2, 3], dtype=np.int64)
    protected_shape = _initializer(graph, array, "protected_shape")
    registry.capture(registry.observe(array), protected_shape, array)
    first = _op(
        graph,
        "Reshape",
        [base, protected_shape],
        "first",
        shape=(6,) if inverse else (2, 3),
    )
    if inverse:
        target = _initializer(graph, np.asarray([2, 3], dtype=np.int64), "target")
        alias = _op(graph, "Reshape", [first, target], "restored")
    else:
        alias = first
    result = _op(graph, "Add", [base, alias], "result")
    model = _finish(registry, graph, result)
    assert not can_replace_node(alias.producer(), base)
    simplify_graph(model, registry)
    registry.validate_ir(model)
    assert result.producer().inputs[1] is alias
    assert protected_shape.const_value.metadata_props == {"source": "same opaque tag"}


@pytest.mark.parametrize(
    "source_shape,target,allowzero,proved",
    [
        ((2, 3), (-1, 3), None, True),
        ((2, 3), (0, 3), None, True),
        ((0, 3), (0, 3), 1, True),
        ((), (), None, True),
        ((0, 3), (0, -1), None, False),
        ((0, 3), (0, -1), 1, False),
    ],
    ids=[
        "infer",
        "zero-copy",
        "literal-empty",
        "scalar",
        "ambiguous-copy",
        "ambiguous-empty",
    ],
)
def test_static_reshape_proof_obeys_inferred_and_zero_dimension_semantics(
    source_shape: tuple[int, ...],
    target: tuple[int, ...],
    allowzero: int | None,
    proved: bool,
) -> None:
    registry, graph, source, base = _captured_graph(source_shape)
    shape_value = _initializer(graph, np.asarray(target, dtype=np.int64), "shape")
    attributes = [] if allowzero is None else [ir.AttrInt64("allowzero", allowzero)]
    alias = _op(
        graph,
        "Reshape",
        [base, shape_value],
        "alias",
        shape=source_shape,
        attributes=attributes,
    )
    result = _op(graph, "Add", [base, alias], "result", shape=source_shape)
    model = _finish(registry, graph, result)
    assert can_replace_node(alias.producer(), base) is proved
    simplify_graph(model, registry)
    registry.validate_ir(model)
    assert (result.producer().inputs[1] is base) is proved
    if proved:
        proto = ir.to_proto(model)
        onnx.checker.check_model(proto, full_check=True)
        x = np.zeros(source_shape, dtype=np.float32)
        expected = 2 * (x - np.asarray(source.const_value.numpy()))
        np.testing.assert_array_equal(_run(proto, x)[0], expected)


def test_overridable_shape_initializer_is_not_a_static_reshape_proof() -> None:
    registry, graph, _, base = _captured_graph()
    shape_value = _initializer(graph, np.asarray([2, 3], dtype=np.int64), "shape")
    graph.inputs.append(shape_value)
    alias = _op(graph, "Reshape", [base, shape_value], "alias")
    result = _op(graph, "Add", [base, alias], "result")
    model = _finish(registry, graph, result)
    assert not can_replace_node(alias.producer(), base)
    simplify_graph(model, registry)
    registry.validate_ir(model)
    assert result.producer().inputs[1] is alias


def test_public_reshape_output_keeps_its_declared_name_and_type() -> None:
    weight = jnp.asarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    model = to_onnx(
        lambda x: jnp.reshape(x + weight, x.shape),
        [(2, 3)],
        output_names=["public_result"],
        initializer_metadata=lambda source: (
            {"source": "weight"} if source is weight else None
        ),
    )
    assert model.graph.node[-1].op_type == "Reshape"
    output = model.graph.output[0]
    assert output.name == "public_result"
    assert output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert [dimension.dim_value for dimension in output.type.tensor_type.shape.dim] == [
        2,
        3,
    ]
    onnx.checker.check_model(model, full_check=True)
    x = np.ones((2, 3), dtype=np.float32)
    np.testing.assert_array_equal(_run(model, x)[0], x + np.asarray(weight))


@pytest.mark.parametrize(
    "kind", ["custom-domain", "stateful", "tensor-attribute", "operand-order"]
)
def test_unsupported_cse_candidates_cannot_enter_the_verified_rewrite_path(
    kind: str,
) -> None:
    registry, graph, source, base = _captured_graph()
    inputs = [graph.inputs[0], source]
    attributes: list[ir.Attr] = []
    domain = "application.custom" if kind == "custom-domain" else ""
    op_type = "RandomUniform" if kind == "stateful" else "Sub"
    if kind == "stateful":
        inputs = []
        attributes = [ir.AttrInt64s("shape", [2, 3])]
    elif kind == "tensor-attribute":
        attributes = [ir.AttrTensor("opaque", ir.tensor(np.ones((), dtype=np.float32)))]
    first = _op(graph, op_type, inputs, "first", attributes=attributes, domain=domain)
    second_inputs = list(reversed(inputs)) if kind == "operand-order" else inputs
    second = _op(
        graph, op_type, second_inputs, "second", attributes=attributes, domain=domain
    )
    pair = _op(graph, "Mul", [first, second], "pair")
    result = _op(graph, "Add", [base, pair], "result")
    model = _finish(registry, graph, result)
    assert not can_replace_node(second.producer(), first)
    assert not registry.replace_uses_with_verified_value(
        model, second.producer(), first
    )
    simplify_graph(model, registry)
    registry.validate_ir(model)
    # The forward Sub may itself share the earlier base computation. Reversing
    # its operands must still retain a distinct second computation.
    canonical = base if kind == "operand-order" else first
    assert (
        pair.producer().inputs[0] is canonical and pair.producer().inputs[1] is second
    )
