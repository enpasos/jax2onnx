# tests/extra_tests/framework/test_initializer_metadata_materialization.py

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import InitializerMetadataError, onnx_function, to_onnx
from jax2onnx.converter.function_scope import FunctionScope
from jax2onnx.converter.ir_builder import IRBuilder
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


def _node_tensors(nodes: Any) -> Iterator[onnx.TensorProto]:
    for node in nodes:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from attribute.g.initializer
                yield from _node_tensors(attribute.g.node)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph in attribute.graphs:
                    yield from graph.initializer
                    yield from _node_tensors(graph.node)


def _tensors(model: onnx.ModelProto) -> list[onnx.TensorProto]:
    return [
        *model.graph.initializer,
        *_node_tensors(model.graph.node),
        *(
            tensor
            for function in model.functions
            for tensor in _node_tensors(function.node)
        ),
    ]


def _props(tensor: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.metadata_props}


def _run(model: onnx.ModelProto, x: np.ndarray) -> list[np.ndarray]:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if x.size == 0:
        # ORT 1.24.1's RemoveNode optimizer rejects valid zero-extent reshapes.
        # Execute the checked model without that optimizer on empty fixtures.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, {session.get_inputs()[0].name: x})


@pytest.mark.parametrize("value", [2.0, 5.0])
@pytest.mark.parametrize("cast_source", [False, True])
@pytest.mark.parametrize("direct_use", [False, True])
def test_full_retains_annotated_scalar_and_every_consuming_use(
    value: float, cast_source: bool, direct_use: bool
) -> None:
    weight = jnp.asarray(value, dtype=jnp.float16 if cast_source else jnp.float32)
    tags = {"source": "full.scalar", "opaque/empty": ""}

    def apply(x: jax.Array) -> tuple[jax.Array, ...]:
        fill = weight.astype(jnp.float32) if cast_source else weight
        filled = x + jnp.full((2,), fill)
        return (x + weight, filled) if direct_use else (filled,)

    model = to_onnx(
        apply,
        [jax.ShapeDtypeStruct((2,), jnp.float32)],
        initializer_metadata=lambda source: tags if source is weight else None,
    )
    annotated = [tensor for tensor in _tensors(model) if _props(tensor)]
    assert len(annotated) == 1
    parameter = annotated[0]
    assert _props(parameter) == tags
    assert tuple(parameter.dims) == ()
    x = np.asarray([0.5, 1.0], dtype=np.float32)
    for actual, expected in zip(_run(model, x), apply(jnp.asarray(x))):
        np.testing.assert_array_equal(actual, np.asarray(expected))

    # An independent artifact oracle: the full branch must consume the tagged
    # scalar even when another output already keeps that scalar live.
    parameter.raw_data = np.asarray(
        value + 3, dtype=np.float16 if cast_source else np.float32
    ).tobytes()
    for actual in _run(model, x):
        np.testing.assert_array_equal(actual, x + np.float32(value + 3))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("return_mode", ["proto", "ir"])
def test_function_local_reshape_preserves_exact_source_and_runtime_uses(
    nested: bool, return_mode: str
) -> None:
    # Distinct signed zeros and a noncanonical NaN payload test storage bits,
    # independently of numerical comparisons that equate NaN/zero variants.
    bits = np.asarray(
        [0x00000000, 0x80000000, 0x7FC01234, 0x3F800000, 0x40000000, 0x40400000],
        dtype=np.uint32,
    )
    weight = jnp.asarray(bits.view(np.float32).reshape(2, 3))
    tags = {"source": "function.reshape"}

    @onnx_function
    class Inner:
        def __call__(self, x: jax.Array) -> jax.Array:
            return x + jnp.reshape(weight, (3, 2))

    inner = Inner()

    @onnx_function
    class Outer:
        def __call__(self, x: jax.Array) -> jax.Array:
            return inner(x)

    outer = Outer()
    target = outer if nested else inner
    exported = to_onnx(
        lambda x: target(x),
        [(3, 2)],
        initializer_metadata=lambda source: tags if source is weight else None,
        return_mode=return_mode,
    )
    model = ir.to_proto(exported) if isinstance(exported, ir.Model) else exported
    annotated = [tensor for tensor in _tensors(model) if _props(tensor)]
    assert annotated and all(_props(tensor) == tags for tensor in annotated)
    for tensor in annotated:
        assert onnx.numpy_helper.to_array(tensor).dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(
            onnx.numpy_helper.to_array(tensor).reshape(-1).view(np.uint32), bits
        )
    views = [tensor for tensor in annotated if tuple(tensor.dims) == (3, 2)]
    assert len(views) == 1
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    np.testing.assert_allclose(
        _run(model, x)[0], np.asarray(target(jnp.asarray(x))), equal_nan=True
    )
    views[0].raw_data = np.full((3, 2), 7, np.float32).tobytes()
    np.testing.assert_array_equal(_run(model, x)[0], x + 7)


@pytest.mark.parametrize("annotate_neighbor", [False, True])
@pytest.mark.parametrize("reshape", [False, True])
def test_equal_sources_retain_independent_artifact_uses(
    annotate_neighbor: bool, reshape: bool
) -> None:
    left = jnp.asarray(np.ones((2, 3), dtype=np.float32))
    right = jnp.asarray(np.ones((2, 3), dtype=np.float32))
    assert left is not right
    tags = {"source": "identical labels do not imply identity"}

    def metadata(source: object) -> dict[str, str] | None:
        return (
            tags if source is left or (annotate_neighbor and source is right) else None
        )

    def apply(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        if reshape:
            return x + jnp.reshape(left, (3, 2)), x + jnp.reshape(right, (3, 2))
        return x + left, x + right

    shape = (3, 2) if reshape else (2, 3)
    model = to_onnx(apply, [shape], initializer_metadata=metadata)
    parameters = [
        tensor
        for tensor in _tensors(model)
        if tensor.data_type == onnx.TensorProto.FLOAT and np.prod(tensor.dims) == 6
    ]
    assert len(parameters) == 2
    assert sum(_props(tensor) == tags for tensor in parameters) == (
        2 if annotate_neighbor else 1
    )
    x = np.arange(6, dtype=np.float32).reshape(shape)
    baseline = _run(model, x)
    for parameter in parameters:
        original = parameter.SerializeToString()
        parameter.raw_data = np.full(tuple(parameter.dims), 4, np.float32).tobytes()
        changed = _run(model, x)
        differs = [
            not np.array_equal(after, before)
            for after, before in zip(changed, baseline)
        ]
        assert sum(differs) == 1
        for actual, touched in zip(changed, differs):
            np.testing.assert_array_equal(actual, x + (4 if touched else 1))
        parameter.ParseFromString(original)


def test_generated_equal_scalar_keeps_independent_artifact_use() -> None:
    weight = jnp.asarray(2.0, dtype=jnp.float32)
    tags = {"source": "captured scalar"}
    model = to_onnx(
        lambda x: (x + weight, x + 2.0),
        [(2,)],
        initializer_metadata=lambda source: tags if source is weight else None,
    )
    annotated = [tensor for tensor in _tensors(model) if _props(tensor) == tags]
    assert len(annotated) == 1
    x = np.asarray([0.5, 1.0], dtype=np.float32)
    for actual in _run(model, x):
        np.testing.assert_array_equal(actual, x + 2)
    annotated[0].raw_data = np.asarray(5, dtype=np.float32).tobytes()
    actual = _run(model, x)
    np.testing.assert_array_equal(actual[0], x + 5)
    np.testing.assert_array_equal(actual[1], x + 2)


@pytest.mark.parametrize("frontend", ["numpy", "lax"])
def test_empty_capture_reshape_uses_literal_zero_extent(frontend: str) -> None:
    weight = jnp.asarray(np.empty((1, 0, 2), dtype=np.float32))
    tags = {"source": "empty.reshape"}

    def apply(x: jax.Array) -> jax.Array:
        reshaped = (
            jnp.reshape(weight, (0,))
            if frontend == "numpy"
            else jax.lax.reshape(weight, (0,))
        )
        return x + reshaped

    model = to_onnx(
        apply,
        [(0,)],
        initializer_metadata=lambda source: tags if source is weight else None,
    )
    annotated = [tensor for tensor in _tensors(model) if _props(tensor) == tags]
    assert len(annotated) == 1
    assert onnx.numpy_helper.to_array(annotated[0]).size == 0
    actual = _run(model, np.empty((0,), dtype=np.float32))[0]
    assert actual.shape == (0,) and actual.dtype == np.dtype(np.float32)


def test_graph_assembly_cannot_replace_a_source_use_before_final_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    original = IRBuilder.to_ir_model

    def corrupt(builder: IRBuilder, *args: Any, **kwargs: Any) -> ir.Model:
        graph = builder.graph
        fabricated = ir.val(
            "fabricated",
            dtype=ir.DataType.FLOAT,
            shape=(2,),
            const_value=ir.tensor(np.asarray([2.0, 3.0], dtype=np.float32)),
        )
        graph.register_initializer(fabricated)
        output = ir.val("replacement", dtype=ir.DataType.FLOAT, shape=(2,))
        graph.append(
            ir.Node("", "Mul", [graph.inputs[0], fabricated], outputs=[output])
        )
        graph.outputs[1] = output
        return original(builder, *args, **kwargs)

    monkeypatch.setattr(IRBuilder, "to_ir_model", corrupt)
    with pytest.raises(InitializerMetadataError) as error:
        to_onnx(
            lambda x: (x + weight, x * weight),
            [(2,)],
            initializer_metadata=lambda source: (
                {"source": "weight"} if source is weight else None
            ),
        )
    assert error.value.reason == "preservation_failed"


def test_function_assembly_cannot_replace_an_intermediate_source_use_before_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = jnp.asarray([2.0, 3.0], dtype=jnp.float32)

    @onnx_function
    class Block:
        def __call__(self, x: jax.Array) -> jax.Array:
            return (x + weight) + (x * weight)

    block = Block()
    original = FunctionScope.end

    def corrupt(scope: FunctionScope, outputs: list[ir.Value]) -> ir.Function:
        graph = scope.ctx.builder.graph
        multiply = next(node for node in graph if node.op_type == "Mul")
        tensor = ir.tensor(np.asarray([2.0, 3.0], dtype=np.float32))
        fabricated = ir.val(
            "fabricated", dtype=ir.DataType.FLOAT, shape=(2,), const_value=tensor
        )
        output = ir.val("replacement", dtype=ir.DataType.FLOAT, shape=(2,))
        graph.insert_before(
            multiply,
            [
                ir.Node(
                    "",
                    "Constant",
                    [],
                    outputs=[fabricated],
                    attributes=[ir.AttrTensor("value", tensor)],
                ),
                ir.Node("", "Mul", [graph.inputs[0], fabricated], outputs=[output]),
            ],
        )
        for use in tuple(multiply.outputs[0].uses()):
            use.node.replace_input_with(use.idx, output)
        return original(scope, outputs)

    monkeypatch.setattr(FunctionScope, "end", corrupt)
    with pytest.raises(InitializerMetadataError) as error:
        to_onnx(
            lambda x: block(x),
            [(2,)],
            initializer_metadata=lambda source: (
                {"source": "weight"} if source is weight else None
            ),
        )
    assert error.value.reason == "preservation_failed"


@pytest.mark.parametrize("fault", ["output_order", "tensor_copy"])
def test_protective_graph_clone_must_keep_source_and_use_correspondence(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    left = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    right = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    original = ir.Graph.clone

    def corrupt(graph: ir.Graph, *args: Any, **kwargs: Any) -> ir.Graph:
        cloned = original(graph, *args, **kwargs)
        if fault == "output_order":
            first, second = cloned.outputs
            cloned.outputs[0], cloned.outputs[1] = second, first
        else:
            value = next(iter(cloned.initializers.values()))
            value.const_value = ir.tensor(np.asarray([2.0, 3.0], dtype=np.float32))
        return cloned

    monkeypatch.setattr(ir.Graph, "clone", corrupt)
    with pytest.raises(InitializerMetadataError) as error:
        to_onnx(
            lambda x: (x + left, x * right),
            [(2,)],
            return_mode="ir",
            initializer_metadata=lambda source: (
                {"source": "same"} if source is left or source is right else None
            ),
        )
    assert error.value.reason == "preservation_failed"


def test_production_payload_reads_use_guarded_materialization_interfaces() -> None:
    """Stop common bypasses while keeping the independent runtime invariant tests.

    This is a syntactic policy for production payload APIs, not a proof of arbitrary
    Python dataflow. The converter's equation and final provenance checks remain
    necessary, including for plugin-generated constants.
    """
    package = Path(__file__).resolve().parents[3] / "jax2onnx"
    paths = [
        *package.joinpath("plugins").rglob("*.py"),
        *package.joinpath("converter").rglob("*.py"),
    ]
    violations: list[str] = []
    for path in paths:
        relative = path.relative_to(package)
        if (
            relative.parts[:2] == ("plugins", "examples")
            or path.name == "_post_check_onnx_graph.py"
            # The exact storage validator deliberately reads source bits. It
            # never manufactures replacement tensors from those reads.
            or relative == Path("converter/initializer_metadata.py")
        ):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "get_const_tensor" for alias in node.names
            ):
                violations.append(
                    f"{relative}:{node.lineno}: unguarded tensor reader import"
                )
            for _field, statements in ast.iter_fields(node):
                if not isinstance(statements, list):
                    continue
                for index, statement in enumerate(statements):
                    if not (
                        isinstance(statement, ast.Assign)
                        and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Attribute)
                        and statement.value.func.attr == "clone"
                    ):
                        continue
                    following = (
                        statements[index + 1] if index + 1 < len(statements) else None
                    )
                    if not (
                        isinstance(following, ast.Expr)
                        and isinstance(following.value, ast.Call)
                        and isinstance(following.value.func, ast.Name)
                        and following.value.func.id == "register_graph_clone"
                        and len(following.value.args) == 2
                        and ast.dump(following.value.args[0])
                        == ast.dump(statement.value.func.value)
                        and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and isinstance(following.value.args[1], ast.Name)
                        and following.value.args[1].id == statement.targets[0].id
                    ):
                        violations.append(
                            f"{relative}:{statement.lineno}: graph clone lacks provenance registration"
                        )
        tensor_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "const_value"
            )
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            location = f"{relative}:{node.lineno}"
            if any(keyword.arg == "allow_annotated" for keyword in node.keywords):
                if relative != Path("plugins/_utils.py"):
                    violations.append(
                        f"{location}: unapproved annotated materialization"
                    )
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in {"numpy", "__array__", "get_const_tensor"}
            ):
                violations.append(f"{location}: indirect unguarded tensor reader")
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "get_const_tensor":
                    violations.append(f"{location}: unguarded IR tensor lookup")
                if node.func.attr == "numpy":
                    receiver = node.func.value
                    name = (
                        receiver.id
                        if isinstance(receiver, ast.Name)
                        else (
                            receiver.attr if isinstance(receiver, ast.Attribute) else ""
                        )
                    )
                    if not (
                        name.endswith("dtype")
                        or name.endswith("_dtype")
                        or name.endswith("_enum")
                    ):
                        violations.append(f"{location}: direct tensor.numpy()")
                if node.func.attr in {"asarray", "array", "frombuffer"} and node.args:
                    payload = node.args[0]
                    if any(
                        (
                            isinstance(child, ast.Attribute)
                            and child.attr == "const_value"
                        )
                        or (isinstance(child, ast.Name) and child.id in tensor_names)
                        for child in ast.walk(payload)
                    ):
                        violations.append(
                            f"{location}: direct constant payload array conversion"
                        )
    assert not violations, "\n".join(violations)
