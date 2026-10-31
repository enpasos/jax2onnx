# tests/extra_tests/converter/test_source_optimizer_precision.py

"""Optimizer eligibility preserves the bounded SOURCE operation contract."""

from collections.abc import Iterator
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator
from numpy.typing import NDArray
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.ir_optimizations import optimize_graph
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
)
from tests.extra_tests.converter.precision_runtime_support import (
    SampleAcceptanceError,
    SampleStatus,
    check_cpu_sample,
    require_sample_acceptance,
)
from tests.extra_tests.converter.precision_test_support import export_source_dtypes


@onnx_function
def _rsqrt_exact(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * (np.float32(1) / jax.lax.sqrt(y))


@onnx_function
def _rsqrt_near_one(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * (np.float32(1.000001) / jax.lax.sqrt(y))


@onnx_function
def _rsqrt_shared(x: jax.Array, y: jax.Array) -> tuple[jax.Array, ...]:
    root = jax.lax.sqrt(y)
    inverse = np.float32(1) / root
    product = x * inverse
    return root, inverse, product, product + np.float32(1)


@pytest.fixture(autouse=True)
def _restore_jax_config() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    previous_matmul_precision = jax.config.jax_default_matmul_precision
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)
        jax.config.update("jax_default_matmul_precision", previous_matmul_precision)


@pytest.mark.parametrize("numerator", [np.float32(1), np.float32(1.000001)])
@pytest.mark.parametrize("enable_rewrite", [False, True], ids=["flag-off", "flag-on"])
@pytest.mark.parametrize("inside_function", [False, True], ids=["top", "function"])
@pytest.mark.parametrize("optimize", [False, True], ids=["ort-off", "ort-on"])
def test_source_rsqrt_keeps_numerator_and_intermediate_rounding(
    numerator: np.float32,
    enable_rewrite: bool,
    inside_function: bool,
    optimize: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("JAX2ONNX_ENABLE_REWRITE_MUL_RSQRT", str(int(enable_rewrite)))

    def arithmetic(x: jax.Array, y: jax.Array) -> jax.Array:
        return x * (numerator / jax.lax.sqrt(y))

    def model(x: jax.Array, y: jax.Array) -> jax.Array:
        if inside_function:
            return _rsqrt_exact(x, y) if numerator == 1 else _rsqrt_near_one(x, y)
        return arithmetic(x, y)

    # Exact one is also unsafe to reassociate: FP32 reciprocal rounding remains
    # observable even after correcting a near-one constant matcher.
    x: NDArray[np.float32] = np.asarray([3 if numerator == 1 else 1], np.float32)
    y: NDArray[np.float32] = np.asarray([2 if numerator == 1 else 1], np.float32)
    expected = np.asarray(arithmetic(jnp.asarray(x), jnp.asarray(y)))
    direct_division = np.divide(x, np.sqrt(y), dtype=np.float32)
    assert not np.array_equal(expected, direct_division)
    specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in (x, y)]
    previous_x64 = bool(jax.config.x64_enabled)
    exported = export_source_dtypes(model, specs)
    assert len(exported.functions) == int(inside_function)
    graph = exported.functions[0] if inside_function else exported.graph
    operations = [node.op_type for node in graph.node]
    assert (
        operations.count("Sqrt")
        == operations.count("Div")
        == operations.count("Mul")
        == 1
    )
    path = tmp_path / "source_optimizer.onnx"
    onnx.save_model(exported, path)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    feeds = {
        value.name: data
        for value, data in zip(reloaded.graph.input, (x, y), strict=True)
    }
    (reference,) = ReferenceEvaluator(reloaded).run(None, feeds)
    np.testing.assert_array_equal(reference, expected, strict=True)
    check = check_cpu_sample(reloaded, (x, y), (expected,), optimize=optimize)
    if check.status is SampleStatus.FAILED:
        # ORT's own optimizer still reassociates this unobserved reciprocal.
        # Preserving source structure does not grant that target acceptance.
        # An ORT release that preserves the sample's result may be accepted.
        assert optimize and numerator == 1
        assert check.outputs is not None
        np.testing.assert_array_equal(check.outputs[0], direct_division, strict=True)
        with pytest.raises(SampleAcceptanceError, match="failed"):
            require_sample_acceptance(check)
    else:
        require_sample_acceptance(check)
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    assert bool(jax.config.x64_enabled) is previous_x64
    assert os.environ["JAX2ONNX_ENABLE_REWRITE_MUL_RSQRT"] == str(int(enable_rewrite))


@pytest.mark.parametrize("inside_function", [False, True], ids=["top", "function"])
def test_source_optimizer_preserves_shared_and_observable_values(
    inside_function: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAX2ONNX_ENABLE_REWRITE_MUL_RSQRT", "1")

    def arithmetic(x: jax.Array, y: jax.Array) -> tuple[jax.Array, ...]:
        root = jax.lax.sqrt(y)
        inverse = np.float32(1) / root
        product = x * inverse
        return root, inverse, product, product + np.float32(1)

    def model(x: jax.Array, y: jax.Array) -> tuple[jax.Array, ...]:
        return _rsqrt_shared(x, y) if inside_function else arithmetic(x, y)

    exported = export_source_dtypes(model, [jax.ShapeDtypeStruct((1,), np.float32)] * 2)
    assert len(exported.functions) == int(inside_function)
    graph = exported.functions[0] if inside_function else exported.graph
    nodes = {node.op_type: node for node in graph.node}
    assert nodes["Div"].input[1] == nodes["Sqrt"].output[0]
    assert nodes["Mul"].input[1] == nodes["Div"].output[0]
    assert nodes["Add"].input[0] == nodes["Mul"].output[0]
    output_names = (
        list(graph.output)
        if inside_function
        else [value.name for value in graph.output]
    )
    assert output_names[:3] == [
        nodes[kind].output[0] for kind in ("Sqrt", "Div", "Mul")
    ]
    x: NDArray[np.float32] = np.asarray([3], np.float32)
    y: NDArray[np.float32] = np.asarray([2], np.float32)
    expected = tuple(
        np.asarray(value) for value in arithmetic(jnp.asarray(x), jnp.asarray(y))
    )
    for optimize in (False, True):
        require_sample_acceptance(
            check_cpu_sample(exported, (x, y), expected, optimize=optimize)
        )


@pytest.mark.parametrize("enable_rewrite", [False, True], ids=["flag-off", "flag-on"])
@pytest.mark.parametrize("inside_function", [False, True], ids=["top", "function"])
def test_legacy_rsqrt_opt_in_and_source_policy_are_isolated(
    enable_rewrite: bool,
    inside_function: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX2ONNX_ENABLE_REWRITE_MUL_RSQRT", str(int(enable_rewrite)))

    def arithmetic(x: jax.Array, y: jax.Array) -> jax.Array:
        return x * (np.float32(1) / jax.lax.sqrt(y))

    def model(x: jax.Array, y: jax.Array) -> jax.Array:
        return _rsqrt_exact(x, y) if inside_function else arithmetic(x, y)

    specs = [jax.ShapeDtypeStruct((1,), np.float32)] * 2
    # Reuse each callable so tracing caches cannot hide a policy leak.
    for source in (True, False, True, False):
        exported = (
            export_source_dtypes(model, specs)
            if source
            else to_onnx(model, specs, enable_double_precision=False)
        )
        graph = exported.functions[0] if inside_function else exported.graph
        operations = [node.op_type for node in graph.node]
        assert ("Mul" not in operations) == (enable_rewrite and not source)
        assert active_precision_policy() is PrecisionPolicy.LEGACY
    direct = ir.from_proto(export_source_dtypes(model, specs))
    optimize_graph(direct)
    direct_graph = (
        next(iter(direct.functions.values())).graph if inside_function else direct.graph
    )
    assert ("Mul" not in [node.op_type for node in direct_graph]) == enable_rewrite


def _constant(name: str, array: np.ndarray) -> ir.Value:
    tensor = ir.tensor(array)
    return ir.val(name, tensor.dtype, array.shape, const_value=tensor)


def _graph_fixture(kind: str) -> ir.Graph:
    x = ir.val("x", ir.DataType.FLOAT, [1, 1, 2])
    output = ir.val("public_output", ir.DataType.FLOAT, x.shape)
    output.metadata_props["source"] = "preserve"
    if kind == "swish":
        sigmoid = ir.val("sigmoid", ir.DataType.FLOAT, x.shape)
        nodes = [
            ir.Node("", "Sigmoid", [x], outputs=[sigmoid]),
            ir.Node("", "Mul", [x, sigmoid], outputs=[output]),
        ]
        initializers = []
    elif kind == "conv_bias":
        weight = _constant("weight", np.ones((1, 1, 1), np.float32))
        bias = _constant("bias", np.ones((1, 1, 1), np.float32))
        convolved = ir.val("convolved", ir.DataType.FLOAT, x.shape)
        nodes = [
            ir.Node("", "ConvTranspose", [x, weight], outputs=[convolved]),
            ir.Node("", "Add", [convolved, bias], outputs=[output]),
        ]
        initializers = [weight, bias]
    elif kind in {"transpose_reduce", "transpose_reduce_axes"}:
        transposed = ir.val("transposed", ir.DataType.FLOAT, [1, 2, 1])
        reduced = ir.val("reduced", ir.DataType.FLOAT, [1, 1, 1])
        initializers = (
            [_constant("axes", np.asarray([1], np.int64))]
            if kind == "transpose_reduce_axes"
            else []
        )
        output.shape = ir.Shape([1, 1, 1])
        nodes = [
            ir.Node(
                "",
                "Transpose",
                [x],
                [ir.AttrInt64s("perm", [0, 2, 1])],
                outputs=[transposed],
            ),
            ir.Node(
                "",
                "ReduceMean",
                [transposed, *initializers],
                [ir.AttrInt64("keepdims", 1)],
                outputs=[reduced],
            ),
            ir.Node(
                "",
                "Transpose",
                [reduced],
                [ir.AttrInt64s("perm", [0, 2, 1])],
                outputs=[output],
            ),
        ]
    else:
        assert kind == "structural"
        cast = ir.val("cast", ir.DataType.FLOAT, x.shape)
        relu = ir.val("relu", ir.DataType.FLOAT, x.shape)
        shape = _constant("shape", np.asarray([1, 1, 2], np.int64))
        nodes = [
            ir.Node(
                "",
                "Cast",
                [x],
                [ir.AttrInt64("to", int(ir.DataType.FLOAT))],
                outputs=[cast],
            ),
            ir.Node("", "Relu", [cast], outputs=[relu]),
            ir.Node("", "Reshape", [relu, shape], outputs=[output]),
        ]
        initializers = [shape]
    return ir.Graph(
        [x],
        [output],
        nodes=nodes,
        initializers=initializers,
        opset_imports={"": 26},
        name=f"source_optimizer_{kind}",
    )


def _model_fixture(kind: str, scope: str) -> tuple[ir.Model, ir.Graph]:
    graph = _graph_fixture(kind)
    if scope == "top":
        return ir.Model(graph, ir_version=10), graph
    x = graph.inputs[0]
    output = graph.outputs[0]
    top_x = ir.val("top_x", x.dtype, x.shape)
    top_output = ir.val("top_output", output.dtype, output.shape)
    if scope == "function":
        # Function bodies cannot use graph initializers. Preserve the same
        # values through Constant nodes so the actual Function runner is tested.
        for name, value in list(graph.initializers.items()):
            del graph.initializers[name]
            assert value.const_value is not None
            constant = ir.Node(
                "",
                "Constant",
                [],
                [ir.AttrTensor("value", value.const_value)],
                outputs=[value],
            )
            graph.insert_before(next(iter(graph)), constant)
        fn = ir.Function("test", "optimizer_body", graph=graph, attributes=())
        top = ir.Graph(
            [top_x],
            [top_output],
            nodes=[ir.Node("test", "optimizer_body", [top_x], outputs=[top_output])],
            opset_imports={"": 26, "test": 1},
            name="function_call",
        )
        return ir.Model(top, ir_version=10, functions=[fn]), graph
    assert scope == "branch" and kind == "conv_bias"
    graph.inputs.clear()
    # Capture the same outer value in both branches. Conv bias folding already
    # traverses subgraphs; selecting SOURCE must gate that existing path too.
    top_x = x
    else_output = ir.val("else_output", x.dtype, x.shape)
    otherwise = ir.Graph(
        [],
        [else_output],
        nodes=[ir.Node("", "Identity", [x], outputs=[else_output])],
        name="otherwise",
    )
    condition = _constant("condition", np.asarray(True))
    top = ir.Graph(
        [top_x],
        [top_output],
        nodes=[
            ir.Node(
                "",
                "If",
                [condition],
                [
                    ir.AttrGraph("then_branch", graph),
                    ir.AttrGraph("else_branch", otherwise),
                ],
                outputs=[top_output],
            )
        ],
        initializers=[condition],
        opset_imports={"": 26},
        name="branch_call",
    )
    return ir.Model(top, ir_version=10), graph


@pytest.mark.parametrize(
    ("kind", "scope"),
    [
        (kind, scope)
        for kind in ("swish", "conv_bias", "transpose_reduce")
        for scope in ("top", "function")
    ]
    + [("conv_bias", "branch")],
)
def test_source_excludes_unaudited_math_passes_and_legacy_default_keeps_them(
    kind: str,
    scope: str,
) -> None:
    source, source_graph = _model_fixture(kind, scope)
    original_operations = [node.op_type for node in source_graph]
    original_output = source_graph.outputs[0]
    optimize_graph(source, precision_policy=PrecisionPolicy.SOURCE)
    assert [node.op_type for node in source_graph] == original_operations
    assert source_graph.outputs[0] is original_output
    assert original_output.metadata_props["source"] == "preserve"
    onnx.checker.check_model(ir.to_proto(source), full_check=True)
    legacy, legacy_graph = _model_fixture(kind, scope)
    optimize_graph(legacy)
    operations = [node.op_type for node in legacy_graph]
    if kind == "swish":
        assert operations == ["Swish"]
    elif kind == "conv_bias":
        assert "Add" not in operations
        assert (
            len(
                next(
                    node for node in legacy_graph if node.op_type == "ConvTranspose"
                ).inputs
            )
            == 3
        )
    else:
        assert "Transpose" not in operations
    onnx.checker.check_model(ir.to_proto(legacy), full_check=True)


@pytest.mark.parametrize("scope", ["top", "function"])
def test_source_still_runs_structural_cleanup(scope: str) -> None:
    model, graph = _model_fixture("structural", scope)
    optimize_graph(model, precision_policy=PrecisionPolicy.SOURCE)
    assert [node.op_type for node in graph if node.op_type != "Constant"] == ["Relu"]
    relu = next(node for node in graph if node.op_type == "Relu")
    assert graph.outputs[0] is relu.outputs[0]
    assert relu.inputs[0] is graph.inputs[0]
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def test_source_preserves_explicit_reduction_axes_inside_function() -> None:
    model, graph = _model_fixture("transpose_reduce_axes", "function")
    original_axes = next(node for node in graph if node.op_type == "ReduceMean").inputs[
        1
    ]
    optimize_graph(model, precision_policy=PrecisionPolicy.SOURCE)
    reducer = next(node for node in graph if node.op_type == "ReduceMean")
    assert reducer.inputs[1] is original_axes
    assert sum(node.op_type == "Transpose" for node in graph) == 2
    assert not graph.initializers
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
