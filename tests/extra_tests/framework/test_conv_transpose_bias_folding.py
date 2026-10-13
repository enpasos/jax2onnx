# tests/extra_tests/framework/test_conv_transpose_bias_folding.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from flax import nnx
import jax
from jax import export as jax_export
import jax.numpy as jnp
from onnx.reference import ReferenceEvaluator

from jax2onnx import to_onnx
from jax2onnx.converter.conv_transpose_bias_folding import (
    fold_conv_transpose_constant_bias,
)
from jax2onnx.converter.ir_optimizations import optimize_graph


@dataclass
class _Fixture:
    model: ir.Model
    conv: ir.Node
    add: ir.Node
    bias: ir.Value
    output: ir.Value


def _constant(name: str, array: np.ndarray) -> ir.Value:
    tensor = ir.tensor(array)
    return ir.val(name, tensor.dtype, array.shape, const_value=tensor)


def _fixture(
    *,
    rank: int = 4,
    dtype: type = np.float32,
    bias_shape: tuple[int, ...] | None = None,
    group: int = 1,
    symbolic_batch: bool = False,
    reverse_add: bool = False,
    bias_source: str = "initializer",
    existing_bias: bool = False,
    optional_bias: bool = False,
    separate_pad: bool = False,
    export_conv_output: bool = False,
    second_consumer: bool = False,
) -> _Fixture:
    spatial_rank = rank - 2
    channels = 6
    batch = "B" if symbolic_batch else 2
    shape = (batch, channels, *([3] * spatial_rank))
    numpy_dtype = np.dtype(dtype)
    ir_dtype = {
        np.dtype(np.float32): ir.DataType.FLOAT,
        np.dtype(np.float64): ir.DataType.DOUBLE,
        np.dtype(np.float16): ir.DataType.FLOAT16,
        np.dtype(ml_dtypes.bfloat16): ir.DataType.BFLOAT16,
    }[numpy_dtype]
    x = ir.val("x", ir_dtype, (batch, 4, *([3] * spatial_rank)))
    weight_array = (
        np.random.default_rng(4)
        .normal(size=(4, channels // group, *([3] * spatial_rank)))
        .astype(dtype)
    )
    weights = _constant("weights", weight_array / np.asarray(8, dtype=dtype))
    if bias_shape is None:
        bias_shape = (channels, *([1] * spatial_rank))
    values = np.linspace(-0.75, 0.75, max(1, np.prod(bias_shape)), dtype=dtype)
    bias = _constant("channel_bias", values.reshape(bias_shape))
    initializers = [weights]
    inputs = [x]
    nodes = []
    if bias_source in {"initializer", "overridable"}:
        initializers.append(bias)
    if bias_source in {"dynamic", "overridable"}:
        inputs.append(bias)
        if bias_source == "dynamic":
            bias.const_value = None
    elif bias_source == "constant":
        nodes.append(
            ir.Node(
                "",
                "Constant",
                [],
                attributes=[
                    ir.Attr("value", ir.AttributeType.TENSOR, bias.const_value)
                ],
                outputs=[bias],
            )
        )
    conv_inputs = [x, weights]
    if existing_bias:
        first_bias = _constant("first_bias", np.ones(channels, dtype=dtype))
        initializers.append(first_bias)
        conv_inputs.append(first_bias)
    elif optional_bias:
        conv_inputs.append(None)
    conv_output = ir.val("conv_output", ir_dtype, shape)
    conv = ir.Node(
        "",
        "ConvTranspose",
        conv_inputs,
        attributes=[
            ir.Attr("group", ir.AttributeType.INT, group),
            ir.Attr("strides", ir.AttributeType.INTS, [1] * spatial_rank),
            ir.Attr("pads", ir.AttributeType.INTS, [1] * (2 * spatial_rank)),
        ],
        outputs=[conv_output],
    )
    nodes.append(conv)
    add_input = conv_output
    if separate_pad:
        pads = _constant(
            "pads", np.asarray([0] * rank + [0, 0] + [1] * spatial_rank, np.int64)
        )
        initializers.append(pads)
        shape = (batch, channels, *([4] * spatial_rank))
        add_input = ir.val("padded_output", ir_dtype, shape)
        nodes.append(ir.Node("", "Pad", [conv_output, pads], outputs=[add_input]))
    output = ir.val("public_output", ir_dtype, shape)
    output.doc_string = "Public biased convolution result"
    output.metadata_props["test_metadata"] = "preserve"
    add_inputs = [bias, add_input] if reverse_add else [add_input, bias]
    add = ir.Node("", "Add", add_inputs, outputs=[output])
    nodes.append(add)
    outputs = [output]
    if export_conv_output:
        outputs.append(conv_output)
    if second_consumer:
        branch_output = ir.val("unbiased_branch", ir_dtype, conv_output.shape)
        nodes.append(ir.Node("", "Neg", [conv_output], outputs=[branch_output]))
        outputs.append(branch_output)
    graph = ir.Graph(
        inputs,
        outputs,
        nodes=nodes,
        initializers=initializers,
        opset_imports={"": 23},
        name="conv_transpose_bias",
    )
    return _Fixture(ir.Model(graph, ir_version=10), conv, add, bias, output)


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("full_rank_bias", [False, True])
@pytest.mark.parametrize("reverse_add", [False, True])
def test_channel_bias_folding_preserves_output_and_broadcasting(
    rank: int, dtype: type, full_rank_bias: bool, reverse_add: bool
) -> None:
    bias_shape = (6, *([1] * (rank - 2)))
    if full_rank_bias:
        bias_shape = (1, *bias_shape)
    fixture = _fixture(
        rank=rank, dtype=dtype, bias_shape=bias_shape, reverse_add=reverse_add
    )
    graph = fixture.model.graph
    old_shape, old_type = fixture.output.shape, fixture.output.type

    fold_conv_transpose_constant_bias(graph)

    assert [node.op_type for node in graph if node.op_type != "Constant"] == [
        "ConvTranspose"
    ]
    conv = fixture.output.producer()
    assert len(conv.inputs) == 3
    assert conv.inputs[2].shape == ir.Shape([6])
    np.testing.assert_array_equal(
        ir.convenience.get_const_tensor(conv.inputs[2]).numpy(),
        fixture.bias.const_value.numpy().reshape(6),
    )
    assert graph.outputs[0] is fixture.output
    assert conv.outputs[0] is fixture.output
    assert fixture.output.name == "public_output"
    assert fixture.output.shape == old_shape
    assert fixture.output.type == old_type
    assert fixture.output.doc_string == "Public biased convolution result"
    assert fixture.output.metadata_props["test_metadata"] == "preserve"
    onnx.checker.check_model(ir.to_proto(fixture.model), full_check=True)


@pytest.mark.parametrize("bias_source", ["constant", "initializer"])
def test_fold_supports_constant_nodes_and_optional_missing_bias(
    bias_source: str,
) -> None:
    fixture = _fixture(bias_source=bias_source, optional_bias=True)

    fold_conv_transpose_constant_bias(fixture.model.graph)

    assert "Add" not in [node.op_type for node in fixture.model.graph]
    assert len(fixture.output.producer().inputs) == 3
    onnx.checker.check_model(ir.to_proto(fixture.model), full_check=True)


@pytest.mark.parametrize(
    "options",
    [
        {"dtype": np.float16},
        {"dtype": ml_dtypes.bfloat16},
        {"bias_shape": ()},
        {"bias_shape": (6,)},
        {"bias_shape": (1, 6, 3, 1)},
        {"bias_shape": (2, 6, 1, 1)},
        {"bias_shape": (1, 1, 1, 3)},
        {"bias_source": "dynamic"},
        {"bias_source": "overridable"},
        {"existing_bias": True},
        {"separate_pad": True},
        {"export_conv_output": True},
        {"second_consumer": True},
    ],
    ids=[
        "fp16_rounding",
        "bf16_rounding",
        "scalar",
        "spatial_vector",
        "spatial_bias",
        "batch_bias",
        "last_axis_bias",
        "dynamic_bias",
        "overridable_initializer",
        "existing_conv_bias",
        "output_pad",
        "exported_unbiased_output",
        "shared_unbiased_output",
    ],
)
def test_unsafe_patterns_remain_unchanged(options: dict) -> None:
    fixture = _fixture(**options)
    graph = fixture.model.graph
    original_nodes = tuple(graph)
    original_outputs = tuple(graph.outputs)

    fold_conv_transpose_constant_bias(graph)

    assert tuple(graph) == original_nodes
    assert tuple(graph.outputs) == original_outputs
    assert len(fixture.conv.inputs) == 2 or options.get("existing_bias")


@pytest.mark.parametrize("custom_node", ["conv", "add"])
def test_custom_domain_nodes_are_never_folded(custom_node: str) -> None:
    fixture = _fixture()
    getattr(fixture, custom_node).domain = "custom"

    fold_conv_transpose_constant_bias(fixture.model.graph)

    assert list(fixture.model.graph) == [fixture.conv, fixture.add]


def test_mismatched_bias_dtype_is_not_folded() -> None:
    fixture = _fixture()
    double_bias = _constant(
        "double_bias", fixture.bias.const_value.numpy().astype(np.float64)
    )
    fixture.model.graph.initializers.add(double_bias)
    fixture.add.replace_input_with(1, double_bias)

    fold_conv_transpose_constant_bias(fixture.model.graph)

    assert list(fixture.model.graph) == [fixture.conv, fixture.add]


def test_unbiased_output_captured_by_subgraph_is_not_modified() -> None:
    fixture = _fixture()
    graph = fixture.model.graph
    unbiased = fixture.conv.outputs[0]

    def branch(name: str) -> ir.Graph:
        output = ir.val(name, ir.DataType.FLOAT, unbiased.shape)
        return ir.Graph(
            [],
            [output],
            nodes=[ir.Node("", "Identity", [unbiased], outputs=[output])],
            name=name,
        )

    condition = _constant("condition", np.asarray(True, dtype=np.bool_))
    graph.initializers.add(condition)
    capture = ir.val("captured_unbiased", ir.DataType.FLOAT, unbiased.shape)
    if_node = ir.Node(
        "",
        "If",
        [condition],
        attributes=[
            ir.Attr("then_branch", ir.AttributeType.GRAPH, branch("then_output")),
            ir.Attr("else_branch", ir.AttributeType.GRAPH, branch("else_output")),
        ],
        outputs=[capture],
    )
    graph.append(if_node)
    graph.outputs.append(capture)

    fold_conv_transpose_constant_bias(graph)

    assert list(graph) == [fixture.conv, fixture.add, if_node]
    onnx.checker.check_model(ir.to_proto(fixture.model), full_check=True)


def _reshape_vector_bias(
    fixture: _Fixture, *, overridable_shape: bool = False
) -> ir.Value:
    graph = fixture.model.graph
    array = fixture.bias.const_value.numpy()
    vector = _constant("immutable_bias_vector", array.reshape(-1))
    new_shape = _constant("bias_shape", np.asarray(array.shape, dtype=np.int64))
    del graph.initializers[fixture.bias.name]
    graph.initializers.add(vector)
    graph.initializers.add(new_shape)
    if overridable_shape:
        graph.inputs.append(new_shape)
    fixture.bias.const_value = None
    graph.insert_before(
        fixture.add,
        ir.Node("", "Reshape", [vector, new_shape], outputs=[fixture.bias]),
    )
    return vector


def test_immutable_vector_behind_constant_reshape_is_reused() -> None:
    fixture = _fixture()
    vector = _reshape_vector_bias(fixture)
    baseline = ir.to_proto(fixture.model)

    optimize_graph(fixture.model)

    assert [node.op_type for node in fixture.model.graph] == ["ConvTranspose"]
    assert fixture.output.producer().inputs[2] is vector
    x = np.random.default_rng(3).normal(size=(2, 4, 3, 3)).astype(np.float32)
    np.testing.assert_allclose(
        _run(ir.to_proto(fixture.model), x), _run(baseline, x), rtol=1e-6, atol=1e-6
    )


def test_overridable_reshape_shape_does_not_enable_bias_folding() -> None:
    fixture = _fixture()
    _reshape_vector_bias(fixture, overridable_shape=True)
    original_nodes = tuple(fixture.model.graph)

    fold_conv_transpose_constant_bias(fixture.model.graph)

    assert tuple(fixture.model.graph) == original_nodes


def test_folding_inside_function_preserves_signature_and_local_constants() -> None:
    fixture = _fixture()
    graph = fixture.model.graph
    for value in list(graph.initializers.values()):
        del graph.initializers[value.name]
        graph.insert_before(
            next(iter(graph)),
            ir.Node(
                "",
                "Constant",
                [],
                outputs=[value],
                attributes=[
                    ir.Attr("value", ir.AttributeType.TENSOR, value.const_value)
                ],
            ),
        )
    unused_parameter = ir.val("unused_parameter", ir.DataType.FLOAT, ())
    graph.inputs.append(unused_parameter)
    function = ir.Function(
        domain="custom", name="biased_transpose", graph=graph, attributes=()
    )
    top_input = ir.val("top_input", ir.DataType.FLOAT, (2, 4, 3, 3))
    unused_argument = _constant("unused_argument", np.asarray(0, np.float32))
    top_output = ir.val("top_output", ir.DataType.FLOAT, (2, 6, 3, 3))
    top = ir.Graph(
        [top_input],
        [top_output],
        nodes=[
            ir.Node(
                "custom",
                "biased_transpose",
                [top_input, unused_argument],
                outputs=[top_output],
            )
        ],
        initializers=[unused_argument],
        opset_imports={"": 23, "custom": 1},
        name="function_call",
    )
    model = ir.Model(top, ir_version=10, functions=[function])

    optimize_graph(model)

    assert [value.name for value in graph.inputs] == ["x", "unused_parameter"]
    assert not graph.initializers
    assert "Add" not in [node.op_type for node in graph]
    assert fixture.output.producer().op_type == "ConvTranspose"
    assert fixture.output.producer().inputs[2].producer().op_type == "Constant"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def test_folding_recurses_into_if_branch_graphs() -> None:
    branches = [_fixture(), _fixture(reverse_add=True)]
    x = branches[0].model.graph.inputs[0]
    branches[1].conv.replace_input_with(0, x)
    for branch in branches:
        branch.model.graph.inputs.clear()
    condition = ir.val("condition", ir.DataType.BOOL, ())
    output = ir.val("branch_result", ir.DataType.FLOAT, (2, 6, 3, 3))
    if_node = ir.Node(
        "",
        "If",
        [condition],
        outputs=[output],
        attributes=[
            ir.Attr("then_branch", ir.AttributeType.GRAPH, branches[0].model.graph),
            ir.Attr("else_branch", ir.AttributeType.GRAPH, branches[1].model.graph),
        ],
    )
    model = ir.Model(
        ir.Graph(
            [x, condition],
            [output],
            nodes=[if_node],
            opset_imports={"": 23},
            name="branches",
        ),
        ir_version=10,
    )

    optimize_graph(model)

    for branch in branches:
        assert "Add" not in [node.op_type for node in branch.model.graph]
        assert branch.output.producer().op_type == "ConvTranspose"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def _run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    (output,) = session.run(None, {"x": x})
    return output


def _reference_run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    # ORT's CPU ConvTranspose implementation does not accept DOUBLE. Keep the
    # optimizer's double-precision parity covered through ONNX's evaluator.
    onnx.checker.check_model(model, full_check=True)
    (output,) = ReferenceEvaluator(model).run(None, {"x": x})
    return output


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("symbolic_batch", [False, True], ids=["static", "symbolic"])
def test_fold_has_numerical_parity(dtype: type, symbolic_batch: bool) -> None:
    # Exercise grouped convolution on the actual CPU runtime. ONNX's reference
    # evaluator currently has a grouped ConvTranspose weight-slicing bug.
    fixture = _fixture(
        dtype=dtype,
        group=2 if dtype is np.float32 else 1,
        symbolic_batch=symbolic_batch,
    )
    baseline = ir.to_proto(fixture.model)

    optimize_graph(fixture.model)

    assert [
        node.op_type for node in fixture.model.graph if node.op_type != "Constant"
    ] == ["ConvTranspose"]
    folded = ir.to_proto(fixture.model)
    for batch in (1, 2, 4) if symbolic_batch else (2,):
        x = np.random.default_rng(batch).normal(size=(batch, 4, 3, 3)).astype(dtype)
        runner = _run if dtype is np.float32 else _reference_run
        before = runner(baseline, x)
        after = runner(folded, x)
        assert after.dtype == np.dtype(dtype)
        np.testing.assert_allclose(
            after,
            before,
            rtol=1e-6 if dtype is np.float32 else 1e-13,
            atol=1e-6 if dtype is np.float32 else 1e-13,
        )


def test_bias_fold_does_not_move_bias_across_zero_output_padding() -> None:
    fixture = _fixture(separate_pad=True)
    baseline = ir.to_proto(fixture.model)

    optimize_graph(fixture.model)

    assert "Add" in [node.op_type for node in fixture.model.graph]
    x = np.zeros((2, 4, 3, 3), dtype=np.float32)
    before = _run(baseline, x)
    after = _run(ir.to_proto(fixture.model), x)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(
        after[0, :, -1, -1], np.linspace(-0.75, 0.75, 6, dtype=np.float32)
    )


def _optimized_model(model: onnx.ModelProto, path: Path) -> onnx.ModelProto:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(path)
    options.intra_op_num_threads = 1
    ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return onnx.load(path)


def test_fold_allows_conv_transpose_batch_norm_runtime_fusion(tmp_path: Path) -> None:
    fixture = _fixture()
    graph = fixture.model.graph
    bn_inputs = [fixture.output]
    for name, array in (
        ("bn_scale", np.full(6, 1.1, np.float32)),
        ("bn_bias", np.full(6, 0.2, np.float32)),
        ("bn_mean", np.full(6, 0.1, np.float32)),
        ("bn_variance", np.full(6, 0.8, np.float32)),
    ):
        value = _constant(name, array)
        graph.initializers.add(value)
        bn_inputs.append(value)
    bn_output = ir.val("bn_output", ir.DataType.FLOAT, fixture.output.shape)
    graph.append(ir.Node("", "BatchNormalization", bn_inputs, outputs=[bn_output]))
    graph.outputs[0] = bn_output
    before = _optimized_model(ir.to_proto(fixture.model), tmp_path / "before.onnx")

    fold_conv_transpose_constant_bias(graph)

    after = _optimized_model(ir.to_proto(fixture.model), tmp_path / "after.onnx")
    if not any(node.op_type == "Add" for node in before.graph.node):
        pytest.skip("this runtime already folds the separate channel bias")
    assert not any(node.op_type == "Add" for node in after.graph.node)
    if any(node.op_type == "BatchNormalization" for node in after.graph.node):
        pytest.skip("this runtime does not fuse ConvTranspose and BatchNormalization")
    assert any(node.op_type == "BatchNormalization" for node in before.graph.node)
    x = np.random.default_rng(9).normal(size=(2, 4, 3, 3)).astype(np.float32)
    np.testing.assert_allclose(_run(after, x), _run(before, x), rtol=2e-6, atol=2e-6)


def _layout_fixture(*, reverse_add: bool = False) -> tuple[_Fixture, ir.Node, ir.Node]:
    fixture = _fixture(bias_shape=(6,), reverse_add=reverse_add)
    graph = fixture.model.graph
    raw = fixture.conv.outputs[0]
    view_shape = (2, 3, 3, 6)
    transposed = ir.val("channels_last", ir.DataType.FLOAT, view_shape)
    before = ir.Node(
        "",
        "Transpose",
        [raw],
        outputs=[transposed],
        attributes=[ir.Attr("perm", ir.AttributeType.INTS, [0, 2, 3, 1])],
    )
    graph.insert_before(fixture.add, before)
    fixture.add.replace_input_with(1 if reverse_add else 0, transposed)
    fixture.output.name = "added_channels_last"
    fixture.output.shape = ir.Shape(view_shape)
    public_output = ir.val("restored_channels_first", ir.DataType.FLOAT, raw.shape)
    public_output.doc_string = "Preserved inverse-layout result"
    after = ir.Node(
        "",
        "Transpose",
        [fixture.output],
        outputs=[public_output],
        attributes=[ir.Attr("perm", ir.AttributeType.INTS, [0, 3, 1, 2])],
    )
    graph.append(after)
    graph.outputs[0] = public_output
    fixture.output = public_output
    return fixture, before, after


@pytest.mark.parametrize("reverse_add", [False, True])
def test_inverse_layout_pair_folds_channel_bias_with_cpu_parity(
    reverse_add: bool,
) -> None:
    fixture, _, _ = _layout_fixture(reverse_add=reverse_add)
    baseline = ir.to_proto(fixture.model)
    public_output = fixture.output

    fold_conv_transpose_constant_bias(fixture.model.graph)

    assert [node.op_type for node in fixture.model.graph] == ["ConvTranspose"]
    assert fixture.model.graph.outputs[0] is public_output
    assert public_output.producer().inputs[2] is fixture.bias
    assert public_output.name == "restored_channels_first"
    assert public_output.doc_string == "Preserved inverse-layout result"
    x = np.random.default_rng(8).normal(size=(2, 4, 3, 3)).astype(np.float32)
    np.testing.assert_allclose(
        _run(ir.to_proto(fixture.model), x), _run(baseline, x), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    "observable",
    [
        "wrong_permutation",
        "returned_view",
        "branched_view",
        "returned_add",
        "branched_add",
    ],
)
def test_layout_pair_keeps_observable_intermediates_and_requires_inverse_permutations(
    observable: str,
) -> None:
    fixture, before, after = _layout_fixture()
    graph = fixture.model.graph
    if observable == "wrong_permutation":
        # Swapping equally sized spatial dimensions retains the stamped shape,
        # but still changes data and must not be treated as an inverse pair.
        after.attributes["perm"] = ir.Attr("perm", ir.AttributeType.INTS, [0, 3, 2, 1])
    else:
        observed = (
            before.outputs[0] if observable.endswith("view") else fixture.add.outputs[0]
        )
        if observable.startswith("returned"):
            graph.outputs.append(observed)
        else:
            branch = ir.val("observable_branch", ir.DataType.FLOAT, observed.shape)
            graph.append(ir.Node("", "Neg", [observed], outputs=[branch]))
            graph.outputs.append(branch)
    original_nodes = tuple(graph)

    fold_conv_transpose_constant_bias(graph)

    assert tuple(graph) == original_nodes
    onnx.checker.check_model(ir.to_proto(fixture.model), full_check=True)


@pytest.mark.parametrize("symbolic_batch", [False, True], ids=["static", "symbolic"])
def test_nnx_transposed_convolution_bias_enables_batch_norm_fusion_and_matches_jax(
    symbolic_batch: bool, tmp_path: Path
) -> None:
    conv = nnx.ConvTranspose(
        3,
        6,
        (3, 3),
        strides=(2, 2),
        padding="VALID",
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        bias_init=nnx.initializers.constant(0.25),
        rngs=nnx.Rngs(0),
    )
    batch_norm = nnx.BatchNorm(
        6,
        use_running_average=True,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        scale_init=nnx.initializers.constant(1.1),
        bias_init=nnx.initializers.constant(-0.2),
        rngs=nnx.Rngs(1),
    )

    def fn(x: jax.Array) -> jax.Array:
        return batch_norm(conv(x))

    shape = jax_export.symbolic_shape("B, 3, 4, 3") if symbolic_batch else (2, 3, 4, 3)
    model = to_onnx(fn, [jax.ShapeDtypeStruct(shape, jnp.float32)])
    onnx.checker.check_model(model, full_check=True)
    transpose_convs = [
        node for node in model.graph.node if node.op_type == "ConvTranspose"
    ]
    assert len(transpose_convs) == 1
    assert len(transpose_convs[0].input) == 3
    assert any(node.op_type == "BatchNormalization" for node in model.graph.node)
    optimized = _optimized_model(model, tmp_path / "nnx_optimized.onnx")
    if any(node.op_type == "BatchNormalization" for node in optimized.graph.node):
        pytest.skip("this runtime does not fuse ConvTranspose and BatchNormalization")
    assert not any(node.op_type == "Add" for node in optimized.graph.node)

    for batch in (1, 2, 4) if symbolic_batch else (2,):
        x = (
            np.random.default_rng(batch)
            .normal(size=(batch, 3, 4, 3))
            .astype(np.float32)
        )
        expected = np.asarray(fn(jnp.asarray(x)))
        # Export-generated input names differ from the explicit IR fixtures.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        (actual,) = session.run(None, {session.get_inputs()[0].name: x})
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
