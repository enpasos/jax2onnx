# tests/extra_tests/converter/test_typed_builder.py

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from jax2onnx.converter.ir_builder import (
    IRBuilder,
    JAX_CALLSITE_METADATA_KEY,
    PLUGIN_METADATA_KEY,
)
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.typing_support import IRBuilderProtocol
from jax2onnx.ir_utils import const_value_to_numpy, numpy_dtype_to_ir
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.jax.lax.neg import NegPlugin
from jax2onnx.plugins.jax.lax.top_k import TopKPlugin


def test_typed_single_output_keeps_output_identity_and_metadata() -> None:
    builder: IRBuilderProtocol = IRBuilder(opset=18, enable_stacktrace_metadata=True)
    builder.set_current_jax_traceback("caller.py:17 (export)")
    builder.set_current_plugin_identifier("demo.Plugin.lower", "42")
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3])
    out = ir.val("result", dtype=ir.DataType.FLOAT, shape=[3])
    builder.inputs.append(x)

    result: ir.Value = builder.op(
        "Neg",
        [x],
        output=out,
        name="negate",
        version=18,
        doc_string="typed output",
        metadata_props={"purpose": "regression"},
    )

    node = result.producer()
    assert node is not None
    assert result is out
    assert node.graph is builder.graph
    assert node.name == "negate"
    assert node.doc_string == "typed output"
    assert result.dtype == ir.DataType.FLOAT
    assert list(result.shape) == [3]
    assert node.metadata_props["purpose"] == "regression"
    assert node.metadata_props[JAX_CALLSITE_METADATA_KEY] == "export:17"
    assert node.metadata_props[PLUGIN_METADATA_KEY] == "Plugin.lower:42"


def test_typed_multi_output_accepts_optional_inputs_and_explicit_values() -> None:
    concrete = IRBuilder(opset=18)
    builder: IRBuilderProtocol = concrete
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[4])
    first = ir.val("first", dtype=ir.DataType.FLOAT, shape=[2])
    second = ir.val("second", dtype=ir.DataType.FLOAT, shape=[2])
    builder.inputs.append(x)

    outputs: Sequence[ir.Value] = builder.op_multi_out(
        "Split",
        [x, None],
        {"axis": 0, "num_outputs": 2},
        outputs=[first, second],
        version=18,
    )

    assert outputs[0] is first
    assert outputs[1] is second
    node = first.producer()
    assert node is not None and node is second.producer()
    assert list(node.inputs) == [x, None]
    assert node.attributes["axis"].as_int() == 0
    assert node.attributes["num_outputs"].as_int() == 2
    assert node.graph is builder.graph
    assert ("", 18) in concrete.used_opsets


@pytest.mark.parametrize("output_count", [0, 2])
def test_typed_multi_output_preserves_tape_output_count(output_count: int) -> None:
    builder = IRBuilder(opset=18)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[4])

    outputs = builder.op_multi_out("Split", [x], num_outputs=output_count)

    assert len(outputs) == output_count
    assert len(builder.nodes) == 1


def test_typed_multi_output_rejects_ambiguous_arity_before_mutation() -> None:
    builder = IRBuilder(opset=18)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[4])

    with pytest.raises(ValueError, match="Both num_outputs and outputs"):
        builder.op_multi_out("Split", [x], num_outputs=1, outputs=[ir.Value()])

    assert len(builder.nodes) == 0
    assert not builder.used_opsets


def test_typed_attributes_use_public_ir_attr_constructors() -> None:
    builder: IRBuilderProtocol = IRBuilder(opset=18)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 3])

    result = builder.op("Transpose", [x], {"perm": ir.AttrInt64s("perm", [1, 0])})

    node = result.producer()
    assert node is not None
    assert node.attributes["perm"].as_ints() == (1, 0)


def test_neg_pilot_preserves_allocated_output() -> None:
    eqn = jax.make_jaxpr(lambda x: -x)(jnp.ones((3,), dtype=jnp.float32)).jaxpr.eqns[0]
    ctx = IRContext(opset=18)
    x = ctx.add_input_for_invar(eqn.invars[0], 0)
    output = ctx.allocate_value_for_var(eqn.outvars[0], name_hint="neg_result")

    NegPlugin().lower(ctx, eqn)

    result = ctx.require_value_for_var(eqn.outvars[0])
    assert result is output
    assert result.name == "neg_result"
    assert result.dtype == ir.DataType.FLOAT
    assert list(result.shape) == [3]
    assert len(ctx.builder.graph) == 1
    node = ctx.builder.graph[0]
    assert node.op_type == "Neg"
    assert list(node.inputs) == [x]
    assert list(node.outputs) == [output]


def test_top_k_pilot_keeps_two_outputs_and_index_cast() -> None:
    eqn = jax.make_jaxpr(lambda x: jax.lax.top_k(x, 2))(
        jnp.ones((4, 6), dtype=jnp.float32)
    ).jaxpr.eqns[0]
    ctx = IRContext(opset=18)
    x = ctx.add_input_for_invar(eqn.invars[0], 0)

    TopKPlugin().lower(ctx, eqn)

    values = ctx.require_value_for_var(eqn.outvars[0])
    indices = ctx.require_value_for_var(eqn.outvars[1])
    assert [node.op_type for node in ctx.builder.graph] == ["TopK", "Cast"]
    top_k, cast = ctx.builder.graph
    assert top_k.inputs[0] is x
    assert top_k.outputs[0] is values
    assert top_k.outputs[1].dtype == ir.DataType.INT64
    assert cast.outputs[0] is indices
    assert values.dtype == ir.DataType.FLOAT
    assert indices.dtype == ir.DataType.INT32
    assert list(values.shape) == list(indices.shape) == [4, 2]
    assert values.name == "TopK_Values_0"
    assert indices.name == "topk_indices_0"
    assert top_k.attributes["axis"].as_int() == 1
    assert top_k.attributes["largest"].as_int() == 1
    assert top_k.attributes["sorted"].as_int() == 1
    k = ctx.builder.graph.initializers["topk_k_0"].const_value
    assert k is not None
    np.testing.assert_array_equal(k.numpy(), [2])


@pytest.mark.parametrize("operator", ["neg", "top_k"])
def test_pilot_relowering_preserves_existing_results(operator: str) -> None:
    operation = (lambda x: -x) if operator == "neg" else lambda x: jax.lax.top_k(x, 2)
    plugin = NegPlugin() if operator == "neg" else TopKPlugin()
    (eqn,) = jax.make_jaxpr(operation)(jnp.ones((3,), dtype=jnp.float32)).jaxpr.eqns
    ctx = IRContext(opset=21)
    x = ctx.add_input_for_invar(eqn.invars[0], 0)
    plugin.lower(ctx, eqn)
    original = [ctx.require_value_for_var(var) for var in eqn.outvars]
    names = [value.name for value in original]
    producers = [value.producer() for value in original]

    plugin.lower(ctx, eqn)

    replacements = [ctx.require_value_for_var(var) for var in eqn.outvars]
    for old, new, name, producer in zip(original, replacements, names, producers):
        assert new is not old
        assert old.name == name
        assert old.producer() is producer
        assert new.producer() is not producer
        assert new.type == old.type
        assert new.shape == old.shape
    ctx.builder.outputs.extend([*original, *replacements])
    model = ir.to_proto(
        ctx.builder.to_ir_model(name="repeated_lowering", ir_version=10)
    )
    data = np.asarray([1.0, 4.0, 2.0], dtype=np.float32)
    actual = ReferenceEvaluator(model).run(None, {x.name: data})
    expected = (
        [-data]
        if operator == "neg"
        else [np.asarray([4.0, 2.0], np.float32), np.asarray([1, 2], np.int32)]
    )
    for result, reference in zip(actual, [*expected, *expected]):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("binding", ["input", "initializer"])
def test_output_allocation_preserves_input_and_initializer_bindings(
    binding: str,
) -> None:
    (eqn,) = jax.make_jaxpr(lambda x: -x)(jnp.ones((3,), dtype=jnp.float32)).jaxpr.eqns
    ctx = IRContext(opset=21)
    if binding == "input":
        source = ctx.add_input_for_invar(eqn.invars[0], 0)
    else:
        source = ctx.bind_const_for_var(eqn.invars[0], np.ones((3,), np.float32))
    ctx.bind_value_for_var(eqn.outvars[0], source)
    assert ctx.get_value_for_var(eqn.outvars[0]) is source

    NegPlugin().lower(ctx, eqn)

    result = ctx.require_value_for_var(eqn.outvars[0])
    assert result is not source
    assert source.producer() is None
    assert source.is_graph_input() if binding == "input" else source.is_initializer()
    assert result.producer().inputs[0] is source
    assert ctx.get_value_for_var(eqn.invars[0]) is source


@pytest.mark.parametrize(
    "source_dtype,target_dtype", [(np.float64, np.float32), (np.float32, np.float64)]
)
def test_parameter_adaptation_preserves_shared_initializer_payload(
    source_dtype, target_dtype
) -> None:
    # Both consumers must retain their own precision, including a deliberate
    # narrowing and the exact zero sign. The initializer remains independently
    # observable before and after serialization.
    payload = np.asarray([1.0 + 2**-30, -0.0], dtype=source_dtype)
    original_bytes = payload.tobytes()
    ctx = IRContext(opset=23)
    source = ctx.builder.add_initializer_from_array(name="shared", array=payload)
    like = ir.val("like", dtype=numpy_dtype_to_ir(target_dtype), shape=[2])
    ctx.builder.inputs.append(like)

    adapted = cast_param_like(ctx, source, like)

    assert adapted is not source
    assert source.dtype == numpy_dtype_to_ir(source_dtype)
    assert const_value_to_numpy(source).tobytes() == original_bytes
    assert adapted.dtype == like.dtype
    assert adapted.shape == source.shape
    ctx.builder.outputs.extend([source, adapted])
    model = ir.serde.serialize_model(ctx.builder.to_ir_model(name="shared_parameter"))
    observed, converted = ReferenceEvaluator(model).run(
        None, {"like": np.zeros((2,), dtype=target_dtype)}
    )
    assert observed.dtype == np.dtype(source_dtype)
    assert observed.tobytes() == original_bytes
    expected = payload.astype(target_dtype)
    np.testing.assert_array_equal(converted, expected)
    np.testing.assert_array_equal(np.signbit(converted), np.signbit(expected))


@pytest.mark.parametrize("dtype", [None, "unsupported-dtype"])
def test_missing_or_unsupported_source_dtype_is_not_substituted(dtype) -> None:
    with pytest.raises(TypeError, match="(Missing traced|Unsupported source) dtype"):
        numpy_dtype_to_ir(dtype)


@pytest.mark.parametrize(
    "change", ["dtype", "zero_sign", "nan_payload", "identical_nan"]
)
def test_initializer_reuse_preserves_exact_typed_payload_and_connections(
    change,
) -> None:
    if change == "dtype":
        original = np.asarray([1], np.float32)
        duplicate = original.astype(np.float64)
    elif change == "zero_sign":
        original = np.asarray([-0.0], np.float32)
        duplicate = np.asarray([0.0], np.float32)
    else:
        original = np.asarray([0x7FC00001], np.uint32).view(np.float32)
        payload = 0x7FC00002 if change == "nan_payload" else 0x7FC00001
        duplicate = np.asarray([payload], np.uint32).view(np.float32)

    builder = IRBuilder(opset=23)
    canonical = builder.add_initializer_from_array(name="shared", array=original)
    consumer = builder.op("Identity", [canonical])
    other = ir.Value(
        name="shared",
        type=ir.TensorType(numpy_dtype_to_ir(duplicate.dtype)),
        shape=ir.Shape(duplicate.shape),
        const_value=ir.tensor(duplicate),
    )
    if change == "identical_nan":
        builder.initializers.append(other)
    else:
        with pytest.raises(ValueError, match="different data"):
            builder.initializers.append(other)
    assert builder.graph.initializers["shared"] is canonical
    assert consumer.producer().inputs[0] is canonical
    assert const_value_to_numpy(canonical).dtype == original.dtype
    assert const_value_to_numpy(canonical).tobytes() == original.tobytes()
