# tests/extra_tests/converter/test_typed_builder.py

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx.converter.ir_builder import (
    IRBuilder,
    JAX_CALLSITE_METADATA_KEY,
    PLUGIN_METADATA_KEY,
)
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.typing_support import IRBuilderProtocol
from jax2onnx.plugins.jax.lax.neg import NegPlugin
from jax2onnx.plugins.jax.lax.top_k import TopKPlugin


def test_typed_single_output_keeps_output_identity_and_metadata() -> None:
    builder: IRBuilderProtocol = IRBuilder(
        opset=18, enable_double_precision=False, enable_stacktrace_metadata=True
    )
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
    concrete = IRBuilder(opset=18, enable_double_precision=False)
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
    builder = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[4])

    outputs = builder.op_multi_out("Split", [x], num_outputs=output_count)

    assert len(outputs) == output_count
    assert len(builder.nodes) == 1


def test_typed_multi_output_rejects_ambiguous_arity_before_mutation() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[4])

    with pytest.raises(ValueError, match="Both num_outputs and outputs"):
        builder.op_multi_out("Split", [x], num_outputs=1, outputs=[ir.Value()])

    assert len(builder.nodes) == 0
    assert not builder.used_opsets


def test_typed_attributes_use_public_ir_attr_constructors() -> None:
    builder: IRBuilderProtocol = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 3])

    result = builder.op("Transpose", [x], {"perm": ir.AttrInt64s("perm", [1, 0])})

    node = result.producer()
    assert node is not None
    assert node.attributes["perm"].as_ints() == (1, 0)


def test_neg_pilot_preserves_allocated_output() -> None:
    eqn = jax.make_jaxpr(lambda x: -x)(jnp.ones((3,), dtype=jnp.float32)).jaxpr.eqns[0]
    ctx = IRContext(opset=18, enable_double_precision=False)
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
    ctx = IRContext(opset=18, enable_double_precision=False)
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
