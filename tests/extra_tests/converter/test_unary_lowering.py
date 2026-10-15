# tests/extra_tests/converter/test_unary_lowering.py

from __future__ import annotations

import jax
import numpy as np
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator
import pytest

from jax2onnx.converter.ir_builder import PLUGIN_METADATA_KEY
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins.jax.numpy.ceil import JnpCeilPlugin
from jax2onnx.plugins.jax.numpy.exp import JnpExpPlugin
from jax2onnx.plugins.jax.numpy.tanh import JnpTanhPlugin


@pytest.mark.parametrize(
    "plugin_class,dtype,expected_ops,expected_dtype,hint,expected_fn",
    [
        (JnpExpPlugin, np.int32, ["Cast", "Exp"], ir.DataType.FLOAT, "jnp_exp", np.exp),
        (JnpCeilPlugin, np.int32, ["Identity"], ir.DataType.INT32, "jnp_ceil", np.ceil),
        (JnpExpPlugin, np.float16, ["Exp"], ir.DataType.FLOAT16, "jnp_exp", np.exp),
        (JnpTanhPlugin, np.float32, ["Tanh"], ir.DataType.FLOAT, "jnp_tanh", np.tanh),
    ],
)
def test_shared_unary_lowering_preserves_output_and_promotion(
    plugin_class, dtype, expected_ops, expected_dtype, hint, expected_fn
) -> None:
    with plugin_class.plugin_binding():
        jaxpr = jax.make_jaxpr(lambda x: plugin_class._PRIM.bind(x))(
            jax.ShapeDtypeStruct((3,), dtype)
        ).jaxpr
    (eqn,) = jaxpr.eqns
    ctx = IRContext(opset=21, enable_double_precision=False, stacktrace_metadata=True)
    ctx.builder.set_current_plugin_identifier("test.Unary.lower", "17")
    x = ctx.add_input_for_invar(eqn.invars[0], 0)
    output = ctx.allocate_value_for_var(eqn.outvars[0], name_hint="provided_output")

    plugin_class().lower(ctx, eqn)

    result = ctx.require_value_for_var(eqn.outvars[0])
    assert result is output
    assert result.name == "provided_output"
    assert result.dtype == expected_dtype
    assert result.shape == ir.Shape([3])
    assert [node.op_type for node in ctx.builder.graph] == expected_ops
    assert all(
        node.metadata_props[PLUGIN_METADATA_KEY].startswith("Unary.lower:")
        for node in ctx.builder.graph
    )
    assert ctx.fresh_name(f"{hint}_in") == f"{hint}_in_1"
    assert ctx.fresh_name(f"{hint}_out") == f"{hint}_out_1"

    ctx.add_outputs_from_vars(eqn.outvars)
    model = ir.to_proto(ctx.builder.to_ir_model(name="unary", ir_version=10))
    data = np.asarray([-1, 0, 1], dtype=dtype)
    (actual,) = ReferenceEvaluator(model).run(None, {x.name: data})
    expected = expected_fn(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def test_shared_unary_lowering_can_replace_an_existing_produced_binding() -> None:
    with JnpTanhPlugin.plugin_binding():
        jaxpr = jax.make_jaxpr(lambda x: JnpTanhPlugin._PRIM.bind(x))(
            jax.ShapeDtypeStruct((3,), np.float32)
        ).jaxpr
    (eqn,) = jaxpr.eqns
    ctx = IRContext(opset=21, enable_double_precision=False)
    ctx.add_input_for_invar(eqn.invars[0], 0)
    plugin = JnpTanhPlugin()
    plugin.lower(ctx, eqn)
    original = ctx.require_value_for_var(eqn.outvars[0])
    original_producer = original.producer()

    plugin.lower(ctx, eqn)

    replacement = ctx.require_value_for_var(eqn.outvars[0])
    assert replacement is not original
    assert replacement.producer() is not original_producer
    assert original.producer() is original_producer
    assert original.name == "jnp_tanh_out_0"
    assert replacement.name == "jnp_tanh_out_2"
    assert replacement.type == original.type
    assert replacement.shape == original.shape
