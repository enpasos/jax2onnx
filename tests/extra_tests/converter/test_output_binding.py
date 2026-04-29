# tests/extra_tests/converter/test_output_binding.py

from __future__ import annotations

from types import SimpleNamespace

import onnx_ir as ir
import pytest

from jax2onnx.converter.output_binding import finalize_eqn_lowering_outputs


def _ctx_with_graph_input(value: ir.Value):
    ctx = SimpleNamespace(
        builder=SimpleNamespace(
            inputs=[value],
            initializers=[],
            nodes=[],
            _var2val={},
        )
    )
    ctx.bind_value_for_var = lambda var, bound: ctx.builder._var2val.__setitem__(
        var, bound
    )
    return ctx


def _ctx_with_node_output(value: ir.Value):
    input_value = ir.Value(name="input")
    node = ir.Node("", "Identity", [input_value], outputs=[value], name="identity")
    ctx = SimpleNamespace(
        builder=SimpleNamespace(
            inputs=[input_value],
            initializers=[],
            nodes=[node],
            _var2val={},
        )
    )
    ctx.bind_value_for_var = lambda var, bound: ctx.builder._var2val.__setitem__(
        var, bound
    )
    return ctx


def test_finalize_eqn_lowering_outputs_binds_returned_value() -> None:
    outvar = object()
    returned_value = ir.Value(name="returned")
    ctx = _ctx_with_graph_input(returned_value)
    eqn = SimpleNamespace(outvars=[outvar])

    finalize_eqn_lowering_outputs(
        ctx,
        eqn,
        returned_value,
        primitive_name="sample",
        eqn_index=0,
    )

    assert ctx.builder._var2val[outvar] is returned_value


def test_finalize_eqn_lowering_outputs_accepts_name_matched_graph_value() -> None:
    outvar = object()
    produced_value = ir.Value(name="produced")
    bound_value = ir.Value(name="produced")
    ctx = _ctx_with_node_output(produced_value)
    ctx.builder._var2val[outvar] = bound_value
    eqn = SimpleNamespace(outvars=[outvar])

    finalize_eqn_lowering_outputs(
        ctx,
        eqn,
        None,
        primitive_name="sample",
        eqn_index=0,
    )


def test_finalize_eqn_lowering_outputs_rejects_disconnected_name() -> None:
    outvar = object()
    produced_value = ir.Value(name="produced")
    bound_value = ir.Value(name="other")
    ctx = _ctx_with_node_output(produced_value)
    ctx.builder._var2val[outvar] = bound_value
    eqn = SimpleNamespace(outvars=[outvar])

    with pytest.raises(
        RuntimeError,
        match="bound output 0 to disconnected value 'other'",
    ):
        finalize_eqn_lowering_outputs(
            ctx,
            eqn,
            None,
            primitive_name="sample",
            eqn_index=0,
        )


def test_finalize_eqn_lowering_outputs_reports_unbound_output() -> None:
    outvar = object()
    ctx = _ctx_with_graph_input(ir.Value(name="input"))
    eqn = SimpleNamespace(outvars=[outvar])

    with pytest.raises(
        RuntimeError,
        match="Primitive 'sample' at equation 2 did not bind output 0",
    ):
        finalize_eqn_lowering_outputs(
            ctx,
            eqn,
            None,
            primitive_name="sample",
            eqn_index=2,
        )
