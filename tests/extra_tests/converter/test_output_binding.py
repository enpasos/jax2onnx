# tests/extra_tests/converter/test_output_binding.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx._compat.jax import DropVar, JaxprEqn, Literal, Var
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.plugins.jax.lax.add import lower_add
from jax2onnx.converter.output_binding import (
    assert_eqn_inputs_bound,
    assert_eqn_outputs_bound,
    finalize_eqn_lowering_outputs,
)


def _context_and_equation() -> tuple[IRContext, JaxprEqn]:
    jaxpr = jax.make_jaxpr(lambda x: x + x)(
        jax.ShapeDtypeStruct((2,), jnp.float32)
    ).jaxpr
    ctx = IRContext(opset=21, enable_double_precision=False, input_specs=[])
    ctx.add_input_for_invar(jaxpr.invars[0], 0)
    return ctx, jaxpr.eqns[0]


def test_finalize_eqn_lowering_outputs_binds_returned_value() -> None:
    ctx, eqn = _context_and_equation()
    returned_value = ctx.builder.inputs[0]

    finalize_eqn_lowering_outputs(
        ctx, eqn, returned_value, primitive_name="sample", eqn_index=0
    )

    assert ctx.lookup_value_for_var(eqn.outvars[0]) is returned_value


def test_legacy_name_binding_resolves_to_the_actual_graph_value() -> None:
    ctx, eqn = _context_and_equation()
    produced_value = ctx.builder.Identity(ctx.builder.inputs[0])
    bound_value = ir.Value(name=produced_value.name)
    ctx.bind_value_for_var(eqn.outvars[0], bound_value)

    finalize_eqn_lowering_outputs(ctx, eqn, None, primitive_name="sample", eqn_index=0)

    assert ctx.lookup_value_for_var(eqn.outvars[0]) is produced_value


def test_canonical_binding_rejects_a_disconnected_same_name_value() -> None:
    ctx, eqn = _context_and_equation()
    produced_value = ctx.builder.Identity(ctx.builder.inputs[0])
    ctx.bind_value_for_var(eqn.outvars[0], ir.Value(name=produced_value.name))

    with pytest.raises(RuntimeError, match="bound output 0 to disconnected value"):
        assert_eqn_outputs_bound(ctx, eqn, primitive_name="sample", eqn_index=0)


def test_finalize_eqn_lowering_outputs_rejects_disconnected_name() -> None:
    ctx, eqn = _context_and_equation()
    ctx.builder.Identity(ctx.builder.inputs[0])
    ctx.bind_value_for_var(eqn.outvars[0], ir.Value(name="other"))

    with pytest.raises(
        RuntimeError, match="bound output 0 to disconnected value 'other'"
    ):
        finalize_eqn_lowering_outputs(
            ctx, eqn, None, primitive_name="sample", eqn_index=0
        )


def test_finalize_eqn_lowering_outputs_reports_unbound_output() -> None:
    ctx, eqn = _context_and_equation()

    with pytest.raises(
        RuntimeError, match="Primitive 'sample' at equation 2 did not bind output 0"
    ):
        finalize_eqn_lowering_outputs(
            ctx, eqn, None, primitive_name="sample", eqn_index=2
        )


@pytest.mark.parametrize("return_all", [True, False])
def test_legacy_multiple_outputs_preserve_existing_bindings(return_all: bool) -> None:
    ctx, eqn = _context_and_equation()
    second_var = Var(eqn.outvars[0].aval)
    eqn = eqn.replace(outvars=[eqn.outvars[0], second_var])
    existing = ctx.builder.inputs[0]
    produced = ctx.builder.Identity(existing)
    ctx.bind_value_for_var(eqn.outvars[0], existing)
    result = (existing, produced) if return_all else [produced]

    finalize_eqn_lowering_outputs(
        ctx, eqn, result, primitive_name="sample", eqn_index=0
    )

    assert ctx.lookup_value_for_var(eqn.outvars[0]) is existing
    assert ctx.lookup_value_for_var(second_var) is produced


def test_legacy_multiple_outputs_skip_dropvars() -> None:
    ctx, eqn = _context_and_equation()
    drop = DropVar(eqn.outvars[0].aval)
    eqn = eqn.replace(outvars=[drop, eqn.outvars[0]])
    produced = ctx.builder.Identity(ctx.builder.inputs[0])

    finalize_eqn_lowering_outputs(
        ctx, eqn, [produced], primitive_name="sample", eqn_index=0
    )

    assert ctx.lookup_value_for_var(eqn.outvars[1]) is produced
    assert ctx.lookup_value_for_var(drop) is None


@pytest.mark.parametrize("result", [False, "name", [ir.Value(name="a"), None]])
def test_legacy_outputs_reject_invalid_result_types(result: object) -> None:
    ctx, eqn = _context_and_equation()

    with pytest.raises(TypeError, match="returned unsupported lowering result"):
        finalize_eqn_lowering_outputs(
            ctx, eqn, result, primitive_name="sample", eqn_index=0
        )


def test_legacy_outputs_report_wrong_result_count() -> None:
    ctx, eqn = _context_and_equation()
    input_value = ctx.builder.inputs[0]

    with pytest.raises(RuntimeError, match="returned 2 value.*1 non-drop outvar"):
        finalize_eqn_lowering_outputs(
            ctx, eqn, [input_value, input_value], primitive_name="sample", eqn_index=0
        )


def test_native_outputs_reject_removed_producers_even_if_name_is_reused() -> None:
    ctx, eqn = _context_and_equation()
    input_value = ctx.builder.inputs[0]
    removed_value = ctx.builder.Identity(input_value)
    producer = removed_value.producer()
    assert producer is not None
    ctx.builder.graph.remove(producer)
    replacement = ctx.builder.Identity(input_value)
    replacement.name = removed_value.name
    ctx.bind_value_for_var(eqn.outvars[0], removed_value)

    with pytest.raises(RuntimeError, match="disconnected value"):
        assert_eqn_outputs_bound(ctx, eqn, primitive_name="sample", eqn_index=0)


def test_native_outputs_reject_foreign_graph_values_with_local_names() -> None:
    ctx, eqn = _context_and_equation()
    foreign_ctx, _ = _context_and_equation()
    ctx.bind_value_for_var(eqn.outvars[0], foreign_ctx.builder.inputs[0])
    assert ctx.builder.inputs[0].name == foreign_ctx.builder.inputs[0].name

    with pytest.raises(RuntimeError, match="disconnected value"):
        assert_eqn_outputs_bound(ctx, eqn, primitive_name="sample", eqn_index=0)


def _literal_context(
    policy: PrecisionPolicy, *, bound: bool = True
) -> tuple[IRContext, JaxprEqn]:
    native = jax.make_jaxpr(lambda value: jax.lax.add(value, np.int32(1)))(
        jax.ShapeDtypeStruct((), np.int32)
    )
    ctx = IRContext(
        opset=23, enable_double_precision=False, input_specs=[], precision_policy=policy
    )
    if bound:
        ctx.add_input_for_invar(native.jaxpr.invars[0], 0)
    equation = native.jaxpr.eqns[0]
    assert isinstance(equation.invars[1], Literal)
    return ctx, equation


@pytest.mark.parametrize("policy", (PrecisionPolicy.SOURCE, PrecisionPolicy.LEGACY))
def test_literal_preflight_preserves_source_and_legacy_materialization(
    policy: PrecisionPolicy,
) -> None:
    ctx, equation = _literal_context(policy)
    assert not ctx.builder.graph.initializers and not tuple(ctx.builder.graph)
    assert_eqn_inputs_bound(ctx, equation, primitive_name="add", eqn_index=0)
    preflight_count = 0 if policy is PrecisionPolicy.SOURCE else 1
    assert len(ctx.builder.graph.initializers) == preflight_count
    assert not tuple(ctx.builder.graph)
    lower_add(ctx, equation)
    assert len(ctx.builder.graph.initializers) == preflight_count + 1
    operation = next(iter(ctx.builder.graph))
    assert operation.op_type == "Add"
    literal = operation.inputs[1]
    assert literal is not None and literal.is_initializer()
    assert literal.const_value is not None
    storage = literal.const_value.numpy()
    assert storage.dtype == np.dtype(np.int32) and storage.shape == ()
    assert storage.tobytes() == np.asarray(1, np.int32).tobytes()


def test_source_literal_constant_folder_needs_no_ir_materialization() -> None:
    ctx, equation = _literal_context(PrecisionPolicy.SOURCE)
    literal = equation.invars[1]
    assert isinstance(literal, Literal)
    folded = ctx.try_evaluate_const(literal)
    assert folded is not None
    assert folded.dtype == np.dtype(np.int32) and folded.shape == ()
    assert folded.tobytes() == np.asarray(1, np.int32).tobytes()
    assert_eqn_inputs_bound(ctx, equation, primitive_name="add", eqn_index=0)
    assert not ctx.builder.graph.initializers and not tuple(ctx.builder.graph)


@pytest.mark.parametrize("policy", (PrecisionPolicy.SOURCE, PrecisionPolicy.LEGACY))
def test_literal_preflight_still_rejects_an_unbound_native_variable(
    policy: PrecisionPolicy,
) -> None:
    ctx, equation = _literal_context(policy, bound=False)
    with pytest.raises(RuntimeError, match="unbound input 0"):
        assert_eqn_inputs_bound(ctx, equation, primitive_name="add", eqn_index=3)
    assert not ctx.builder.graph.initializers and not tuple(ctx.builder.graph)


@pytest.mark.parametrize("policy", (PrecisionPolicy.SOURCE, PrecisionPolicy.LEGACY))
def test_literal_preflight_keeps_dropvar_semantics(policy: PrecisionPolicy) -> None:
    ctx, equation = _literal_context(policy, bound=False)
    drop = DropVar(equation.invars[0].aval)
    equation = equation.replace(invars=[drop, equation.invars[1]])
    assert_eqn_inputs_bound(ctx, equation, primitive_name="add", eqn_index=0)
    assert len(ctx.builder.graph.initializers) == (
        0 if policy is PrecisionPolicy.SOURCE else 1
    )
    assert not tuple(ctx.builder.graph)
