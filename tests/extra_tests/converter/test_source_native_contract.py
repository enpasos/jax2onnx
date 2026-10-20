# tests/extra_tests/converter/test_source_native_contract.py

"""Strict native inference shared by existing SOURCE loop/scientific owners.

These are abstract-contract tests. They establish no new operation, numerical,
provider or opset admission; existing owner tests retain child/capture checks.
"""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import jax
from jax import export as jax_export
from jax._src import core as native_core
from jax._src import state
from jax._src.mesh import AbstractMesh, use_abstract_mesh
from jax.experimental.xla_metadata import set_xla_metadata
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np
import pytest

from jax2onnx._compat.jax import (
    Primitive,
    ShapedArray,
    Var,
    infer_native_abstract_results,
)
from jax2onnx.converter import source_native_contract as contract
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import SourcePrecisionError, _Site


_FAMILIES = ("scan", "while", "scatter", "scatter-add", "scatter-mul", "remat2")


@pytest.fixture(autouse=True)
def _restore_context() -> Iterator[None]:
    context, policy = native_core.JaxprEqnContext(), active_precision_policy()
    x64 = jax.config.x64_enabled
    with jax.enable_x64(True):
        yield
    assert native_core.JaxprEqnContext() is context
    assert jax.config.x64_enabled is x64
    assert active_precision_policy() is policy


def _trace(kind: str, dtype: Any = np.float64) -> Any:
    spec = jax.ShapeDtypeStruct((4,), dtype)
    if kind == "scan":

        def target(value: Any) -> Any:
            return jax.lax.scan(
                lambda carry, _: (carry + dtype(1), carry), value, None, length=2
            )

        native = jax.make_jaxpr(target)(spec)
    elif kind == "while":

        def target(value: Any) -> Any:
            return jax.lax.while_loop(
                lambda carry: carry[0] < np.int32(2),
                lambda carry: (carry[0] + np.int32(1), carry[1] + dtype(1)),
                (np.int32(0), value),
            )

        native = jax.make_jaxpr(target)(spec)
    elif kind == "remat2":
        native = jax.make_jaxpr(jax.checkpoint(lambda value: (value, -value)))(spec)
    else:
        operation = {
            "scatter": jax.lax.scatter,
            "scatter-add": jax.lax.scatter_add,
            "scatter-mul": jax.lax.scatter_mul,
        }[kind]
        dimensions = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )

        def target(value: Any, indices: Any, updates: Any) -> Any:
            return operation(
                value,
                indices,
                updates,
                dimensions,
                unique_indices=True,
                mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
            )

        native = jax.make_jaxpr(target)(
            spec,
            jax.ShapeDtypeStruct((2, 1), np.int32),
            jax.ShapeDtypeStruct((2,), dtype),
        )
    body = native.jaxpr  # Native 0.11 returns itself; unwrap only once.
    assert len(body.eqns) == 1
    equation = body.eqns[0]
    assert equation.primitive.name == kind
    return equation


def _parameters(eqn: Any) -> set[str]:
    if eqn.primitive is jax.lax.scan_p:
        return {"jaxpr", "reverse", "length", "unroll"} | (
            {"ft_in", "ft_out"}
            if "ft_in" in eqn.params
            else {"num_consts", "num_carry"}
        )
    if eqn.primitive is jax.lax.while_p:
        return {"cond_jaxpr", "body_jaxpr", "cond_nconsts", "body_nconsts"}
    if eqn.primitive in (
        jax.lax.scatter_p,
        jax.lax.scatter_add_p,
        jax.lax.scatter_mul_p,
    ):
        return {
            "dimension_numbers",
            "indices_are_sorted",
            "unique_indices",
            "mode",
            "update_jaxpr",
            "update_consts",
        }
    assert eqn.primitive.name == "remat2"
    return {"jaxpr", "prevent_cse", "differentiated", "policy"}


def _equation(eqn: Any, **changes: Any) -> Any:
    fields = {
        "primitive": eqn.primitive,
        "invars": eqn.invars,
        "outvars": eqn.outvars,
        "params": eqn.params,
        "effects": eqn.effects,
        "ctx": eqn.ctx,
    }
    fields.update(changes)
    return SimpleNamespace(**fields)


def _validate(
    eqn: Any, *, primitive: Any = None, parameters: set[str] | None = None
) -> None:
    contract.validate_native_results(
        eqn,
        _Site("native-contract", 21, eqn.primitive.name),
        primitive=eqn.primitive if primitive is None else primitive,
        parameters=_parameters(eqn) if parameters is None else parameters,
    )


def _bindings(eqn: Any) -> tuple[Any, ...]:
    return (
        id(eqn.params),
        tuple((name, id(value)) for name, value in eqn.params.items()),
        tuple((id(var), id(var.aval)) for var in (*eqn.invars, *eqn.outvars)),
        id(eqn.ctx),
    )


def _forged_aval(
    aval: Any, *, aval_type: type[Any] = ShapedArray, **changes: Any
) -> Any:
    # A fresh annotation bypasses native constructor canonicalization without
    # mutating the interned aval shared by the genuine trace.
    result = object.__new__(aval_type)
    for name in (
        "shape",
        "dtype",
        "weak_type",
        "sharding",
        "manual_axis_type",
        "memory_space",
        "layout",
        "_stripped_weak_type",
    ):
        if hasattr(aval, name):
            object.__setattr__(result, name, getattr(aval, name))
    for name, value in changes.items():
        object.__setattr__(result, name, value)
    return result


@pytest.mark.parametrize("kind", _FAMILIES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_selected_native_rules_preserve_bindings_through_exporter_worlds(
    kind: str, dtype: Any
) -> None:
    from jax2onnx.converter.conversion_api import _activate_plugin_worlds

    eqn = _trace(kind, dtype)
    before = _bindings(eqn)
    avals, effects = infer_native_abstract_results(eqn, eqn.primitive)
    assert type(avals) is tuple and len(avals) == len(eqn.outvars)
    assert not effects and not eqn.effects
    assert all(
        (actual.dtype, actual.shape, actual.weak_type)
        == (var.aval.dtype, var.aval.shape, var.aval.weak_type)
        for actual, var in zip(avals, eqn.outvars, strict=True)
    )
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        _validate(eqn)
    assert _bindings(eqn) == before


@pytest.mark.parametrize("kind", _FAMILIES)
def test_replaced_primitive_evaluator_cannot_certify_forged_outputs(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    eqn = _trace(kind)
    calls = []

    def echo(*args: Any, **params: Any) -> Any:
        calls.append((args, params))
        return [var.aval for var in eqn.outvars], frozenset()

    monkeypatch.setattr(eqn.primitive, "abstract_eval", echo)
    _validate(eqn)
    wrong = list(eqn.outvars)
    wrong[0] = Var(wrong[0].aval.update(dtype=np.float32))
    with pytest.raises(SourcePrecisionError):
        _validate(_equation(eqn, outvars=wrong))
    assert not calls


def test_multiple_result_normalization_uses_the_frozen_native_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eqn = _trace("scan")
    assert eqn.primitive.multiple_results is True and len(eqn.outvars) == 2
    monkeypatch.setattr(eqn.primitive, "multiple_results", False)
    avals, effects = infer_native_abstract_results(eqn, eqn.primitive)
    assert type(avals) is tuple and len(avals) == 2 and not effects
    _validate(eqn)


@pytest.mark.parametrize(
    "kind", ["same-name", "unselected-native", "unknown-parameter"]
)
def test_unowned_identity_or_parameters_fail_before_the_oracle(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    eqn = _trace("remat2")
    primitive, parameters = eqn.primitive, _parameters(eqn)
    if kind == "same-name":
        eqn = _equation(eqn, primitive=Primitive(eqn.primitive.name))
    elif kind == "unselected-native":
        eqn = jax.make_jaxpr(lambda value: value + np.float64(1))(
            jax.ShapeDtypeStruct((4,), np.float64)
        ).jaxpr.eqns[0]
        primitive, parameters = eqn.primitive, set(eqn.params)
    else:
        eqn = _equation(eqn, params={**eqn.params, "unknown-native-field": object()})
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise AssertionError("unowned input reached the inference boundary")

    if kind == "unselected-native":
        # The compatibility resolver, rather than the converter wrapper, owns
        # the six-rule selection. A primitive's method must never be invoked.
        monkeypatch.setattr(primitive, "abstract_eval", forbidden)
    else:
        monkeypatch.setattr(contract, "infer_native_abstract_results", forbidden)
    with pytest.raises(SourcePrecisionError):
        _validate(eqn, primitive=primitive, parameters=parameters)
    assert not calls


@pytest.mark.parametrize("kind", ["scan", "scatter"])
@pytest.mark.parametrize(
    "corruption",
    [
        "missing-result",
        "extra-result",
        "dtype",
        "shape",
        "weakness",
        "nonboolean-weakness",
        "negative-dimension",
        "boolean-dimension",
    ],
)
def test_declared_result_contract_is_stricter_than_native_typematch(
    kind: str, corruption: str
) -> None:
    eqn = _trace(kind)
    outputs = list(eqn.outvars)
    aval = outputs[0].aval
    if corruption == "missing-result":
        outputs.pop()
    elif corruption == "extra-result":
        outputs.append(outputs[0])
    else:
        changes = {
            "dtype": {"dtype": np.dtype(np.float32)},
            "shape": {"shape": (5,)},
            "weakness": {"weak_type": not aval.weak_type},
            "nonboolean-weakness": {"weak_type": 1},
            "negative-dimension": {"shape": (-1,)},
            "boolean-dimension": {"shape": (True,)},
        }[corruption]
        outputs[0] = Var(_forged_aval(aval, **changes))
    with pytest.raises(SourcePrecisionError):
        _validate(_equation(eqn, outvars=outputs))


def test_same_symbol_name_from_a_foreign_scope_is_not_native_shape_equality() -> None:
    local = jax_export.symbolic_shape("B", scope=jax_export.SymbolicScope())
    foreign = jax_export.symbolic_shape("B", scope=jax_export.SymbolicScope())
    assert str(local[0]) == str(foreign[0]) == "B"
    eqn = jax.make_jaxpr(jax.checkpoint(lambda value: -value))(
        jax.ShapeDtypeStruct(local, np.float64)
    ).jaxpr.eqns[0]
    _validate(eqn)
    wrong = Var(eqn.outvars[0].aval.update(shape=foreign))
    with pytest.raises(SourcePrecisionError):
        _validate(_equation(eqn, outvars=[wrong]))


@pytest.mark.parametrize("role", ["input", "declared", "inferred"])
@pytest.mark.parametrize(
    "corruption",
    [
        "mesh",
        "partition-spec",
        "manual-axis",
        "manual-lookalike",
        "host-memory",
        "any-memory",
        "foreign-memory",
        "aval-lookalike",
        "ref",
        "token",
    ],
)
def test_nondefault_metadata_is_rejected_on_each_side_of_inference(
    role: str, corruption: str
) -> None:
    eqn = _trace("remat2")
    aval = eqn.invars[0].aval
    changes: dict[str, Any] = {}
    if corruption == "mesh":
        changes["sharding"] = NamedSharding(
            AbstractMesh((2,), ("worker",)), PartitionSpec(None)
        )
    elif corruption == "partition-spec":
        changes["sharding"] = NamedSharding(aval.sharding.mesh, PartitionSpec())
    elif corruption == "manual-axis":
        changes["manual_axis_type"] = native_core.ManualAxisType(
            varying=frozenset({"worker"})
        )
    elif corruption == "manual-lookalike":
        changes["manual_axis_type"] = SimpleNamespace(
            empty=True, varying=frozenset(), unreduced=frozenset(), reduced=frozenset()
        )
    elif corruption.endswith("memory"):
        changes["memory_space"] = {
            "host-memory": native_core.MemorySpace.Host,
            "any-memory": native_core.MemorySpace.Any,
            "foreign-memory": "Device",
        }[corruption]
    bad = _forged_aval(aval, **changes)
    if corruption == "aval-lookalike":
        bad = SimpleNamespace(
            shape=aval.shape, dtype=aval.dtype, weak_type=aval.weak_type
        )
    elif corruption == "ref":
        bad = state.AbstractRef(aval)
    elif corruption == "token":
        bad = native_core.abstract_token
    if role == "input":
        eqn = _equation(eqn, invars=[SimpleNamespace(aval=bad)])
    elif role == "declared":
        eqn = _equation(eqn, outvars=[SimpleNamespace(aval=bad), eqn.outvars[1]])
    else:
        # Only the inferred annotation changes. The canonical remat evaluator
        # reads this real native body's output aval; no substitute oracle runs.
        body = eqn.params["jaxpr"]
        body = body.replace(outvars=[SimpleNamespace(aval=bad), body.outvars[1]])
        eqn = _equation(eqn, params={**eqn.params, "jaxpr": body})
    with pytest.raises(SourcePrecisionError):
        _validate(eqn)


def test_primary_layout_metadata_is_not_silently_erased() -> None:
    eqn = _trace("remat2")
    aval = eqn.invars[0].aval
    if not hasattr(aval, "layout"):
        assert not hasattr(eqn.ctx, "cur_layout_mode")
        return  # Secondary native avals genuinely have no layout role.
    from jax._src.layout import LayoutMode, use_layout_mode

    for role in ("input", "declared", "inferred"):
        bad = _forged_aval(aval, layout=object())
        if role == "input":
            changed = _equation(eqn, invars=[SimpleNamespace(aval=bad)])
        elif role == "declared":
            changed = _equation(
                eqn, outvars=[SimpleNamespace(aval=bad), eqn.outvars[1]]
            )
        else:
            body = eqn.params["jaxpr"]
            changed = _equation(
                eqn,
                params={
                    **eqn.params,
                    "jaxpr": body.replace(
                        outvars=[SimpleNamespace(aval=bad), body.outvars[1]]
                    ),
                },
            )
        with pytest.raises(SourcePrecisionError):
            _validate(changed)
    with use_layout_mode(LayoutMode.JAX):
        context = native_core.JaxprEqnContext()
    with pytest.raises(SourcePrecisionError):
        _validate(_equation(eqn, ctx=context))


@pytest.mark.parametrize(
    "context", [None, SimpleNamespace(manager=None), "native-context"]
)
def test_foreign_equation_context_does_not_supply_a_manager(context: Any) -> None:
    eqn = _trace("remat2")
    with pytest.raises(SourcePrecisionError):
        _validate(_equation(eqn, ctx=context))


def test_nonempty_native_equation_mesh_is_not_default_context() -> None:
    eqn = _trace("remat2")
    mesh = AbstractMesh((2,), ("worker",))
    with use_abstract_mesh(mesh):
        context = native_core.JaxprEqnContext()
    assert type(context) is native_core.JaxprEqnContext
    assert context.cur_abstract_mesh is mesh and not mesh.empty
    with pytest.raises(SourcePrecisionError):
        _validate(_equation(eqn, ctx=context))


def test_shaped_array_subclass_cannot_supply_an_ordinary_native_signature() -> None:
    class DerivedAval(ShapedArray):
        __slots__ = ()

    eqn = _trace("remat2")
    aval = _forged_aval(eqn.invars[0].aval, aval_type=DerivedAval)
    assert isinstance(aval, ShapedArray) and type(aval) is not ShapedArray
    with pytest.raises(SourcePrecisionError):
        _validate(_equation(eqn, invars=[SimpleNamespace(aval=aval)]))


@pytest.mark.parametrize("role", ["input", "declared"])
def test_variable_quantization_metadata_is_not_erased(role: str) -> None:
    eqn = _trace("remat2")
    var = SimpleNamespace(aval=eqn.invars[0].aval, initial_qdd=object(), final_qdd=None)
    changes = (
        {"invars": [var]} if role == "input" else {"outvars": [var, eqn.outvars[1]]}
    )
    with pytest.raises(SourcePrecisionError):
        _validate(_equation(eqn, **changes))


@pytest.mark.parametrize("fail", [False, True])
def test_recorded_native_context_is_active_and_restored_on_both_exits(
    fail: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    with jax.threefry_partitionable(False), set_xla_metadata(contract_role="recorded"):
        eqn = _trace("remat2")
    original = native_core.positional_effects
    observed = []

    def effects(body: Any) -> Any:
        observed.append(native_core.JaxprEqnContext())
        assert observed[-1].threefry_partitionable is False
        assert observed[-1].xla_metadata["contract_role"] == "recorded"
        if fail:
            raise ValueError("controlled canonical-evaluator failure")
        return original(body)

    monkeypatch.setattr(native_core, "positional_effects", effects)
    with jax.threefry_partitionable(True), set_xla_metadata(contract_role="caller"):
        before = native_core.JaxprEqnContext()
        if fail:
            with pytest.raises(SourcePrecisionError):
                _validate(eqn)
        else:
            _validate(eqn)
        assert native_core.JaxprEqnContext() is before
    assert len(observed) == 1


@pytest.mark.parametrize(
    "declaration", ["native-effects", "forged-empty", "forged-extra"]
)
def test_native_and_declared_effects_must_both_be_empty(declaration: str) -> None:
    called = []

    def effect(value: Any) -> Any:
        jax.debug.callback(lambda result: called.append(result), value)
        return value

    effectful = jax.make_jaxpr(jax.checkpoint(effect))(
        jax.ShapeDtypeStruct((4,), np.float64)
    ).jaxpr.eqns[0]
    assert effectful.effects and not called
    if declaration == "forged-extra":
        eqn = _equation(_trace("remat2"), effects=effectful.effects)
    elif declaration == "forged-empty":
        eqn = _equation(effectful, effects=frozenset())
    else:
        eqn = effectful
    with pytest.raises(SourcePrecisionError):
        _validate(eqn)
    assert not called


def test_native_qdd_wrapper_is_not_an_ordinary_tensor() -> None:
    eqn = _trace("remat2")
    wrapper = getattr(native_core, "AvalMutableQDD", None)
    if wrapper is None:
        assert not hasattr(native_core, "AvalMutableQDD")
        return  # Primary removed this native carrier.
    bad = wrapper(eqn.invars[0].aval, None)
    with pytest.raises(SourcePrecisionError):
        _validate(_equation(eqn, invars=[SimpleNamespace(aval=bad)]))
