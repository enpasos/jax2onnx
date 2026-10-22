# tests/extra_tests/converter/test_source_native_activation.py

"""Native SOURCE activation, exact owned nesting, and genuine Function state."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from functools import partial
import inspect
from types import CellType, CodeType, FunctionType, ModuleType, SimpleNamespace
from typing import Any
import weakref

from flax import linen, nnx
import jax
from jax import export as jax_export
import jax.numpy as jnp
from jax._src import custom_derivatives as native_cd, linear_util as native_lu
from jax._src.interpreters import partial_eval as native_pe
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import (
    ClosedJaxpr,
    Jaxpr,
    Literal,
    PRNGKeyDType,
    Var,
    _native_context_fields,
)
from jax2onnx._source_state_preparation import prepare_nnx_call
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins import _patching, plugin_system as ps
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec, apply_patches
from jax2onnx.plugins.plugin_system import FunctionPlugin, PrimitiveLeafPlugin
from tests.extra_tests.converter.test_source_mlp_runtime_acceptance import (
    _runtime_configuration,
)
from tests.extra_tests.converter.test_source_model_native_bindings import _fresh, _same
from tests.extra_tests.converter.test_source_native_tile import (
    _check_one_artifact,
    _sample,
)
from tests.extra_tests.converter.test_source_state_preparation import (
    _clone,
    _native_words,
)


@contextmanager
def _world(policy: PrecisionPolicy, *, body: bool = False) -> Iterator[None]:
    activate = (
        ps._activate_full_plugin_worlds_for_body if body else _activate_plugin_worlds
    )
    with precision_policy_scope(policy), activate():
        yield


def _body(closed: Any) -> Any:
    inner = getattr(closed, "jaxpr", None)
    return inner if inner is not None and inner is not closed else closed


def _array(value: Any) -> tuple[Any, ...]:
    if hasattr(value, "dtype") and isinstance(value.dtype, PRNGKeyDType):
        return (
            "key",
            str(jax.random.key_impl(value)),
            _array(jax.random.key_data(value)),
        )
    array = np.asarray(value)
    return (array.dtype.str, array.shape, array.tobytes())


def _dimension(value: Any) -> Any:
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value), id(value._scope)


def _aval(value: Any) -> tuple[Any, ...]:
    return (
        type(value),
        value.dtype,
        tuple(_dimension(d) for d in value.shape),
        bool(value.weak_type),
        getattr(value, "sharding", None),
        getattr(value, "memory_space", None),
    )


def _native_nested_code(owner: Any, name: str) -> CodeType:
    assert type(owner) is FunctionType
    candidates = [
        code
        for code in owner.__code__.co_consts
        if type(code) is CodeType and code.co_name == name
    ]
    assert len(candidates) == 1, "Unknown native custom-JVP metadata protocol"
    return candidates[0]


_NATIVE_CUSTOM_JVP = native_cd.custom_jvp_call_p
_NATIVE_MEMOIZE = native_pe._memoize
_NATIVE_JVP_OWNER = native_pe.DynamicJaxprTrace.process_custom_jvp_call
_NATIVE_MEMO_CODE = _native_nested_code(_NATIVE_MEMOIZE, "memoized")
_NATIVE_JVP_THUNK_CODE = _native_nested_code(_NATIVE_JVP_OWNER, "jvp_jaxpr_thunk")


def _debug_recipe(debug: Any) -> Any:
    assert type(debug) is native_lu.DebugInfo
    assert debug._fields == ("traced_for", "func_src_info", "arg_names", "result_paths")
    assert type(debug.traced_for) is str and type(debug.func_src_info) is str
    assert debug.arg_names is None or (
        type(debug.arg_names) is tuple and all(type(v) is str for v in debug.arg_names)
    )
    assert debug.result_paths is None or (
        type(debug.result_paths) is tuple
        and all(type(v) is str for v in debug.result_paths)
    )
    return type(debug), tuple(debug)


def _native_jvp_recipe(wrapped: Any, equation: Any) -> Any:
    """Validate only known native derivative metadata; never invoke its thunk.

    Forward JAXPR/SSA/params/context stay exact. Only fresh empty memoization
    state, its sentinel, and trace-instance identity are excluded here. This
    protocol establishes no derivative numerical equivalence.
    """
    assert native_cd.custom_jvp_call_p is _NATIVE_CUSTOM_JVP
    assert native_pe._memoize is _NATIVE_MEMOIZE
    assert native_pe.DynamicJaxprTrace.process_custom_jvp_call is _NATIVE_JVP_OWNER
    assert equation.primitive is _NATIVE_CUSTOM_JVP
    assert set(equation.params) == {
        "call_jaxpr",
        "jvp_jaxpr_fun",
        "num_consts",
        "symbolic_zeros",
    }
    assert type(wrapped) is native_lu.WrappedFun
    assert native_lu.WrappedFun.__slots__ == (
        "f",
        "f_transformed",
        "transforms",
        "stores",
        "params",
        "in_type",
        "debug_info",
    )
    assert all(
        type(getattr(wrapped, name)) is tuple and not getattr(wrapped, name)
        for name in ("transforms", "stores", "params")
    )
    assert wrapped.in_type is None
    function = wrapped.f
    assert type(function) is FunctionType
    assert function.__code__ is _NATIVE_MEMO_CODE
    assert function.__globals__ is vars(native_pe)
    assert function.__defaults__ is function.__kwdefaults__ is None
    assert function.__code__.co_freevars == ("cells", "fn", "sentinel")
    assert function.__closure__ is not None and len(function.__closure__) == 3
    cells, thunk, sentinel = (cell.cell_contents for cell in function.__closure__)
    assert type(cells) is dict and not cells and type(sentinel) is object
    transformed = wrapped.f_transformed
    assert type(transformed) is partial and transformed.func is function
    assert transformed.args == () and transformed.keywords == {}
    assert type(thunk) is FunctionType and thunk.__code__ is _NATIVE_JVP_THUNK_CODE
    assert thunk.__globals__ is vars(native_pe)
    assert thunk.__defaults__ is thunk.__kwdefaults__ is None
    assert thunk.__code__.co_freevars == ("in_avals", "in_tangent_avals", "jvp", "self")
    assert thunk.__closure__ is not None and len(thunk.__closure__) == 4
    in_avals, tangent_avals, jvp, trace = (
        cell.cell_contents for cell in thunk.__closure__
    )
    assert type(in_avals) is type(tangent_avals) is list
    assert type(jvp) is native_lu.WrappedFun
    assert type(trace) is native_pe.DynamicJaxprTrace
    num_consts = equation.params["num_consts"]
    assert type(num_consts) is int and 0 <= num_consts <= len(equation.invars)
    assert type(equation.params["symbolic_zeros"]) is bool
    expected = tuple(var.aval for var in equation.invars[num_consts:])
    assert tuple(_aval(aval) for aval in in_avals) == tuple(
        _aval(aval) for aval in expected
    )
    assert tuple(_aval(aval) for aval in tangent_avals) == tuple(
        _aval(aval.to_tangent_aval()) for aval in expected
    )
    return (
        type(wrapped),
        _NATIVE_MEMO_CODE,
        id(vars(native_pe)),
        _NATIVE_JVP_THUNK_CODE,
        tuple(_aval(aval) for aval in in_avals),
        tuple(_aval(aval) for aval in tangent_avals),
        type(jvp),
        type(trace),
        _debug_recipe(wrapped.debug_info),
        _debug_recipe(jvp.debug_info),
    )


def _cond_direct_aliases(
    equation: Any, recipes: tuple[Any, ...]
) -> tuple[tuple[Any, ...], tuple[int, ...], dict[int, int]]:
    """Hoist only common pure direct native branch formal references.

    Recipe input tokens are established from actual Var identity, including
    exact genuine Function argument/result substitution. Every branch node,
    operand, context and non-alias result remains present and unchanged.
    """
    assert equation.primitive is jax.lax.cond_p
    assert set(equation.params) == {"branches"}, "Unknown canonical cond schema"
    branches = equation.params["branches"]
    assert type(branches) is tuple and len(branches) >= 2
    assert len(recipes) == len(branches)
    assert all(
        type(branch) is ClosedJaxpr and type(_body(branch)) is Jaxpr
        for branch in branches
    )
    positions = tuple(range(len(equation.outvars)))
    if equation.effects or any(
        _body(branch).effects or recipe[2]
        for branch, recipe in zip(branches, recipes, strict=True)
    ):
        return recipes, positions, {}
    operands = equation.invars[1:]
    for branch, recipe in zip(branches, recipes, strict=True):
        body = _body(branch)
        assert len(body.invars) == len(operands)
        assert all(
            type(formal) is Var and _aval(formal.aval) == _aval(actual.aval)
            for formal, actual in zip(body.invars, operands, strict=True)
        )
        assert len(body.outvars) == len(recipe[1]) == len(equation.outvars)
        assert all(
            _aval(formal.aval) == _aval(actual.aval)
            for formal, actual in zip(body.outvars, equation.outvars, strict=True)
        )
    aliases: dict[int, int] = {}
    for position in positions:
        references = tuple(recipe[1][position] for recipe in recipes)
        if not all(
            type(ref) is tuple
            and len(ref) == 3
            and ref[0] == "input"
            and type(ref[1]) is int
            for ref in references
        ):
            continue
        formal_index = references[0][1]
        if not all(ref[1] == formal_index for ref in references):
            continue
        assert 0 <= formal_index < len(operands)
        signature = _aval(operands[formal_index].aval)
        assert all(ref[2] == signature for ref in references)
        assert _aval(equation.outvars[position].aval) == signature
        assert all(
            _aval(_body(branch).invars[formal_index].aval) == signature
            for branch in branches
        )
        aliases[position] = formal_index
    if not aliases or len(aliases) == len(positions):
        # An all-result conditional elimination needs a separate proof.
        return recipes, positions, {}
    retained = tuple(position for position in positions if position not in aliases)
    normalized = tuple(
        (recipe[0], tuple(recipe[1][position] for position in retained), recipe[2])
        for recipe in recipes
    )
    return normalized, retained, aliases


def _recipe(closed: Any, *, expand_functions: bool = False) -> Any:
    """Compare actual SSA/primitive identities; expand only owned Functions.

    Child JIT/cond/loop scopes stay separate native equations. Constants and
    literals retain their exact payloads; symbolic dimensions retain scope
    identities. This helper neither admits primitives nor exports a model.
    """
    nodes: list[Any] = []

    def parameter(value: Any) -> Any:
        candidate = _body(value)
        if hasattr(candidate, "eqns") and hasattr(candidate, "invars"):
            return _recipe(value, expand_functions=expand_functions)
        if isinstance(value, Mapping):
            return type(value), tuple((k, parameter(v)) for k, v in value.items())
        if isinstance(value, (tuple, list)):
            return type(value), tuple(parameter(v) for v in value)
        if isinstance(value, (jax.Array, np.ndarray, np.generic)):
            return _array(value)
        if isinstance(value, float):
            return _array(np.float64(value))
        if type(value).__module__ == "jax._src.export.shape_poly":
            return _dimension(value)
        return value

    def enter(graph: Any, formals: tuple[Any, ...] | None = None) -> tuple[Any, ...]:
        body = _body(graph)
        env = {
            id(var): ("input", i, _aval(var.aval)) for i, var in enumerate(body.invars)
        }
        if formals is not None:
            assert len(formals) == len(body.invars)
            env.update(
                (id(var), ref) for var, ref in zip(body.invars, formals, strict=True)
            )
        constants = tuple(getattr(graph, "consts", ()))
        assert len(constants) == len(body.constvars)
        env.update(
            (id(var), ("constant", _aval(var.aval), _array(value)))
            for var, value in zip(body.constvars, constants, strict=True)
        )

        def reference(var: Any) -> Any:
            return (
                ("literal", _aval(var.aval), _array(var.val))
                if isinstance(var, Literal)
                else env[id(var)]
            )

        for equation in body.eqns:
            inputs = tuple(reference(v) for v in equation.invars)
            plugin = ps.PLUGIN_REGISTRY.get(equation.primitive.name)
            if (
                expand_functions
                and isinstance(plugin, FunctionPlugin)
                and equation.primitive is plugin._source_primitive
            ):
                params = dict(equation.params)
                trace = params.pop(ps._SOURCE_TRACE)
                assert tuple(trace.output_avals) == tuple(
                    v.aval for v in equation.outvars
                )
                active = ps._IN_FUNCTION_BUILD.get()
                token = ps._IN_FUNCTION_BUILD.set(active | {plugin.name})
                try:
                    with _world(PrecisionPolicy.SOURCE, body=True), trace.activate():
                        child = jax.make_jaxpr(
                            lambda *xs: jax.tree.leaves(
                                plugin._source_call(xs, params, trace=trace)
                            )
                        )(
                            *(
                                jax.ShapeDtypeStruct(
                                    v.aval.shape,
                                    v.aval.dtype,
                                    weak_type=v.aval.weak_type,
                                )
                                for v in equation.invars
                            )
                        )
                    assert frozenset(_body(child).effects) == frozenset(
                        equation.effects
                    )
                    outputs = enter(child, inputs)
                finally:
                    ps._IN_FUNCTION_BUILD.reset(token)
                assert len(outputs) == len(equation.outvars)
                env.update(
                    (id(v), ref)
                    for v, ref in zip(equation.outvars, outputs, strict=True)
                )
                continue
            position = len(nodes)
            retained = tuple(range(len(equation.outvars)))
            aliases: dict[int, int] = {}
            if equation.primitive is _NATIVE_CUSTOM_JVP:
                derivative = _native_jvp_recipe(
                    equation.params.get("jvp_jaxpr_fun"), equation
                )
                parameters = (
                    type(equation.params),
                    tuple(
                        (
                            name,
                            derivative if name == "jvp_jaxpr_fun" else parameter(value),
                        )
                        for name, value in equation.params.items()
                    ),
                )
            elif equation.primitive is jax.lax.cond_p:
                branches = equation.params.get("branches")
                assert type(branches) is tuple
                recipes = tuple(
                    _recipe(branch, expand_functions=expand_functions)
                    for branch in branches
                )
                recipes, retained, aliases = _cond_direct_aliases(equation, recipes)
                parameters = (
                    type(equation.params),
                    tuple(
                        (
                            name,
                            (
                                (type(branches), recipes)
                                if name == "branches"
                                else parameter(value)
                            ),
                        )
                        for name, value in equation.params.items()
                    ),
                )
            else:
                parameters = parameter(equation.params)
            outputs = tuple(_aval(equation.outvars[index].aval) for index in retained)
            nodes.append(
                (
                    equation.primitive,
                    parameters,
                    inputs,
                    outputs,
                    frozenset(equation.effects),
                    _native_context_fields(equation.ctx),
                )
            )
            env.update(
                (id(equation.outvars[index]), ("result", position, compact, signature))
                for compact, (index, signature) in enumerate(
                    zip(retained, outputs, strict=True)
                )
            )
            env.update(
                (id(equation.outvars[index]), inputs[formal + 1])
                for index, formal in aliases.items()
            )
        return tuple(reference(v) for v in body.outvars)

    outputs = enter(closed)
    return tuple(nodes), outputs, frozenset(_body(closed).effects)


def _numeric_body(x: Any, indices: Any) -> Any:
    value = jnp.maximum(x + 0.5, 0)
    left, right = jnp.split(value, 2, axis=-1)
    joined = jnp.concatenate((left, right), axis=-1)
    return (
        jnp.take(joined, indices, axis=-1, mode="clip"),
        nnx.relu(-joined),
        jnp.tile(left, (1, 2)),
    )


@onnx_function(name="NativeActivationFunction")
def _named_activation(x: Any, indices: Any) -> Any:
    return _numeric_body(x, indices)


def _call_named_activation(x: Any, indices: Any) -> Any:
    return _named_activation(x, indices)


@onnx_function(name="NativeActivationSingleResult")
def _named_single_activation(x: Any, indices: Any) -> Any:
    return jnp.take(
        jnp.maximum(x + np.float32(0.5), np.float32(0)), indices, axis=-1, mode="clip"
    )


def _call_named_single_activation(x: Any, indices: Any) -> Any:
    return _named_single_activation(x, indices)


@onnx_function(name="NativeConditionalAlias")
def _conditional_alias_leaf(x: Any, y: Any, *, subtract: bool) -> Any:
    return (x - y if subtract else x + y), x


def _conditional_alias_control(flag: Any, x: Any, y: Any) -> Any:
    return jax.lax.cond(
        flag,
        lambda values: _conditional_alias_leaf(*values, subtract=False),
        lambda values: _conditional_alias_leaf(*values, subtract=True),
        (x, y),
    )


@onnx_function(name="NativeActivationStateOwner")
class _StateOwner(nnx.Module):
    def __init__(self) -> None:
        self.rngs = nnx.Rngs(dropout=jax.random.key(41, impl="threefry2x32"))
        self.alias = self.rngs
        self.dropout = nnx.Dropout(rate=0.5, rngs=self.rngs)
        self.scale = nnx.Param(jnp.asarray(2, np.float32))

    def __call__(self, x: Any, *, deterministic: bool = True) -> Any:
        return self.dropout(x * self.scale[...], deterministic=deterministic), x


@pytest.fixture(autouse=True)
def _exact_patch_stack() -> Iterator[None]:
    ps.import_all_plugins()
    assert not ps._PATCH_STATE
    assert not _patching._OWNED_PATCHES
    yield
    assert not ps._PATCH_STATE
    assert not _patching._OWNED_PATCHES


def _namespace() -> dict[tuple[Any, str], Any]:
    return {
        (target, attr): vars(target).get(attr, _patching._MISSING)
        for target, attr in (
            (jnp, "maximum"),
            (jnp, "tile"),
            (jnp, "split"),
            (jnp, "concatenate"),
            (nnx, "relu"),
            (nnx.Conv, "__call__"),
            (linen.Conv, "__call__"),
            (inspect.getmodule(_call_named_activation), "_named_activation"),
            (
                inspect.getmodule(_call_named_single_activation),
                "_named_single_activation",
            ),
            (_StateOwner, "__call__"),
        )
    }


def _assert_namespace(expected: dict[tuple[Any, str], Any]) -> None:
    assert all(
        vars(target).get(attr, _patching._MISSING) is value
        for (target, attr), value in expected.items()
    )


def _patch_snapshot() -> tuple[Any, Any]:
    return (
        {
            key: (state, state["count"], dict(state))
            for key, state in ps._PATCH_STATE.items()
        },
        {
            plugin: (plugin._orig_fn, plugin._unbound_orig_fn)
            for plugin in ps.PLUGIN_REGISTRY.values()
            if isinstance(plugin, FunctionPlugin)
        },
    )


def _assert_patch_snapshot(snapshot: tuple[Any, Any]) -> None:
    states, fields = snapshot
    assert ps._PATCH_STATE.keys() == states.keys()
    for key, (state, count, contents) in states.items():
        assert ps._PATCH_STATE[key] is state
        assert state["count"] == count
        assert state.keys() == contents.keys()
        assert all(state[name] is value for name, value in contents.items())
    for plugin, (original, unbound) in fields.items():
        assert plugin._orig_fn is original
        assert plugin._unbound_orig_fn is unbound


@pytest.mark.parametrize("body", (False, True), ids=("conversion", "function-body"))
def test_source_never_evaluates_any_numeric_leaf_factory_or_patch_info(
    monkeypatch: pytest.MonkeyPatch, body: bool
) -> None:
    attempts: list[str] = []

    class Hostile(PrimitiveLeafPlugin):
        @classmethod
        def binding_specs(cls) -> list[Any]:
            attempts.append("binding_specs")
            raise AssertionError("SOURCE evaluated a numeric leaf factory")

        @staticmethod
        def patch_info() -> dict[str, Any]:
            attempts.append("patch_info")
            raise AssertionError("SOURCE evaluated leaked numeric patch_info")

        @classmethod
        def ensure_abstract_eval_bound(cls) -> None:
            attempts.append("abstract_eval_setup")

        def lower(self, ctx: Any, eqn: Any) -> None:
            raise AssertionError("This activation control is never lowered")

    monkeypatch.setitem(ps.PLUGIN_REGISTRY, "test::hostile-numeric-leaf", Hostile())
    before = _namespace()
    with _world(PrecisionPolicy.SOURCE, body=body):
        assert attempts == ["abstract_eval_setup"]
        # The genuine Function bindings are the only owned namespace changes.
        for (target, attr), value in before.items():
            if (target, attr) not in ps._PATCH_STATE:
                assert vars(target).get(attr, _patching._MISSING) is value
        assert ps._PATCH_STATE
        assert all(
            isinstance(state["owner"], FunctionPlugin)
            for state in ps._PATCH_STATE.values()
        )
    _assert_namespace(before)
    assert attempts == ["abstract_eval_setup"]


def test_source_skips_assign_specs_as_well_as_monkey_patch_specs() -> None:
    target = ModuleType("activation_assign_control")
    target.original = object()
    original = target.original
    installed = object()
    factories: list[Any] = []

    def factory(value: Any) -> Any:
        factories.append(value)
        return installed

    class Leaf(PrimitiveLeafPlugin):
        @classmethod
        def binding_specs(cls) -> list[Any]:
            return [
                AssignSpec(target, "new", installed),
                MonkeyPatchSpec(target, "original", factory),
            ]

        def lower(self, ctx: Any, eqn: Any) -> None:
            raise AssertionError("not lowered")

    with precision_policy_scope(PrecisionPolicy.SOURCE), Leaf.plugin_binding():
        assert target.original is original
        assert "new" not in vars(target)
    assert factories == []
    with precision_policy_scope(PrecisionPolicy.LEGACY), Leaf.plugin_binding():
        assert target.original is installed and target.new is installed
    assert target.original is original and "new" not in vars(target)
    assert factories == [original]


def _graph_identity_snapshot(closed: Any) -> Any:
    def parameter(value: Any) -> Any:
        candidate = _body(value)
        if hasattr(candidate, "eqns") and hasattr(candidate, "invars"):
            return _graph_identity_snapshot(value)
        if isinstance(value, Mapping):
            return id(value), tuple(
                (key, parameter(child)) for key, child in value.items()
            )
        if isinstance(value, (tuple, list)):
            return id(value), tuple(parameter(child) for child in value)
        return id(value)

    body = _body(closed)
    return (
        id(body),
        tuple(map(id, body.invars)),
        tuple(map(id, body.outvars)),
        tuple(map(id, body.constvars)),
        tuple(map(id, getattr(closed, "consts", ()))),
        tuple(
            (
                id(e),
                tuple(map(id, e.invars)),
                tuple(map(id, e.outvars)),
                parameter(e.params),
                id(e.ctx),
                e.effects,
            )
            for e in body.eqns
        ),
        body.effects,
    )


def test_cond_common_direct_alias_matches_native_native_source_without_graph_mutation() -> (
    None
):
    specs = [
        jax.ShapeDtypeStruct((), np.bool_),
        *(jax.ShapeDtypeStruct((2, 4), np.float32) for _ in range(2)),
    ]
    first = jax.make_jaxpr(_fresh(_conditional_alias_control))(*specs)
    second = jax.make_jaxpr(_fresh(_conditional_alias_control))(*specs)
    with _world(PrecisionPolicy.SOURCE):
        activated = jax.make_jaxpr(_fresh(_conditional_alias_control))(*specs)
    snapshots = [
        _graph_identity_snapshot(value) for value in (first, second, activated)
    ]
    assert _recipe(first) == _recipe(second)
    assert _recipe(first) == _recipe(activated, expand_functions=True)
    assert [
        _graph_identity_snapshot(value) for value in (first, second, activated)
    ] == snapshots


def _cond_alias_probe(
    *, branch_choice: str = "same", producer: str | None = None
) -> Any:
    spec = jax.ShapeDtypeStruct((2, 4), np.float32)
    template = jax.make_jaxpr(
        lambda flag, x, y: jax.lax.cond(
            flag, lambda v: v[0] + v[1], lambda v: v[0] - v[1], (x, y)
        )
    )(jax.ShapeDtypeStruct((), np.bool_), spec, spec)
    equation = next(e for e in _body(template).eqns if e.primitive is jax.lax.cond_p)
    branches = []
    for index, branch in enumerate(equation.params["branches"]):
        body = _body(branch)
        if producer is None:
            alias = body.invars[
                1 if branch_choice == "different-formal" and index == 1 else 0
            ]
            outputs = [body.outvars[0], alias]
            if branch_choice == "swapped-output" and index == 1:
                outputs.reverse()
            body = body.replace(outvars=outputs)
            branch = ClosedJaxpr(body, tuple(getattr(branch, "consts", ())))
        else:

            def derived(x: Any, y: Any) -> Any:
                if producer == "copy":
                    alias = jax.lax.copy_p.bind(x)
                elif producer == "cast":
                    alias = jax.lax.convert_element_type(
                        jax.lax.convert_element_type(x, np.float16), np.float32
                    )
                elif producer == "arithmetic":
                    alias = x + np.float32(0)
                else:
                    alias = jax.lax.cond(y[0, 0] > 0, lambda _: x, lambda _: y, None)
                return x + y, alias

            branch = jax.make_jaxpr(derived)(spec, spec)
        branches.append(branch)
    return equation.replace(
        params={"branches": tuple(branches)},
        outvars=[equation.outvars[0], Var(equation.outvars[0].aval)],
    )


def test_cond_alias_hoist_preserves_every_branch_node_and_compacts_only_return_positions() -> (
    None
):
    equation = _cond_alias_probe()
    recipes = tuple(_recipe(branch) for branch in equation.params["branches"])
    snapshots = tuple(
        _graph_identity_snapshot(branch) for branch in equation.params["branches"]
    )
    normalized, retained, aliases = _cond_direct_aliases(equation, recipes)
    assert retained == (0,) and aliases == {1: 0}
    for before, after in zip(recipes, normalized, strict=True):
        assert after[0] is before[0] and after[2] is before[2]
        assert after[1] == (before[1][0],)
    assert (
        tuple(
            _graph_identity_snapshot(branch) for branch in equation.params["branches"]
        )
        == snapshots
    )


def test_cond_all_output_alias_elimination_remains_outside_this_partial_hoist() -> None:
    equation = _cond_alias_probe()
    branches = tuple(
        ClosedJaxpr(
            _body(branch).replace(outvars=list(_body(branch).invars)),
            tuple(getattr(branch, "consts", ())),
        )
        for branch in equation.params["branches"]
    )
    equation = equation.replace(params={"branches": branches})
    recipes = tuple(_recipe(branch) for branch in branches)
    normalized, retained, aliases = _cond_direct_aliases(equation, recipes)
    assert normalized is recipes and retained == (0, 1) and aliases == {}


def test_cond_actual_formal_to_outer_operand_mapping_requires_exact_arity() -> None:
    equation = _cond_alias_probe()
    recipes = tuple(_recipe(branch) for branch in equation.params["branches"])
    changed = equation.replace(invars=[*equation.invars, equation.invars[-1]])
    with pytest.raises(AssertionError):
        _cond_direct_aliases(changed, recipes)


@pytest.mark.parametrize("choice", ("different-formal", "swapped-output"))
def test_cond_differing_formal_or_output_order_is_not_a_common_alias(
    choice: str,
) -> None:
    equation = _cond_alias_probe(branch_choice=choice)
    recipes = tuple(_recipe(branch) for branch in equation.params["branches"])
    normalized, retained, aliases = _cond_direct_aliases(equation, recipes)
    assert normalized is recipes and retained == (0, 1) and aliases == {}


@pytest.mark.parametrize("producer", ("copy", "cast", "arithmetic", "conditional"))
def test_cond_computed_same_shaped_result_cannot_be_erased_as_a_direct_alias(
    producer: str,
) -> None:
    equation = _cond_alias_probe(producer=producer)
    recipes = tuple(_recipe(branch) for branch in equation.params["branches"])
    assert all(recipe[0] for recipe in recipes)
    normalized, retained, aliases = _cond_direct_aliases(equation, recipes)
    assert normalized is recipes and retained == (0, 1) and aliases == {}


@pytest.mark.parametrize(
    "kind",
    (
        "weakness",
        "symbol-scope",
        "sharding",
        "memory-space",
        "extra-parameter",
        "nonpure",
    ),
)
def test_cond_alias_correspondence_rejects_unproven_type_schema_or_effects(
    kind: str,
) -> None:
    equation = _cond_alias_probe()
    recipes = tuple(_recipe(branch) for branch in equation.params["branches"])
    if kind == "nonpure":
        effect = object()
        branch = equation.params["branches"][0]
        changed = ClosedJaxpr(
            _body(branch).replace(effects=frozenset({effect})),
            tuple(getattr(branch, "consts", ())),
        )
        equation = equation.replace(
            params={"branches": (changed, *equation.params["branches"][1:])}
        )
        recipes = tuple(_recipe(branch) for branch in equation.params["branches"])
        normalized, retained, aliases = _cond_direct_aliases(equation, recipes)
        assert normalized is recipes and retained == (0, 1) and aliases == {}
        return
    if kind == "extra-parameter":
        equation = equation.replace(params={**equation.params, "unknown": False})
    else:
        node_list, outputs, effects = recipes[0]
        ref = outputs[1]
        signature = list(ref[2])
        field = {"weakness": 3, "symbol-scope": 2, "sharding": 4, "memory-space": 5}[
            kind
        ]
        signature[field] = (
            True if kind == "weakness" else ("unproven-native-correspondence", object())
        )
        outputs = (outputs[0], (ref[0], ref[1], tuple(signature)))
        recipes = ((node_list, outputs, effects), *recipes[1:])
    with pytest.raises(AssertionError):
        _cond_direct_aliases(equation, recipes)


def _native_custom_jvp_equation() -> Any:
    closed = jax.make_jaxpr(_fresh(lambda x: nnx.relu(x)))(
        jax.ShapeDtypeStruct((2, 4), np.float32)
    )
    return next(e for e in _body(closed).eqns if e.primitive is _NATIVE_CUSTOM_JVP)


def _replace_function_closure(function: Any, **changes: Any) -> Any:
    assert function.__closure__ is not None
    values = dict(
        zip(
            function.__code__.co_freevars,
            (c.cell_contents for c in function.__closure__),
            strict=True,
        )
    )
    assert changes.keys() <= values.keys()
    values.update(changes)
    return FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        tuple(CellType(values[name]) for name in function.__code__.co_freevars),
    )


def _replace_wrapped(wrapped: Any, **changes: Any) -> Any:
    assert changes.keys() <= set(native_lu.WrappedFun.__slots__)
    return native_lu.WrappedFun(
        *(
            changes.get(name, getattr(wrapped, name))
            for name in native_lu.WrappedFun.__slots__
        )
    )


def test_native_independent_traces_and_source_match_exact_concrete_forward_recipe() -> (
    None
):
    specs = [
        jax.ShapeDtypeStruct((2, 4), np.float32),
        jax.ShapeDtypeStruct((2,), np.int32),
    ]
    first = jax.make_jaxpr(_fresh(_numeric_body))(*specs)
    second = jax.make_jaxpr(_fresh(_numeric_body))(*specs)
    assert _recipe(first) == _recipe(second)
    with _world(PrecisionPolicy.SOURCE):
        source = jax.make_jaxpr(_fresh(_numeric_body))(*specs)
    assert _recipe(first) == _recipe(source)


@pytest.mark.parametrize(
    "kind",
    (
        "wrapper-type",
        "function-code",
        "function-globals",
        "partial-target",
        "partial-args",
        "partial-kwargs",
        "transforms",
        "stores",
        "params",
        "in_type",
        "memo-cells",
        "memo-sentinel",
        "thunk-code",
        "thunk-globals",
        "thunk-inputs",
        "thunk-tangents",
        "thunk-jvp",
        "thunk-trace",
        "debug-type",
        "extra-param",
        "missing-param",
    ),
)
def test_native_derivative_metadata_protocol_rejects_hostile_or_unknown_structure(
    kind: str,
) -> None:
    equation = _native_custom_jvp_equation()
    wrapped = equation.params["jvp_jaxpr_fun"]
    assert _native_jvp_recipe(wrapped, equation)
    function = wrapped.f
    if kind == "wrapper-type":
        hostile = SimpleNamespace(
            **{name: getattr(wrapped, name) for name in native_lu.WrappedFun.__slots__}
        )
    elif kind == "function-code":
        f = FunctionType((lambda *xs: xs).__code__, vars(native_pe))
        hostile = _replace_wrapped(wrapped, f=f, f_transformed=partial(f))
    elif kind == "function-globals":
        f = FunctionType(
            function.__code__, dict(function.__globals__), closure=function.__closure__
        )
        hostile = _replace_wrapped(wrapped, f=f, f_transformed=partial(f))
    elif kind.startswith("partial-"):
        transformed = {
            "partial-target": partial(lambda: None),
            "partial-args": partial(function, 1),
            "partial-kwargs": partial(function, unexpected=True),
        }[kind]
        hostile = _replace_wrapped(wrapped, f_transformed=transformed)
    elif kind in {"transforms", "stores", "params", "in_type"}:
        hostile = _replace_wrapped(wrapped, **{kind: (object(),)})
    elif kind.startswith("memo-"):
        changes = (
            {"cells": {"unexpected-cache-entry": object()}}
            if kind == "memo-cells"
            else {"sentinel": 0}
        )
        f = _replace_function_closure(function, **changes)
        hostile = _replace_wrapped(wrapped, f=f, f_transformed=partial(f))
    elif kind.startswith("thunk-"):
        thunk = dict(
            zip(
                function.__code__.co_freevars,
                (c.cell_contents for c in function.__closure__),
                strict=True,
            )
        )["fn"]
        if kind == "thunk-code":

            def hostile_thunk() -> None:
                return None

            thunk = hostile_thunk
        elif kind == "thunk-globals":
            thunk = FunctionType(
                thunk.__code__, dict(thunk.__globals__), closure=thunk.__closure__
            )
        else:
            field = {
                "thunk-inputs": "in_avals",
                "thunk-tangents": "in_tangent_avals",
                "thunk-jvp": "jvp",
                "thunk-trace": "self",
            }[kind]
            thunk = _replace_function_closure(
                thunk,
                **{
                    field: (
                        [] if kind in {"thunk-inputs", "thunk-tangents"} else object()
                    )
                },
            )
        f = _replace_function_closure(function, fn=thunk)
        hostile = _replace_wrapped(wrapped, f=f, f_transformed=partial(f))
    elif kind == "debug-type":
        hostile = _replace_wrapped(wrapped, debug_info=object())
    else:
        parameters = dict(equation.params)
        if kind == "extra-param":
            parameters["unknown-derivative-protocol"] = object()
        else:
            parameters.pop("symbolic_zeros")
        equation = equation.replace(params=parameters)
        hostile = wrapped
    with pytest.raises(AssertionError):
        _native_jvp_recipe(hostile, equation)


def test_forward_recipe_retains_literal_ssa_debug_and_native_context_obligations() -> (
    None
):
    spec = jax.ShapeDtypeStruct((2, 4), np.float32)
    first = jax.make_jaxpr(lambda x: x + np.float32(0.5))(spec)
    changed = jax.make_jaxpr(lambda x: x + np.float32(0.75))(spec)
    assert _recipe(first) != _recipe(changed)
    pair = jax.make_jaxpr(lambda x, y: x + y)(spec, spec)
    equation = _body(pair).eqns[0]
    swapped = equation.replace(invars=list(reversed(equation.invars)))
    assert _recipe(pair) != _recipe(_body(pair).replace(eqns=[swapped]))
    original_context = equation.ctx
    changed_context = copy(equation)
    changed_context.ctx = None
    assert changed_context is not equation and changed_context.ctx is None
    assert equation.ctx is original_context
    assert all(
        getattr(changed_context, name) is getattr(equation, name)
        for name in type(equation).__slots__
        if name != "ctx"
    )
    with pytest.raises(TypeError, match="actual recorded equation context"):
        _recipe(_body(pair).replace(eqns=[changed_context]))
    assert equation.ctx is original_context
    custom = _native_custom_jvp_equation()
    wrapped = custom.params["jvp_jaxpr_fun"]
    debug = wrapped.debug_info._replace(
        func_src_info=wrapped.debug_info.func_src_info + " changed"
    )
    assert _native_jvp_recipe(wrapped, custom) != _native_jvp_recipe(
        _replace_wrapped(wrapped, debug_info=debug), custom
    )


def test_source_keeps_captured_numeric_aliases_and_native_primitive_registration() -> (
    None
):
    maximum, relu = jnp.maximum, nnx.relu
    native_abstract_eval = jax.lax.add_p.abstract_eval

    def function(x: Any) -> Any:
        return maximum(x + np.float32(0.5), np.float32(0)), relu(x)

    spec = jax.ShapeDtypeStruct((2, 4), np.float32)
    native = jax.make_jaxpr(_fresh(function))(spec)
    with _world(PrecisionPolicy.SOURCE):
        assert jnp.maximum is maximum and nnx.relu is relu
        assert jax.lax.add_p.abstract_eval is native_abstract_eval
        activated = jax.make_jaxpr(_fresh(function))(spec)
        assert _recipe(activated) == _recipe(native)
    assert jax.lax.add_p.abstract_eval is native_abstract_eval


def test_source_same_policy_nesting_keeps_exact_function_wrapper_and_fields() -> None:
    module = inspect.getmodule(_call_named_activation)
    key = module, "_named_activation"
    with _world(PrecisionPolicy.SOURCE):
        wrapper = module._named_activation
        snapshot = _patch_snapshot()
        count = ps._PATCH_STATE[key]["count"]
        with _world(PrecisionPolicy.SOURCE):
            assert module._named_activation is wrapper
            assert ps._PATCH_STATE[key]["count"] > count
        assert module._named_activation is wrapper
        _assert_patch_snapshot(snapshot)


def test_source_genuine_function_patch_info_failure_is_explicit_and_restores_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = ps.PLUGIN_REGISTRY[
        f"onnx_fn::{_named_activation.__module__}.{_named_activation.__name__}"
    ]
    with _world(PrecisionPolicy.LEGACY):
        snapshot, namespace = _patch_snapshot(), _namespace()

        def failure() -> Any:
            raise RuntimeError("required genuine Function patch_info failed")

        monkeypatch.setattr(plugin, "patch_info", failure)
        with pytest.raises(
            RuntimeError, match="required genuine Function patch_info failed"
        ):
            with _world(PrecisionPolicy.SOURCE):
                raise AssertionError("entry must fail")
        _assert_namespace(namespace)
        _assert_patch_snapshot(snapshot)


@pytest.mark.parametrize("x64", (False, True))
@pytest.mark.parametrize("named", (False, True), ids=("native", "genuine-function"))
def test_full_source_world_matches_independent_native_typed_program(
    x64: bool, named: bool
) -> None:
    with _runtime_configuration(x64):
        batch = jax_export.symbolic_shape("B")[0]
        specs = [
            jax.ShapeDtypeStruct((batch, 4), np.float32),
            jax.ShapeDtypeStruct((2,), np.int32),
        ]
        function = _call_named_activation if named else _numeric_body
        native = jax.make_jaxpr(_fresh(function))(*specs)
        owners = _namespace()
        resolved = {(target, attr): getattr(target, attr) for target, attr in owners}
        with _world(PrecisionPolicy.SOURCE):
            activated = jax.make_jaxpr(_fresh(function))(*specs)
            assert _recipe(activated, expand_functions=True) == _recipe(native)
            native_alias = owners[(jnp, "maximum")]
            assert jnp.maximum is native_alias
            assert nnx.Conv.__call__ is resolved[(nnx.Conv, "__call__")]
            assert linen.Conv.__call__ is resolved[(linen.Conv, "__call__")]
            assert (
                vars(nnx.Conv).get("__call__", _patching._MISSING)
                is owners[(nnx.Conv, "__call__")]
            )
            assert (
                vars(linen.Conv).get("__call__", _patching._MISSING)
                is owners[(linen.Conv, "__call__")]
            )
        _assert_namespace(owners)
        assert all(
            getattr(target, attr) is value for (target, attr), value in resolved.items()
        )
        validate_source_jaxpr(activated, 23)
        if named:
            assert any(
                isinstance(ps.PLUGIN_REGISTRY.get(e.primitive.name), FunctionPlugin)
                for e in _body(activated).eqns
            )
        samples = [
            _sample(
                function,
                (
                    (np.arange(size * 4, dtype=np.float32).reshape(size, 4) - 8) / 8,
                    np.asarray([-1, 5], np.int32),
                ),
            )
            for size in (2, 5)
        ]
        model = ui._to_onnx(
            function, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        if named:
            owner = next(
                f for f in model.functions if f.name == "NativeActivationFunction"
            )
            assert any(
                n.domain == owner.domain
                and n.op_type == owner.name
                and n.overload == owner.overload
                for n in model.graph.node
            )
        _check_one_artifact(model, samples)


@pytest.mark.parametrize("outer", (PrecisionPolicy.SOURCE, PrecisionPolicy.LEGACY))
@pytest.mark.parametrize("failure", (False, True), ids=("success", "body-failure"))
def test_nested_policy_worlds_restore_exact_outer_wrappers_counts_and_fields(
    outer: PrecisionPolicy, failure: bool
) -> None:
    inner = (
        PrecisionPolicy.LEGACY
        if outer is PrecisionPolicy.SOURCE
        else PrecisionPolicy.SOURCE
    )
    before = _namespace()
    specs = [
        jax.ShapeDtypeStruct((2, 4), np.float32),
        jax.ShapeDtypeStruct((2,), np.int32),
    ]
    independent_native = jax.make_jaxpr(_fresh(_call_named_single_activation))(*specs)
    with _world(outer):
        jax.make_jaxpr(_fresh(_call_named_single_activation))(*specs)
        snapshot, namespace = _patch_snapshot(), _namespace()
        try:
            with _world(inner, body=True):
                if inner is PrecisionPolicy.SOURCE:
                    assert jnp.maximum is before[(jnp, "maximum")]
                    assert nnx.relu is before[(nnx, "relu")]
                    trace = jax.make_jaxpr(_fresh(_call_named_single_activation))(
                        *specs
                    )
                    assert _recipe(trace, expand_functions=True) == _recipe(
                        independent_native
                    )
                else:
                    assert jnp.maximum is not before[(jnp, "maximum")]
                if failure:
                    raise RuntimeError("nested activation body failure")
        except RuntimeError as error:
            assert failure and str(error) == "nested activation body failure"
        _assert_namespace(namespace)
        if outer is PrecisionPolicy.SOURCE:
            trace = jax.make_jaxpr(_fresh(_call_named_single_activation))(*specs)
            assert _recipe(trace, expand_functions=True) == _recipe(independent_native)
        # LEGACY nested calls historically update original-call fields. SOURCE
        # suspension must restore them byte-for-byte for the outer LEGACY owner.
        if inner is PrecisionPolicy.SOURCE:
            _assert_patch_snapshot(snapshot)
        else:
            states, _ = snapshot
            assert all(
                ps._PATCH_STATE[k] is st and st["count"] == count
                for k, (st, count, _) in states.items()
            )
    _assert_namespace(before)


@pytest.mark.parametrize("failure", (False, True), ids=("success", "factory-failure"))
def test_inherited_function_descriptor_and_missing_numeric_assignment_restore_exactly(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    class Parent:
        def __call__(self, x: Any) -> Any:
            return x

    class Child(Parent):
        pass

    module = ModuleType("activation_missing_control")
    installed = object()

    class Leaf(PrimitiveLeafPlugin):
        @classmethod
        def binding_specs(cls) -> list[Any]:
            return [AssignSpec(module, "introduced", installed)]

        def lower(self, ctx: Any, eqn: Any) -> None:
            raise AssertionError("not lowered")

    plugin = FunctionPlugin("onnx_fn::activation-inherited-control", Child)
    original_factory = plugin.patch_info()["patch_function"]

    def factory(original: Any) -> Any:
        if failure and ps.active_precision_policy() is PrecisionPolicy.SOURCE:
            raise RuntimeError("SOURCE Function factory failure")
        return original_factory(original)

    monkeypatch.setattr(
        plugin,
        "patch_info",
        lambda: {"patch_targets": [Child], "patch_function": factory},
    )
    monkeypatch.setitem(ps.PLUGIN_REGISTRY, plugin.name, plugin)
    monkeypatch.setitem(ps.PLUGIN_REGISTRY, "test::missing-assignment", Leaf())
    parent = dict(vars(Parent))
    assert "__call__" not in vars(Child) and "introduced" not in vars(module)
    with _world(PrecisionPolicy.LEGACY):
        snapshot, wrapper = _patch_snapshot(), vars(Child)["__call__"]
        try:
            with _world(PrecisionPolicy.SOURCE):
                assert not failure
                assert "introduced" not in vars(module)
                assert plugin._unbound_orig_fn is Parent.__call__
        except RuntimeError as error:
            assert failure and str(error) == "SOURCE Function factory failure"
        assert vars(Child)["__call__"] is wrapper
        assert module.introduced is installed
        _assert_patch_snapshot(snapshot)
    assert "__call__" not in vars(Child) and "introduced" not in vars(module)
    assert all(vars(Parent)[key] is value for key, value in parent.items())


def test_source_preserves_unrelated_caller_patch_and_rejects_function_ownership_conflict() -> (
    None
):
    original = _namespace()
    caller = object()
    with _world(PrecisionPolicy.LEGACY):
        with apply_patches([AssignSpec(jnp, "maximum", caller)]):
            with _world(PrecisionPolicy.SOURCE):
                assert jnp.maximum is caller
            assert jnp.maximum is caller
        module = inspect.getmodule(_call_named_activation)
        with apply_patches([AssignSpec(module, "_named_activation", caller)]):
            snapshot = _patch_snapshot()
            with pytest.raises(
                RuntimeError, match="Function patch ownership conflicts"
            ):
                with _world(PrecisionPolicy.SOURCE):
                    raise AssertionError("ownership conflict must fail on entry")
            assert module._named_activation is caller
            _assert_patch_snapshot(snapshot)
    _assert_namespace(original)


@pytest.mark.parametrize(
    "order", (("ordinary", "source", "legacy"), ("legacy", "source", "ordinary"))
)
def test_native_source_legacy_warm_cache_orders_never_clear_user_jit_cache(
    order: tuple[str, ...],
) -> None:
    traced: list[str] = []
    phase = "initial"

    def function(x: Any) -> Any:
        traced.append(phase)
        return jax.lax.neg(x)

    compiled = jax.jit(function)
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    _same(compiled(x), -x)
    _same(compiled(x), -x)
    initial = tuple(traced)
    assert len(initial) == 1
    for phase in order:
        if phase == "ordinary":
            _same(compiled(x), -x)
        else:
            policy = (
                PrecisionPolicy.SOURCE if phase == "source" else PrecisionPolicy.LEGACY
            )
            with _world(policy):
                _same(compiled(x), -x)
                _same(compiled(x), -x)
                assert jax.make_jaxpr(compiled)(x)
    phase = "ordinary-after-worlds"
    _same(compiled(x), -x)
    assert phase not in traced
    assert tuple(traced[:1]) == initial


def test_delayed_function_impl_and_nested_vmap_use_exact_per_call_native_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = inspect.getmodule(_call_named_activation)
    spec = jax.ShapeDtypeStruct((2, 4), np.float32)
    index_spec = jax.ShapeDtypeStruct((2,), np.int32)
    x, indices = np.arange(8, dtype=np.float32).reshape(2, 4), np.asarray(
        [0, 3], np.int32
    )
    original = module._named_activation

    def alternate(x: Any, indices: Any) -> Any:
        value = x + np.float32(2)
        return jnp.take(value, indices, axis=-1, mode="clip"), nnx.relu(value), value

    equations = []
    expected = [original(x, indices), alternate(x, indices)]
    with _world(PrecisionPolicy.LEGACY):
        outer = _patch_snapshot()
        with _world(PrecisionPolicy.SOURCE):
            closed = jax.make_jaxpr(_fresh(_call_named_activation))(spec, index_spec)
            equations.append(
                next(
                    e
                    for e in closed.eqns
                    if isinstance(
                        ps.PLUGIN_REGISTRY.get(e.primitive.name), FunctionPlugin
                    )
                )
            )
        _assert_patch_snapshot(outer)
    monkeypatch.setattr(module, "_named_activation", alternate)
    with _world(PrecisionPolicy.SOURCE):
        closed = jax.make_jaxpr(_fresh(_call_named_activation))(spec, index_spec)
        equations.append(
            next(
                e
                for e in closed.eqns
                if isinstance(ps.PLUGIN_REGISTRY.get(e.primitive.name), FunctionPlugin)
            )
        )
    assert equations[0].primitive is equations[1].primitive
    first, second = (e.params[ps._SOURCE_TRACE] for e in equations)
    assert first.output_avals == second.output_avals
    assert first.call != second.call and first != second
    assert (
        hash(first.call) != hash(second.call)
        or first.call.identity != second.call.identity
    )
    with _world(PrecisionPolicy.LEGACY):
        outer, namespace = _patch_snapshot(), _namespace()
        for equation, reference in zip(equations, expected, strict=True):
            plugin = ps.PLUGIN_REGISTRY[equation.primitive.name]
            result = plugin._source_primitive_impl(x, indices, **equation.params)
            _same(result, list(jax.tree.leaves(reference)))
            values, dimensions = plugin._source_batching_rule(
                (np.stack((x, x)), indices), (0, ps.NOT_MAPPED), **equation.params
            )
            assert dimensions == [0] * len(values)
            _same(
                values,
                [np.stack((value, value)) for value in jax.tree.leaves(reference)],
            )
            # A vmap around a vmap exercises delayed batching after restoration.
            nested = jax.vmap(
                jax.vmap(
                    lambda row: equation.primitive.bind(row, indices, **equation.params)
                )
            )
            _same(
                nested(np.stack((np.stack((x, x)), np.stack((x, x))))),
                [
                    np.stack((np.stack((value, value)), np.stack((value, value))))
                    for value in jax.tree.leaves(reference)
                ],
            )
            _assert_namespace(namespace)
            _assert_patch_snapshot(outer)


def test_source_callable_capsule_keeps_unhashable_array_receiver_without_instance_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class Receiver:
        offset: Any

        def __call__(self, x: Any) -> Any:
            return x + self.offset

    owner = Receiver(jnp.asarray([1, 2], np.float32))
    reference = weakref.ref(owner)
    capsule = ps._SourceFunctionCall.capture(
        owner.__call__, instance=True, receiver=owner
    )
    assert capsule == ps._SourceFunctionCall.capture(
        owner.__call__, instance=True, receiver=owner
    )
    assert hash(capsule) == hash(
        ps._SourceFunctionCall.capture(owner.__call__, instance=True, receiver=owner)
    )
    plugin = FunctionPlugin("onnx_fn::source-strong-array-receiver", Receiver)
    monkeypatch.setitem(ps.PLUGIN_REGISTRY, plugin.name, plugin)
    receivers = [owner]
    with _world(PrecisionPolicy.SOURCE):
        closed = jax.make_jaxpr(lambda x: receivers[0](x))(
            jax.ShapeDtypeStruct((2,), np.float32)
        )
        equation = next(
            e for e in closed.eqns if e.primitive is plugin._source_primitive
        )
    receivers.clear()
    key = equation.params["instance_key"]
    retained = ps.INSTANCE_MAP2.pop(key)
    del owner
    assert reference() is capsule.receiver is retained
    try:
        _same(
            plugin._source_primitive_impl(
                np.asarray([3, 4], np.float32), **equation.params
            ),
            [np.asarray([4, 6], np.float32)],
        )
    finally:
        ps.INSTANCE_MAP2[key] = retained


def _assert_function_result_provenance(
    artifact: Any,
    methods: Mapping[tuple[str, str, str], Any],
    expected: tuple[Any, ...],
    input_name: str,
    *,
    unique: bool,
) -> None:
    """Resolve exact returned call results through transparent Identity only."""
    definitions = {(f.domain, f.name, f.overload) for f in artifact.functions}
    assert definitions == methods.keys()
    nodes = tuple(artifact.graph.node)
    calls = {
        id(node): node
        for node in nodes
        if (node.domain, node.op_type, node.overload) in definitions
    }
    assert len(definitions) == (3 if unique else 2)
    assert len(calls) in ((3,) if unique else (2, 3))
    assert all(
        tuple(node.input) == (input_name,) and len(node.output) == 1
        for node in calls.values()
    )
    producers = {name: node for node in nodes for name in node.output}
    assert len(producers) == sum(len(node.output) for node in nodes)
    assert len(artifact.graph.output) == len(expected) == 3
    tokens = []
    for result, method in zip(artifact.graph.output, expected, strict=True):
        name, seen = result.name, set()
        while name in producers:
            assert name not in seen
            seen.add(name)
            producer = producers[name]
            if (producer.domain, producer.op_type) != ("", "Identity"):
                break
            assert len(producer.input) == len(producer.output) == 1
            assert producer.output[0] == name
            name = producer.input[0]
        assert name in producers
        producer = producers[name]
        assert id(producer) in calls and producer.output[0] == name
        identifier = producer.domain, producer.op_type, producer.overload
        assert methods[identifier] == method
        tokens.append((identifier, tuple(producer.input), 0))
    if unique:
        assert len(set(tokens)) == 3
    else:
        # Duplicate call CSE or repeated calls to one exact pure definition and
        # the same operand have the same returned-result provenance.
        assert tokens[0] == tokens[2] and tokens[0] != tokens[1]


def _function_result_probe() -> tuple[Any, Any, Any]:
    helper = onnx.helper
    methods = {
        ("exact.methods", "original", ""): (11, 13),
        ("exact.methods", "alternate", ""): (12, 13),
    }
    definitions = [
        helper.make_function(
            domain,
            name,
            ["x"],
            ["y"],
            [helper.make_node("Identity", ["x"], ["y"])],
            [helper.make_opsetid("", 23)],
        )
        for domain, name, _ in methods
    ]
    nodes = [
        helper.make_node(
            "original",
            ["features"],
            ["first"],
            domain="exact.methods",
            name="original-call",
        ),
        helper.make_node(
            "alternate",
            ["features"],
            ["second"],
            domain="exact.methods",
            name="alternate-call",
        ),
        helper.make_node("Identity", ["first"], ["third"], name="result-alias"),
    ]

    def value(name: str) -> Any:
        return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [2])

    graph = helper.make_graph(
        nodes,
        "method-results",
        [value("features"), value("other")],
        [value(name) for name in ("first", "second", "third")],
    )
    model = helper.make_model(graph, functions=definitions)
    expected = (
        methods[("exact.methods", "original", "")],
        methods[("exact.methods", "alternate", "")],
        methods[("exact.methods", "original", "")],
    )
    return model, methods, expected


def test_exact_function_output_provenance_accepts_shared_call_cse_and_identity_alias() -> (
    None
):
    model, methods, expected = _function_result_probe()
    _assert_function_result_provenance(
        model, methods, expected, "features", unique=False
    )


@pytest.mark.parametrize(
    "kind",
    ("method", "swapped-output", "unrelated-producer", "different-operand", "cast"),
)
def test_exact_function_output_provenance_rejects_wrong_method_port_or_producer(
    kind: str,
) -> None:
    model, methods, expected = _function_result_probe()
    if kind == "method":
        model.graph.node[1].op_type = "original"
    elif kind == "swapped-output":
        model.graph.output[0].name, model.graph.output[1].name = (
            model.graph.output[1].name,
            model.graph.output[0].name,
        )
    elif kind == "unrelated-producer":
        model.graph.node[2].input[0] = "features"
    elif kind == "different-operand":
        model.graph.node[0].input[0] = "other"
    else:
        model.graph.node[2].op_type = "Cast"
        model.graph.node[2].attribute.append(
            onnx.helper.make_attribute("to", onnx.TensorProto.FLOAT)
        )
    with pytest.raises(AssertionError):
        _assert_function_result_provenance(
            model, methods, expected, "features", unique=False
        )


@pytest.mark.parametrize("unique", (False, True), ids=("shared", "unique"))
def test_same_receiver_distinct_native_methods_own_distinct_exported_functions(
    monkeypatch: pytest.MonkeyPatch, unique: bool
) -> None:
    @dataclass
    class Receiver:
        offset: Any

        def __call__(self, x: Any) -> Any:
            return x + self.offset

    def alternate(self: Any, x: Any) -> Any:
        return x - self.offset

    owner = Receiver(jnp.asarray([1, 2], np.float32))
    original = Receiver.__call__
    plugin = FunctionPlugin(
        "onnx_fn::source-same-receiver-native-methods", Receiver, unique=unique
    )
    monkeypatch.setitem(ps.PLUGIN_REGISTRY, plugin.name, plugin)
    spec = jax.ShapeDtypeStruct((2,), np.float32)

    def capture() -> Any:
        with _world(PrecisionPolicy.SOURCE):
            closed = jax.make_jaxpr(lambda x: owner(x))(spec)
        return next(e for e in closed.eqns if e.primitive is plugin._source_primitive)

    first, repeated = capture(), capture()
    monkeypatch.setattr(Receiver, "__call__", alternate)
    second = capture()
    traces = [e.params[ps._SOURCE_TRACE] for e in (first, repeated, second)]
    assert traces[0].call == traces[1].call
    assert traces[0].call != traces[2].call
    assert traces[0].output_avals == traces[2].output_avals
    assert all(t.call.receiver is owner for t in traces)
    context = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE, deployment_target=None
    )
    captures = ps.FunctionCaptures((), {}, ())
    keys = [
        plugin._function_key(owner, equation, captures, ctx=context, source_trace=trace)
        for equation, trace in zip((first, repeated, second), traces, strict=True)
    ]
    if unique:
        # This unregistered opaque receiver deliberately prevents cache reuse.
        assert len(set(keys)) == 3
    else:
        assert keys[0] == keys[1] and keys[0] != keys[2]

    def retained_calls(x: Any) -> Any:
        return tuple(
            equation.primitive.bind(x, **equation.params)[0]
            for equation in (first, second, repeated)
        )

    x = np.asarray([4, 8], np.float32)
    expected = (original(owner, x), alternate(owner, x), original(owner, x))
    _same(retained_calls(x), expected)

    def independent_native(x: Any) -> Any:
        return original(owner, x), alternate(owner, x), original(owner, x)

    sample = _sample(independent_native, (x,))
    _same(sample.expected_outputs, expected)
    methods = {}
    original_build = plugin._build_function

    def retain_build(*args: Any, **kwargs: Any) -> Any:
        definition = original_build(*args, **kwargs)
        trace = kwargs["source_trace"]
        identifier = definition.domain, definition.name, definition.overload
        assert identifier not in methods
        methods[identifier] = trace.call.identity
        return definition

    with monkeypatch.context() as observer:
        observer.setattr(plugin, "_build_function", retain_build)
        artifact = ui._to_onnx(
            retained_calls, [spec], precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
    _assert_function_result_provenance(
        artifact,
        methods,
        (traces[0].call.identity, traces[2].call.identity, traces[1].call.identity),
        artifact.graph.input[0].name,
        unique=unique,
    )
    _check_one_artifact(artifact, [sample])


def test_source_rebuilds_function_that_captured_an_owned_numeric_shim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Receiver:
        def __call__(self, x: Any) -> Any:
            return x + np.float32(1)

    original = Receiver.__call__
    numeric_calls: list[Any] = []

    def numeric_shim(self: Any, x: Any) -> Any:
        numeric_calls.append(x)
        return x + np.float32(100)

    class Leaf(PrimitiveLeafPlugin):
        @classmethod
        def binding_specs(cls) -> list[Any]:
            return [MonkeyPatchSpec(Receiver, "__call__", lambda before: numeric_shim)]

        def lower(self, ctx: Any, eqn: Any) -> None:
            raise AssertionError("not lowered")

    owner = Receiver()
    native = jax.make_jaxpr(lambda x: original(owner, x))(
        jax.ShapeDtypeStruct((2,), np.float32)
    )
    plugin = FunctionPlugin("onnx_fn::source-owned-numeric-capture", Receiver)
    monkeypatch.setitem(ps.PLUGIN_REGISTRY, plugin.name, plugin)
    with (
        precision_policy_scope(PrecisionPolicy.LEGACY),
        Leaf.plugin_binding(),
        ps.apply_monkey_patches(),
    ):
        assert plugin._unbound_orig_fn is numeric_shim
        outer, wrapper = _patch_snapshot(), Receiver.__call__
        with _world(PrecisionPolicy.SOURCE):
            assert plugin._unbound_orig_fn is original
            activated = jax.make_jaxpr(lambda x: owner(x))(
                jax.ShapeDtypeStruct((2,), np.float32)
            )
            assert _recipe(activated, expand_functions=True) == _recipe(native)
        assert Receiver.__call__ is wrapper
        _assert_patch_snapshot(outer)
        equation = next(
            e for e in activated.eqns if e.primitive is plugin._source_primitive
        )
        _same(
            plugin._source_primitive_impl(
                np.asarray([3, 4], np.float32), **equation.params
            ),
            [np.asarray([4, 5], np.float32)],
        )
        assert Receiver.__call__ is wrapper
        _assert_patch_snapshot(outer)
    assert Receiver.__call__ is original
    assert numeric_calls == []


@pytest.mark.parametrize("weak", (False, True))
def test_source_keeps_native_weak_types_and_symbol_scope(weak: bool) -> None:
    batch = jax_export.symbolic_shape("B")[0]
    specs = [
        jax.ShapeDtypeStruct((batch, 2), np.float32),
        jax.ShapeDtypeStruct((), np.float32, weak_type=weak),
    ]

    def function(x: Any, scalar: Any) -> Any:
        return x + scalar, scalar + 1

    native = jax.make_jaxpr(_fresh(function))(*specs)
    with _world(PrecisionPolicy.SOURCE):
        activated = jax.make_jaxpr(_fresh(function))(*specs)
    assert _recipe(activated) == _recipe(native)
    assert _body(activated).invars[0].aval.shape[0]._scope is batch._scope
    assert _body(activated).invars[1].aval.weak_type is weak


@pytest.mark.parametrize("x64", (False, True))
def test_full_source_activation_preserves_actual_named_nnx_state_rng_and_runtime(
    x64: bool,
) -> None:
    with _runtime_configuration(x64):
        owner = _StateOwner()
        initial = _native_words(owner)
        prepared = prepare_nnx_call(owner)
        fresh = prepared.fresh_original()
        assert fresh.alias is fresh.rngs and fresh.rngs is not owner.rngs
        batch = jax_export.symbolic_shape("B")[0]
        specs = [
            jax.ShapeDtypeStruct((batch, 4), np.float32),
            jax.ShapeDtypeStruct((), np.bool_),
            *(jax.ShapeDtypeStruct(word.shape, word.dtype) for word in initial),
        ]
        native = jax.make_jaxpr(_fresh(prepared))(*specs)
        with _world(PrecisionPolicy.SOURCE):
            activated = jax.make_jaxpr(_fresh(prepared))(*specs)
            assert _recipe(activated, expand_functions=True) == _recipe(native)
        samples = []
        compiled = jax.jit(prepared)
        for size in (2, 5):
            reference_owner = _clone(owner)
            state = prepared.initial_words
            x = np.arange(size * 4, dtype=np.float32).reshape(size, 4) / 8
            for flag in (True, False, False, True):
                expected = reference_owner(x, deterministic=flag)
                values = (x, np.asarray(flag), *(np.asarray(word) for word in state))
                eager = prepared(*values)
                _same(eager[0], expected)
                _same(tuple(eager[1:]), _native_words(reference_owner))
                _same(compiled(*values), eager)
                _same(compiled(*values), eager)
                samples.append(_sample(prepared, values))
                state = tuple(eager[1:])
            _same(prepared.concrete(True, x, *prepared.initial_words)[1:], initial)
        _same(_native_words(owner), initial)
        model = ui._to_onnx(
            prepared, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        assert any(f.name == "NativeActivationStateOwner" for f in model.functions)
        _check_one_artifact(model, samples)


def test_native_unaudited_primitive_still_fails_existing_source_validation() -> None:
    closed = jax.make_jaxpr(jax.lax.neg)(jax.ShapeDtypeStruct((2, 4), np.float32))
    # Canonical native ownership alone never grants finite profile admission.
    # Corrupt the actual native equation at the canonical boundary.
    from jax2onnx._compat.jax import Primitive

    unknown = Primitive("test::native-source-unknown")
    equation = closed.eqns[0].replace(primitive=unknown)
    corrupted = _body(closed).replace(eqns=[equation])
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(corrupted, 23)
