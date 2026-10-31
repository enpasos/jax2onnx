# tests/extra_tests/converter/test_source_literal_cond.py

"""Finite native literal-cond transport; exact comparisons do not certify kernels."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
import json
from pathlib import Path
from typing import Any

import jax
from jax import export
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import (
    ClosedJaxpr,
    Literal,
    Primitive,
    Var,
    is_selected_prng_key_dtype,
    selected_prng_key_words,
)
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.lowering_dispatch import lower_equation_with_plugin
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.plugins.jax.lax.cond import (
    CondPlugin,
    _attach_selected_branch,
    _same_native_aval,
    _source_literal_index,
)
from jax2onnx.plugins.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
)
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)


@onnx_function(name="SourceLiteralCell")
def _source_literal_named_cell(value: Any, state: Any, *, index: int, x64: bool) -> Any:
    return _model(index, "root", x64)(value, state)


@onnx_function(name="UnselectedLiteralUnsupportedBody")
def _source_literal_unsupported_body(value: Any) -> Any:
    return jax.lax.erf(value)


_SOURCE_LITERAL_FUNCTION_TARGETS = (
    _source_literal_named_cell,
    _source_literal_unsupported_body,
)


@pytest.fixture(autouse=True)
def _module_function_bindings() -> Iterator[None]:
    targets = {
        f"{target.__module__}.{target.__name__}": target
        for target in _SOURCE_LITERAL_FUNCTION_TARGETS
    }
    plugins = {name: ONNX_FUNCTION_PLUGIN_REGISTRY[name] for name in targets}

    def unchanged() -> None:
        own_names = {
            name
            for name in ONNX_FUNCTION_PLUGIN_REGISTRY
            if name.startswith(__name__ + ".")
        }
        assert own_names == set(targets)
        for name, target in targets.items():
            assert globals()[target.__name__] is target
            assert ONNX_FUNCTION_PLUGIN_REGISTRY[name] is plugins[name]
            assert PLUGIN_REGISTRY.get("onnx_fn::" + name) is plugins[name]
            assert plugins[name].target is target

    unchanged()
    try:
        yield
    finally:
        unchanged()


@pytest.fixture(autouse=True)
def _native_context() -> Iterator[None]:
    states = (
        jax.enable_x64,
        jax.default_matmul_precision,
        jax.numpy_dtype_promotion,
        jax.numpy_rank_promotion,
        jax.default_prng_impl,
        jax.threefry_partitionable,
        jax.debug_key_reuse,
    )
    names = (
        "jax_enable_x64",
        "jax_default_matmul_precision",
        "jax_numpy_dtype_promotion",
        "jax_numpy_rank_promotion",
        "jax_default_prng_impl",
        "jax_threefry_partitionable",
        "jax_debug_key_reuse",
    )
    globals_before = tuple(state.get_global() for state in states)
    effective_before = tuple(getattr(jax.config, name) for name in names)
    policy_before = active_precision_policy()
    try:
        with ExitStack() as stack:
            for state, value in zip(
                states,
                (False, None, "standard", "allow", "threefry2x32", True, False),
                strict=True,
            ):
                stack.enter_context(state(value))
            yield
    finally:
        # Restore globals too, including if a failed call changed one inside a context.
        for name, value in zip(names, globals_before, strict=True):
            jax.config.update(name, value)
        assert tuple(state.get_global() for state in states) == globals_before
        assert tuple(getattr(jax.config, name) for name in names) == effective_before
        assert active_precision_policy() is policy_before


def _host(value: Any) -> np.ndarray:
    if is_selected_prng_key_dtype(getattr(value, "dtype", None)):
        return np.asarray(selected_prng_key_words(value))
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.complexfloating):
        return np.stack((array.real, array.imag), axis=-1)
    return array


def _same(actual: Any, expected: Any) -> None:
    a, b = _host(actual), _host(expected)
    assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


def _fresh(fn: Any) -> Any:
    def call(*args: Any) -> Any:
        return fn(*args)

    return call


def _nodes(nodes: Any) -> Iterator[Any]:
    for node in nodes:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _nodes(attr.g.node)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for graph in attr.graphs:
                    yield from _nodes(graph.node)


def _observe(fn: Any, feeds: list[tuple[Any, ...]], path: Path, *, ifs: int) -> Any:
    # Freeze the actual finite criterion before constructing any target session.
    (path / "criterion.json").write_text(
        json.dumps(
            {
                "criterion": "every-output-exact-physical-dtype-shape-bytes",
                "reference_calls": [
                    "eager",
                    "warmed-jit-first",
                    "warmed-jit-replay",
                    "fresh-jit",
                ],
                "optimizations": ["disabled", "all"],
                "claim": "observed finite equality only",
            },
            indent=2,
        )
    )
    # Keep the actual feed owners; copies below are independent evidence only.
    owner_feeds = [tuple(_host(value) for value in feed) for feed in feeds]
    before_feeds = [tuple(value.copy() for value in owners) for owners in owner_feeds]
    for number, before in enumerate(before_feeds):
        np.savez(
            path / f"feed-{number}.npz",
            **{f"input_{i}": value for i, value in enumerate(before)},
        )
    spec = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in feeds[0]]
    trace = jax.make_jaxpr(_fresh(fn))(*spec)
    (path / "native-trace.json").write_text(
        json.dumps(typed_trace_manifest(trace), indent=2)
    )
    np.savez(
        path / "native-constants.npz",
        **{f"constant_{i}": _host(v) for i, v in enumerate(trace.consts)},
    )
    (path / "native-constant-types.json").write_text(
        json.dumps(
            [
                {
                    "dtype": str(getattr(v, "dtype", type(v).__name__)),
                    "logical_shape": list(getattr(v, "shape", ())),
                    "key_implementation": (
                        str(jax.random.key_impl(v))
                        if is_selected_prng_key_dtype(getattr(v, "dtype", None))
                        else None
                    ),
                }
                for v in trace.consts
            ],
            indent=2,
        )
    )
    warmed = jax.jit(_fresh(fn))
    references = []
    native_outputs = []
    target_outputs = []
    for number, feed in enumerate(feeds):
        outputs = []
        for call_number, call in enumerate((fn, warmed, warmed, jax.jit(_fresh(fn)))):
            sample = tuple(jax.tree.leaves(call(*feed)))
            outputs.append(sample)
            np.savez(
                path / f"native-{number}-{call_number}.npz",
                **{f"output_{i}": _host(v) for i, v in enumerate(sample)},
            )
            # Freeze the original eager result before another call can affect aliases.
            if call_number == 0:
                references.append(tuple(_host(value).copy() for value in sample))
            after = tuple(_host(value) for value in feed)
            np.savez(
                path / f"native-feed-after-{number}-{call_number}.npz",
                **{f"input_{i}": value for i, value in enumerate(after)},
            )
            assert len(sample) == len(references[number])
            for actual, expected in zip(sample, references[number], strict=True):
                _same(actual, expected)
            for value, before in zip(after, before_feeds[number], strict=True):
                _same(value, before)
        native_outputs.append(outputs)
    model = ui._to_onnx(fn, spec, opset=23, precision_policy=PrecisionPolicy.SOURCE)
    onnx.checker.check_model(model, full_check=True)
    payload = model.SerializeToString()
    (path / "source.onnx").write_bytes(payload)
    all_nodes = [
        node
        for root in [model.graph.node, *(f.node for f in model.functions)]
        for node in _nodes(root)
    ]
    assert sum(node.op_type == "If" for node in all_nodes) == ifs
    report = get_precision_report(model)
    assert report is not None
    checks = {check.category: check.status for check in report.checks}
    assert checks["representation"] is CheckStatus.VALIDATED
    assert checks["lowering"] is CheckStatus.VALIDATED
    assert checks["runtime-execution"] is CheckStatus.UNVERIFIED
    assert checks["numerical-acceptance"] is CheckStatus.UNVERIFIED
    for mode, level in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = level
        session = ort.InferenceSession(
            payload, sess_options=options, providers=["CPUExecutionProvider"]
        )
        for number, (feed, expected) in enumerate(zip(feeds, references, strict=True)):
            owners = owner_feeds[number]
            actual = session.run(
                None,
                dict(zip((v.name for v in model.graph.input), owners, strict=True)),
            )
            # Save this session's actual arrays before any comparison.
            np.savez(
                path / f"actual-{mode}-{number}.npz",
                **{f"output_{i}": v for i, v in enumerate(actual)},
            )
            target_outputs.append(actual)
            np.savez(
                path / f"actual-feed-after-{mode}-{number}.npz",
                **{f"input_{i}": value for i, value in enumerate(owners)},
            )
            assert len(actual) == len(expected)
            for value, reference in zip(actual, expected, strict=True):
                _same(value, reference)
            for value, before in zip(owners, before_feeds[number], strict=True):
                _same(value, before)
    return model


def _assert_dynamic_duplicate_bridges(model: Any) -> None:
    conditional = next(node for node in model.graph.node if node.op_type == "If")
    branches = {attribute.name: attribute.g for attribute in conditional.attribute}
    outer = {value.name for value in model.graph.input} | {
        value.name for value in model.graph.initializer
    }
    for branch in branches.values():
        assert len(branch.output) == len(conditional.output)
        assert len({value.name for value in branch.output}) == len(branch.output)
        defined = outer | {value.name for value in branch.initializer}
        for node in branch.node:
            assert all(name in defined for name in node.input if name)
            assert all(name and name not in defined for name in node.output)
            defined.update(node.output)
    branch = branches["else_branch"]
    producers = {name: node for node in branch.node for name in node.output}
    alias_ports = branch.output[1:3]
    first, second = (producers[value.name] for value in alias_ports)
    assert first is not second and first.op_type == second.op_type == "Identity"
    assert len(first.input) == 1 and list(first.input) == list(second.input)
    assert producers[first.input[0]].op_type == "Neg"
    for value in alias_ports:
        tensor = value.type.tensor_type
        assert tensor.elem_type == onnx.TensorProto.FLOAT
        assert [(dim.dim_value, dim.dim_param) for dim in tensor.shape.dim] == [(2, "")]


def _model(index: int, scope: str, x64: bool) -> Any:
    constant = np.asarray([1.25, -2.5], np.float32)
    wide = np.asarray([1 + 2**-30, -2.5], np.float64 if x64 else np.float32)

    def branch(sign: int, value: Any, state: Any) -> Any:
        signed = value if sign == 1 else -value
        return (
            signed + constant,
            value,
            value,
            (value + np.float32(1)) - value,
            signed.astype(wide.dtype) + wide,
            np.float32(sign * 0.5),
            state + np.uint32(1),
            state,
        )

    def cell(value: Any, state: Any) -> Any:
        return jax.lax.cond(
            bool(index),
            lambda v, s: branch(1, v, s),
            lambda v, s: branch(-1, v, s),
            value,
            state,
        )

    if scope == "named":
        # Resolve the module global inside activation; a captured original callable
        # would bypass the Function patch even if its name existed in the module.
        return lambda value, state: _source_literal_named_cell(
            value, state, index=index, x64=x64
        )
    if scope == "nested":
        return lambda value, state: jax.lax.cond(
            True, cell, lambda v, s: cell(-v, s), value, state
        )
    if scope == "dynamic-if":
        return lambda value, state, flag: jax.lax.cond(
            flag, cell, lambda v, s: cell(-v, s), value, state
        )
    if scope == "loop":

        def loop(value: Any, state: Any) -> Any:
            def step(_: Any, carry: Any) -> Any:
                result = cell(*carry)
                return result[0], result[6]

            result = jax.lax.fori_loop(0, 2, step, (value, state))
            return (*cell(*result), value, state)

        return loop
    return cell


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("scope", ["root", "named", "nested", "dynamic-if", "loop"])
def test_original_literal_cond_scopes_keep_native_outputs_aliases_and_state(
    index: int,
    x64: bool,
    scope: str,
    tmp_path: Path,
) -> None:
    with jax.enable_x64(x64):
        fn = _model(index, scope, x64)
        x = np.asarray([1e8, -0.0], np.float32)
        state = np.asarray([7, 19], np.uint32)
        feeds = [(x, state)]
        if scope == "dynamic-if":
            feeds = [
                (x, state, np.asarray(flag)) for flag in (True, False, False, True)
            ]
        model = _observe(fn, feeds, tmp_path, ifs=int(scope == "dynamic-if"))
        assert len(model.graph.output) == (10 if scope == "loop" else 8)
        if scope == "dynamic-if":
            _assert_dynamic_duplicate_bridges(model)
        if scope == "named":
            assert any(node.domain for node in model.graph.node)
            assert model.functions
            assert any(
                node.domain == function.domain and node.op_type == function.name
                for node in model.graph.node
                for function in model.functions
            )
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                active = jax.make_jaxpr(_fresh(fn))(
                    *(
                        jax.ShapeDtypeStruct(value.shape, value.dtype)
                        for value in feeds[0]
                    )
                )
                assert any(
                    eqn.primitive.name
                    == f"onnx_fn::{__name__}._source_literal_named_cell"
                    for eqn in active.jaxpr.eqns
                )
        if scope == "loop":
            assert any(node.op_type == "Loop" for node in _nodes(model.graph.node))


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("x64", [False, True])
def test_literal_cond_preserves_packed_values_and_actual_typed_key_constant(
    index: int,
    x64: bool,
    tmp_path: Path,
) -> None:
    with jax.enable_x64(x64):
        key = jax.random.key(17, impl="threefry2x32")
        logical = (str(key.dtype), str(jax.random.key_impl(key)), tuple(key.shape))
        words = np.asarray(selected_prng_key_words(key)).copy()
        constant = np.asarray([1.25 + 0.5j, -2.5 - 0.0j], np.complex64)

        def fn(value: Any) -> Any:
            return jax.lax.cond(
                bool(index),
                lambda v: (v, v, constant, key, np.float32(0.5)),
                lambda v: (constant, v, constant, key, np.float32(-0.5)),
                value,
            )

        (tmp_path / "logical-key.json").write_text(json.dumps(logical))
        model = _observe(fn, [(constant * np.complex64(2),)], tmp_path, ifs=0)
        report = get_precision_report(model)
        assert report is not None
        logical_output = report.traced_outputs[3]
        assert logical_output.dtype == key.dtype.name == "key<fry>"
        assert logical_output.shape == () and logical_output.weak_type is False
        physical_output = report.exported_outputs[3]
        assert physical_output.dtype == "uint32" and physical_output.shape == (2,)
        tensor = model.graph.output[3].type.tensor_type
        assert tensor.elem_type == onnx.TensorProto.UINT32
        assert [(dim.dim_value, dim.dim_param) for dim in tensor.shape.dim] == [(2, "")]
        name = model.graph.output[3].name
        producers = {value: node for node in model.graph.node for value in node.output}
        while name in producers and producers[name].op_type == "Identity":
            assert len(producers[name].input) == 1
            name = producers[name].input[0]
        initializers = {value.name: value for value in model.graph.initializer}
        if name in initializers:
            stored_words = onnx.numpy_helper.to_array(initializers[name])
        else:
            producer = producers[name]
            assert producer.op_type == "Constant"
            value = next(
                attribute.t
                for attribute in producer.attribute
                if attribute.name == "value"
            )
            stored_words = onnx.numpy_helper.to_array(value)
        _same(stored_words, words)
        assert (
            str(key.dtype),
            str(jax.random.key_impl(key)),
            tuple(key.shape),
        ) == logical
        _same(selected_prng_key_words(key), words)


def _equation(index: int = 0) -> Any:
    graph = jax.make_jaxpr(
        lambda x: jax.lax.cond(
            bool(index),
            lambda v: (v + np.float32(1), v, v, np.float32(0.5)),
            lambda v: (-v, v, v, np.float32(-0.5)),
            x,
        )
    )(jax.ShapeDtypeStruct((2,), np.float32))
    return next(e for e in graph.jaxpr.eqns if e.primitive is jax.lax.cond_p)


def _context(eqn: Any) -> tuple[Any, Any]:
    ctx = IRContext(
        opset=23,
        enable_double_precision=False,
        precision_policy=PrecisionPolicy.SOURCE,
        jaxpr_validator=validate_source_jaxpr,
    )
    var = eqn.invars[1]
    rep = representation_for_aval(var.aval)
    value = ir.Value(
        name="original_input",
        type=ir.TensorType(rep.ir_dtype),
        shape=ir.Shape(rep.physical_shape),
    )
    ctx.builder.inputs.append(value)
    ctx.bind_value_for_var(var, value)
    return ctx, value


def _lower(ctx: Any, eqn: Any) -> None:
    lower_equation_with_plugin(
        CondPlugin(),
        ctx=ctx,
        eqn=eqn,
        primitive_name="cond",
        eqn_index=0,
        source="literal-test",
    )


def test_actual_reused_branch_jaxpr_keeps_separate_graph_ownership_and_outer_values() -> (
    None
):
    eqn = _equation()
    branches = eqn.params["branches"]
    arity = len(eqn.outvars)
    assert arity > 0
    ctx, first = _context(eqn)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        _lower(ctx, eqn)
        first_outputs = tuple(ctx.require_value_for_var(var) for var in eqn.outvars)
        second = ir.Value(name="second_input", type=first.type, shape=first.shape)
        ctx.builder.inputs.append(second)
        ctx.bind_value_for_var(eqn.invars[1], second)
        # The exact equation and each exact native branch object are reused.
        assert eqn.params["branches"] is branches
        assert all(
            a is b for a, b in zip(eqn.params["branches"], branches, strict=True)
        )
        _lower(ctx, eqn)
    second_outputs = tuple(ctx.require_value_for_var(var) for var in eqn.outvars)
    assert len(first_outputs) == len(second_outputs) == arity
    assert all(a is not b for a, b in zip(first_outputs, second_outputs, strict=True))
    assert len({id(v) for v in (*first_outputs, *second_outputs)}) == 2 * arity
    nodes = tuple(ctx.builder.graph)
    assert all(node.graph is ctx.builder.graph for node in nodes)
    values = [*ctx.builder.inputs, *(v for node in nodes for v in node.outputs)]
    assert len({v.name for v in values}) == len(values)
    assert all(
        v.producer().op_type == "Identity" for v in (*first_outputs, *second_outputs)
    )
    assert any(node.op_type == "Identity" and node.inputs[0] is first for node in nodes)
    assert any(
        node.op_type == "Identity" and node.inputs[0] is second for node in nodes
    )
    assert first.name == "original_input" and first.producer() is None
    assert second.name == "second_input" and second.producer() is None


@pytest.mark.parametrize("index", [0, 1])
def test_unselected_actual_named_function_body_still_retraces_and_rejects_erf(
    index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fn(value: Any) -> Any:
        true, false = (
            ((lambda v: v + np.float32(1)), _source_literal_unsupported_body)
            if index
            else (_source_literal_unsupported_body, (lambda v: -v))
        )
        return jax.lax.cond(bool(index), true, false, value)

    spec = jax.ShapeDtypeStruct((2,), np.float32)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        trace = jax.make_jaxpr(_fresh(fn))(spec)
        # Raw call-marker validation is insufficient; actual lowering must retrace.
        validate_source_jaxpr(trace.jaxpr, 23)
        cond = next(e for e in trace.jaxpr.eqns if e.primitive is jax.lax.cond_p)
        dead = cond.params["branches"][1 - index].jaxpr
        assert any(e.primitive.name.startswith("onnx_fn::") for e in dead.eqns)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid dead Function must fail before a target session")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    with pytest.raises(SourcePrecisionError) as captured:
        ui._to_onnx(fn, [spec], opset=23, precision_policy=PrecisionPolicy.SOURCE)
    assert captured.value.primitive == "erf"


def test_default_legacy_literal_keeps_its_original_if_structure() -> None:
    def fn(x: Any) -> Any:
        return jax.lax.cond(False, lambda v: -v, lambda v: v + np.float32(1), x)

    model = ui._to_onnx(fn, [jax.ShapeDtypeStruct((2,), np.float32)], opset=23)
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    assert any(node.op_type == "If" for node in model.graph.node)


@pytest.mark.parametrize(
    "corruption",
    [
        "negative",
        "two",
        "bool",
        "float",
        "wide-payload",
        "vector",
        "weak",
        "wrong-primitive",
        "params",
        "platform-literal",
        "one-branch",
        "branch-list",
        "signature",
        "branch-weak",
        "effects",
        "const-arity",
    ],
)
def test_native_literal_guard_rejects_corrupt_actual_equations(corruption: str) -> None:
    eqn = _equation()
    selector = eqn.invars[0]
    aval = selector.aval
    if corruption in {
        "negative",
        "two",
        "bool",
        "float",
        "wide-payload",
        "vector",
        "weak",
    }:
        values = {
            "negative": np.int32(-1),
            "two": np.int32(2),
            "bool": np.bool_(False),
            "float": np.float32(0),
            "wide-payload": np.int64(0),
            "vector": np.asarray([0], np.int32),
            "weak": np.int32(0),
        }
        if corruption == "weak":
            aval = aval.update(weak_type=True)
        eqn = eqn.replace(invars=[Literal(values[corruption], aval), *eqn.invars[1:]])
    elif corruption == "wrong-primitive":
        eqn = eqn.replace(primitive=Primitive("cond"))
    elif corruption in {"params", "platform-literal", "one-branch", "branch-list"}:
        params = dict(eqn.params)
        if corruption == "params":
            params["unreviewed"] = True
        elif corruption == "platform-literal":
            params["branches_platforms"] = (("mosaic",), ("cpu",))
        elif corruption == "one-branch":
            params["branches"] = params["branches"][:1]
        else:
            params["branches"] = list(params["branches"])
        eqn = eqn.replace(params=params)
    elif corruption == "effects":
        eqn = eqn.replace(effects=frozenset({"unexpected-effect"}))
    else:
        closed = eqn.params["branches"][0]
        body = closed.jaxpr
        if corruption in {"signature", "branch-weak"}:
            formal = body.invars[0]
            changed = formal.aval.update(
                **(
                    {"dtype": np.dtype(np.float64)}
                    if corruption == "signature"
                    else {"weak_type": True}
                )
            )
            body = body.replace(invars=[Var(changed)])
            closed = ClosedJaxpr(body, list(closed.consts))
        else:
            # Public native const list corruption: old ClosedJaxpr arity mismatch;
            # newer unified Jaxpr also shifts the formal boundary and must reject.
            closed = ClosedJaxpr(body, list(closed.consts))
            closed.consts.append(np.float32(3))
        eqn = eqn.replace(params={"branches": (closed, eqn.params["branches"][1])})
    with pytest.raises(ValueError):
        _source_literal_index(eqn)


def test_equal_symbol_spelling_does_not_replace_actual_native_scope_identity() -> None:
    left = export.symbolic_shape("B", scope=export.SymbolicScope())[0]
    right = export.symbolic_shape("B", scope=export.SymbolicScope())[0]
    first = jax.ShapeDtypeStruct((left, 2), np.float32)
    second = jax.ShapeDtypeStruct((right, 2), np.float32)
    assert not _same_native_aval(first, second)


@pytest.mark.parametrize(
    "corruption",
    ["operand", "staged-type", "staged-unbound", "clone-unbound", "allocated"],
)
def test_physical_transport_rejects_drift_before_parent_attachment(
    corruption: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eqn = _equation()
    ctx, value = _context(eqn)
    plugin = CondPlugin()
    if corruption == "operand":
        value.type = ir.TensorType(ir.DataType.DOUBLE)
        with pytest.raises(ValueError):
            plugin.lower(ctx, eqn)
        assert not tuple(ctx.builder.graph)
        return
    closed = eqn.params["branches"][0]
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        graph = plugin._build_branch_graph(
            ctx,
            closed.jaxpr,
            closed.consts,
            [value],
            prefix="physical_control",
            source_literal=True,
        )
    assert not graph.initializers and not graph.inputs
    if corruption == "staged-type":
        graph.outputs[0].type = ir.TensorType(ir.DataType.DOUBLE)
    elif corruption == "staged-unbound":
        old = graph.outputs[0]
        graph.outputs[0] = ir.Value(name="disconnected", type=old.type, shape=old.shape)
    elif corruption == "clone-unbound":
        original_clone = ir.Graph.clone

        def bad_clone(self: Any, **kwargs: Any) -> Any:
            result = original_clone(self, **kwargs)
            old = result.outputs[0]
            result.outputs[0] = ir.Value(
                name="clone_disconnected", type=old.type, shape=old.shape
            )
            return result

        monkeypatch.setattr(ir.Graph, "clone", bad_clone)
    else:
        allocate = ctx.allocate_value_for_var

        def bad_allocate(var: Any, **kwargs: Any) -> Any:
            result = allocate(var, **kwargs)
            result.type = ir.TensorType(ir.DataType.DOUBLE)
            return result

        monkeypatch.setattr(ctx, "allocate_value_for_var", bad_allocate)
    with pytest.raises(ValueError):
        _attach_selected_branch(ctx, graph, eqn)
    assert not tuple(ctx.builder.graph)


def test_function_child_materializes_array_literal_and_selected_key_as_constants() -> (
    None
):
    key = jax.random.key(31, impl="threefry2x32")
    array = np.asarray([1.25, -2.5], np.float32)
    closed = jax.make_jaxpr(lambda x: (x, array, key, np.float32(0.5)))(
        jax.ShapeDtypeStruct((2,), np.float32)
    )
    ctx = IRContext(
        opset=23,
        enable_double_precision=False,
        precision_policy=PrecisionPolicy.SOURCE,
        jaxpr_validator=validate_source_jaxpr,
    )
    rep = representation_for_aval(closed.jaxpr.invars[0].aval)
    value = ir.Value(
        name="captured_original",
        type=ir.TensorType(rep.ir_dtype),
        shape=ir.Shape(rep.physical_shape),
    )
    ctx.builder.inputs.append(value)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        graph = CondPlugin()._build_branch_graph(
            ctx,
            closed.jaxpr,
            closed.consts,
            [value],
            prefix="constant_proof",
            source_literal=True,
        )
    assert not graph.initializers and not graph.inputs
    assert all(
        v.producer() is not None and v.producer().graph is graph for v in graph.outputs
    )
    constants = [
        node.attributes["value"].as_tensor().numpy()
        for node in graph
        if node.op_type == "Constant"
    ]
    for expected in (
        array,
        np.asarray(selected_prng_key_words(key)),
        np.asarray(0.5, np.float32),
    ):
        assert any(
            (v.dtype, v.shape, v.tobytes())
            == (expected.dtype, expected.shape, expected.tobytes())
            for v in constants
        )
