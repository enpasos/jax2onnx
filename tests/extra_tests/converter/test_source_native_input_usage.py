# tests/extra_tests/converter/test_source_native_input_usage.py

"""Actual native-unused proof and directed provenance/dataflow corruptions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import FunctionPlugin
from tests.extra_tests.converter.source_integer_structure import (
    _called_scopes,
    check_integer_callback_counterpart,
)
from tests.extra_tests.converter.source_native_input_usage import (
    NativeInputUsage,
    _lexical_usage,
    check_native_input_usage,
    native_unused_positions,
)


@onnx_function(name="NativeUsageLeft")
def _usage_left(x: Any, y: Any, flag: Any) -> Any:
    return x + y


@onnx_function(name="NativeUsageRight")
def _usage_right(x: Any, y: Any, flag: Any) -> Any:
    return x - y


@onnx_function(name="NativeUsageOuter")
def _usage_outer(x: Any, y: Any, flag: Any) -> Any:
    first = _usage_left(x, y, flag)
    second = _usage_right(x, y, flag)
    return first, second, _usage_left(y, x, flag)


def _usage_root(x: Any, y: Any, flag: Any) -> Any:
    return _usage_outer(x, y, flag)


@onnx_function(name="NativeUsageStateKeyword")
class _StateKeyword(nnx.Module):
    def __init__(self) -> None:
        self.total = nnx.Variable(jnp.zeros((2,), np.float32))

    def __call__(self, value: Any, *, increment: Any) -> Any:
        self.total[...] = self.total[...] + increment
        return value - self.total[...]


class _PortSubclass(int):
    pass


class _OrderSubclass(tuple):
    pass


def _export() -> tuple[NativeInputUsage, Any]:
    proof = NativeInputUsage()
    with proof.observe():
        artifact = ui._to_onnx(
            _usage_root,
            [
                jax.ShapeDtypeStruct((2,), np.float32),
                jax.ShapeDtypeStruct((2,), np.float32),
                jax.ShapeDtypeStruct((), np.bool_),
            ],
            precision_policy=PrecisionPolicy.SOURCE,
            opset=23,
        )
    return proof, artifact


def _all_nodes(model: Any) -> list[Any]:
    return [*model.graph.node, *(node for f in model.functions for node in f.node)]


def test_actual_nested_cached_same_shaped_callees_preserve_unused_native_declarations() -> (
    None
):
    proof, artifact = _export()
    result = proof.seal(artifact)
    assert result["actual_native_function_definitions"] == 3
    assert result["actual_native_function_calls"] == 4
    assert {row["definition"][1] for row in result["native_unused_formals"]} == {
        "NativeUsageLeft",
        "NativeUsageRight",
    }
    assert {row["position"] for row in result["native_unused_formals"]} == {2}
    assert result["root_unused_exemptions"] == []
    left = next(f for f in artifact.functions if f.name == "NativeUsageLeft")
    calls = [
        n
        for n in _all_nodes(artifact)
        if (n.domain, n.op_type, n.overload) == (left.domain, left.name, left.overload)
    ]
    assert len(calls) == 2 and len(left.input) == 3
    assert all(len(node.input) == 3 for node in calls)
    raw = expect_graph(
        [], no_unused_inputs=True, no_unused_function_inputs=True, search_functions=True
    )
    assert raw(artifact) is False
    with pytest.raises(ValueError, match="without native proof"):
        _called_scopes(artifact)
    assert _called_scopes(artifact, native_input_usage=proof)
    amended = check_integer_callback_counterpart(
        raw, artifact, native_input_usage=proof
    )
    assert amended["native_unused_correspondence"] == result
    assert raw(artifact) is False
    forbidden = expect_graph([], must_absent=["Add"], search_functions=True)
    with pytest.raises(ValueError, match="preserved original callback"):
        check_integer_callback_counterpart(
            forbidden, artifact, native_input_usage=proof
        )


@pytest.mark.parametrize(
    "kind",
    ("extra-formal", "extra-root", "swapped-call", "arity", "overload", "other-body"),
)
def test_exact_artifact_mapping_rejects_mutation_before_proof_seal(kind: str) -> None:
    proof, artifact = _export()
    definition = next(f for f in artifact.functions if f.name == "NativeUsageLeft")
    identifier = definition.domain, definition.name, definition.overload
    calls = [
        node
        for node in _all_nodes(artifact)
        if (node.domain, node.op_type, node.overload) == identifier
    ]
    if kind == "extra-formal":
        definition.input.append("unproven_extra")
        for node in calls:
            node.input.append(node.input[0])
    elif kind == "extra-root":
        extra = artifact.graph.input.add()
        extra.CopyFrom(artifact.graph.input[0])
        extra.name = "unproven_root"
    elif kind == "swapped-call":
        calls[0].input[0], calls[0].input[1] = calls[0].input[1], calls[0].input[0]
    elif kind == "arity":
        calls[0].input.pop()
    elif kind == "overload":
        definition.overload = "different_actual_overload"
        for node in calls:
            node.overload = definition.overload
    else:
        next(node for node in definition.node if node.op_type == "Add").op_type = "Sub"
    with pytest.raises(ValueError, match="exact observed final IR model"):
        proof.seal(artifact)


def test_native_used_formal_disconnection_fails_even_with_matching_final_ir_artifact() -> (
    None
):
    proof, _ = _export()
    frame = next(
        frame
        for frame in proof._definitions.values()
        if frame.definition.name == "NativeUsageLeft"
    )
    graph = frame.definition.graph
    left, right = graph.inputs[:2]
    changed = False
    for node in graph:
        for position, value in enumerate(node.inputs):
            if value is left:
                node.replace_input_with(position, right)
                changed = True
    assert changed and 0 not in frame.unused
    corrupted = ir.to_proto(proof._ir_model)
    with pytest.raises(
        ValueError, match="Native-used Function formal became disconnected"
    ):
        proof.seal(corrupted)


def test_proof_rejects_different_artifact_and_stale_mutation_after_seal() -> None:
    proof, artifact = _export()
    proof.seal(artifact)
    copy = onnx.ModelProto()
    copy.CopyFrom(artifact)
    with pytest.raises(ValueError, match="stale or different"):
        proof.check(copy)
    artifact.doc_string = "mutation after a successful proof"
    with pytest.raises(ValueError, match="stale or different"):
        proof.check(artifact)


def test_native_formal_use_counts_identity_results_state_keys_and_lexical_references() -> (
    None
):
    captured = jnp.asarray([1, 2], np.float32)
    closed = jax.make_jaxpr(lambda x, unused: x + captured)(
        jax.ShapeDtypeStruct((2,), np.float32), jax.ShapeDtypeStruct((), np.bool_)
    )
    assert closed.jaxpr.constvars
    assert native_unused_positions(closed) == {1}
    passthrough = jax.make_jaxpr(lambda x, key: (x + np.float32(1), key))(
        jax.ShapeDtypeStruct((2,), np.float32), jax.random.key(5)
    )
    assert native_unused_positions(passthrough) == set()
    x, unused = object(), object()
    inner = SimpleNamespace(invars=[], outvars=[unused], eqns=[], effects=frozenset())
    equation = SimpleNamespace(
        invars=[x], params={"nested": ({"body": inner},)}, effects=frozenset()
    )
    outer = SimpleNamespace(
        invars=[x, unused], outvars=[x], eqns=[equation], effects=frozenset()
    )
    assert native_unused_positions(outer) == set()
    outer.effects = frozenset({"unmodeled-effect"})
    outer.eqns = []
    assert native_unused_positions(outer) == set()


def test_onnx_lexical_output_capture_and_direct_output_are_used() -> None:
    helper = onnx.helper
    tensor = onnx.TensorProto
    captured = helper.make_tensor_value_info("captured", tensor.FLOAT, [2])
    condition = helper.make_tensor_value_info("condition", tensor.BOOL, [])
    branch = helper.make_graph([], "lexical", [], [captured])
    graph = helper.make_graph(
        [
            helper.make_node(
                "If", ["condition"], ["result"], then_branch=branch, else_branch=branch
            )
        ],
        "root",
        [captured, condition],
        [helper.make_tensor_value_info("result", tensor.FLOAT, [2])],
    )
    assert _lexical_usage(graph) == (set(), set())
    direct = helper.make_graph([], "direct", [captured], [captured])
    assert _lexical_usage(direct) == (set(), set())
    model = helper.make_model(graph)
    assert check_native_input_usage(model)["mode"] == "strict-no-proof"
    graph.input.add().CopyFrom(
        helper.make_tensor_value_info("extra", tensor.FLOAT, [2])
    )
    assert _lexical_usage(graph)[0] == {"extra"}


def test_exact_build_frames_restore_after_error_and_reject_wrong_body_owner() -> None:
    proof = NativeInputUsage()
    first, second = object(), object()
    ctx = SimpleNamespace(
        require_value_for_var=lambda value: value,
        precision_policy=PrecisionPolicy.SOURCE,
    )
    equation = SimpleNamespace(invars=[])
    captures = SimpleNamespace(dynamic=[])
    body = SimpleNamespace(
        invars=[], outvars=[], constvars=[], eqns=[], effects=frozenset()
    )
    with pytest.raises(RuntimeError, match="directed nested build failure"):
        with proof.build(
            first, ctx, None, equation, captures, source_trace=object()
        ) as outer:
            with proof.build(
                second, ctx, None, equation, captures, source_trace=object()
            ):
                with pytest.raises(ValueError, match="exact active build frame"):
                    proof.retain_body(first, body)
                proof.retain_body(second, body)
                raise RuntimeError("directed nested build failure")
            proof.retain_definition(outer, object())
    assert not proof._stack and not proof._definitions
    with pytest.raises(ValueError, match="exact active build frame"):
        proof.retain_body(first, body)


def test_actual_state_keyword_port_metadata_is_retained_and_forwarded_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphdef, state = nnx.split(_StateKeyword())
    leaves, tree = jax.tree.flatten(state)
    assert len(leaves) == 1

    def target(value: Any, increment: Any, total: Any) -> Any:
        owner = nnx.merge(graphdef, jax.tree.unflatten(tree, [total]))
        result = owner(value, increment=increment)
        return result, owner.total[...]

    proof = NativeInputUsage()
    original = FunctionPlugin._build_function
    observed = []

    def build(
        plugin: Any,
        ctx: Any,
        callee: Any,
        equation: Any,
        captures: Any,
        *,
        source_trace: Any = None,
        source_body_order: tuple[int, ...] = (),
    ) -> Any:
        assert proof._stack[-1].source_body_order is source_body_order
        assert source_body_order == (0, 2, 1)
        observed.append(source_body_order)
        return original(
            plugin,
            ctx,
            callee,
            equation,
            captures,
            source_trace=source_trace,
            source_body_order=source_body_order,
        )

    monkeypatch.setattr(FunctionPlugin, "_build_function", build)
    with proof.observe():
        artifact = ui._to_onnx(
            target,
            [jax.ShapeDtypeStruct((2,), np.float32)] * 3,
            precision_policy=PrecisionPolicy.SOURCE,
            opset=23,
        )
    assert observed == [(0, 2, 1)]
    result = proof.seal(artifact)
    assert result["actual_native_function_definitions"] == 1
    assert result["actual_native_function_calls"] == 1
    assert result["root_unused_exemptions"] == []
    assert not proof._stack


@pytest.mark.parametrize(
    "order",
    [
        [0, 1, 2],
        (0, 0, 2),
        (0, 1),
        (0, 1, 3),
        (0, False, 2),
        (0, 1.0, 2),
        (0, _PortSubclass(1), 2),
        _OrderSubclass((0, 1, 2)),
    ],
)
def test_native_build_rejects_malformed_known_port_metadata_without_binding(
    order: Any,
) -> None:
    proof = NativeInputUsage()
    reads = []
    ctx = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        require_value_for_var=lambda value: reads.append(value),
    )
    equation = SimpleNamespace(invars=[object(), object()])
    captures = SimpleNamespace(dynamic=[SimpleNamespace(value=object())])
    with pytest.raises(ValueError, match="source port metadata.*complete permutation"):
        with proof.build(
            object(),
            ctx,
            None,
            equation,
            captures,
            source_trace=object(),
            source_body_order=order,
        ):
            pytest.fail("invalid source port metadata entered a build frame")
    assert reads == [] and proof._stack == [] and proof._definitions == {}


def test_native_build_preserves_empty_default_and_rejects_unknown_keyword() -> None:
    proof = NativeInputUsage()
    ctx = SimpleNamespace(precision_policy=PrecisionPolicy.SOURCE)
    equation, captures = SimpleNamespace(invars=[]), SimpleNamespace(dynamic=[])
    with proof.build(
        object(), ctx, None, equation, captures, source_trace=object()
    ) as frame:
        assert frame.source_body_order == () and frame.parent_values == ()
    assert not proof._stack
    with pytest.raises(TypeError, match="unexpected keyword"):
        with proof.build(
            object(),
            ctx,
            None,
            equation,
            captures,
            source_trace=object(),
            unknown_port_metadata=(),
        ):
            pytest.fail("unknown metadata keyword entered a build frame")
    assert not proof._stack
