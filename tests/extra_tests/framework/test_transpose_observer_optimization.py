# tests/extra_tests/framework/test_transpose_observer_optimization.py

"""Layout rewrites preserve outputs/captures before changing any value's layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx.converter import ir_optimizations as optimizer
from jax2onnx.converter.ir_optimizations import (
    optimize_graph,
    remove_redundant_transpose_add_forests_ir,
    remove_redundant_transpose_pairs_ir,
)
from jax2onnx.converter.precision_policy import PrecisionPolicy


_P = (0, 2, 3, 1)
_Q = (0, 3, 1, 2)
_SHAPE = (1, 4, 2, 3)
_CANONICAL = (1, 2, 3, 4)
_DATA = (np.arange(24, dtype=np.float32).reshape(_SHAPE) - 11) / np.float32(4)


@dataclass
class _Fixture:
    model: ir.Model
    graph: ir.Graph
    input_transposes: list[ir.Node]
    region: list[ir.Node]
    canonical: ir.Value
    intermediate: ir.Value


def _v(
    name: str, shape: tuple[int, ...], dtype: ir.DataType = ir.DataType.FLOAT
) -> ir.Value:
    return ir.val(name, dtype=dtype, shape=shape)


def _constant(name: str, data: Any) -> ir.Value:
    data = np.asarray(data)
    return ir.val(
        name,
        dtype=ir.DataType.from_numpy(data.dtype),
        shape=data.shape,
        const_value=ir.tensor(data),
    )


def _transpose(value: ir.Value, name: str, perm: tuple[int, ...]) -> ir.Node:
    return ir.Node(
        "",
        "Transpose",
        [value],
        outputs=[_v(name, tuple(value.shape[index] for index in perm))],
        attributes=[ir.AttrInt64s("perm", perm)],
    )


def _build(op: str = "Neg", observer: str = "none", scalar: bool = False) -> _Fixture:
    x = _v("in_0", _SHAPE)
    first = _transpose(x, "canonical", _P)
    canonical = first.outputs[0]
    canonical.metadata_props["observer"] = "keep original NHWC"
    inputs = [x]
    nodes = [first]
    transposes = [first]
    initializers = []
    if op == "Add":
        if scalar:
            side = _constant("scalar", np.float32(2))
            initializers.append(side)
        else:
            side_input = _v("in_1", _SHAPE)
            inputs.append(side_input)
            second = _transpose(side_input, "side_canonical", _P)
            transposes.append(second)
            nodes.append(second)
            side = second.outputs[0]
        operands = [canonical, side]
    else:
        operands = [canonical]
    intermediate = _v("intermediate", _CANONICAL)
    region = ir.Node("", op, operands, outputs=[intermediate])
    nodes.append(region)
    last = _transpose(intermediate, "result", _Q)
    nodes.append(last)
    outputs = [last.outputs[0]]
    observed = (
        last.outputs[0]
        if observer.startswith("output")
        else intermediate if observer.startswith("intermediate") else canonical
    )
    observed_shape = _SHAPE if observer.startswith("output") else _CANONICAL
    if observer.endswith("output"):
        outputs.append(observed)
    elif observer == "input-shared":
        copied = _v("shared", _CANONICAL)
        nodes.append(ir.Node("", "Identity", [observed], outputs=[copied]))
        outputs.append(copied)
    elif observer.endswith("if"):
        choose = _v("choose", (), ir.DataType.BOOL)
        inputs.append(choose)
        direct = observer in {"input-name-only-if", "output-name-only-if"}
        branches = []
        for prefix in ("then", "else"):
            if direct:
                # A valid lexical name reference can lack the parent IR use link.
                reference = _v(observed.name, observed_shape)
                out = _v(prefix + "_out", observed_shape)
                branch = ir.Graph(
                    [],
                    [out],
                    nodes=[ir.Node("", "Identity", [reference], outputs=[out])],
                    name=prefix,
                )
            else:
                out = _v(prefix + "_out", observed_shape)
                branch = ir.Graph(
                    [],
                    [out],
                    nodes=[ir.Node("", "Identity", [observed], outputs=[out])],
                    name=prefix,
                )
            branches.append(branch)
        nested = _v("nested", observed_shape)
        nodes.append(
            ir.Node(
                "",
                "If",
                [choose],
                outputs=[nested],
                attributes=[
                    ir.AttrGraph("then_branch", branches[0]),
                    ir.AttrGraph("else_branch", branches[1]),
                ],
            )
        )
        outputs.append(nested)
    elif observer == "input-loop":
        trip = _constant("trip", np.int64(2))
        condition = _constant("condition", np.asarray(True))
        initial = _constant("initial", np.zeros(_CANONICAL, np.float32))
        initializers.extend((trip, condition, initial))
        iteration = _v("iteration", (), ir.DataType.INT64)
        cond = _v("cond", (), ir.DataType.BOOL)
        carry = _v("carry", _CANONICAL)
        captured = _v("captured", _CANONICAL)
        body = ir.Graph(
            [iteration, cond, carry],
            [cond, captured],
            nodes=[ir.Node("", "Identity", [observed], outputs=[captured])],
            name="body",
        )
        nested = _v("loop_result", _CANONICAL)
        nodes.append(
            ir.Node(
                "",
                "Loop",
                [trip, condition, initial],
                outputs=[nested],
                attributes=[ir.AttrGraph("body", body)],
            )
        )
        outputs.append(nested)
    graph = ir.Graph(
        inputs,
        outputs,
        nodes=nodes,
        initializers=initializers,
        opset_imports={"": 23},
        name="observed_layout",
    )
    return _Fixture(
        ir.Model(graph, ir_version=10),
        graph,
        transposes,
        [region],
        canonical,
        intermediate,
    )


def _run(model: ir.Model, optimize: bool) -> list[np.ndarray]:
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    values = {
        "in_0": _DATA,
        "in_1": np.flip(_DATA, axis=3).copy(),
        "choose": np.asarray(True),
    }
    return session.run(
        None, {value.name: values[value.name] for value in session.get_inputs()}
    )


def _apply(fixture: _Fixture, runner: str) -> None:
    if runner == "pairs":
        remove_redundant_transpose_pairs_ir(fixture.graph)
    elif runner == "add-forest":
        remove_redundant_transpose_add_forests_ir(fixture.graph)
    else:
        optimize_graph(
            fixture.model,
            precision_policy=(
                PrecisionPolicy.SOURCE if runner == "source" else PrecisionPolicy.LEGACY
            ),
        )


def _before(model: ir.Model) -> dict[bool, list[np.ndarray]]:
    return {mode: _run(model, mode) for mode in (False, True)}


def _parity(model: ir.Model, before: dict[bool, list[np.ndarray]]) -> None:
    for mode, expected in before.items():
        actual = _run(model, mode)
        for left, right in zip(actual, expected, strict=True):
            assert left.dtype == right.dtype and left.shape == right.shape
            assert left.tobytes() == right.tobytes()


@pytest.mark.parametrize(
    ("runner", "op", "observer"),
    [
        ("pairs", "Neg", "input-output"),
        ("pairs", "Neg", "input-shared"),
        ("pairs", "Neg", "input-if"),
        ("pairs", "Neg", "input-name-only-if"),
        ("pairs", "Neg", "input-loop"),
        ("pairs", "Neg", "intermediate-output"),
        ("pairs", "Sigmoid", "intermediate-output"),
        ("pairs", "Add", "intermediate-output"),
        ("add-forest", "Add", "intermediate-output"),
        ("add-forest", "Add", "intermediate-if"),
        ("source", "Neg", "input-output"),
        ("legacy", "Neg", "input-output"),
        ("source", "Neg", "input-if"),
        ("source", "Neg", "input-loop"),
        ("source", "Add", "intermediate-if"),
    ],
)
def test_observed_original_layout_values_remain_connected_and_unchanged(
    runner: str, op: str, observer: str
) -> None:
    fixture = _build(op, observer)
    before = _before(fixture.model)
    canonical = fixture.canonical
    intermediate = fixture.intermediate
    _apply(fixture, runner)
    onnx.checker.check_model(ir.to_proto(fixture.model), full_check=True)
    assert canonical.shape == ir.Shape(_CANONICAL)
    assert canonical.metadata_props["observer"] == "keep original NHWC"
    assert fixture.input_transposes[0].graph is fixture.graph
    if observer.startswith("intermediate"):
        assert intermediate.shape == ir.Shape(_CANONICAL)
        assert fixture.region[0].inputs[0] is canonical
    _parity(fixture.model, before)
    if op in {"Neg", "Add"}:
        expected = -_DATA if op == "Neg" else _DATA + np.flip(_DATA, axis=3)
        np.testing.assert_array_equal(_run(fixture.model, False)[0], expected)


@pytest.mark.parametrize(
    ("runner", "op", "scalar"),
    [
        ("pairs", "Neg", False),
        ("pairs", "Add", True),
        ("add-forest", "Add", False),
        ("source", "Add", False),
        ("legacy", "Neg", False),
    ],
)
def test_safe_isolated_elementwise_patterns_still_fold(
    runner: str, op: str, scalar: bool
) -> None:
    fixture = _build(op, scalar=scalar)
    before = _before(fixture.model)
    count = len(fixture.graph)
    _apply(fixture, runner)
    assert len(fixture.graph) < count
    assert not any(node.op_type == "Transpose" for node in fixture.graph)
    expected = (
        -_DATA
        if op == "Neg"
        else _DATA + (np.float32(2) if scalar else np.flip(_DATA, axis=3))
    )
    np.testing.assert_array_equal(_run(fixture.model, False)[0], expected)
    _parity(fixture.model, before)


@pytest.mark.parametrize("position", ["input", "intermediate"])
@pytest.mark.parametrize("ownership", ["removed-local", "removed-parent", "attached"])
def test_only_live_local_or_reachable_child_users_preserve_original_layout(
    position: str, ownership: str
) -> None:
    fixture = _build(
        observer=f"{position}-if" if ownership != "removed-local" else "none"
    )
    observed = fixture.canonical if position == "input" else fixture.intermediate
    if ownership == "removed-local":
        unused = ir.Node("", "Identity", [observed], outputs=[_v("unused", _CANONICAL)])
        fixture.graph.append(unused)
        fixture.graph.remove(unused)
        assert unused.graph is None and (unused, 0) in observed.uses()
    elif ownership == "removed-parent":
        parent = fixture.graph[-1]
        assert parent.op_type == "If"
        assert fixture.graph.outputs.pop() is parent.outputs[0]
        fixture.graph.remove(parent)
        assert parent.graph is None
        child_users = [
            user for user, _ in observed.uses() if user.op_type == "Identity"
        ]
        assert len(child_users) == 2
        assert all(
            user.graph is not None and user.graph is not fixture.graph
            for user in child_users
        )
    before = _before(fixture.model)
    _apply(fixture, "source")
    transposes = [node for node in fixture.graph if node.op_type == "Transpose"]
    if ownership == "attached":
        assert len(transposes) == (1 if position == "input" else 2)
        assert fixture.input_transposes[0].graph is fixture.graph
        assert observed.shape == ir.Shape(_CANONICAL)
    else:
        assert not transposes
        assert fixture.input_transposes[0].graph is None
    np.testing.assert_array_equal(_run(fixture.model, False)[0], -_DATA)
    _parity(fixture.model, before)


@pytest.mark.parametrize("shared", [False, True])
def test_direct_inverse_pairs_keep_observed_input_values(shared: bool) -> None:
    fixture = _build(observer="input-output")
    region = fixture.region[0]
    output_transpose = fixture.graph[2]
    output_transpose.replace_input_with(0, fixture.canonical)
    fixture.graph.remove(region)
    if shared:
        copied = _v("copied", _CANONICAL)
        fixture.graph.append(
            ir.Node("", "Identity", [fixture.canonical], outputs=[copied])
        )
        fixture.graph.outputs.append(copied)
    before = _before(fixture.model)
    remove_redundant_transpose_pairs_ir(fixture.graph)
    assert fixture.input_transposes[0].graph is fixture.graph
    assert output_transpose.graph is None
    _parity(fixture.model, before)


@pytest.mark.parametrize(
    ("runner", "op", "observer"),
    [
        ("pairs", "Neg", "output-name-only-if"),
        ("source", "Neg", "output-name-only-if"),
        ("legacy", "Neg", "output-name-only-if"),
        ("add-forest", "Add", "output-name-only-if"),
        ("pairs", "Add", "output-name-only-if"),
        ("source", "Add", "output-name-only-if"),
        ("legacy", "Add", "output-name-only-if"),
        ("source", "Neg", "output-if"),
    ],
)
def test_inverse_output_captures_are_checked_before_any_layout_mutation(
    runner: str, op: str, observer: str
) -> None:
    fixture = _build(op, observer)
    observed = fixture.graph.outputs[0]
    output_transpose = observed.producer()
    assert output_transpose is not None
    if observer == "output-name-only-if":
        assert not observed.uses()
    before = _before(fixture.model)
    expected = -_DATA if op == "Neg" else _DATA + np.flip(_DATA, axis=3)
    for values in before.values():
        for value in values:
            np.testing.assert_array_equal(value, expected)
    _apply(fixture, runner)
    assert output_transpose.graph is fixture.graph
    assert fixture.region[0].inputs[0] is fixture.canonical
    assert fixture.intermediate.shape == ir.Shape(_CANONICAL)
    _parity(fixture.model, before)


@pytest.mark.parametrize("shared", [False, True])
def test_direct_inverse_output_name_capture_survives_single_and_multi_use_paths(
    shared: bool,
) -> None:
    fixture = _build(observer="output-name-only-if")
    output_transpose = fixture.graph[2]
    output_transpose.replace_input_with(0, fixture.canonical)
    fixture.graph.remove(fixture.region[0])
    if shared:
        copied = _v("copied", _CANONICAL)
        fixture.graph.append(
            ir.Node("", "Identity", [fixture.canonical], outputs=[copied])
        )
        fixture.graph.outputs.append(copied)
    assert not output_transpose.outputs[0].uses()
    before = _before(fixture.model)
    np.testing.assert_array_equal(before[False][0], _DATA)
    np.testing.assert_array_equal(before[False][1], _DATA)
    remove_redundant_transpose_pairs_ir(fixture.graph)
    assert output_transpose.graph is fixture.graph
    assert fixture.input_transposes[0].graph is fixture.graph
    _parity(fixture.model, before)


@pytest.mark.parametrize("scope", ["function", "child"])
def test_scope_ownership_and_nonrecursive_processing(scope: str) -> None:
    fixture = _build(observer="input-output")
    graph = fixture.graph
    if scope == "function":
        function = ir.Function("local", "Observed", graph=graph, attributes=())
        x = _v("in_0", _SHAPE)
        outs = [_v("out", _SHAPE), _v("canonical_out", _CANONICAL)]
        root = ir.Graph(
            [x],
            outs,
            nodes=[ir.Node("local", "Observed", [x], outputs=outs)],
            opset_imports={"": 23, "local": 1},
            name="function_root",
        )
        model = ir.Model(root, ir_version=10, functions=[function])
        before = _before(model)
        optimize_graph(model, precision_policy=PrecisionPolicy.SOURCE)
        assert list(function.graph.inputs) == [graph.inputs[0]]
    else:
        choose = _v("choose", (), ir.DataType.BOOL)
        # If branches capture graph inputs lexically, with no formal parameters.
        source = graph.inputs.pop()
        root_input = _v("in_0", _SHAPE)
        graph[0].replace_input_with(0, root_input)
        other = _build(observer="input-output").graph
        other.inputs.pop()
        other[0].replace_input_with(0, root_input)
        outs = [_v("out", _SHAPE), _v("canonical_out", _CANONICAL)]
        root = ir.Graph(
            [root_input, choose],
            outs,
            nodes=[
                ir.Node(
                    "",
                    "If",
                    [choose],
                    outputs=outs,
                    attributes=[
                        ir.AttrGraph("then_branch", graph),
                        ir.AttrGraph("else_branch", other),
                    ],
                )
            ],
            opset_imports={"": 23},
            name="child_root",
        )
        model = ir.Model(root, ir_version=10)
        before = _before(model)
        # This graph pass stays nonrecursive; explicit child invocation is safe.
        remove_redundant_transpose_pairs_ir(root)
        assert fixture.region[0].inputs[0] is fixture.canonical
        remove_redundant_transpose_pairs_ir(graph)
        assert all(node.graph is other for node in list(other))
        assert source.name == root_input.name
    assert fixture.input_transposes[0].graph is graph
    _parity(model, before)


@pytest.mark.parametrize("custom", ["Transpose", "Neg"])
def test_custom_domain_names_do_not_supply_standard_layout_proofs(custom: str) -> None:
    fixture = _build()
    if custom == "Neg":
        formal = _v("value", _CANONICAL)
        result = _v("result", _CANONICAL)
        body = ir.Graph(
            [formal],
            [result],
            nodes=[ir.Node("", "Neg", [formal], outputs=[result])],
            opset_imports={"": 23},
        )
        target = fixture.region[0]
        attributes = ()
    else:
        formal = _v("value", _SHAPE)
        transpose = _transpose(formal, "result", _P)
        body = ir.Graph(
            [formal], list(transpose.outputs), nodes=[transpose], opset_imports={"": 23}
        )
        target = fixture.input_transposes[0]
        attributes = (ir.AttrInt64s("perm", _P),)
    target.domain = "custom"
    function = ir.Function("custom", custom, graph=body, attributes=attributes)
    fixture.model = ir.Model(fixture.graph, ir_version=10, functions=[function])
    fixture.model.opset_imports["custom"] = 1
    before = _before(fixture.model)
    original = tuple(fixture.graph)
    remove_redundant_transpose_pairs_ir(fixture.graph)
    assert tuple(fixture.graph) == original
    _parity(fixture.model, before)


def test_orphan_cleanup_still_removes_a_genuinely_unused_transpose() -> None:
    from jax2onnx.converter.ir_optimizations import remove_orphan_transposes_ir

    value = _v("in_0", _SHAPE)
    unused = _transpose(value, "unused", _P)
    graph = ir.Graph(
        [value],
        [value],
        nodes=[unused],
        name="unused_transpose",
        opset_imports={"": 23},
    )
    model = ir.Model(graph, ir_version=10)
    before = _before(model)
    remove_orphan_transposes_ir(graph)
    assert unused.graph is None and len(graph) == 0
    _parity(model, before)


@pytest.mark.parametrize("observer", ["input-name-only-if", "output-name-only-if"])
def test_native_naming_preserves_valid_lexical_alias_objects_and_metadata(
    observer: str,
) -> None:
    fixture = _build(observer=observer)
    aliases = [
        attr.as_graph()[0].inputs[0]
        for node in fixture.graph
        if node.op_type == "If"
        for attr in node.attributes.values()
    ]
    definition = (
        fixture.canonical if observer.startswith("input") else fixture.graph.outputs[0]
    )
    for alias in aliases:
        assert alias is not None and alias is not definition
        alias.metadata_props["lexical"] = "preserve"
    before = _before(fixture.model)
    shapes = [alias.shape for alias in aliases]
    types = [alias.type for alias in aliases]
    optimizer._run_name_fix_pass(fixture.model)
    for index, alias in enumerate(aliases):
        assert alias.name == definition.name and alias.producer() is None
        assert alias.shape is shapes[index] and alias.type is types[index]
        assert alias.metadata_props["lexical"] == "preserve"
    _parity(fixture.model, before)


def test_lexical_alias_uses_final_ancestor_name_after_injected_naming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Injected final rename checks the invariant, not native naming behavior."""
    fixture = _build(observer="output-name-only-if")
    definition = fixture.graph.outputs[0]
    before = _before(fixture.model)
    native = optimizer.common_passes.NameFixPass
    calls = []

    def naming(model: ir.Model) -> None:
        calls.append(model)
        native()(model)
        definition.name = "final_ancestor_name"

    monkeypatch.setattr(optimizer.common_passes, "NameFixPass", lambda: naming)
    optimizer._run_name_fix_pass(fixture.model)
    assert calls == [fixture.model]
    for node in fixture.graph:
        if node.op_type == "If":
            for attr in node.attributes.values():
                assert attr.as_graph()[0].inputs[0].name == "final_ancestor_name"
    _parity(fixture.model, before)


def _naming_scope(
    references: list[ir.Value],
    *,
    inputs: tuple[ir.Value, ...] = (),
    initializers: tuple[ir.Value, ...] = (),
    nodes: tuple[ir.Node, ...] = (),
) -> ir.Graph:
    """Collection-only scope fixtures need no executable envelope operator."""
    body = list(nodes)
    for index, reference in enumerate(references):
        body.append(
            ir.Node("", "Identity", [reference], outputs=[_v(f"out_{index}", _SHAPE)])
        )
    return ir.Graph(inputs, [], nodes=body, initializers=initializers)


def _naming_envelope(
    children: list[ir.Graph], outputs: tuple[ir.Value, ...] = ()
) -> ir.Node:
    return ir.Node(
        "naming_fixture",
        "Envelope",
        [],
        outputs=outputs,
        attributes=[ir.AttrGraphs("bodies", children)],
    )


@pytest.mark.parametrize("kind", ["formal", "initializer", "local"])
def test_name_snapshot_collection_respects_shadows_siblings_and_nested_graphs(
    kind: str,
) -> None:
    """Collection controls cover GRAPHS and shadowing, not operator execution."""
    ancestor = _v("shared", _SHAPE)
    genuine = _v("shared", _SHAPE)
    local_alias = _v("shared", _SHAPE)
    deep = _naming_scope([genuine])
    first = _naming_scope([], nodes=(_naming_envelope([deep]),))
    if kind == "formal":
        second = _naming_scope([local_alias], inputs=(_v("shared", _SHAPE),))
    elif kind == "initializer":
        second = _naming_scope(
            [local_alias], initializers=(_constant("shared", _DATA),)
        )
    else:
        local = ir.Node("", "Neg", [ancestor], outputs=[_v("shared", _SHAPE)])
        second = _naming_scope([local_alias], nodes=(local,))
    graph = _naming_scope(
        [], inputs=(ancestor,), nodes=(_naming_envelope([first, second]),)
    )
    references = {}
    optimizer._snapshot_lexical_name_references(graph, {}, references, set())
    assert references == {genuine: ancestor}
    assert genuine is not local_alias and local_alias.name == "shared"


@pytest.mark.parametrize("definition", ["enclosing", "later"])
def test_name_snapshot_collection_does_not_bind_invisible_parent_results(
    definition: str,
) -> None:
    alias = _v("future", _SHAPE)
    child = _naming_scope([alias])
    future = _v("future", _SHAPE)
    envelope = _naming_envelope([child], (future,) if definition == "enclosing" else ())
    nodes = [envelope]
    if definition == "later":
        nodes.append(ir.Node("", "Identity", [_v("source", _SHAPE)], outputs=[future]))
    graph = ir.Graph([], [], nodes=nodes)
    references = {}
    optimizer._snapshot_lexical_name_references(graph, {}, references, set())
    assert not references and alias.name == "future"


def test_native_function_naming_preserves_its_own_nested_lexical_capture() -> None:
    fixture = _build(observer="output-name-only-if")
    function = ir.Function("local", "Captured", graph=fixture.graph, attributes=())
    inputs = [_v("in_0", _SHAPE), _v("choose", (), ir.DataType.BOOL)]
    outputs = [_v("result", _SHAPE), _v("nested", _SHAPE)]
    graph = ir.Graph(
        inputs,
        outputs,
        nodes=[ir.Node("local", "Captured", inputs, outputs=outputs)],
        opset_imports={"": 23, "local": 1},
        name="function_lexical_root",
    )
    model = ir.Model(graph, ir_version=10, functions=[function])
    before = _before(model)
    optimizer._run_name_fix_pass(model)
    local_definition = fixture.graph.outputs[0]
    assert local_definition is not outputs[0]
    for node in fixture.graph:
        if node.op_type == "If":
            for attr in node.attributes.values():
                alias = attr.as_graph()[0].inputs[0]
                assert alias.name == local_definition.name and alias is not outputs[0]
    _parity(model, before)


def test_name_snapshot_collection_keeps_function_roots_independent() -> None:
    """An unbound Function reference is a negative collection control."""
    root_definition = _v("model_only", _SHAPE)
    alias = _v("model_only", _SHAPE)
    function_graph = _naming_scope(
        [], nodes=(_naming_envelope([_naming_scope([alias])]),)
    )
    references = {}
    optimizer._snapshot_lexical_name_references(
        _naming_scope([], inputs=(root_definition,)), {}, references, set()
    )
    optimizer._snapshot_lexical_name_references(function_graph, {}, references, set())
    assert not references


@pytest.mark.parametrize("conflict", ["ancestor", "local"])
@pytest.mark.parametrize("reverse", [False, True])
def test_ambiguous_shared_lexical_reference_fails_before_native_naming(
    conflict: str, reverse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    ancestor = _v("shared", _SHAPE)
    alias = _v("shared", _SHAPE)
    first = _naming_scope([alias])
    if conflict == "ancestor":
        second = _naming_scope(
            [],
            inputs=(_v("shared", _SHAPE),),
            nodes=(_naming_envelope([_naming_scope([alias])]),),
        )
    else:
        second = _naming_scope([alias], inputs=(alias,))
    children = [second, first] if reverse else [first, second]
    graph = _naming_scope([], inputs=(ancestor,), nodes=(_naming_envelope(children),))
    model = ir.Model(graph, ir_version=10)
    original = tuple(graph)
    calls = []
    monkeypatch.setattr(
        optimizer.common_passes, "NameFixPass", lambda: calls.append("unexpected")
    )
    with pytest.raises(ValueError, match="Ambiguous nested lexical reference 'shared'"):
        optimizer._run_name_fix_pass(model)
    assert not calls and tuple(graph) == original
    assert alias.name == ancestor.name == "shared"
