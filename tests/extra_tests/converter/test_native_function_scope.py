# tests/extra_tests/converter/test_native_function_scope.py

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import onnx_ir as ir
from onnx_ir.tape import Tape
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx import user_interface as ui
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    validate_source_samples,
)
from jax2onnx.converter.conversion_api import _apply_late_ir_attr_overrides
from jax2onnx.converter.function_scope import (
    FunctionKey,
    FunctionRegistry,
    FunctionScope,
)
from jax2onnx.converter.ir_context import IRContext


def test_function_finalization_owns_its_body_and_preserves_metadata():
    parent = IRContext(opset=21, enable_double_precision=False)
    parent_input = ir.Value(
        name="x", type=ir.TensorType(ir.DataType.FLOAT16), shape=ir.Shape(("B", 3))
    )
    parent.builder.inputs.append(parent_input)
    scope = FunctionScope(parent, name="Activation", domain="custom.Activation.1")
    formal = scope.begin([parent_input])[0]
    formal.metadata_props["formal"] = "preserved"
    staging_graph = scope.ctx.builder.graph
    staging_graph.doc_string = "Function body documentation"
    staging_graph.metadata_props["body"] = "preserved"
    output = ir.Value(name="activated", type=formal.type, shape=formal.shape)
    output.metadata_props["result"] = "preserved"
    scope.ctx.builder.op(
        "LeakyRelu",
        [formal],
        {"alpha": 0.2},
        name="activation",
        doc_string="Node documentation",
        metadata_props={"node": "preserved"},
        output=output,
    )
    scope.ctx.add_node_attr_override("activation", {"alpha": 0.4})

    function = scope.end([output])

    assert function.graph is not staging_graph
    assert function.inputs[0] is not formal
    assert function.inputs[0].graph is function.graph
    assert function.outputs[0] is not output
    assert function.outputs[0].graph is function.graph
    assert formal.graph is staging_graph and output.graph is staging_graph
    assert parent_input.graph is parent.builder.graph
    assert not tuple(parent.builder.graph)
    assert function.inputs[0].dtype == function.outputs[0].dtype == ir.DataType.FLOAT16
    assert function.inputs[0].shape == function.outputs[0].shape == ir.Shape(("B", 3))
    assert function.inputs[0].metadata_props["formal"] == "preserved"
    assert function.outputs[0].metadata_props["result"] == "preserved"
    assert function.graph.doc_string == "Function body documentation"
    assert function.graph.metadata_props["body"] == "preserved"
    (node,) = tuple(function.graph)
    assert node.graph is function.graph and node.inputs[0] is function.inputs[0]
    assert node.outputs[0] is function.outputs[0]
    assert node.doc_string == "Node documentation"
    assert node.metadata_props["node"] == "preserved"
    assert function.opset_imports == {"": 21, "custom.Activation.1": 21}

    # Late overrides apply to the finalized function, without mutating staging IR.
    model = ir.Model(parent.builder.graph, ir_version=10)
    model.functions[function.identifier()] = function
    _apply_late_ir_attr_overrides(model, parent)
    assert node.attributes["alpha"].as_float() == pytest.approx(0.4)
    assert tuple(staging_graph)[0].attributes["alpha"].as_float() == pytest.approx(0.2)
    staging_graph.metadata_props["body"] = "changed after finalization"
    assert function.graph.metadata_props["body"] == "preserved"


def test_function_scope_finalizes_once_and_registry_retains_the_native_definition():
    parent = IRContext(opset=21, enable_double_precision=False)
    scope = FunctionScope(parent, name="Identity", domain="custom.Identity.1")
    with pytest.raises(RuntimeError, match="begin.*before end"):
        scope.end([])
    formal = scope.begin([ir.Value(name="x")])[0]
    function = scope.end([formal])
    with pytest.raises(RuntimeError, match="already sealed"):
        scope.end([formal])
    with pytest.raises(RuntimeError, match="only be called once"):
        scope.begin([])

    key = FunctionKey("Identity", (), ())
    registry = FunctionRegistry()
    registry.put(key, function)
    assert registry.get(key) is function
    assert registry.all() == [function]


@onnx_function(unique=True)
def _shared_native_definition(x):
    return jnp.square(x) + 1.0


def _call_shared_native_definition(x):
    return _shared_native_definition(x), _shared_native_definition(x + 1.0)


def test_repeated_calls_publish_the_same_native_definition_stored_in_the_registry(
    monkeypatch,
):
    definitions = []
    original_put = FunctionRegistry.put

    def record_definition(registry, key, function):
        definitions.append(function)
        original_put(registry, key, function)

    monkeypatch.setattr(FunctionRegistry, "put", record_definition)
    model = to_onnx(
        _call_shared_native_definition,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        return_mode="ir",
    )

    assert len(definitions) == 1
    (function,) = definitions
    assert isinstance(function, ir.Function)
    assert model.functions[function.identifier()] is function
    calls = [node for node in model.graph if node.domain == function.domain]
    assert len(calls) == 2
    assert all(node.op_type == function.name for node in calls)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def _scope_definitions(graph):
    """Return definitions once by identity, including empty nested graphs."""
    seen = set()
    for owner in (graph, *graph.subgraphs()):
        for value in (
            *owner.inputs,
            *owner.initializers.values(),
            *(output for node in owner for output in node.outputs),
        ):
            if id(value) not in seen:
                seen.add(id(value))
                yield value


def _scope_identity(builder, value, *, name=None):
    return builder.op(
        "Identity",
        [value],
        output=ir.Value(name=name, type=value.type, shape=value.shape),
    )


def test_function_namespace_covers_automatic_nested_values_without_mutating_captures():
    parent = IRContext(opset=23, enable_double_precision=False)
    features = ir.Value(
        name="features", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2,))
    )
    flag = ir.Value(
        name="flag", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape(())
    )
    parent.builder.inputs.extend([features, flag])
    payload = np.asarray([0x80000000, 0x7FC01234], np.uint32).view(np.float32)
    borrowed = parent.builder.add_initializer_from_array(name="borrowed", array=payload)
    scope = FunctionScope(parent, "OwnsAutomaticValues", "test.owned.1")
    formal, condition = scope.begin([features, flag])
    trip_count = scope.ctx.builder.add_initializer_from_scalar(
        name="trip_count", value=np.int64(1)
    )
    iteration = ir.Value(
        name="iteration",
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(()),
        const_value=trip_count.const_value,
    )
    body_condition = ir.Value(
        name="condition", type=condition.type, shape=condition.shape
    )
    carry = ir.Value(name="carry", type=formal.type, shape=formal.shape)
    alias = ir.Value(
        name="borrowed_alias",
        type=borrowed.type,
        shape=borrowed.shape,
        const_value=borrowed.const_value,
    )
    body = ir.Graph(
        inputs=[iteration, body_condition, carry],
        outputs=[],
        nodes=[],
        initializers=[iteration, alias],
        name="body",
        opset_imports={"": 23},
    )
    tape = Tape(body)
    current = carry
    for _ in range(555):
        current = tape.op(
            "Identity", [current], output=ir.Value(type=carry.type, shape=carry.shape)
        )
    captured = tape.op(
        "Identity", [borrowed], output=ir.Value(type=carry.type, shape=carry.shape)
    )
    body.outputs.extend([body_condition, current, captured])
    outputs = scope.ctx.builder.op_multi_out(
        "Loop",
        [trip_count, condition, formal],
        {"body": ir.AttrGraph("body", body)},
        outputs=[
            ir.Value(type=formal.type, shape=formal.shape),
            ir.Value(type=formal.type, shape=ir.Shape((1, 2))),
        ],
    )
    before = list(_scope_definitions(scope.ctx.builder.graph))
    assert (
        sum(bool(value.name and value.name.startswith("val_")) for value in before)
        >= 555
    )
    borrowed_tensor = borrowed.const_value
    trip_tensor = trip_count.const_value
    assert borrowed_tensor is not None and trip_tensor is not None
    tensor_names = (borrowed_tensor.name, trip_tensor.name)
    parent_names = (features.name, flag.name, borrowed.name)
    old_nodes = [(node.op_type, dict(node.attributes)) for node in body]

    function = scope.end(list(outputs))

    assert [value.name for value in function.inputs] == ["f_in_0", "f_in_1"]
    local = list(_scope_definitions(function.graph))[2:]
    local_names = [value.name for value in local]
    assert all(name and "/local_" in name for name in local_names)
    assert len(local_names) == len(set(local_names))
    assert not set(local_names).intersection(parent_names)
    assert (features.name, flag.name, borrowed.name) == parent_names
    assert (borrowed_tensor.name, trip_tensor.name) == tensor_names
    assert borrowed.const_value is borrowed_tensor
    assert borrowed_tensor.tobytes() == payload.tobytes()
    cloned_body = next(function.graph.subgraphs())
    assert [(node.op_type, dict(node.attributes)) for node in cloned_body] == old_nodes
    assert tuple(cloned_body)[-1].inputs[0] is borrowed
    for owner in (function.graph, *function.graph.subgraphs()):
        assert all(value.graph is owner for value in owner.inputs)
        assert all(node.graph is owner for node in owner)
        assert all(value.graph is owner for node in owner for value in node.outputs)
        assert all(name == value.name for name, value in owner.initializers.items())
    assert len(cloned_body.initializers) == 2
    assert cloned_body.inputs[0] is next(
        value
        for value in cloned_body.initializers.values()
        if value.dtype == ir.DataType.INT64
    )
    cloned_alias = next(
        value
        for value in cloned_body.initializers.values()
        if value.dtype == ir.DataType.FLOAT
    )
    assert cloned_alias.const_value is not borrowed_tensor
    assert cloned_alias.const_value.tobytes() == borrowed_tensor.tobytes()
    assert all(
        a is not b
        for a, b in zip(before, _scope_definitions(function.graph), strict=True)
    )


def test_function_namespace_distinguishes_domains_nested_scopes_and_ambiguous_pairs():
    parent = IRContext(opset=23, enable_double_precision=False)
    x = ir.Value(
        name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))
    )
    parent.builder.inputs.append(x)
    pairs = [("a.b", "Same"), ("a_b", "Same"), ("a", "b_Same"), ("a_b", "Same")]
    scopes = [FunctionScope(parent, name, domain) for domain, name in pairs]
    functions = []
    prefixes = []
    for scope in scopes:
        formal = scope.begin([x])[0]
        context_name = scope.ctx.fresh_name("context_probe")
        builder_name = scope.ctx.builder.fresh_name("builder_probe")
        assert context_name.rsplit("/", 1)[0] == builder_name.rsplit("/", 1)[0]
        prefixes.append(context_name.rsplit("/", 1)[0])
        output = _scope_identity(scope.ctx.builder, formal)
        functions.append(scope.end([output]))
    outer = FunctionScope(parent, "Outer", "nested.1")
    outer_formal = outer.begin([x])[0]
    inner = FunctionScope(outer.ctx, "Same", "a.b")
    inner_formal = inner.begin([outer_formal])[0]
    inner_function = inner.end([_scope_identity(inner.ctx.builder, inner_formal)])
    called = outer.ctx.builder.op(
        inner_function.name,
        [outer_formal],
        domain=inner_function.domain,
        output=ir.Value(type=outer_formal.type, shape=outer_formal.shape),
    )
    functions.extend([inner_function, outer.end([called])])
    names = []
    for function in functions:
        assert [value.name for value in function.inputs] == ["f_in_0"]
        definitions = list(_scope_definitions(function.graph))[1:]
        names.extend(value.name for value in definitions)
        assert all(value.graph is function.graph for value in definitions)
    assert len(prefixes) == len(set(prefixes))
    assert len(names) == len(set(names))
    outer_call = tuple(functions[-1].graph)[0]
    assert (outer_call.domain, outer_call.op_type) == (
        inner_function.domain,
        inner_function.name,
    )
    assert outer_call.inputs[0] is functions[-1].inputs[0]
    assert inner_function.inputs[0] is not outer_formal
    assert inner_function.inputs[0].graph is inner_function.graph


def test_function_namespace_retries_occupied_names_without_mutating_externals():
    parent = IRContext(opset=23, enable_double_precision=False)
    scope = FunctionScope(parent, "Collision", "test.collision")
    prefix = scope.ctx.fresh_name("namespace_probe").rsplit("/", 1)[0]
    occupied = [f"{prefix}/local_0_0", f"{prefix}/local_1_0"]
    parent_value = ir.Value(
        name=occupied[0], type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2,))
    )
    parent.builder.inputs.append(parent_value)
    payload = np.asarray([0x80000000, 0x7FC01234], np.uint32).view(np.float32)
    parent_constant = parent.builder.add_initializer_from_array(
        name=occupied[1], array=payload
    )
    tensor = parent_constant.const_value
    assert tensor is not None
    tensor_name = tensor.name
    formals = scope.begin([parent_value, parent_constant])
    output = _scope_identity(scope.ctx.builder, formals[0])

    function = scope.end([output])

    assert function.outputs[0].name == f"{prefix}/local_2_0"
    assert function.outputs[0].name not in occupied
    assert [value.name for value in function.inputs] == ["f_in_0", "f_in_1"]
    assert parent.builder.inputs[0] is parent_value
    assert parent.builder.graph.initializers[occupied[1]] is parent_constant
    assert [parent_value.name, parent_constant.name] == occupied
    assert parent_constant.const_value is tensor
    assert tensor.name == tensor_name
    assert tensor.tobytes() == payload.tobytes()


def test_function_namespace_detaches_shared_proto_tensor_name_storage():
    parent = IRContext(opset=23, enable_double_precision=False)
    payload = np.asarray([0x80000000, 0x7FC01234], np.uint32).view(np.float32)
    proto = onnx.numpy_helper.from_array(payload, name="borrowed_proto")
    proto.doc_string = "Keep the backing proto documentation"
    entry = proto.metadata_props.add()
    entry.key, entry.value = "origin", "parent tensor"
    tensor = ir.TensorProtoTensor(proto)
    tensor.metadata_props["runtime"] = "preserved"
    tensor.meta["pass_note"] = "preserved"
    parent_value = ir.Value(
        name="borrowed_proto",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape((2,)),
        const_value=tensor,
    )
    parent.builder.graph.initializers.add(parent_value)
    scope = FunctionScope(parent, "ProtoTensor", "test.proto_tensor")
    scope.begin([])
    alias = ir.Value(
        name="local_alias",
        type=parent_value.type,
        shape=parent_value.shape,
        const_value=tensor,
    )
    scope.ctx.builder.op(
        "Constant",
        [],
        {"value": ir.AttrTensor("value", tensor)},
        output=alias,
    )
    before = proto.SerializeToString()
    tensor_name = tensor.name
    tensor_metadata = dict(tensor.metadata_props)
    tensor_meta = dict(tensor.meta)

    function = scope.end([alias])

    assert parent.builder.graph.initializers["borrowed_proto"] is parent_value
    assert parent_value.name == "borrowed_proto"
    assert parent_value.const_value is tensor and tensor.raw is proto
    assert tensor.name == tensor_name
    assert proto.SerializeToString() == before
    assert dict(tensor.metadata_props) == tensor_metadata
    assert dict(tensor.meta) == tensor_meta
    detached = function.outputs[0].const_value
    assert isinstance(detached, ir.TensorProtoTensor)
    assert detached is not tensor and detached.raw is not proto
    assert detached.name == function.outputs[0].name
    assert detached.dtype == tensor.dtype and detached.shape == tensor.shape
    assert detached.tobytes() == tensor.tobytes() == payload.tobytes()
    assert detached.doc_string == tensor.doc_string
    assert dict(detached.metadata_props) == tensor_metadata
    assert dict(detached.meta) == tensor_meta
    assert tuple(function.graph)[0].attributes["value"].as_tensor() is tensor
    assert function.outputs[0].graph is function.graph
    assert alias.graph is scope.ctx.builder.graph


def test_function_nested_count_and_key_wires_keep_distinct_physical_signatures():
    parent = IRContext(opset=23, enable_double_precision=False)
    condition = ir.Value(
        name="gate", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape(())
    )
    count = ir.Value(
        name="count", type=ir.TensorType(ir.DataType.UINT32), shape=ir.Shape(())
    )
    key = ir.Value(
        name="key", type=ir.TensorType(ir.DataType.UINT32), shape=ir.Shape((2,))
    )
    parent.builder.inputs.extend([condition, count, key])
    caller = parent.create_child_context(scope="control_flow", prefix="cond_then")
    collision = "cond_then_0/cond_capture_2"
    caller_count = _scope_identity(caller.builder, count, name=collision)
    caller_key = _scope_identity(caller.builder, key, name="caller_key")
    scope = FunctionScope(caller, "CountKey", "test.count_key.1")
    formal_condition, formal_count, formal_key = scope.begin(
        [condition, caller_count, caller_key]
    )
    branches = []
    for prefix in ("inner_then", "inner_else"):
        branch = scope.ctx.create_child_context(scope="control_flow", prefix=prefix)
        local_key = _scope_identity(branch.builder, formal_key, name=collision)
        local_count = _scope_identity(branch.builder, formal_count, name="inner_count")
        branch.builder.outputs.extend([local_count, local_key])
        branches.append(branch.builder.graph)
    outputs = scope.ctx.builder.op_multi_out(
        "If",
        [formal_condition],
        {
            "then_branch": ir.AttrGraph("then_branch", branches[0]),
            "else_branch": ir.AttrGraph("else_branch", branches[1]),
        },
        outputs=[
            ir.Value(type=count.type, shape=count.shape),
            ir.Value(type=key.type, shape=key.shape),
        ],
    )
    function = scope.end(list(outputs))
    call_outputs = caller.builder.op_multi_out(
        function.name,
        [condition, caller_count, caller_key],
        domain=function.domain,
        outputs=[
            ir.Value(type=count.type, shape=count.shape),
            ir.Value(type=key.type, shape=key.shape),
        ],
    )
    caller.builder.outputs.extend(call_outputs)
    otherwise = parent.create_child_context(scope="control_flow", prefix="otherwise")
    otherwise.builder.outputs.extend(
        [
            _scope_identity(otherwise.builder, count),
            _scope_identity(otherwise.builder, key),
        ]
    )
    final_outputs = parent.builder.op_multi_out(
        "If",
        [condition],
        {
            "then_branch": ir.AttrGraph("then_branch", caller.builder.graph),
            "else_branch": ir.AttrGraph("else_branch", otherwise.builder.graph),
        },
        outputs=[
            ir.Value(type=count.type, shape=count.shape),
            ir.Value(type=key.type, shape=key.shape),
        ],
    )
    parent.builder.outputs.extend(final_outputs)
    model = ir.Model(parent.builder.graph, ir_version=10)
    model.functions[function.identifier()] = function
    parent.builder.graph.opset_imports[function.domain] = 23
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    assert list(proto.functions[0].input) == ["f_in_0", "f_in_1", "f_in_2"]
    outer_if = next(node for node in proto.graph.node if node.op_type == "If")
    caller_proto = next(
        attr.g for attr in outer_if.attribute if attr.name == "then_branch"
    )
    call = next(node for node in caller_proto.node if node.domain == function.domain)
    assert list(call.input) == ["gate", collision, "caller_key"]
    assert caller_proto.node[0].input == ["count"]
    assert caller_proto.node[0].output == [collision]
    assert caller_count.dtype == ir.DataType.UINT32 and caller_count.shape == ir.Shape(
        ()
    )
    inner_if = next(node for node in proto.functions[0].node if node.op_type == "If")
    for attr in inner_if.attribute:
        if attr.type != onnx.AttributeProto.GRAPH:
            continue
        assert not attr.g.input
        identities = [node for node in attr.g.node if node.op_type == "Identity"]
        assert [list(node.input) for node in identities] == [["f_in_2"], ["f_in_1"]]
        assert collision not in {name for node in identities for name in node.output}
        assert list(attr.g.output[0].type.tensor_type.shape.dim) == []
        assert [
            dim.dim_value for dim in attr.g.output[1].type.tensor_type.shape.dim
        ] == [2]
        assert all(
            value.type.tensor_type.elem_type == onnx.TensorProto.UINT32
            for value in attr.g.output
        )
        assert [value.name for value in attr.g.output] == [
            identities[1].output[0],
            identities[0].output[0],
        ]
    assert model.functions[function.identifier()] is function


@onnx_function(name="ScopeSentinelSelect")
def _scope_sentinel_select(x, sentinel, inner):
    return jax.lax.cond(
        inner,
        lambda operands: jax.lax.add(*operands),
        lambda operands: jax.lax.sub(*operands),
        (x, sentinel),
    )


def _scope_sentinel_target(x, left, right, outer, inner):
    return jax.lax.cond(
        outer,
        lambda _: _scope_sentinel_select(x, left, inner),
        lambda _: _scope_sentinel_select(x, right, inner),
        operand=None,
    )


def _serialized_scope_names(nodes, initializers=()):
    definitions = [tensor.name for tensor in initializers]
    definitions.extend(name for node in nodes for name in node.output if name)
    assert len(definitions) == len(set(definitions))
    names = set(definitions)
    for node in nodes:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                child_names = _serialized_scope_names(attr.g.node, attr.g.initializer)
                assert names.isdisjoint(child_names)
                names.update(child_names)
    return names


def _identity_origins(nodes, initial):
    origins = dict(initial)
    for node in nodes:
        if node.op_type in {"Identity", "Cast"} and not node.domain:
            assert len(node.input) == len(node.output) == 1
            if node.input[0] in origins:
                origins[node.output[0]] = origins[node.input[0]]
    return origins


def _assert_sentinel_serialized_wires(model):
    functions = {(fn.domain, fn.name): fn for fn in model.functions}
    selected = [fn for fn in model.functions if fn.name == "ScopeSentinelSelect"]
    assert selected
    root_names = _serialized_scope_names(model.graph.node, model.graph.initializer)
    all_local_names = set()
    for function in selected:
        assert list(function.input) == ["f_in_0", "f_in_1", "f_in_2"]
        names = _serialized_scope_names(function.node)
        assert names and all("/local_" in name for name in names)
        assert names.isdisjoint(root_names | all_local_names | set(function.input))
        all_local_names.update(names)
        inner_if = next(node for node in function.node if node.op_type == "If")
        for attr, operator in (("then_branch", "Add"), ("else_branch", "Sub")):
            branch = next(value.g for value in inner_if.attribute if value.name == attr)
            assert not branch.input
            origins = _identity_origins(
                branch.node, {name: name for name in function.input}
            )
            (arithmetic,) = [node for node in branch.node if node.op_type == operator]
            assert [origins[name] for name in arithmetic.input] == ["f_in_0", "f_in_1"]
            assert len(branch.output) == 1
            tail_origins = _identity_origins(
                branch.node,
                {**origins, arithmetic.output[0]: "arithmetic"},
            )
            assert tail_origins[branch.output[0].name] == "arithmetic"
    outer_if = next(node for node in model.graph.node if node.op_type == "If")
    for attr, sentinel in (("then_branch", "left"), ("else_branch", "right")):
        branch = next(value.g for value in outer_if.attribute if value.name == attr)
        assert not branch.input
        origins = _identity_origins(
            branch.node,
            {name: name for name in ("features", "left", "right", "outer", "inner")},
        )
        (call,) = [
            node for node in branch.node if (node.domain, node.op_type) in functions
        ]
        assert [origins[name] for name in call.input] == ["features", sentinel, "inner"]
        assert functions[(call.domain, call.op_type)] in selected


@pytest.mark.parametrize(
    "policy,x64",
    [
        (PrecisionPolicy.SOURCE, False),
        (PrecisionPolicy.SOURCE, True),
        (PrecisionPolicy.LEGACY, False),
    ],
    ids=["source-f32-default", "source-f32-x64", "legacy-f32-default"],
)
def test_if_function_if_same_shape_captures_keep_distinct_sentinel_values(
    policy, x64, tmp_path
):
    with jax.enable_x64(x64):
        assert active_precision_policy() is PrecisionPolicy.LEGACY
        references = []
        evidence = {}
        compiled = jax.jit(_scope_sentinel_target)
        for offset in (0, 64):
            features = np.arange(6, dtype=np.float32).reshape(2, 3) + np.float32(offset)
            left = np.full((2, 3), 8, np.float32)
            right = np.full((2, 3), 32, np.float32)
            assert left.shape == right.shape == features.shape
            assert left.dtype == right.dtype == features.dtype
            assert left.tobytes() != right.tobytes()
            family_outputs = []
            for outer in (False, True):
                for inner in (False, True):
                    values = (
                        features,
                        left,
                        right,
                        np.asarray(outer),
                        np.asarray(inner),
                    )
                    eager = np.asarray(_scope_sentinel_target(*values))
                    jitted = np.asarray(compiled(*values))
                    expected = features + (left if outer else right) * np.float32(
                        1 if inner else -1
                    )
                    assert (
                        eager.dtype
                        == jitted.dtype
                        == expected.dtype
                        == np.dtype(np.float32)
                    )
                    assert eager.shape == jitted.shape == expected.shape == (2, 3)
                    assert eager.tobytes() == jitted.tobytes() == expected.tobytes()
                    sample_index = len(references)
                    for name, value in zip(
                        ("features", "left", "right", "outer", "inner"),
                        values,
                        strict=True,
                    ):
                        evidence[f"sample_{sample_index}_{name}"] = value
                    evidence[f"sample_{sample_index}_eager"] = eager
                    evidence[f"sample_{sample_index}_jit"] = jitted
                    evidence[f"sample_{sample_index}_analytic"] = expected
                    references.append((values, eager.copy()))
                    family_outputs.append(eager.tobytes())
            assert len(set(family_outputs)) == 4
        np.savez(tmp_path / "native_references.npz", **evidence)
        samples = (
            [
                SourceSample(values, [expected], SampleCriteria("bitwise"))
                for values, expected in references
            ]
            if policy is PrecisionPolicy.SOURCE
            else []
        )
        model = ui._to_onnx(
            _scope_sentinel_target,
            [
                jax.ShapeDtypeStruct((2, 3), np.float32),
                jax.ShapeDtypeStruct((2, 3), np.float32),
                jax.ShapeDtypeStruct((2, 3), np.float32),
                jax.ShapeDtypeStruct((), np.bool_),
                jax.ShapeDtypeStruct((), np.bool_),
            ],
            input_names=["features", "left", "right", "outer", "inner"],
            precision_policy=policy,
            opset=23,
        )
        assert isinstance(model, onnx.ModelProto)
        onnx.checker.check_model(model, full_check=True)
        _assert_sentinel_serialized_wires(model)
        payload = model.SerializeToString()
        (tmp_path / "model.onnx").write_bytes(payload)
        for optimization in ("disabled", "all"):
            if policy is PrecisionPolicy.SOURCE:
                result = validate_source_samples(
                    model, samples=samples, optimization=optimization
                )
                assert result.status is CheckStatus.VALIDATED, result
                assert len(result.samples) == 8
                result.require_sample_acceptance()
            else:
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    if optimization == "disabled"
                    else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                session = ort.InferenceSession(
                    payload, options, providers=["CPUExecutionProvider"]
                )
                for values, expected in references:
                    (actual,) = session.run(
                        None,
                        dict(
                            zip(
                                ("features", "left", "right", "outer", "inner"),
                                values,
                                strict=True,
                            )
                        ),
                    )
                    assert (
                        actual.dtype == expected.dtype
                        and actual.shape == expected.shape
                    )
                    assert actual.tobytes() == expected.tobytes()
        assert model.SerializeToString() == payload
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _assert_native_flat_result_values(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if expected.dtype.kind != "f":
        np.testing.assert_array_equal(actual, expected)
        return
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    infinite = np.isinf(expected)
    np.testing.assert_array_equal(
        np.signbit(actual[infinite]), np.signbit(expected[infinite])
    )
    zeros = expected == 0
    np.testing.assert_array_equal(actual[zeros], expected[zeros])
    np.testing.assert_array_equal(
        np.signbit(actual[zeros]), np.signbit(expected[zeros])
    )
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(
        actual[finite].tobytes(order="C"), expected[finite].tobytes(order="C")
    )


class _FlatOwnedExportResult:
    def __init__(self, value, owner):
        self.value, self.owner = value, owner


jax.tree_util.register_pytree_node(
    _FlatOwnedExportResult,
    lambda result: ((result.value,), result.owner),
    lambda owner, values: _FlatOwnedExportResult(values[0], owner),
)


@onnx_function
class _FlatAuxExportOwner:
    def __init__(self):
        self.weights = np.asarray([1.25, -2.5], np.float32)

    def __call__(self, x):
        return _FlatOwnedExportResult(-x, self)


@onnx_function
class _NNXFlatAuxExportOwner(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.asarray([1.25, -2.5], np.float32))
        self.rngs = nnx.Rngs(17)

    def __call__(self, x):
        return _FlatOwnedExportResult(-x * self.weight[...], self)


@pytest.mark.parametrize("fail", [False, True], ids=["completed", "failed"])
def test_source_nnx_function_custom_aux_exports_release_modules_and_buffers(
    fail, monkeypatch
):
    import gc
    import weakref

    from jax2onnx.plugins import plugin_system as ps
    from jax2onnx.plugins._source_function_state import (
        FunctionOperandLayout,
        FunctionStateCall,
    )
    from jax2onnx.plugins.flax.nnx._source_function_state import NNXFunctionState
    from tests.extra_tests.converter.test_source_mlp_state_preparation import (
        array_snapshot,
    )

    plugin = ps.ONNX_FUNCTION_PLUGIN_REGISTRY[
        ps._qualname_of_target(_NNXFlatAuxExportOwner)
    ]
    plugin_fields = plugin._orig_fn, plugin._unbound_orig_fn
    method = _NNXFlatAuxExportOwner.__call__
    original_invoke = NNXFunctionState.invoke
    original_lower = ps.FunctionPlugin._lower_in_context
    original_trace = ps.FunctionPlugin._trace_function_body
    owner_reference = None
    clone_references = []
    completed_traces = 0
    observed_layouts = 0
    sentinel_observed = False
    sentinel = "abort after SOURCE NNX custom-aux body trace"

    def observe_invoke(self, arguments, keywords, state):
        result, updated, effects = original_invoke(self, arguments, keywords, state)
        if self.method is method:
            assert isinstance(result, _FlatOwnedExportResult)
            assert isinstance(result.owner, _NNXFlatAuxExportOwner)
            assert result.owner is not owner_reference()
            assert self.state_count == len(state) == len(updated) == 3
            assert effects == ()
            # Do not retain the result, adapter, clone Variables or traced leaves.
            clone_references.append(weakref.ref(result.owner))
        return result, updated, effects

    def observe_lower(self, context, equation, parameters):
        nonlocal observed_layouts
        if self is plugin:
            metadata = parameters[ps._SOURCE_TRACE]
            adapter = parameters[ps._SOURCE_STATE]
            layout = parameters[ps._FUNCTION_LAYOUT]
            keys = parameters[ps._SOURCE_KEY_REUSE]
            assert metadata.call is None
            assert isinstance(adapter, NNXFunctionState) and adapter.method is method
            assert adapter.effects == () and not equation.effects
            assert isinstance(layout, FunctionOperandLayout)
            assert layout.positional_count == layout.argument_count == 1
            assert layout.state_count == adapter.state_count == 3
            layout.check(equation.invars)
            assert len(equation.outvars) == 1 + layout.state_count
            assert not keys.failed and not keys.sinks and not keys.sources
            observed_layouts += 1
        return original_lower(self, context, equation, parameters)

    def checked_trace(self, callee, equation, captures, **kwargs):
        nonlocal completed_traces
        if self is plugin:
            assert isinstance(callee, FunctionStateCall)
            assert isinstance(callee.adapter, NNXFunctionState)
            assert callee.adapter.method is method and callee.adapter.effects == ()
            assert callee.positional_count == 1
            assert kwargs["source_trace"].call is None
            assert kwargs["precision_policy"] is PrecisionPolicy.SOURCE
        body = original_trace(self, callee, equation, captures, **kwargs)
        if self is plugin:
            completed_traces += 1
            assert not body.jaxpr.effects
            assert len(body.jaxpr.invars) == len(body.jaxpr.outvars) == 4
            if fail:
                raise RuntimeError(sentinel)
        return body

    monkeypatch.setattr(NNXFunctionState, "invoke", observe_invoke)
    monkeypatch.setattr(ps.FunctionPlugin, "_lower_in_context", observe_lower)
    monkeypatch.setattr(ps.FunctionPlugin, "_trace_function_body", checked_trace)

    def export_owned():
        nonlocal owner_reference, sentinel_observed
        owner = _NNXFlatAuxExportOwner()
        owner_reference = weakref.ref(owner)
        buffers = tuple(
            weakref.ref(value) for value in jax.tree_util.tree_leaves(nnx.state(owner))
        )
        assert len(buffers) == 3
        before = array_snapshot(owner)
        variable_ids = (
            id(owner.weight),
            id(owner.rngs.default.key),
            id(owner.rngs.default.count),
        )
        assert np.asarray(owner.rngs.default.count[...]).item() == 0

        def call(x):
            return owner(x).value

        artifact = None
        try:
            with jax.enable_x64(False):
                artifact = ui._to_onnx(
                    call,
                    [jax.ShapeDtypeStruct((2,), np.float32)],
                    precision_policy=PrecisionPolicy.SOURCE,
                    return_mode="ir",
                )
        except RuntimeError as error:
            assert fail and type(error) is RuntimeError and str(error) == sentinel
            sentinel_observed = True
        else:
            assert not fail
            assert isinstance(artifact, ir.Model) and len(artifact.functions) == 1
            onnx.checker.check_model(ir.to_proto(artifact), full_check=True)
        finally:
            assert array_snapshot(owner) == before
            assert variable_ids == (
                id(owner.weight),
                id(owner.rngs.default.key),
                id(owner.rngs.default.count),
            )
        # No owner, Variable, state tuple, result or exception escapes this helper.
        return artifact, owner_reference, buffers

    artifact, reference, buffers = export_owned()
    assert completed_traces == observed_layouts == 1
    assert sentinel_observed is fail and clone_references
    assert plugin._orig_fn is plugin_fields[0]
    assert plugin._unbound_orig_fn is plugin_fields[1]
    gc.collect()
    assert reference() is None
    assert all(reference() is None for reference in (*buffers, *clone_references))
    if not fail:
        assert isinstance(artifact, ir.Model) and len(artifact.functions) == 1


def test_source_nnx_function_custom_aux_closed_jaxpr_releases_modules_and_buffers(
    monkeypatch,
):
    import gc
    import weakref

    from jax2onnx._compat.jax import Literal
    from jax2onnx.converter.conversion_api import _activate_plugin_worlds
    from jax2onnx.converter.precision_policy import precision_policy_scope
    from jax2onnx.plugins import plugin_system as ps
    from jax2onnx.plugins._source_function_state import (
        FunctionOperandLayout,
        FunctionStateCall,
    )
    from jax2onnx.plugins.flax.nnx._source_function_state import NNXFunctionState
    from tests.extra_tests.converter.test_source_mlp_state_preparation import (
        array_snapshot,
    )

    plugin = ps.ONNX_FUNCTION_PLUGIN_REGISTRY[
        ps._qualname_of_target(_NNXFlatAuxExportOwner)
    ]
    plugin_fields = plugin._orig_fn, plugin._unbound_orig_fn
    method = _NNXFlatAuxExportOwner.__call__
    original_invoke = NNXFunctionState.invoke
    owner_reference = None
    clone_references = []

    def observe_invoke(self, arguments, keywords, state):
        result, updated, effects = original_invoke(self, arguments, keywords, state)
        if self.method is method:
            assert isinstance(result, _FlatOwnedExportResult)
            assert isinstance(result.owner, _NNXFlatAuxExportOwner)
            assert result.owner is not owner_reference()
            assert self.state_count == len(state) == len(updated) == 3
            assert effects == ()
            clone_references.append(weakref.ref(result.owner))
        return result, updated, effects

    monkeypatch.setattr(NNXFunctionState, "invoke", observe_invoke)

    def captured_owned():
        nonlocal owner_reference
        owner = _NNXFlatAuxExportOwner()
        owner_reference = weakref.ref(owner)
        buffers = tuple(
            weakref.ref(value) for value in jax.tree_util.tree_leaves(nnx.state(owner))
        )
        assert len(buffers) == 3
        before = array_snapshot(owner)
        variable_ids = (
            id(owner.weight),
            id(owner.rngs.default.key),
            id(owner.rngs.default.count),
        )
        assert np.asarray(owner.rngs.default.count[...]).item() == 0

        def call(x):
            return owner(x).value

        with (
            jax.enable_x64(False),
            precision_policy_scope(PrecisionPolicy.SOURCE),
            _activate_plugin_worlds(),
        ):
            closed = jax.make_jaxpr(call)(jax.ShapeDtypeStruct((2,), np.float32))
        assert array_snapshot(owner) == before
        assert variable_ids == (
            id(owner.weight),
            id(owner.rngs.default.key),
            id(owner.rngs.default.count),
        )
        return closed, owner_reference, buffers

    def exercise(closed, buffers):
        (equation,) = closed.jaxpr.eqns
        assert equation.primitive is plugin._source_primitive
        metadata = equation.params[ps._SOURCE_TRACE]
        adapter = equation.params[ps._SOURCE_STATE]
        layout = equation.params[ps._FUNCTION_LAYOUT]
        keys = equation.params[ps._SOURCE_KEY_REUSE]
        assert metadata.call is None
        assert isinstance(adapter, NNXFunctionState) and adapter.method is method
        assert adapter.effects == () and not equation.effects
        assert isinstance(layout, FunctionOperandLayout)
        assert layout.positional_count == layout.argument_count == 1
        assert layout.state_count == adapter.state_count == len(buffers) == 3
        layout.check(equation.invars)
        assert len(equation.outvars) == 1 + layout.state_count
        assert not keys.failed and not keys.sinks and not keys.sources
        key_inputs = {
            index
            for index, variable in enumerate(equation.invars)
            if jax.dtypes.issubdtype(variable.aval.dtype, jax.dtypes.prng_key)
        }
        key_outputs = {
            index
            for index, variable in enumerate(equation.outvars)
            if jax.dtypes.issubdtype(variable.aval.dtype, jax.dtypes.prng_key)
        }
        assert {source for source, _ in keys.forwards} == key_inputs
        assert {target for _, target in keys.forwards} == key_outputs
        # These are legitimate closed-Jaxpr numeric constants, not owner metadata.
        assert closed.consts
        for reference in buffers:
            value = reference()
            assert value is None or any(value is item for item in closed.consts)
        x = np.asarray([-2.0, 3.0], np.float32)
        environment = dict(zip(closed.jaxpr.constvars, closed.consts, strict=True))
        (argument,) = closed.jaxpr.invars
        environment[argument] = x
        operands = tuple(
            atom.val if isinstance(atom, Literal) else environment[atom]
            for atom in equation.invars
        )
        native_call = FunctionStateCall(adapter, layout.positional_count, keys)
        native_result, native_state = native_call(*operands)
        assert isinstance(native_result, _FlatOwnedExportResult)
        assert all(
            after is before
            for before, after in zip(
                operands[layout.argument_count :], native_state, strict=True
            )
        )
        actual = equation.primitive.impl(*operands, **equation.params)
        axes = (0, *(None for _ in range(layout.state_count)))
        nested = jax.vmap(
            jax.vmap(
                lambda *values: equation.primitive.bind(*values, **equation.params),
                in_axes=axes,
            ),
            in_axes=axes,
        )
        batched = nested(np.broadcast_to(x, (2, 3, 2)), *operands[1:])
        for observed, batch_shape in ((actual, ()), (batched, (2, 3))):
            assert len(observed) == 1 + layout.state_count
            _assert_native_flat_result_values(
                np.asarray(observed[0]),
                np.broadcast_to(
                    np.asarray(native_result.value),
                    (*batch_shape, *native_result.value.shape),
                ),
            )
            for actual_state, expected_state in zip(
                observed[1:], native_state, strict=True
            ):
                assert actual_state.dtype == expected_state.dtype
                assert actual_state.shape == (*batch_shape, *expected_state.shape)
                is_key = jax.dtypes.issubdtype(
                    expected_state.dtype, jax.dtypes.prng_key
                )
                actual_data = np.asarray(
                    jax.random.key_data(actual_state) if is_key else actual_state
                )
                expected_data = np.asarray(
                    jax.random.key_data(expected_state) if is_key else expected_state
                )
                _assert_native_flat_result_values(
                    actual_data,
                    np.broadcast_to(
                        expected_data, (*batch_shape, *expected_data.shape)
                    ),
                )

    closed, reference, buffers = captured_owned()
    assert clone_references
    assert plugin._orig_fn is plugin_fields[0]
    assert plugin._unbound_orig_fn is plugin_fields[1]
    gc.collect()
    assert reference() is None
    assert all(reference() is None for reference in clone_references)
    exercise(closed, buffers)
    assert plugin._orig_fn is plugin_fields[0]
    assert plugin._unbound_orig_fn is plugin_fields[1]
    del closed
    gc.collect()
    assert reference() is None
    assert all(reference() is None for reference in (*buffers, *clone_references))


@pytest.mark.parametrize("fail", [False, True], ids=["completed", "failed"])
def test_source_function_custom_aux_exports_release_owner_and_weights(
    fail, monkeypatch
):
    import gc
    import weakref

    from jax2onnx.plugins.plugin_system import (
        FunctionPlugin,
        ONNX_FUNCTION_PLUGIN_REGISTRY,
        _qualname_of_target,
    )

    plugin = ONNX_FUNCTION_PLUGIN_REGISTRY[_qualname_of_target(_FlatAuxExportOwner)]
    plugin_fields = plugin._orig_fn, plugin._unbound_orig_fn
    original_trace = FunctionPlugin._trace_function_body
    completed_traces = 0
    sentinel_observed = False
    sentinel = "abort after SOURCE custom-aux body trace"

    def checked_trace(self, *args, **kwargs):
        nonlocal completed_traces
        closed = original_trace(self, *args, **kwargs)
        if self is plugin:
            completed_traces += 1
            if fail:
                raise RuntimeError(sentinel)
        return closed

    monkeypatch.setattr(FunctionPlugin, "_trace_function_body", checked_trace)

    def export_owned():
        nonlocal sentinel_observed
        owner = _FlatAuxExportOwner()
        references = weakref.ref(owner), weakref.ref(owner.weights)

        def call(x):
            return owner(x).value

        artifact = None
        try:
            with jax.enable_x64(False):
                artifact = ui._to_onnx(
                    call,
                    [jax.ShapeDtypeStruct((2,), np.float32)],
                    precision_policy=PrecisionPolicy.SOURCE,
                    return_mode="ir",
                )
        except RuntimeError as error:
            assert fail and type(error) is RuntimeError and str(error) == sentinel
            sentinel_observed = True
        else:
            assert not fail
            assert isinstance(artifact, ir.Model) and len(artifact.functions) == 1
            onnx.checker.check_model(ir.to_proto(artifact), full_check=True)
        return artifact, references

    artifact, references = export_owned()
    assert completed_traces == 1 and sentinel_observed is fail
    assert plugin._orig_fn is plugin_fields[0]
    assert plugin._unbound_orig_fn is plugin_fields[1]
    gc.collect()
    assert all(reference() is None for reference in references)
    if not fail:
        # Keep the actual artifact live beyond release of all Python owners.
        assert isinstance(artifact, ir.Model) and len(artifact.functions) == 1


def test_source_function_custom_aux_live_equation_owns_only_its_lifetime():
    import gc
    import weakref

    from jax2onnx.converter.conversion_api import _activate_plugin_worlds
    from jax2onnx.converter.precision_policy import precision_policy_scope
    from jax2onnx.plugins.plugin_system import (
        ONNX_FUNCTION_PLUGIN_REGISTRY,
        _SOURCE_TRACE,
        _qualname_of_target,
    )

    plugin = ONNX_FUNCTION_PLUGIN_REGISTRY[_qualname_of_target(_FlatAuxExportOwner)]
    plugin_fields = plugin._orig_fn, plugin._unbound_orig_fn

    def owned_equation():
        owner = _FlatAuxExportOwner()
        references = weakref.ref(owner), weakref.ref(owner.weights)

        def call(x):
            return owner(x).value

        with (
            jax.enable_x64(False),
            precision_policy_scope(PrecisionPolicy.SOURCE),
            _activate_plugin_worlds(),
        ):
            closed = jax.make_jaxpr(call)(jax.ShapeDtypeStruct((2,), np.float32))
        (equation,) = closed.jaxpr.eqns
        return equation, references

    def exercise(equation, references):
        metadata = equation.params[_SOURCE_TRACE]
        assert metadata.call.receiver is references[0]()
        assert metadata.call.original.__self__ is references[0]()
        x = np.asarray([-2.0, 3.0], np.float32)
        (actual,) = equation.primitive.impl(x, **equation.params)
        _assert_native_flat_result_values(np.asarray(actual), -x)
        nested = jax.vmap(
            jax.vmap(lambda row: equation.primitive.bind(row, **equation.params))
        )
        values = np.broadcast_to(x, (2, 3, 2))
        (actual,) = nested(values)
        _assert_native_flat_result_values(np.asarray(actual), -values)

    equation, references = owned_equation()
    gc.collect()
    assert all(reference() is not None for reference in references)
    exercise(equation, references)
    assert plugin._orig_fn is plugin_fields[0]
    assert plugin._unbound_orig_fn is plugin_fields[1]
    del equation
    gc.collect()
    assert all(reference() is None for reference in references)
