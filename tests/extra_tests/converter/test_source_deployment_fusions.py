# tests/extra_tests/converter/test_source_deployment_fusions.py

"""Exact private fusion correspondence; these structural proofs grant no numerics."""

from copy import deepcopy
import struct

import pytest

from jax2onnx._deployment_fusions import (
    Attribute,
    FusionUnfinished,
    Graph,
    Node,
    Registry,
    TensorData,
    ValueType,
    View,
    expand_proven_fusions,
    _FP32_POST_MUL_SINGLE_PANEL,
)


def _node(tag, operation, inputs, outputs, attributes=None, domain=""):
    return Node(
        f"jax2onnx_src_{tag:06d}",
        domain,
        operation,
        "",
        tuple(inputs),
        tuple(outputs),
        attributes or {},
    )


def _matrix_pair(bias="vector", layout=False):
    roles = {
        f"jax2onnx_src_{tag:06d}": (operation, "", "root")
        for tag, operation in [
            (1, "MatMul"),
            (2, "Add"),
            (3, "Reshape"),
            (4, "Expand"),
            (5, "Transpose"),
        ]
    }
    weights = TensorData(1, (2, 2), struct.pack("<4f", 1, 2, 3, 4))
    bias_shape, bias_values = {
        "scalar": ((), (5,)),
        "vector": ((2,), (5, 6)),
        "row": ((1, 2), (5, 6)),
    }[bias]
    bias_value = TensorData(
        1, bias_shape, struct.pack("<" + "f" * len(bias_values), *bias_values)
    )
    row_values = (5, 5) if bias == "scalar" else bias_values
    target_bias = TensorData(1, (1, 2), struct.pack("<2f", *row_values))
    types = {name: ValueType(1, (2, 2)) for name in ("x", "other", "mat", "z", "out")}
    source = Graph(
        "source",
        ("x", "other"),
        ("out" if layout else "z",),
        dict(types),
        {"w": weights, "bias": bias_value},
        [
            _node(1, "MatMul", ("x", "w"), ("mat",)),
            _node(2, "Add", ("mat", "bias"), ("z",)),
        ],
    )
    target = Graph(
        "target",
        ("x", "other"),
        source.outputs,
        {**types, "z": ValueType(None, None)},
        {"w": weights, "bias": target_bias},
        [
            _node(
                1,
                "Gemm",
                ("x", "w", "bias"),
                ("z",),
                {
                    "alpha": Attribute("FLOAT", 1.0),
                    "beta": Attribute("FLOAT", 1.0),
                    "transA": Attribute("INT", 0),
                    "transB": Attribute("INT", 0),
                },
            )
        ],
    )
    if layout:
        transpose = _node(
            5, "Transpose", ("z",), ("out",), {"perm": Attribute("INTS", (1, 0))}
        )
        source.nodes.append(transpose)
        target.nodes.append(deepcopy(transpose))
    return source, target, roles


@pytest.mark.parametrize("bias", ["scalar", "vector", "row"])
def test_exact_gemm_bias_index_map_and_unknown_metadata_are_proved(bias):
    source, target, roles = _matrix_pair(bias)
    original = deepcopy((source, target))
    virtual, kinds = expand_proven_fusions(source, target, roles)
    assert kinds == ("Gemm",)
    assert [node.op for node in virtual.nodes] == ["MatMul", "Add"]
    assert virtual.types["z"] == ValueType(1, (2, 2))
    assert (source, target) == original
    assert virtual.nodes[0].inputs[0] == "x"
    assert (
        virtual.initializers[virtual.nodes[1].inputs[1]].data
        == source.initializers["bias"].data
    )


def test_exact_constant_bias_layout_closure_is_restored_without_arithmetic():
    source, target, roles = _matrix_pair()
    source.initializers["shape"] = TensorData(7, (2,), struct.pack("<2q", 1, 2))
    source.nodes[1:1] = [
        _node(3, "Reshape", ("bias", "shape"), ("reshaped",)),
        _node(4, "Expand", ("reshaped", "shape"), ("expanded",)),
    ]
    source.nodes[-1].inputs = ("mat", "expanded")
    virtual, kinds = expand_proven_fusions(source, target, roles)
    assert kinds == ("Gemm",)
    assert [node.op for node in virtual.nodes] == ["Reshape", "Expand", "MatMul", "Add"]
    assert virtual.nodes[-1].inputs[1] == virtual.nodes[1].outputs[0]
    assert virtual.nodes[1].inputs[0] == virtual.nodes[0].outputs[0]


@pytest.mark.parametrize(
    "defect",
    [
        "alpha",
        "transpose",
        "operand",
        "bias-words",
        "bias-map",
        "dynamic-bias",
        "scalar-rank",
        "direct-matrix",
        "cast-between",
        "missing-tag",
        "sibling-port",
    ],
)
def test_unproved_gemm_role_port_storage_and_observable_edges_reject(defect):
    source, target, roles = _matrix_pair()
    if defect == "alpha":
        target.nodes[0].attributes["alpha"] = Attribute("FLOAT", 0.5)
    elif defect == "transpose":
        target.nodes[0].attributes["transA"] = Attribute("INT", 1)
    elif defect == "operand":
        target.nodes[0].inputs = ("other", "w", "bias")
    elif defect == "bias-words":
        target.initializers["bias"] = TensorData(1, (1, 2), struct.pack("<2f", 6, 5))
    elif defect == "bias-map":
        target.initializers["bias"] = TensorData(
            1, (2, 1), target.initializers["bias"].data
        )
    elif defect == "dynamic-bias":
        source.inputs += ("bias",)
        source.types["bias"] = ValueType(1, (2,))
    elif defect == "scalar-rank":
        target.types["z"] = ValueType(1, ())
    elif defect == "direct-matrix":
        source.outputs += ("mat",)
    elif defect == "cast-between":
        roles["jax2onnx_src_000006"] = ("Cast", "", "root")
        source.nodes.insert(
            1, _node(6, "Cast", ("mat",), ("cast",), {"to": Attribute("INT", 1)})
        )
        source.nodes[-1].inputs = ("cast", "bias")
    elif defect == "sibling-port":
        roles["jax2onnx_src_000006"] = ("Neg", "", "root")
        roles["jax2onnx_src_000007"] = ("Add", "", "root")
        for body in (source, target):
            body.outputs = ("final",)
            body.types.update(
                {"neg": ValueType(1, (2, 2)), "final": ValueType(1, (2, 2))}
            )
            body.nodes.extend(
                [
                    _node(6, "Neg", ("x",), ("neg",)),
                    _node(7, "Add", ("z", "neg"), ("final",)),
                ]
            )
        target.nodes[-1].inputs = ("z", "x")
    else:
        target.nodes[0].name = "unbound_Gemm"
    with pytest.raises(FusionUnfinished):
        expand_proven_fusions(source, target, roles)


def test_square_terminal_transpose_order_cannot_disappear_in_shape_equality():
    source, target, roles = _matrix_pair(layout=True)
    assert expand_proven_fusions(source, target, roles)[1] == ("Gemm",)
    target.nodes[-1].attributes["perm"] = Attribute("INTS", (0, 1))
    with pytest.raises(FusionUnfinished, match="terminal consumers"):
        expand_proven_fusions(source, target, roles)


def _gelu_pair():
    roles = {
        f"jax2onnx_src_{tag:06d}": (op, "", "root")
        for tag, op in [(11, "Mul"), (12, "Sigmoid"), (13, "Mul")]
    }
    coefficient = TensorData(1, (), struct.pack("<f", 1.702))
    types = {
        name: ValueType(1, (2, 2))
        for name in ("x", "other", "scaled", "sigmoid", "out")
    }
    source = Graph(
        "source",
        ("x", "other"),
        ("out",),
        types,
        {"alpha": coefficient},
        [
            _node(11, "Mul", ("alpha", "x"), ("scaled",)),
            _node(12, "Sigmoid", ("scaled",), ("sigmoid",)),
            _node(13, "Mul", ("x", "sigmoid"), ("out",)),
        ],
    )
    target = Graph(
        "target",
        source.inputs,
        source.outputs,
        {**types, "out": ValueType(None, None)},
        {},
        [
            _node(
                13,
                "QuickGelu",
                ("x",),
                ("out",),
                {"alpha": Attribute("FLOAT", struct.unpack("<f", coefficient.data)[0])},
                "com.microsoft",
            )
        ],
    )
    return source, target, roles


def test_quick_gelu_restores_all_three_ordered_source_roles_and_exact_alpha():
    source, target, roles = _gelu_pair()
    before = deepcopy((source, target))
    virtual, kinds = expand_proven_fusions(source, target, roles)
    assert kinds == ("QuickGelu",)
    assert [node.name for node in virtual.nodes] != [node.name for node in source.nodes]
    assert [node.op for node in virtual.nodes] == ["Mul", "Sigmoid", "Mul"]
    assert virtual.nodes[0].inputs[1] == virtual.nodes[-1].inputs[0] == "x"
    assert (
        virtual.initializers[virtual.nodes[0].inputs[0]].data
        == source.initializers["alpha"].data
    )
    assert (source, target) == before


@pytest.mark.parametrize(
    "defect",
    [
        "alpha-words",
        "wrong-x",
        "wrong-sigmoid",
        "cast-between",
        "intermediate-output",
        "compute-dtype",
        "output-rewire",
    ],
)
def test_quick_gelu_rejects_changed_complete_pattern_and_observable_ports(defect):
    source, target, roles = _gelu_pair()
    if defect == "alpha-words":
        target.nodes[0].attributes["alpha"] = Attribute("FLOAT", 1.7)
    elif defect == "wrong-x":
        source.nodes[-1].inputs = ("other", "sigmoid")
    elif defect == "wrong-sigmoid":
        source.nodes[1].op = "Tanh"
    elif defect == "cast-between":
        roles["jax2onnx_src_000014"] = ("Cast", "", "root")
        source.nodes.insert(
            2, _node(14, "Cast", ("sigmoid",), ("cast",), {"to": Attribute("INT", 1)})
        )
        source.nodes[-1].inputs = ("x", "cast")
    elif defect == "intermediate-output":
        source.outputs += ("scaled",)
    elif defect == "compute-dtype":
        target.types["out"] = ValueType(11, (2, 2))
    else:
        target.outputs = ("x",)
    with pytest.raises(FusionUnfinished):
        expand_proven_fusions(source, target, roles)


def test_nested_capture_and_same_shaped_wrong_call_binding_stay_distinct():
    source, target, roles = _matrix_pair()
    source.inputs = target.inputs = ()
    source.initializers.pop("w")
    source.initializers.pop("bias")
    target.initializers.pop("w")
    target.initializers.pop("bias")
    outer_source, outer_target, _ = _matrix_pair()
    for body in (outer_source, outer_target):
        body.outputs = ("nested",)
    roles["jax2onnx_src_000020"] = ("If", "", "root")
    roles["jax2onnx_src_000021"] = ("Identity", "", "root")
    other = Graph(
        "other",
        (),
        ("else_output",),
        {"else_output": ValueType(1, (2, 2))},
        {},
        [_node(21, "Identity", ("x",), ("else_output",))],
    )
    outer_source.nodes = [
        _node(
            20,
            "If",
            ("predicate",),
            ("nested",),
            {
                "then_branch": Attribute("GRAPH", source),
                "else_branch": Attribute("GRAPH", deepcopy(other)),
            },
        )
    ]
    outer_target.nodes = [
        _node(
            20,
            "If",
            ("predicate",),
            ("nested",),
            {
                "then_branch": Attribute("GRAPH", target),
                "else_branch": Attribute("GRAPH", deepcopy(other)),
            },
        )
    ]
    for body in (outer_source, outer_target):
        body.inputs += ("predicate",)
        body.types["predicate"] = ValueType(9, ())
    assert expand_proven_fusions(outer_source, outer_target, roles)[1] == ("Gemm",)
    # The child's bias name shadows the parent's initializer name. The actual
    # closure crosses owners twice and must retain lexical SSA identity.
    roles["jax2onnx_src_000022"] = ("Identity", "", "root")
    roles["jax2onnx_src_000023"] = ("Identity", "", "child")
    outer_source.nodes.insert(0, _node(22, "Identity", ("bias",), ("parent_bias",)))
    source.nodes.insert(0, _node(23, "Identity", ("parent_bias",), ("bias",)))
    assert expand_proven_fusions(outer_source, outer_target, roles)[1] == ("Gemm",)
    target.nodes[0].inputs = ("other", "w", "bias")
    with pytest.raises(FusionUnfinished):
        expand_proven_fusions(outer_source, outer_target, roles)


def _folded_scale_pair():
    source, target, roles = _matrix_pair()
    scale = TensorData(1, (2,), struct.pack("<2f", 1.25, 0.75))
    shape = TensorData(7, (2,), struct.pack("<2q", 1, 2))
    for graph in (source, target):
        graph.initializers["scale"] = scale
        graph.outputs += ("scale", "x")
        graph.types["scaled"] = ValueType(1, (2, 2))
    source.initializers["shape"] = shape
    source.nodes[:0] = [
        _node(30, "Reshape", ("scale", "shape"), ("row_scale",)),
        _node(31, "Expand", ("row_scale", "shape"), ("expanded_scale",)),
        _node(32, "Mul", ("x", "expanded_scale"), ("scaled",)),
    ]
    target.initializers["folded_scale"] = TensorData(1, (1, 2), scale.data)
    target.nodes.insert(0, _node(32, "Mul", ("x", "folded_scale"), ("scaled",)))
    source.nodes[3].inputs = ("scaled", "w")
    target.nodes[1].inputs = ("scaled", "w", "bias")
    for tag, operation in ((30, "Reshape"), (31, "Expand"), (32, "Mul")):
        roles[f"jax2onnx_src_{tag:06d}"] = (operation, "", "root")
    return source, target, roles


def test_layout_folded_parameter_origin_is_restored_at_only_the_proved_port():
    source, target, roles = _folded_scale_pair()
    # The independently optimized arithmetic is retained exactly at port0.
    # Only its existing ordered dynamic binding anchors participate here.
    source.nodes[:0] = [
        _node(
            36,
            "Constant",
            (),
            ("one",),
            {"value": Attribute("TENSOR", TensorData(1, (), struct.pack("<f", 1)))},
        ),
        _node(33, "Div", ("one", "other"), ("reciprocal",)),
        _node(34, "Mul", ("x", "reciprocal"), ("dynamic",)),
    ]
    target.nodes.insert(0, _node(33, "Div", ("x", "other"), ("dynamic",)))
    source.nodes[5].inputs = ("dynamic", "expanded_scale")
    target.nodes[1].inputs = ("dynamic", "folded_scale")
    for tag, operation in ((33, "Div"), (34, "Mul"), (36, "Constant")):
        roles[f"jax2onnx_src_{tag:06d}"] = (operation, "", "root")
    before = deepcopy((source, target))
    virtual, kinds = expand_proven_fusions(source, target, roles)
    assert kinds == ("Gemm",)
    consumer = next(
        node for node in virtual.nodes if node.name == "jax2onnx_src_000032"
    )
    assert consumer.inputs[0] == "dynamic" and consumer.attributes == {}
    assert consumer.outputs == ("scaled",)
    assert consumer.inputs[1] != "folded_scale"
    constants = {value.key for value in source.initializers.values()}
    registry = Registry(roles)
    original = View(source, (), registry, constants)
    restored = View(virtual, (), registry, constants)
    assert original.anchors("scaled") == restored.anchors("scaled")
    assert restored.constant(consumer.inputs[1]) == original.constant("expanded_scale")
    assert virtual.initializers["scale"] == source.initializers["scale"]
    assert virtual.initializers["folded_scale"] == target.initializers["folded_scale"]
    assert virtual.outputs == target.outputs
    assert virtual.nodes[0] == target.nodes[0]
    assert (source, target) == before


@pytest.mark.parametrize(
    "defect",
    ["words", "shape", "port", "binding", "ambiguous", "dynamic", "cast"],
)
def test_constant_origin_requires_exact_storage_port_binding_and_closure(defect):
    source, target, roles = _folded_scale_pair()
    if defect == "words":
        target.initializers["folded_scale"] = TensorData(
            1, (1, 2), struct.pack("<2f", 0.75, 1.25)
        )
    elif defect == "shape":
        target.initializers["folded_scale"] = TensorData(
            1, (2, 1), target.initializers["folded_scale"].data
        )
    elif defect == "port":
        target.nodes[0].inputs = ("folded_scale", "x")
    elif defect == "binding":
        target.nodes[0].inputs = ("other", "folded_scale")
    elif defect == "ambiguous":
        source.nodes.insert(
            3, _node(32, "Mul", ("x", "expanded_scale"), ("ambiguous",))
        )
    elif defect == "dynamic":
        source.inputs += ("scale",)
    else:
        source.nodes[0].op = "Cast"
        source.nodes[0].inputs = ("scale",)
        source.nodes[0].attributes = {"to": Attribute("INT", 1)}
    with pytest.raises(FusionUnfinished):
        expand_proven_fusions(source, target, roles)


def _shared_constant_calls():
    weight = TensorData(1, (2, 2), struct.pack("<4f", 1, 2, 3, 4))
    roles = {
        "jax2onnx_src_000040": ("Constant", "", "function"),
        "jax2onnx_src_000041": ("MatMul", "", "function"),
        "jax2onnx_src_000042": ("Neg", "", "unproved"),
    }
    types = {name: ValueType(1, (2, 2)) for name in ("x", "other", "first", "second")}
    source = Graph(
        "source",
        ("x", "other"),
        ("first", "second"),
        dict(types),
        {},
        [
            _node(
                40, "Constant", (), ("weight",), {"value": Attribute("TENSOR", weight)}
            ),
            _node(41, "MatMul", ("x", "weight"), ("first",)),
            _node(41, "MatMul", ("other", "weight"), ("second",)),
        ],
    )
    target = Graph(
        "target",
        source.inputs,
        (*source.outputs, "folded", "unproved"),
        dict(types),
        {"folded": weight},
        [
            _node(41, "MatMul", ("x", "folded"), ("first",)),
            _node(41, "MatMul", ("other", "folded"), ("second",)),
            _node(42, "Neg", ("folded",), ("unproved",)),
        ],
    )
    return source, target, roles


def test_shared_folded_constant_restores_each_actual_call_and_keeps_unproved_uses():
    source, target, roles = _shared_constant_calls()
    before = deepcopy((source, target))
    virtual, kinds = expand_proven_fusions(source, target, roles)
    assert kinds == ()
    matrices = [node for node in virtual.nodes if node.op == "MatMul"]
    assert [node.inputs[0] for node in matrices] == ["x", "other"]
    assert matrices[0].inputs[1] != matrices[1].inputs[1]
    constants = {
        node.outputs[0]: node for node in virtual.nodes if node.op == "Constant"
    }
    for matrix in matrices:
        restored = constants[matrix.inputs[1]]
        assert restored.attributes["value"].value == target.initializers["folded"]
        assert "jax2onnx_src_000040" in restored.name
    assert virtual.nodes[-1] == target.nodes[-1]
    assert virtual.outputs == target.outputs
    assert virtual.initializers == target.initializers
    assert (source, target) == before
    target.nodes[1].inputs = ("x", "folded")
    with pytest.raises(FusionUnfinished, match="reused by optimized bindings"):
        expand_proven_fusions(source, target, roles)


def _batch_fused_pair(cast_back=False, with_gemm=True):
    def control(values):
        return TensorData(
            7, (len(values),), struct.pack("<" + "q" * len(values), *values)
        )

    controls = {
        "expand_shape": control((3, 2, 2, 8)),
        "reduce_axes": control((3,)),
        "starts": control((0, 0, 0, 0)),
        "ends": control((2, 2, 3, 3)),
        "axes": control((0, 1, 2, 3)),
        "final_shape": control((3, 32)),
    }
    probability_dtype = 11 if cast_back else 1
    types = {
        "a": ValueType(1, (3, 2, 1, 8)),
        "prob": ValueType(probability_dtype, (2, 2, 3, 3)),
        "otherprob": ValueType(probability_dtype, (2, 2, 3, 3)),
        "expanded": ValueType(1, (3, 2, 2, 8)),
        "exp": ValueType(probability_dtype, (2, 2, 3, 3)),
        "sum": ValueType(probability_dtype, (2, 2, 3, 1)),
        "div": ValueType(probability_dtype, (2, 2, 3, 3)),
        "data": ValueType(1, (2, 2, 3, 3)),
        "sliced": ValueType(1, (2, 2, 3, 3)),
        "left": ValueType(1, (2, 2, 8, 3)),
        "right": ValueType(1, (2, 2, 3, 3)),
        "batch": ValueType(1, (2, 2, 8, 3)),
        "transposed": ValueType(1, (3, 2, 2, 8)),
        "flat": ValueType(1, (3, 32)),
        "matrix": ValueType(1, (3, 32)),
        "out": ValueType(1, (3, 32)),
    }
    data = "data" if cast_back else "div"
    prefix = [
        _node(100, "Expand", ("a", "expand_shape"), ("expanded",)),
        _node(101, "Exp", ("prob",), ("exp",)),
        _node(
            102,
            "ReduceSum",
            ("exp", "reduce_axes"),
            ("sum",),
            {"keepdims": Attribute("INT", 1)},
        ),
        _node(103, "Div", ("exp", "sum"), ("div",)),
    ]
    if cast_back:
        prefix.append(
            _node(104, "Cast", ("div",), ("data",), {"to": Attribute("INT", 1)})
        )
    prefix.append(_node(105, "Slice", (data, "starts", "ends", "axes"), ("sliced",)))
    source_nodes = [
        *deepcopy(prefix),
        _node(
            106,
            "Transpose",
            ("expanded",),
            ("left",),
            {"perm": Attribute("INTS", (1, 2, 3, 0))},
        ),
        _node(
            107,
            "Transpose",
            ("sliced",),
            ("right",),
            {"perm": Attribute("INTS", (0, 1, 3, 2))},
        ),
        _node(108, "MatMul", ("left", "right"), ("batch",)),
        _node(
            109,
            "Transpose",
            ("batch",),
            ("transposed",),
            {"perm": Attribute("INTS", (3, 0, 1, 2))},
        ),
        _node(110, "Reshape", ("transposed", "final_shape"), ("flat",)),
    ]
    target_nodes = [
        *deepcopy(prefix),
        _node(
            108,
            "FusedMatMul",
            ("expanded", "sliced"),
            ("batch",),
            {
                "alpha": Attribute("FLOAT", 1.0),
                "transA": Attribute("INT", 1),
                "transB": Attribute("INT", 1),
                "transBatchA": Attribute("INT", 1),
                "transBatchB": Attribute("INT", 0),
            },
            "com.microsoft",
        ),
        deepcopy(source_nodes[-2]),
        deepcopy(source_nodes[-1]),
    ]
    target_nodes[2].attributes["noop_with_empty_axes"] = Attribute("INT", 0)
    if cast_back:
        target_nodes[4].attributes["saturate"] = Attribute("INT", 1)
    parameters = {
        "weights": TensorData(1, (32, 32), struct.pack("<f", 1) * 1024),
        "bias": TensorData(1, (1, 32), bytes(4 * 32)),
    }
    if with_gemm:
        source_nodes.extend(
            [
                _node(111, "MatMul", ("flat", "weights"), ("matrix",)),
                _node(112, "Add", ("matrix", "bias"), ("out",)),
            ]
        )
        target_nodes.append(
            _node(
                111,
                "Gemm",
                ("flat", "weights", "bias"),
                ("out",),
                {
                    "alpha": Attribute("FLOAT", 1.0),
                    "beta": Attribute("FLOAT", 1.0),
                    "transA": Attribute("INT", 0),
                    "transB": Attribute("INT", 0),
                },
            )
        )
    outputs = ("out" if with_gemm else "flat", "a", "prob")
    source = Graph(
        "source",
        ("a", "prob", "otherprob"),
        outputs,
        types,
        {**controls, **parameters},
        source_nodes,
    )
    target_types = dict(types)
    for name in ("exp", "sum", "div", "sliced", "batch", "transposed", "flat", "out"):
        target_types[name] = ValueType(None, None)
    if cast_back:
        target_types["data"] = ValueType(1, None)
    target = Graph(
        "target",
        source.inputs,
        outputs,
        target_types,
        deepcopy(source.initializers),
        target_nodes,
    )
    roles = {node.name: (node.op, node.domain, "root") for node in source_nodes}
    return source, target, roles


def test_actual_batch_permutations_and_proved_slice_types_reach_later_gemm():
    for cast_back in (False, True):
        source, target, roles = _batch_fused_pair(cast_back)
        before = deepcopy((source, target))
        virtual, kinds = expand_proven_fusions(source, target, roles)
        assert kinds == ("FusedMatMul", "Gemm")
        assert (source, target) == before
        assert not any(node.op in {"FusedMatMul", "Gemm"} for node in virtual.nodes)
        assert virtual.types["sliced"] == ValueType(1, (2, 2, 3, 3))
        assert virtual.types["batch"] == ValueType(1, (2, 2, 8, 3))
        assert virtual.types["transposed"] == ValueType(1, (3, 2, 2, 8))
        assert virtual.types["flat"] == ValueType(1, (3, 32))
        assert virtual.outputs == target.outputs
        restored = [
            node
            for node in virtual.nodes
            if node.op == "Transpose" and "jax2onnx_src_00010" in node.name
        ]
        assert [node.attributes["perm"].value for node in restored[:2]] == [
            (1, 2, 3, 0),
            (0, 1, 3, 2),
        ]
        if cast_back:
            assert virtual.types["div"] == ValueType(11, (2, 2, 3, 3))
            assert virtual.types["data"] == ValueType(1, (2, 2, 3, 3))
            assert next(node for node in virtual.nodes if node.op == "Cast").inputs == (
                "div",
            )
    source, target, roles = _batch_fused_pair(with_gemm=False)
    assert expand_proven_fusions(source, target, roles)[1] == ("FusedMatMul",)


@pytest.mark.parametrize(
    "defect",
    [
        "batch-flag",
        "last-flag",
        "alpha",
        "square-permutation",
        "slice-control",
        "slice-binding",
        "div-order",
        "reduce-control",
        "ambiguous",
        "known-dtype",
        "known-shape",
    ],
)
def test_batch_fused_correspondence_rejects_unproved_layout_control_and_types(defect):
    source, target, roles = _batch_fused_pair()
    fused = next(node for node in target.nodes if node.op == "FusedMatMul")
    if defect == "batch-flag":
        fused.attributes["transBatchA"] = Attribute("INT", 0)
    elif defect == "last-flag":
        fused.attributes["transB"] = Attribute("INT", 0)
    elif defect == "alpha":
        fused.attributes["alpha"] = Attribute("FLOAT", 0.5)
    elif defect == "square-permutation":
        next(
            node for node in source.nodes if node.name == "jax2onnx_src_000107"
        ).attributes["perm"] = Attribute("INTS", (0, 1, 2, 3))
    elif defect == "slice-control":
        target.initializers["ends"] = TensorData(
            7, (4,), struct.pack("<4q", 2, 2, 3, 2)
        )
    elif defect == "slice-binding":
        next(node for node in target.nodes if node.op == "Exp").inputs = ("otherprob",)
    elif defect == "div-order":
        next(node for node in target.nodes if node.op == "Div").inputs = ("sum", "exp")
    elif defect == "reduce-control":
        target.initializers["reduce_axes"] = TensorData(7, (1,), struct.pack("<q", 2))
    elif defect == "ambiguous":
        source.nodes.insert(8, _node(108, "MatMul", ("left", "right"), ("duplicate",)))
    elif defect == "known-dtype":
        target.types["sliced"] = ValueType(11, (2, 2, 3, 3))
    else:
        target.types["batch"] = ValueType(1, ())
    with pytest.raises(FusionUnfinished):
        expand_proven_fusions(source, target, roles)


def test_actual_proto_batch_dispatch_keeps_no_batch_fast_path():
    import onnx
    from jax2onnx._deployment_fusion_bridge import contains_fusions

    node = onnx.helper.make_node(
        "FusedMatMul", ["a", "b"], ["out"], domain="com.microsoft", transBatchA=0
    )
    model = onnx.helper.make_model(onnx.helper.make_graph([node], "routing", [], []))
    assert not contains_fusions(model)
    flag = model.graph.node[0].attribute[0]
    flag.i = 1
    assert contains_fusions(model)


def _post_mul_fused_pair(inner=8):
    alpha = TensorData(1, (), bytes.fromhex("f304b53e"))
    types = {
        "a": ValueType(1, (3, 2, 2, inner)),
        "b": ValueType(1, (3, 2, 2, inner)),
        "other": ValueType(1, (2, 2, 3, 3)),
        "left": ValueType(1, (2, 2, 3, inner)),
        "right": ValueType(1, (2, 2, inner, 3)),
        "matrix": ValueType(1, (2, 2, 3, 3)),
        "scaled": ValueType(1, (2, 2, 3, 3)),
        "out": ValueType(1, (2, 2, 3, 3)),
    }
    nodes = [
        _node(
            200,
            "Transpose",
            ("a",),
            ("left",),
            {"perm": Attribute("INTS", (1, 2, 0, 3))},
        ),
        _node(
            201,
            "Transpose",
            ("b",),
            ("right",),
            {"perm": Attribute("INTS", (1, 2, 3, 0))},
        ),
        _node(202, "MatMul", ("left", "right"), ("matrix",)),
        _node(203, "Constant", (), ("alpha",), {"value": Attribute("TENSOR", alpha)}),
        _node(204, "Mul", ("matrix", "alpha"), ("scaled",)),
        _node(205, "Add", ("scaled", "bias"), ("out",)),
    ]
    source = Graph(
        "source",
        ("a", "b", "other"),
        ("out", "scaled", "a", "b"),
        types,
        {"bias": TensorData(1, (), struct.pack("<f", 1))},
        nodes,
    )
    target = Graph(
        "target",
        source.inputs,
        source.outputs,
        {**types, "scaled": ValueType(None, None)},
        deepcopy(source.initializers),
        [
            _node(
                202,
                "FusedMatMul",
                ("a", "b"),
                ("scaled",),
                {
                    "alpha": Attribute("FLOAT", struct.unpack("<f", alpha.data)[0]),
                    "transA": Attribute("INT", 0),
                    "transBatchA": Attribute("INT", 1),
                    "transB": Attribute("INT", 1),
                    "transBatchB": Attribute("INT", 1),
                },
                "com.microsoft",
            ),
            deepcopy(nodes[-1]),
        ],
    )
    roles = {node.name: (node.op, node.domain, "root") for node in nodes}
    return source, target, roles


def test_exact_fp32_single_panel_restores_separate_matrix_and_post_mul_rounding():
    source, target, roles = _post_mul_fused_pair()
    before = deepcopy((source, target))
    virtual, kinds = expand_proven_fusions(
        source, target, roles, fp32_post_mul_contract=_FP32_POST_MUL_SINGLE_PANEL
    )
    assert kinds == ("FusedMatMul",)
    assert (source, target) == before
    assert [node.op for node in virtual.nodes] == [
        "Transpose",
        "Transpose",
        "MatMul",
        "Constant",
        "Mul",
        "Add",
    ]
    matrix, constant, scale = virtual.nodes[2:5]
    assert matrix.outputs[0] != "scaled"
    assert scale.outputs == ("scaled",)
    assert scale.inputs == (matrix.outputs[0], constant.outputs[0])
    assert constant.attributes == source.nodes[3].attributes
    assert "jax2onnx_src_000204" in scale.name
    assert virtual.types[matrix.outputs[0]] == ValueType(1, (2, 2, 3, 3))
    assert virtual.types["scaled"] == ValueType(1, (2, 2, 3, 3))
    assert virtual.nodes[-1] == target.nodes[-1]
    assert virtual.outputs == target.outputs
    # Nonunit alpha without a batch transpose still takes this exact proof.
    for port, name in enumerate(("a", "b")):
        logical = source.nodes[2].inputs[port]
        source.types[name] = source.types[logical]
        target.types[name] = source.types[logical]
        source.nodes[2].inputs = tuple(
            name if index == port else value
            for index, value in enumerate(source.nodes[2].inputs)
        )
    del source.nodes[:2]
    target.nodes[0].attributes.update(
        {
            name: Attribute("INT", 0)
            for name in ("transA", "transB", "transBatchA", "transBatchB")
        }
    )
    assert expand_proven_fusions(
        source, target, roles, fp32_post_mul_contract=_FP32_POST_MUL_SINGLE_PANEL
    )[1] == ("FusedMatMul",)


@pytest.mark.parametrize(
    "defect",
    [
        "no-capability",
        "large-k",
        "unknown-k",
        "scale-words",
        "scale-dtype",
        "scale-placement",
        "matrix-output",
        "matrix-capture",
        "terminal-binding",
    ],
)
def test_nonunit_fused_matrix_requires_exact_rounding_scale_use_and_terminal(defect):
    source, target, roles = _post_mul_fused_pair(
        129 if defect == "large-k" else None if defect == "unknown-k" else 8
    )
    contract = _FP32_POST_MUL_SINGLE_PANEL
    expected = (
        "rounding" if defect in {"no-capability", "large-k", "unknown-k"} else None
    )
    if defect == "no-capability":
        contract = None
    elif defect == "scale-words":
        source.nodes[3].attributes["value"] = Attribute(
            "TENSOR", TensorData(1, (), struct.pack("<f", 0.5))
        )
    elif defect == "scale-dtype":
        source.nodes[3].attributes["value"] = Attribute(
            "TENSOR",
            TensorData(
                11, (), struct.pack("<d", target.nodes[0].attributes["alpha"].value)
            ),
        )
    elif defect == "scale-placement":
        source.nodes[4].op = "Div"
        roles[source.nodes[4].name] = ("Div", "", "root")
    elif defect == "matrix-output":
        source.outputs += ("matrix",)
        expected = "another consumer, direct output or lexical capture"
    elif defect == "matrix-capture":
        child = Graph("capture", (), ("matrix",), {}, {}, [])
        source.nodes.append(
            _node(
                206,
                "If",
                ("predicate",),
                ("capture",),
                {
                    "then_branch": Attribute("GRAPH", child),
                    "else_branch": Attribute("GRAPH", deepcopy(child)),
                },
            )
        )
        roles[source.nodes[-1].name] = ("If", "", "root")
        source.inputs += ("predicate",)
        source.types["predicate"] = ValueType(9, ())
        expected = "another consumer, direct output or lexical capture"
    elif defect == "terminal-binding":
        target.nodes[-1].inputs = ("other", "bias")
        expected = "fusion terminal consumers"
    with pytest.raises(FusionUnfinished, match=expected):
        expand_proven_fusions(source, target, roles, fp32_post_mul_contract=contract)


def test_actual_proto_nonunit_dispatch_and_exact_declared_rounding_capability():
    from dataclasses import replace
    import onnx
    from jax2onnx._deployment_fusion_bridge import (
        contains_fusions,
        _reviewed_post_mul_contract,
    )
    from jax2onnx.converter.source_deployment import (
        DEPLOYMENT_METADATA_KEY,
        DeploymentRequirements,
        deployment_requirements_to_json,
        ort_cpu_deployment_target,
    )

    node = onnx.helper.make_node(
        "FusedMatMul", ["a", "b"], ["out"], domain="com.microsoft", alpha=0.5
    )
    model = onnx.helper.make_model(onnx.helper.make_graph([node], "routing", [], []))
    assert contains_fusions(model)
    model.graph.node[0].attribute[0].f = 1.0
    assert not contains_fusions(model)
    assert _reviewed_post_mul_contract(model) is None
    for version, abi in (
        ("1.31.0", "cpython-312"),
        ("1.31.0", "cpython-311"),
        ("1.24.1", "cpython-311"),
    ):
        target = ort_cpu_deployment_target(version, abi)
        onnx.helper.set_model_props(
            model,
            {
                DEPLOYMENT_METADATA_KEY: deployment_requirements_to_json(
                    DeploymentRequirements(target, ())
                )
            },
        )
        assert _reviewed_post_mul_contract(model) == _FP32_POST_MUL_SINGLE_PANEL
        for unreviewed in (
            replace(target, runtime_build=target.runtime_build + ", unreviewed"),
            replace(target, cpu_features=("avx", "avx2", "sse2")),
        ):
            onnx.helper.set_model_props(
                model,
                {
                    DEPLOYMENT_METADATA_KEY: deployment_requirements_to_json(
                        DeploymentRequirements(unreviewed, ())
                    )
                },
            )
            assert _reviewed_post_mul_contract(model) is None


def _repeated_if_pair(shared_predicate=False):
    calls_source, calls_target = [], []
    roles = {
        "jax2onnx_src_000020": ("If", "", "root"),
        "jax2onnx_src_000021": ("Sub", "", "root"),
    }
    for call in range(2):
        source_branches, target_branches = {}, {}
        for branch in ("then_branch", "else_branch"):
            source, target, matrix_roles = _matrix_pair()
            roles.update(matrix_roles)
            for body in (source, target):
                body.inputs = ()
                body.types[f"data{call}"] = ValueType(1, (2, 2))
                for node in body.nodes:
                    node.inputs = tuple(
                        f"data{call}" if name == "x" else name for name in node.inputs
                    )
                # The two branches compute different actual functions.
                if branch == "else_branch":
                    body.initializers["bias"] = TensorData(
                        1, (1, 2), struct.pack("<2f", 7, 8)
                    )
            source_branches[branch] = Attribute("GRAPH", source)
            target_branches[branch] = Attribute("GRAPH", target)
        predicate = "predicate0" if shared_predicate else f"predicate{call}"
        calls_source.append(
            _node(20, "If", (predicate,), (f"call{call}",), source_branches)
        )
        calls_target.append(
            _node(20, "If", (predicate,), (f"call{call}",), target_branches)
        )
    types = {
        name: ValueType(1, (2, 2))
        for name in ("data0", "data1", "call0", "call1", "difference")
    }
    types.update({name: ValueType(9, ()) for name in ("predicate0", "predicate1")})
    inputs = ("data0", "predicate0", "data1", "predicate1")
    consumer = _node(21, "Sub", ("call0", "call1"), ("difference",))
    source = Graph(
        "same",
        inputs,
        ("call0", "call1", "difference"),
        dict(types),
        {},
        [*calls_source, consumer],
    )
    target = Graph(
        "same",
        inputs,
        source.outputs,
        dict(types),
        {},
        [*reversed(calls_target), deepcopy(consumer)],
    )
    return source, target, roles


@pytest.mark.parametrize("shared_predicate", [False, True])
def test_reordered_repeated_if_matches_exact_predicate_and_dynamic_capture(
    shared_predicate,
):
    source, target, roles = _repeated_if_pair(shared_predicate)
    original = deepcopy((source, target))
    virtual, kinds = expand_proven_fusions(source, target, roles)
    assert kinds == ("Gemm",) * 4
    assert all(
        node.op == "MatMul"
        for parent in virtual.nodes
        if parent.op == "If"
        for attribute in parent.attributes.values()
        for node in attribute.value.nodes[:1]
    )
    assert (source, target) == original


@pytest.mark.parametrize(
    "defect", ["capture", "predicate", "outputs", "consumer", "branches", "ambiguous"]
)
def test_repeated_if_routing_keeps_actual_caller_and_branch_proof(defect):
    source, target, roles = _repeated_if_pair()
    first = next(node for node in target.nodes if node.outputs == ("call0",))
    second = next(node for node in target.nodes if node.outputs == ("call1",))
    if defect == "capture":
        branch = first.attributes["then_branch"].value
        branch.nodes[0].inputs = ("data1", *branch.nodes[0].inputs[1:])
    elif defect == "predicate":
        first.inputs, second.inputs = second.inputs, first.inputs
    elif defect == "outputs":
        first.outputs, second.outputs = second.outputs, first.outputs
    elif defect == "consumer":
        target.nodes[-1].inputs = ("call0", "data1")
    elif defect == "branches":
        first.attributes["then_branch"], first.attributes["else_branch"] = (
            first.attributes["else_branch"],
            first.attributes["then_branch"],
        )
    else:
        # Equal discriminators do not authorize choosing/deduplicating a call.
        for body in (source, target):
            for node in body.nodes:
                if node.op != "If":
                    continue
                node.inputs = ("predicate0",)
                for attribute in node.attributes.values():
                    for item in attribute.value.nodes:
                        item.inputs = tuple(
                            "data0" if name == "data1" else name for name in item.inputs
                        )
    with pytest.raises(FusionUnfinished):
        expand_proven_fusions(source, target, roles)


def _slice_table_pair(blank_axes=False):
    def vector(values):
        return TensorData(
            7, (len(values),), struct.pack("<" + "q" * len(values), *values)
        )

    table = TensorData(1, (8, 4), bytes(range(128)))
    source, target, roles = _matrix_pair()
    source.inputs = target.inputs = ("x", "other")
    source.types.update(
        {name: ValueType(1, (2, 3, 4)) for name in ("x", "other", "first", "second")}
    )
    source.types.update({name: ValueType(1, (6, 4)) for name in ("flat", "mat", "z")})
    source.initializers.update(
        {
            "table": table,
            "starts": vector((0, 0)),
            "ends": vector((3, 4)),
            "axes": vector((0, 1)),
            "steps": vector((1, 1)),
            "row_shape": vector((1, 3, 4)),
            "expand_shape": vector((2, 3, 4)),
            "flat_shape": vector((6, 4)),
            "w": TensorData(1, (4, 4), bytes(range(64))),
            "bias": TensorData(1, (4,), bytes(16)),
        }
    )
    slice_inputs = (
        ("table", "starts", "ends", "", "steps")
        if blank_axes
        else ("table", "starts", "ends", "axes")
    )
    prefix = [
        _node(300, "Slice", slice_inputs, ("selected",)),
        _node(301, "Reshape", ("selected", "row_shape"), ("row",)),
        _node(302, "Expand", ("row", "expand_shape"), ("expanded",)),
        _node(303, "Identity", ("expanded",), ("coefficient",)),
        _node(304, "Mul", ("x", "coefficient"), ("first",)),
        _node(304, "Mul", ("other", "coefficient"), ("second",)),
        _node(305, "Reshape", ("first", "flat_shape"), ("flat",)),
    ]
    source.nodes[0].inputs = ("flat", "w")
    source.nodes[:0] = prefix
    source.outputs = ("z", "second", "table")
    target.types = dict(source.types)
    target.initializers = {
        "table": table,
        "folded": TensorData(1, (2, 3, 4), table.data[:48] * 2),
        "w": source.initializers["w"],
        "bias": source.initializers["bias"],
        "flat_shape": source.initializers["flat_shape"],
    }
    target.nodes[:0] = [
        _node(304, "Mul", ("x", "folded"), ("first",)),
        _node(304, "Mul", ("other", "folded"), ("second",)),
        deepcopy(prefix[-1]),
    ]
    target.nodes[-1].inputs = ("flat", "w", "bias")
    target.outputs = source.outputs
    roles.update({node.name: (node.op, node.domain, "root") for node in prefix})
    return source, target, roles


def test_slice_table_bytes_and_original_anchors_restore_at_each_bound_consumer():
    for blank_axes in (False, True):
        source, target, roles = _slice_table_pair(blank_axes)
        before = deepcopy((source, target))
        virtual, kinds = expand_proven_fusions(source, target, roles)
        assert kinds == ("Gemm",)
        assert (source, target) == before
        slices = [node for node in virtual.nodes if node.op == "Slice"]
        assert len(slices) == 2
        if blank_axes:
            assert all(node.inputs[3] == "" and node.inputs[4] for node in slices)
        else:
            assert all(len(node.inputs) == 4 and node.inputs[3] for node in slices)
        constants = {value.key for value in source.initializers.values()}
        view = View(virtual, (), Registry(roles), constants)
        multiplications = [node for node in virtual.nodes if node.op == "Mul"]
        assert [node.inputs[0] for node in multiplications] == ["x", "other"]
        assert multiplications[0].inputs[1] != multiplications[1].inputs[1]
        for multiplication in multiplications:
            coefficient = multiplication.inputs[1]
            assert view.constant(coefficient) == target.initializers["folded"]
            assert ("initializer", source.initializers["table"].key) in view.anchors(
                coefficient
            )
        assert virtual.initializers["folded"] == target.initializers["folded"]
        assert virtual.initializers["table"] == source.initializers["table"]
        assert virtual.outputs == target.outputs


@pytest.mark.parametrize(
    "case",
    [
        # Shape, starts, ends, optional axes/steps, output shape, exact source offsets.
        ((2, 3), (0, 1), (2, 3), None, None, (2, 2), (1, 2, 4, 5)),
        ((2, 3), (2, 1), (-4, -3), (1, 0), (-1, -1), (2, 3), (5, 4, 3, 2, 1, 0)),
        ((2, 3), (0, 1), (3, 2), (-1, -2), (2, 1), (1, 2), (3, 5)),
        (
            (4,),
            (9223372036854775807,),
            (-9223372036854775808,),
            None,
            (-1,),
            (4,),
            (3, 2, 1, 0),
        ),
        (
            (4,),
            (-9223372036854775808,),
            (-9223372036854775808,),
            None,
            (-1,),
            (1,),
            (0,),
        ),
        ((5,), (-3,), (-1,), None, None, (2,), (2, 3)),
        ((4,), (4,), (1,), None, None, (0,), ()),
        ((0, 3), (0,), (2147483647,), (0,), (-1,), (0, 3), ()),
    ],
)
def test_constant_slice_uses_exact_onnx_order_defaults_clipping_and_empty_extents(case):
    shape, starts, ends, axes, steps, expected_shape, offsets = case
    count = 1
    for size in shape:
        count *= size
    data = TensorData(1, shape, bytes(range(count * 4)))
    controls = {"data": data}
    inputs = ["data", "starts", "ends"]
    for name, values in (
        ("starts", starts),
        ("ends", ends),
        ("axes", axes),
        ("steps", steps),
    ):
        if values is not None:
            # One real INT32 route; other cases exercise exact INT64 endpoints.
            dtype = 6 if shape == (0, 3) else 7
            controls[name] = TensorData(
                dtype,
                (len(values),),
                struct.pack("<" + ("i" if dtype == 6 else "q") * len(values), *values),
            )
    if axes is not None or steps is not None:
        inputs.append("axes" if axes is not None else "")
    if steps is not None:
        inputs.append("steps")
    node = _node(310, "Slice", inputs, ("out",))
    graph = Graph("slice", (), ("out",), {}, controls, [node])
    before = deepcopy(graph)
    view = View(
        graph,
        (),
        Registry({node.name: ("Slice", "", "root")}),
        {value.key for value in controls.values()},
    )
    result = view.constant("out")
    assert result == TensorData(
        1,
        expected_shape,
        b"".join(data.data[index * 4 : (index + 1) * 4] for index in offsets),
    )
    assert view.constant("out") is result
    assert graph == before


@pytest.mark.parametrize(
    "defect",
    [
        "bytes",
        "order",
        "control",
        "dtype",
        "dynamic-data",
        "dynamic-control",
        "cycle-data",
        "cycle-control",
    ],
)
def test_constant_slice_unproved_storage_controls_and_cycles_cannot_restore(defect):
    source, target, roles = _slice_table_pair()
    if defect == "bytes":
        old = target.initializers["folded"]
        target.initializers["folded"] = TensorData(
            old.dtype, old.shape, old.data[:-4] + bytes(4)
        )
    elif defect == "order":
        source.initializers["axes"] = TensorData(7, (2,), struct.pack("<2q", 1, 0))
    elif defect == "control":
        source.initializers["ends"] = TensorData(7, (2,), struct.pack("<2q", 2, 4))
    elif defect == "dtype":
        source.initializers["axes"] = TensorData(6, (2,), struct.pack("<2i", 0, 1))
    elif defect == "dynamic-data":
        source.inputs += ("table",)
    elif defect == "dynamic-control":
        source.inputs += ("axes",)
    elif defect == "cycle-data":
        source.nodes[0].inputs = ("selected", *source.nodes[0].inputs[1:])
    else:
        source.nodes[0].inputs = ("table", "selected", *source.nodes[0].inputs[2:])
    if defect in {"dynamic-data", "dynamic-control"}:
        view = View(
            source,
            (),
            Registry(roles),
            {value.key for value in source.initializers.values()},
        )
        assert view.constant("selected") is None
    with pytest.raises(FusionUnfinished):
        expand_proven_fusions(source, target, roles)


def test_refresh_receives_current_private_graph_between_proved_restorations():
    source, target, roles = _batch_fused_pair()
    before = deepcopy((source, target))
    calls = []

    def refresh(current):
        operations = [node.op for node in current.nodes]
        if not calls:
            assert "FusedMatMul" not in operations and "MatMul" in operations
            assert "Gemm" in operations
            assert current.types["exp"] == ValueType(None, None)
        else:
            assert "Gemm" not in operations
            assert current.types["exp"] == ValueType(1, (2, 2, 3, 3))
        result = deepcopy(current)
        result.types["exp"] = ValueType(1, (2, 2, 3, 3))
        calls.append(operations)
        return result

    virtual, kinds = expand_proven_fusions(source, target, roles, refresh_types=refresh)
    assert len(calls) == 2 and kinds == ("FusedMatMul", "Gemm")
    assert virtual.types["exp"] == ValueType(1, (2, 2, 3, 3))
    assert (source, target) == before
    assert expand_proven_fusions(source, target, roles)[1] == kinds
    next(node for node in target.nodes if node.op == "FusedMatMul").attributes[
        "unproved"
    ] = Attribute("INT", 1)
    calls.clear()
    with pytest.raises(FusionUnfinished):
        expand_proven_fusions(source, target, roles, refresh_types=refresh)
    assert not calls


def _proto_refresh_two_fusions(wide):
    import onnx
    from jax2onnx.converter.source_deployment import (
        DEPLOYMENT_METADATA_KEY,
        DeploymentRequirements,
        deployment_requirements_to_json,
        ort_cpu_deployment_target,
    )

    helper, tensor = onnx.helper, onnx.TensorProto

    def node(tag, op, inputs, output, *, domain="", overload="", **attrs):
        return helper.make_node(
            op,
            inputs,
            [output],
            name=f"jax2onnx_src_{tag:06d}",
            domain=domain,
            overload=overload,
            **attrs,
        )

    controls = [
        helper.make_tensor("alpha", tensor.FLOAT, [], [0.5]),
        helper.make_tensor("reduce_axes", tensor.INT64, [1], [3]),
        helper.make_tensor("reshape_shape", tensor.INT64, [4], [2, 2, 3, 1]),
        helper.make_tensor("starts", tensor.INT64, [4], [0, 0, 0, 0]),
        helper.make_tensor("ends", tensor.INT64, [4], [2, 2, 3, 3]),
        helper.make_tensor("axes", tensor.INT64, [4], [0, 1, 2, 3]),
    ]
    first = [
        node(400, "Transpose", ["a"], "left", perm=[1, 2, 0, 3]),
        node(401, "Transpose", ["b"], "right", perm=[1, 2, 3, 0]),
        node(402, "MatMul", ["left", "right"], "matrix"),
        node(403, "Mul", ["matrix", "alpha"], "scaled"),
    ]
    path = []
    data = "scaled"
    if wide:
        path.append(
            node(
                404,
                "Cast",
                [data],
                "wide",
                to=tensor.DOUBLE,
                saturate=1,
                round_mode="up",
            )
        )
        data = "wide"
    path.extend(
        [
            node(405, "ReduceSum", [data, "reduce_axes"], "reduced", keepdims=0),
            node(406, "Reshape", ["reduced", "reshape_shape"], "denominator"),
            node(407, "Div", [data, "denominator"], "probabilities"),
        ]
    )
    data = "probabilities"
    if wide:
        path.append(
            node(
                408,
                "Cast",
                [data],
                "narrow",
                to=tensor.FLOAT,
                saturate=1,
                round_mode="up",
            )
        )
        data = "narrow"
    path.append(node(409, "Slice", [data, "starts", "ends", "axes"], "sliced"))
    last = [
        node(410, "Transpose", ["a"], "last_left", perm=[1, 2, 3, 0]),
        node(411, "Transpose", ["sliced"], "last_right", perm=[0, 1, 3, 2]),
        node(412, "MatMul", ["last_left", "last_right"], "out"),
    ]
    inputs = [
        helper.make_tensor_value_info(name, tensor.FLOAT, [3, 2, 2, 8])
        for name in ("a", "b")
    ]
    outputs = [helper.make_tensor_value_info("out", tensor.FLOAT, None)]
    original = helper.make_model(
        helper.make_graph(
            first + path + last, "two_fusions", inputs, outputs, initializer=controls
        ),
        opset_imports=[
            helper.make_opsetid("", 26),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    target = deepcopy(original)
    del target.graph.node[:]
    target.graph.node.extend(
        [
            node(
                402,
                "FusedMatMul",
                ["a", "b"],
                "scaled",
                domain="com.microsoft",
                alpha=0.5,
                transA=0,
                transB=1,
                transBatchA=1,
                transBatchB=1,
            ),
            *deepcopy(path),
            node(
                412,
                "FusedMatMul",
                ["a", "sliced"],
                "out",
                domain="com.microsoft",
                alpha=1.0,
                transA=1,
                transB=1,
                transBatchA=1,
                transBatchB=0,
            ),
        ]
    )
    target_requirement = DeploymentRequirements(
        ort_cpu_deployment_target("1.31.0", "cpython-312"), ()
    )
    helper.set_model_props(
        original,
        {DEPLOYMENT_METADATA_KEY: deployment_requirements_to_json(target_requirement)},
    )
    roles = {
        value.name: (value.op_type, value.domain, "root")
        for value in original.graph.node
    }
    return original, target, roles


@pytest.mark.parametrize("wide", [False, True])
def test_actual_proto_standard_refresh_reaches_second_fusion_and_preserves_casts(
    wide, monkeypatch
):
    import onnx
    from jax2onnx._deployment_fusion_bridge import prove_deployment_fusions

    original, target, roles = _proto_refresh_two_fusions(wide)
    before = original.SerializeToString(), target.SerializeToString()
    actual_infer = onnx.shape_inference.infer_shapes
    calls = []

    def infer(model, **options):
        assert options == {"strict_mode": True, "data_prop": True}
        calls.append(deepcopy(model))
        return actual_infer(model, **options)

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    virtual, kinds = prove_deployment_fusions(original, target, roles)
    assert kinds == ("FusedMatMul", "FusedMatMul") and len(calls) == 6
    refresh = calls[4]
    assert sum(node.op_type == "FusedMatMul" for node in refresh.graph.node) == 1
    assert any(node.op_type == "MatMul" for node in refresh.graph.node)
    assert any(
        node.op_type == "Mul" and "jax2onnx_src_000403" in node.name
        for node in refresh.graph.node
    )
    assert not any(node.op_type == "FusedMatMul" for node in calls[5].graph.node)
    assert not any(node.op_type == "FusedMatMul" for node in virtual.graph.node)
    types = {
        value.name: value
        for value in (
            *virtual.graph.input,
            *virtual.graph.output,
            *virtual.graph.value_info,
        )
    }
    sliced = types["sliced"].type.tensor_type
    assert sliced.elem_type == onnx.TensorProto.FLOAT
    assert sliced.HasField("shape") and [dim.dim_value for dim in sliced.shape.dim] == [
        2,
        2,
        3,
        3,
    ]
    casts = [
        node.SerializeToString()
        for node in virtual.graph.node
        if node.op_type == "Cast"
    ]
    assert casts == [
        node.SerializeToString() for node in target.graph.node if node.op_type == "Cast"
    ]
    assert (original.SerializeToString(), target.SerializeToString()) == before


def _proto_refresh_lexical_fixture():
    import onnx
    from jax2onnx._deployment_fusion_bridge import _graph

    helper, tensor = onnx.helper, onnx.TensorProto
    inputs = [
        helper.make_tensor_value_info("predicate", tensor.BOOL, []),
        helper.make_tensor_value_info("a", tensor.FLOAT, [2, 2]),
        helper.make_tensor_value_info("b", tensor.FLOAT, [3, 3]),
        helper.make_tensor_value_info("named", tensor.FLOAT, ["B", 2]),
        helper.make_tensor_value_info("captured", tensor.FLOAT, None),
    ]
    branches = [
        helper.make_graph(
            [
                helper.make_node("Identity", [argument], ["same"], name="same_display"),
                helper.make_node("Identity", ["captured"], ["unused"], name="capture"),
            ],
            "same_graph_display",
            [],
            [helper.make_tensor_value_info("same", tensor.FLOAT, None)],
        )
        for argument in ("a", "b")
    ]
    controls = [
        helper.make_tensor(name, tensor.INT64, [2], values, raw=False)
        for name, values in [("starts", [0, 0]), ("ends", [2, 2]), ("steps", [1, 1])]
    ]
    nodes = [
        helper.make_node(
            "LeakyRelu", ["a"], ["activated"], name="activation", alpha=0.0
        ),
        helper.make_node(
            "Cast",
            ["a"],
            ["cast"],
            name="cast",
            to=tensor.FLOAT,
            saturate=1,
            round_mode="up",
        ),
        helper.make_node(
            "Slice", ["a", "starts", "ends", "", "steps"], ["slice"], name="slice"
        ),
        helper.make_node(
            "If",
            ["predicate"],
            ["out"],
            name="controller",
            then_branch=branches[0],
            else_branch=branches[1],
        ),
    ]
    owner = helper.make_graph(
        nodes,
        "lexical",
        inputs,
        [helper.make_tensor_value_info("out", tensor.FLOAT, None)],
        initializer=controls,
        value_info=[helper.make_tensor_value_info("cast", tensor.FLOAT, [2, 2])],
    )
    model = helper.make_model(owner, opset_imports=[helper.make_opsetid("", 26)])
    graph = _graph(owner, {"": 26})
    return graph, model


@pytest.mark.parametrize(
    "defect",
    [
        "operator",
        "ordered-port",
        "optional-slot",
        "float-bits",
        "cast-attribute",
        "constant-control",
        "imports",
        "model-envelope",
        "lexical-child",
        "dtype",
        "rank",
        "dimension",
        "symbol",
    ],
)
def test_actual_proto_refresh_rejects_structure_bits_and_known_type_changes(
    defect, monkeypatch
):
    import onnx
    from jax2onnx._deployment_fusion_bridge import _refresh_virtual_types

    graph, envelope = _proto_refresh_lexical_fixture()
    before = deepcopy(graph), envelope.SerializeToString()

    def infer(model, **options):
        assert options == {"strict_mode": True, "data_prop": True}
        returned = deepcopy(model)
        if defect == "operator":
            returned.graph.node[0].op_type = "Identity"
        elif defect == "ordered-port":
            returned.graph.node[2].input[1], returned.graph.node[2].input[2] = (
                returned.graph.node[2].input[2],
                returned.graph.node[2].input[1],
            )
        elif defect == "optional-slot":
            del returned.graph.node[2].input[3]
        elif defect == "float-bits":
            returned.graph.node[0].attribute[0].f = -0.0
        elif defect == "cast-attribute":
            next(
                item
                for item in returned.graph.node[1].attribute
                if item.name == "round_mode"
            ).s = b"down"
        elif defect == "constant-control":
            returned.graph.initializer[0].int64_data[0] = 1
        elif defect == "imports":
            returned.opset_import[0].version = 25
        elif defect == "model-envelope":
            returned.producer_name = "changed"
        elif defect == "lexical-child":
            returned.graph.node[-1].attribute[0].g.node[0].input[0] = "named"
        else:
            value = returned.graph.input[3 if defect == "symbol" else 1]
            kind = value.type.tensor_type
            if defect == "dtype":
                kind.elem_type = onnx.TensorProto.DOUBLE
            elif defect == "rank":
                kind.shape.ClearField("dim")
                kind.shape.SetInParent()
            elif defect == "dimension":
                kind.shape.dim[0].dim_value = 4
            else:
                kind.shape.dim[0].dim_param = "C"
        return returned

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    with pytest.raises(FusionUnfinished):
        _refresh_virtual_types(graph, envelope)
    assert graph == before[0] and envelope.SerializeToString() == before[1]


def test_actual_proto_refresh_lexical_types_keep_distinct_local_names_and_known_metadata(
    monkeypatch,
):
    import onnx
    from jax2onnx._deployment_fusion_bridge import _refresh_virtual_types

    graph, envelope = _proto_refresh_lexical_fixture()
    before = deepcopy(graph), envelope.SerializeToString()
    result = _refresh_virtual_types(graph, envelope)
    branches = result.nodes[-1].attributes
    assert branches["then_branch"].value.types["same"] == ValueType(1, (2, 2))
    assert branches["else_branch"].value.types["same"] == ValueType(1, (3, 3))

    def omit(model, **options):
        assert options == {"strict_mode": True, "data_prop": True}
        returned = deepcopy(model)
        del returned.graph.value_info[:]
        branch = next(
            attribute.g
            for attribute in returned.graph.node[-1].attribute
            if attribute.name == "then_branch"
        )
        branch.value_info.append(
            onnx.helper.make_tensor_value_info(
                "captured", onnx.TensorProto.FLOAT, [4, 2]
            )
        )
        return returned

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", omit)
    result = _refresh_virtual_types(graph, envelope)
    assert result.types["cast"] == ValueType(1, (2, 2))
    assert result.types["captured"] == ValueType(1, (4, 2))
    assert result.nodes[-1].attributes["then_branch"].value.types[
        "captured"
    ] == ValueType(1, (4, 2))
    assert graph == before[0] and envelope.SerializeToString() == before[1]


def test_actual_proto_refresh_inference_exception_has_no_partial_success(monkeypatch):
    import onnx
    from jax2onnx._deployment_fusion_bridge import _refresh_virtual_types

    graph, envelope = _proto_refresh_lexical_fixture()
    before = deepcopy(graph), envelope.SerializeToString()

    def fail(model, **options):
        assert options == {"strict_mode": True, "data_prop": True}
        model.producer_name = "partial"
        raise ValueError("strict inference failed")

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", fail)
    with pytest.raises(ValueError, match="strict inference failed"):
        _refresh_virtual_types(graph, envelope)
    assert graph == before[0] and envelope.SerializeToString() == before[1]


def _proto_inference_symbol_fixture(symbol=None):
    import onnx
    from jax2onnx._deployment_fusion_bridge import (
        _TargetInferenceSymbols,
        _dimension_symbols,
        _graph,
    )

    helper, tensor = onnx.helper, onnx.TensorProto
    owner = helper.make_graph(
        [
            helper.make_node(
                "MatMul", ["left", "right"], ["out"], domain="", overload=""
            )
        ],
        "fresh_variables",
        [
            helper.make_tensor_value_info("left", tensor.FLOAT, [symbol, 4]),
            helper.make_tensor_value_info("right", tensor.FLOAT, [4, symbol]),
            helper.make_tensor_value_info("later", tensor.FLOAT, [None, 4]),
        ],
        [helper.make_tensor_value_info("out", tensor.FLOAT, [symbol, symbol])],
    )
    envelope = helper.make_model(owner, opset_imports=[helper.make_opsetid("", 26)])
    return (
        _graph(owner, {"": 26}),
        envelope,
        _TargetInferenceSymbols(_dimension_symbols(envelope)),
    )


def _mock_inference_symbol_shapes(monkeypatch, shapes, dtype=1):
    import onnx
    from jax2onnx._deployment_fusion_bridge import _write_type

    def infer(model, **options):
        assert options == {"strict_mode": True, "data_prop": True}
        returned = deepcopy(model)
        for value in (*returned.graph.input, *returned.graph.output):
            if value.name in shapes:
                if shapes[value.name] is None:
                    value.type.tensor_type.ClearField("shape")
                else:
                    _write_type(value, ValueType(dtype, shapes[value.name]))
        return returned

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)


def _shared_symbol_shapes(size):
    return {"left": (size, 4), "right": (4, size), "out": (size, size)}


def test_target_fresh_inference_variable_specializes_every_use_before_matrix_proof(
    monkeypatch,
):
    from jax2onnx._deployment_fusion_bridge import _refresh_virtual_types
    from jax2onnx._deployment_fusions import matrix_dimensions

    graph, envelope, provenance = _proto_inference_symbol_fixture()
    before = deepcopy(graph), envelope.SerializeToString()
    _mock_inference_symbol_shapes(
        monkeypatch, _shared_symbol_shapes("arbitrary_extent")
    )
    current = _refresh_virtual_types(graph, envelope, provenance)
    current = _refresh_virtual_types(current, envelope, provenance)
    assert provenance._next_generation == 1
    _mock_inference_symbol_shapes(monkeypatch, _shared_symbol_shapes(3))
    result = _refresh_virtual_types(current, envelope, provenance)
    assert result.types["left"].shape == (3, 4)
    assert result.types["right"].shape == (4, 3)
    assert result.types["out"].shape == (3, 3)
    matrix_dimensions(result.types["left"], result.types["right"], result.types["out"])
    assert not provenance._active
    assert graph == before[0] and envelope.SerializeToString() == before[1]


@pytest.mark.parametrize("symbol", ["unk__0", "declared_extent"])
@pytest.mark.parametrize("replacement", [3, "different_symbol"])
def test_declared_symbol_spelling_never_grants_inference_specialization(
    symbol, replacement, monkeypatch
):
    from jax2onnx._deployment_fusion_bridge import _refresh_virtual_types

    graph, envelope, provenance = _proto_inference_symbol_fixture(symbol)
    before = deepcopy(graph), envelope.SerializeToString(), dict(provenance.__dict__)
    _mock_inference_symbol_shapes(monkeypatch, _shared_symbol_shapes(replacement))
    with pytest.raises(FusionUnfinished):
        _refresh_virtual_types(graph, envelope, provenance)
    assert graph == before[0] and envelope.SerializeToString() == before[1]
    assert provenance.__dict__ == before[2]


@pytest.mark.parametrize(
    "defect", ["conflict", "partial", "omitted", "rename", "dtype", "rank", "known-dim"]
)
def test_fresh_inference_specialization_requires_complete_consistent_types(
    defect, monkeypatch
):
    from jax2onnx._deployment_fusion_bridge import _refresh_virtual_types

    graph, envelope, provenance = _proto_inference_symbol_fixture()
    symbol = "not_a_reserved_prefix"
    _mock_inference_symbol_shapes(monkeypatch, _shared_symbol_shapes(symbol))
    current = _refresh_virtual_types(graph, envelope, provenance)
    changes = _shared_symbol_shapes(3)
    dtype = 1
    if defect == "conflict":
        changes["right"] = (4, 5)
    elif defect == "partial":
        changes["right"] = (4, symbol)
    elif defect == "omitted":
        changes["out"] = None
    elif defect == "rename":
        changes["right"] = (4, "renamed")
    elif defect == "dtype":
        dtype = 11
    elif defect == "rank":
        changes["left"] = (3,)
    else:
        changes["left"] = (3, 5)
    _mock_inference_symbol_shapes(monkeypatch, changes, dtype)
    before = deepcopy(current), envelope.SerializeToString(), dict(provenance.__dict__)
    with pytest.raises(FusionUnfinished):
        _refresh_virtual_types(current, envelope, provenance)
    assert current == before[0] and envelope.SerializeToString() == before[1]
    assert provenance.__dict__ == before[2]


@pytest.mark.parametrize("replacement", ["same-symbol", "concrete", "missing"])
def test_source_replaced_or_lost_occurrence_cannot_inherit_target_provenance(
    replacement, monkeypatch
):
    from jax2onnx._deployment_fusion_bridge import _refresh_virtual_types

    graph, envelope, provenance = _proto_inference_symbol_fixture()
    symbol = "shared_inference_extent"
    _mock_inference_symbol_shapes(monkeypatch, _shared_symbol_shapes(symbol))
    current = _refresh_virtual_types(graph, envelope, provenance)
    if replacement == "missing":
        del current.types["left"]
    else:
        current.types["left"] = ValueType(
            1, (symbol if replacement == "same-symbol" else 3, 4)
        )
    changes = _shared_symbol_shapes(4)
    if replacement == "concrete":
        changes["left"] = (3, 4)
    _mock_inference_symbol_shapes(monkeypatch, changes)
    before = deepcopy(current), envelope.SerializeToString(), dict(provenance.__dict__)
    with pytest.raises(FusionUnfinished):
        _refresh_virtual_types(current, envelope, provenance)
    assert current == before[0] and envelope.SerializeToString() == before[1]
    assert provenance.__dict__ == before[2]


def test_retired_inference_spelling_starts_new_generation_without_old_binding(
    monkeypatch,
):
    from jax2onnx._deployment_fusion_bridge import _refresh_virtual_types

    graph, envelope, provenance = _proto_inference_symbol_fixture()
    symbol = "reusable_extent"
    _mock_inference_symbol_shapes(monkeypatch, _shared_symbol_shapes(symbol))
    current = _refresh_virtual_types(graph, envelope, provenance)
    first_generation = provenance._active[symbol][0]
    _mock_inference_symbol_shapes(monkeypatch, _shared_symbol_shapes(3))
    current = _refresh_virtual_types(current, envelope, provenance)
    _mock_inference_symbol_shapes(monkeypatch, {"later": (symbol, 4)})
    current = _refresh_virtual_types(current, envelope, provenance)
    assert provenance._active[symbol][0] != first_generation
    assert current.types["left"].shape == (3, 4)
    _mock_inference_symbol_shapes(monkeypatch, {"later": (7, 4)})
    result = _refresh_virtual_types(current, envelope, provenance)
    assert result.types["later"].shape == (7, 4)
    assert result.types["left"].shape == (3, 4)


def test_same_round_new_symbol_use_does_not_receive_retired_binding(monkeypatch):
    from jax2onnx._deployment_fusion_bridge import _refresh_virtual_types

    graph, envelope, provenance = _proto_inference_symbol_fixture()
    symbol = "same_round_extent"
    _mock_inference_symbol_shapes(monkeypatch, _shared_symbol_shapes(symbol))
    current = _refresh_virtual_types(graph, envelope, provenance)
    _mock_inference_symbol_shapes(
        monkeypatch, {**_shared_symbol_shapes(3), "later": (symbol, 4)}
    )
    current = _refresh_virtual_types(current, envelope, provenance)
    assert current.types["left"].shape == (3, 4)
    assert current.types["later"].shape == (symbol, 4)
    _mock_inference_symbol_shapes(monkeypatch, {"later": (7, 4)})
    with pytest.raises(FusionUnfinished):
        _refresh_virtual_types(current, envelope, provenance)


def test_nested_equal_display_names_keep_inference_occurrence_ownership(monkeypatch):
    import onnx
    from jax2onnx._deployment_fusion_bridge import (
        _TargetInferenceSymbols,
        _dimension_symbols,
        _refresh_virtual_types,
        _write_type,
    )

    graph, envelope = _proto_refresh_lexical_fixture()
    provenance = _TargetInferenceSymbols(_dimension_symbols(envelope))
    sizes = ["first_extent", "second_extent"]

    def infer(model, **options):
        assert options == {"strict_mode": True, "data_prop": True}
        returned = deepcopy(model)
        attributes = {item.name: item.g for item in returned.graph.node[-1].attribute}
        for index, name in enumerate(("then_branch", "else_branch")):
            _write_type(
                attributes[name].output[0], ValueType(1, (sizes[index], index + 2))
            )
        return returned

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    current = _refresh_virtual_types(graph, envelope, provenance)
    # The root node index changes between callbacks, while both lexical child
    # graphs and their actual ValueType objects retain their target lineage.
    extra = onnx.helper.make_node(
        "Identity", ["a"], ["additional"], name="same_display", domain="", overload=""
    )
    current.nodes.insert(
        0, Node(extra.name, "", "Identity", "", ("a",), ("additional",), {}, extra)
    )
    sizes[:] = [3, 5]
    result = _refresh_virtual_types(current, envelope, provenance)
    branches = result.nodes[-1].attributes
    assert branches["then_branch"].value.types["same"].shape == (3, 2)
    assert branches["else_branch"].value.types["same"].shape == (5, 3)


@pytest.mark.parametrize("location", ["function", "nested"])
def test_original_function_and_nested_declarations_protect_target_symbol_text(
    location, monkeypatch
):
    import onnx
    from jax2onnx._deployment_fusion_bridge import (
        _TargetInferenceSymbols,
        _dimension_symbols,
        _refresh_virtual_types,
        _write_type,
    )

    graph, envelope, _ = _proto_inference_symbol_fixture()
    symbol = "unk__0"
    if location == "function":
        original = deepcopy(envelope)
        function = onnx.helper.make_function(
            "local",
            "declared",
            ["x"],
            ["y"],
            [onnx.helper.make_node("Identity", ["x"], ["y"])],
            [onnx.helper.make_opsetid("", 26)],
        )
        function.value_info.append(
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [symbol, 4])
        )
        original.functions.append(function)
    else:
        _, original = _proto_refresh_lexical_fixture()
        branch = next(
            item.g
            for item in original.graph.node[-1].attribute
            if item.name == "then_branch"
        )
        _write_type(branch.output[0], ValueType(1, (symbol, 4)))
    original_bytes = original.SerializeToString()
    provenance = _TargetInferenceSymbols(_dimension_symbols(original, envelope))
    assert symbol in provenance._protected
    _mock_inference_symbol_shapes(monkeypatch, _shared_symbol_shapes(symbol))
    current = _refresh_virtual_types(graph, envelope, provenance)
    assert symbol not in provenance._active
    _mock_inference_symbol_shapes(monkeypatch, _shared_symbol_shapes(3))
    with pytest.raises(FusionUnfinished):
        _refresh_virtual_types(current, envelope, provenance)
    assert original.SerializeToString() == original_bytes


@pytest.mark.parametrize("pending", [False, True])
def test_engine_initializes_pristine_copy_before_constants_and_existing_refresh(
    pending, monkeypatch
):
    import jax2onnx._deployment_fusions as engine

    if pending:
        source, target, roles = _matrix_pair()
    else:
        source = Graph(
            "pristine",
            ("x",),
            ("out",),
            {"x": ValueType(1, (2, 2)), "out": ValueType(1, (2, 2))},
            {},
            [_node(900, "Identity", ("x",), ("out",))],
        )
        target = deepcopy(source)
        roles = {source.nodes[0].name: ("Identity", "", "root")}
    before = deepcopy(source), deepcopy(target)
    events = []
    actual_constants = engine.restore_constant_origins

    def initialize(current):
        assert current is not target and current == target
        events.append("initialize")

    def constants(*args):
        assert events == ["initialize"]
        events.append("constants")
        return actual_constants(*args)

    def refresh(current):
        events.append("refresh")
        return current

    monkeypatch.setattr(engine, "restore_constant_origins", constants)
    _, kinds = engine.expand_proven_fusions(
        source,
        target,
        roles,
        initialize_types=initialize,
        refresh_types=refresh,
    )
    assert events == ["initialize", "constants"] + (["refresh"] if pending else [])
    assert kinds == (("Gemm",) if pending else ())
    assert source == before[0] and target == before[1]


def _proto_initial_refresh_fusions():
    import onnx

    original, target, roles = _proto_refresh_two_fusions(True)
    helper, tensor = onnx.helper, onnx.TensorProto
    target.graph.value_info.append(
        helper.make_tensor_value_info("reduced", tensor.DOUBLE, [None, 2, 3])
    )
    for model in (original, target):
        model.graph.node.extend(
            [
                helper.make_node(
                    "Add",
                    ["reduced", "reduced"],
                    ["initial_broadcast"],
                    name="jax2onnx_src_000414",
                    domain="",
                    overload="",
                ),
                helper.make_node(
                    "Identity",
                    ["initial_broadcast"],
                    ["initial_alias"],
                    name="jax2onnx_src_000415",
                    domain="",
                    overload="",
                ),
            ]
        )
    roles.update(
        {
            "jax2onnx_src_000414": ("Add", "", "root"),
            "jax2onnx_src_000415": ("Identity", "", "root"),
        }
    )
    return original, target, roles


@pytest.mark.parametrize("rename_symbol", [False, True])
def test_actual_initial_target_inference_bootstraps_before_first_fusion_refresh(
    rename_symbol, monkeypatch
):
    import onnx
    from jax2onnx._deployment_fusion_bridge import _type, prove_deployment_fusions

    original, target, roles = _proto_initial_refresh_fusions()
    before = original.SerializeToString(), target.SerializeToString()
    actual_infer = onnx.shape_inference.infer_shapes
    calls, initial_symbols = [], []

    def infer(model, **options):
        assert options == {"strict_mode": True, "data_prop": True}
        calls.append((model, model.SerializeToString()))
        inferred = actual_infer(model, **options)
        if len(calls) == 4:
            infos = {value.name: value for value in inferred.graph.value_info}
            symbol = _type(infos["initial_broadcast"]).shape[0]
            assert isinstance(symbol, str) and symbol
            assert _type(infos["initial_alias"]).shape[0] == symbol
            initial_symbols.append(symbol)
            if rename_symbol:
                # Controlled spelling variant, after observing a real standard
                # inference-created variable. The False case returns the exact
                # unchanged standard inference output.
                for value in (
                    *inferred.graph.input,
                    *inferred.graph.output,
                    *inferred.graph.value_info,
                ):
                    for dimension in value.type.tensor_type.shape.dim:
                        if dimension.dim_param == symbol:
                            dimension.dim_param = "initial_arbitrary_extent"
        return inferred

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    virtual, kinds = prove_deployment_fusions(original, target, roles)
    assert kinds == ("FusedMatMul", "FusedMatMul") and len(calls) == 6
    assert len(initial_symbols) == 1
    infos = {value.name: value for value in virtual.graph.value_info}
    assert _type(infos["initial_broadcast"]).shape == (2, 2, 3)
    assert _type(infos["initial_alias"]).shape == (2, 2, 3)
    assert [
        node.SerializeToString()
        for node in virtual.graph.node
        if node.op_type == "Cast"
    ] == [
        node.SerializeToString() for node in target.graph.node if node.op_type == "Cast"
    ]
    assert all(model.SerializeToString() == saved for model, saved in calls)
    assert (original.SerializeToString(), target.SerializeToString()) == before


def _verified_initial_symbol_fixture(symbol="initial_extent"):
    from jax2onnx._deployment_fusion_bridge import _initial_target_types, _write_type

    _, before, _ = _proto_inference_symbol_fixture()
    inferred = deepcopy(before)
    shapes = _shared_symbol_shapes(symbol)
    for value in (*inferred.graph.input, *inferred.graph.output):
        if value.name in shapes:
            _write_type(value, ValueType(1, shapes[value.name]))
    graph, provenance = _initial_target_types(before, inferred, {"": 26}, set())
    current = deepcopy(graph)
    provenance.initialize(current)
    return current, before, provenance


@pytest.mark.parametrize("symbol", ["unk__0", "caller_extent"])
def test_initial_setup_itself_protects_local_declarations_with_empty_external_set(
    symbol, monkeypatch
):
    from jax2onnx._deployment_fusion_bridge import (
        _initial_target_types,
        _refresh_virtual_types,
    )

    _, before, _ = _proto_inference_symbol_fixture(symbol)
    graph, provenance = _initial_target_types(before, deepcopy(before), {"": 26}, set())
    current = deepcopy(graph)
    provenance.initialize(current)
    assert symbol in provenance._protected and symbol not in provenance._active
    _mock_inference_symbol_shapes(monkeypatch, _shared_symbol_shapes(3))
    with pytest.raises(FusionUnfinished):
        _refresh_virtual_types(current, before, provenance)


@pytest.mark.parametrize(
    "defect", ["dtype", "rank", "dimension", "symbol", "structure"]
)
def test_initial_target_guard_rejects_changes_to_declared_types_or_structure(defect):
    from jax2onnx._deployment_fusion_bridge import _initial_target_types, _write_type

    _, before, _ = _proto_inference_symbol_fixture("declared")
    inferred = deepcopy(before)
    if defect == "structure":
        inferred.producer_name = "changed"
    else:
        known = {
            "dtype": ValueType(11, ("declared", 4)),
            "rank": ValueType(1, ("declared",)),
            "dimension": ValueType(1, ("declared", 5)),
            "symbol": ValueType(1, ("renamed", 4)),
        }[defect]
        _write_type(inferred.graph.input[0], known)
    saved = before.SerializeToString(), inferred.SerializeToString()
    with pytest.raises(FusionUnfinished):
        _initial_target_types(before, inferred, {"": 26}, set())
    assert (before.SerializeToString(), inferred.SerializeToString()) == saved


def test_inlined_preinference_symbol_is_protected_before_initial_target_merge(
    monkeypatch,
):
    import onnx
    from jax2onnx._deployment_fusion_bridge import _write_type, prove_deployment_fusions

    _, original, _ = _proto_inference_symbol_fixture()
    target = deepcopy(original)
    before = original.SerializeToString(), target.SerializeToString()
    actual_inline = onnx.inliner.inline_local_functions
    actual_infer = onnx.shape_inference.infer_shapes
    inline_calls, infer_calls = [], []

    def inline(model):
        returned = actual_inline(model)
        inline_calls.append(returned)
        if len(inline_calls) == 2:
            _write_type(returned.graph.input[0], ValueType(1, ("inline_declared", 4)))
        return returned

    def infer(model, **options):
        assert options == {"strict_mode": True, "data_prop": True}
        infer_calls.append(model)
        returned = actual_infer(model, **options)
        if len(infer_calls) == 4:
            _write_type(returned.graph.input[0], ValueType(1, (3, 4)))
        return returned

    monkeypatch.setattr(onnx.inliner, "inline_local_functions", inline)
    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    with pytest.raises(FusionUnfinished):
        prove_deployment_fusions(original, target, {})
    assert len(inline_calls) == 2 and len(infer_calls) == 4
    assert (original.SerializeToString(), target.SerializeToString()) == before


def test_initial_bootstrap_requires_verified_detached_single_use_permission():
    from jax2onnx._deployment_fusion_bridge import (
        _TargetInferenceSymbols,
        _initial_target_types,
    )

    graph, envelope, _ = _proto_inference_symbol_fixture()
    with pytest.raises(FusionUnfinished):
        _TargetInferenceSymbols(set()).initialize(deepcopy(graph))
    verified, provenance = _initial_target_types(
        envelope, deepcopy(envelope), {"": 26}, set()
    )
    state = dict(provenance.__dict__)
    with pytest.raises(FusionUnfinished):
        provenance.initialize(verified)
    assert provenance.__dict__ == state
    current = deepcopy(verified)
    provenance.initialize(current)
    with pytest.raises(FusionUnfinished):
        provenance.initialize(current)


@pytest.mark.parametrize(
    "transition", ["source-replaced", "lost", "conflict", "retired"]
)
def test_initial_generations_retain_postrefresh_lineage_and_lifetime_guards(
    transition, monkeypatch
):
    from jax2onnx._deployment_fusion_bridge import _refresh_virtual_types

    current, envelope, provenance = _verified_initial_symbol_fixture()
    changes = _shared_symbol_shapes(3)
    if transition == "retired":
        first = provenance._active["initial_extent"][0]
        _mock_inference_symbol_shapes(monkeypatch, changes)
        current = _refresh_virtual_types(current, envelope, provenance)
        _mock_inference_symbol_shapes(monkeypatch, {"later": ("initial_extent", 4)})
        current = _refresh_virtual_types(current, envelope, provenance)
        assert provenance._active["initial_extent"][0] != first
        _mock_inference_symbol_shapes(monkeypatch, {"later": (7, 4)})
        result = _refresh_virtual_types(current, envelope, provenance)
        assert result.types["left"].shape == (3, 4)
        assert result.types["later"].shape == (7, 4)
        return
    if transition == "source-replaced":
        current.types["left"] = ValueType(1, ("initial_extent", 4))
    elif transition == "lost":
        del current.types["left"]
    else:
        changes["right"] = (4, 5)
    _mock_inference_symbol_shapes(monkeypatch, changes)
    state = dict(provenance.__dict__)
    saved = deepcopy(current), envelope.SerializeToString()
    with pytest.raises(FusionUnfinished):
        _refresh_virtual_types(current, envelope, provenance)
    assert provenance.__dict__ == state
    assert current == saved[0] and envelope.SerializeToString() == saved[1]


@pytest.mark.parametrize("tamper", ["payload", "type", "port"])
def test_initial_bootstrap_rejects_changed_copy_without_consuming_permission(tamper):
    from jax2onnx._deployment_fusion_bridge import _initial_target_types

    _, envelope, _ = _proto_inference_symbol_fixture()
    verified, provenance = _initial_target_types(
        envelope, deepcopy(envelope), {"": 26}, set()
    )
    current = deepcopy(verified)
    if tamper == "payload":
        current.nodes[0].payload.doc_string = (
            "changed only in ignored descriptor payload"
        )
        assert current == verified
    elif tamper == "type":
        current.types["left"] = ValueType(1, (3, 4))
    else:
        current.nodes[0].inputs = ("right", "left")
    state = dict(provenance.__dict__)
    with pytest.raises(FusionUnfinished):
        provenance.initialize(current)
    assert provenance.__dict__ == state
    provenance.initialize(deepcopy(verified))


def _unit_fused_gemm_pair(defaults):
    types = {
        "a": ValueType(1, (3, 2, 32)),
        "b": ValueType(1, (3, 2, 1)),
        "left": ValueType(1, (3, 32, 2)),
        "batch": ValueType(1, (3, 32, 1)),
        **{
            name: ValueType(1, (3, 32))
            for name in ("flat", "shifted", "normalized", "gemm_input", "cast_input")
        },
        "matrix": ValueType(1, (3, 64)),
        "out": ValueType(1, (3, 64)),
    }
    controls = {
        "shape": TensorData(7, (2,), struct.pack("<2q", -1, 32)),
        "epsilon": TensorData(1, (), struct.pack("<f", 0.00001)),
        "gamma": TensorData(1, (32,), struct.pack("<f", 1.0) * 32),
        "weights": TensorData(1, (32, 64), bytes(4 * 32 * 64)),
        "bias": TensorData(1, (64,), bytes(4 * 64)),
    }
    prefix = [
        _node(
            1000, "Transpose", ("a",), ("left",), {"perm": Attribute("INTS", (0, 2, 1))}
        ),
        _node(1001, "MatMul", ("left", "b"), ("batch",)),
    ]
    middle = [
        _node(1002, "Reshape", ("batch", "shape"), ("flat",)),
        _node(1003, "Add", ("flat", "epsilon"), ("shifted",)),
        _node(1004, "Div", ("flat", "shifted"), ("normalized",)),
        _node(1005, "Mul", ("normalized", "gamma"), ("gemm_input",)),
        _node(
            1006,
            "Cast",
            ("gemm_input",),
            ("cast_input",),
            {
                "to": Attribute("INT", 1),
                "saturate": Attribute("INT", 1),
                "round_mode": Attribute("STRING", b"up"),
            },
        ),
    ]
    suffix = [
        _node(1007, "MatMul", ("cast_input", "weights"), ("matrix",)),
        _node(1008, "Add", ("matrix", "bias"), ("out",)),
    ]
    attributes = {"transA": Attribute("INT", 1)}
    if defaults == "explicit":
        attributes.update(
            {
                "alpha": Attribute("FLOAT", 1.0),
                "transB": Attribute("INT", 0),
                "transBatchA": Attribute("INT", 0),
                "transBatchB": Attribute("INT", 0),
            }
        )
    source = Graph(
        "unit_source",
        ("a", "b"),
        ("out",),
        dict(types),
        dict(controls),
        prefix + middle + suffix,
    )
    target = Graph(
        "unit_target",
        source.inputs,
        source.outputs,
        {**types, "batch": ValueType(None, None), "flat": ValueType(None, None)},
        dict(controls),
        [
            _node(
                1001, "FusedMatMul", ("a", "b"), ("batch",), attributes, "com.microsoft"
            ),
            *deepcopy(middle),
            _node(
                1007,
                "Gemm",
                ("cast_input", "weights", "bias"),
                ("out",),
                {
                    "alpha": Attribute("FLOAT", 1.0),
                    "beta": Attribute("FLOAT", 1.0),
                    "transA": Attribute("INT", 0),
                    "transB": Attribute("INT", 0),
                },
            ),
        ],
    )
    roles = {node.name: (node.op, node.domain, "root") for node in source.nodes}
    return source, target, roles


@pytest.mark.parametrize("defaults", ["explicit", "omitted"])
def test_unit_fused_matmul_restores_before_later_gemm_with_existing_math(defaults):
    from jax2onnx._deployment_fusions import fused_matmul_needs_proof

    source, target, roles = _unit_fused_gemm_pair(defaults)
    assert not fused_matmul_needs_proof(target.nodes[0])
    before = deepcopy((source, target))
    refreshes = []

    def refresh(current):
        refreshes.append(tuple(node.op for node in current.nodes))
        assert current.types["batch"] == ValueType(1, (3, 32, 1))
        assert current.types["flat"] == ValueType(1, (3, 32))
        return current

    virtual, kinds = expand_proven_fusions(source, target, roles, refresh_types=refresh)
    assert kinds == ("FusedMatMul", "Gemm") and len(refreshes) == 2
    assert "FusedMatMul" not in refreshes[0] and "Gemm" in refreshes[0]
    assert "Gemm" not in refreshes[1]
    assert virtual.types["out"] == ValueType(1, (3, 64))
    assert not any(node.op in {"FusedMatMul", "Gemm"} for node in virtual.nodes)
    assert (source, target) == before


@pytest.mark.parametrize(
    "defect",
    [
        "transpose",
        "operands",
        "nonboolean",
        "flag-schema",
        "attribute-schema",
        "dtype",
        "dimension",
        "ambiguous",
    ],
)
def test_newly_selected_unit_fusion_keeps_exact_rejection_before_refresh(defect):
    from jax2onnx._deployment_fusions import fused_matmul_needs_proof

    source, target, roles = _unit_fused_gemm_pair("explicit")
    fused = target.nodes[0]
    if defect == "transpose":
        fused.attributes["transA"] = Attribute("INT", 0)
    elif defect == "operands":
        fused.inputs = ("b", "a")
    elif defect == "nonboolean":
        fused.attributes["transA"] = Attribute("INT", 2)
    elif defect == "flag-schema":
        fused.attributes["transA"] = Attribute("FLOAT", 1.0)
    elif defect == "attribute-schema":
        fused.attributes["unproved"] = Attribute("INT", 1)
    elif defect == "dtype":
        target.types["a"] = ValueType(11, (3, 2, 32))
    elif defect == "dimension":
        target.types["batch"] = ValueType(1, (3, 31, 1))
    else:
        source.nodes.insert(
            2, _node(1001, "MatMul", ("left", "b"), ("duplicate_batch",))
        )
        source.types["duplicate_batch"] = ValueType(1, (3, 32, 1))
    assert not fused_matmul_needs_proof(fused)
    before = deepcopy((source, target))
    refreshes = []

    def refresh(current):
        refreshes.append(current)
        return current

    with pytest.raises(FusionUnfinished):
        expand_proven_fusions(source, target, roles, refresh_types=refresh)
    assert not refreshes and (source, target) == before


def _proto_unit_fused_gemm_pair(defaults):
    import onnx

    source, target, roles = _unit_fused_gemm_pair(defaults)
    helper = onnx.helper

    def node(value):
        return helper.make_node(
            value.op,
            list(value.inputs),
            list(value.outputs),
            name=value.name,
            domain=value.domain,
            overload=value.overload,
            **{name: attribute.value for name, attribute in value.attributes.items()},
        )

    def model(graph, full_annotations):
        return helper.make_model(
            helper.make_graph(
                [node(value) for value in graph.nodes],
                graph.name,
                [
                    helper.make_tensor_value_info(name, 1, source.types[name].shape)
                    for name in graph.inputs
                ],
                [
                    helper.make_tensor_value_info(
                        "out", 1, (3, 64) if full_annotations else None
                    )
                ],
                initializer=[
                    helper.make_tensor(
                        name, value.dtype, value.shape, value.data, raw=True
                    )
                    for name, value in graph.initializers.items()
                ],
                value_info=(
                    [
                        helper.make_tensor_value_info(name, value.dtype, value.shape)
                        for name, value in source.types.items()
                        if name not in (*graph.inputs, *graph.outputs)
                    ]
                    if full_annotations
                    else []
                ),
            ),
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

    return model(source, True), model(target, False), roles


@pytest.mark.parametrize("defaults", ["explicit", "omitted"])
def test_actual_unit_fused_matmul_refresh_supplies_shapes_before_gemm(
    defaults, monkeypatch
):
    import onnx
    from jax2onnx._deployment_fusion_bridge import (
        _type,
        contains_fusions,
        prove_deployment_fusions,
    )

    original, target, roles = _proto_unit_fused_gemm_pair(defaults)
    before = original.SerializeToString(), target.SerializeToString()
    assert contains_fusions(target)
    unit_only = onnx.helper.make_model(
        onnx.helper.make_graph(
            [deepcopy(target.graph.node[0])],
            "unit_only_routing",
            list(target.graph.input),
            [onnx.helper.make_tensor_value_info("batch", 1, None)],
        )
    )
    assert not contains_fusions(unit_only)
    actual_infer = onnx.shape_inference.infer_shapes
    calls, gaps, refreshed = [], [], []

    def infer(model, **options):
        assert options == {"strict_mode": True, "data_prop": True}
        calls.append((model, model.SerializeToString()))
        returned = actual_infer(model, **options)
        infos = {
            value.name: _type(value)
            for value in (*returned.graph.output, *returned.graph.value_info)
        }
        if len(calls) == 4:
            assert "batch" not in infos or infos["batch"].shape is None
            assert "cast_input" not in infos or infos["cast_input"].shape != (3, 32)
            gaps.append(True)
        if len(calls) == 5:
            assert not any(
                node.op_type == "FusedMatMul" for node in returned.graph.node
            )
            assert any(node.op_type == "Gemm" for node in returned.graph.node)
            assert infos["flat"].shape == (3, 32)
            assert infos["cast_input"].shape == (3, 32)
            assert infos["out"].shape == (3, 64)
            refreshed.append(True)
        return returned

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
    virtual, kinds = prove_deployment_fusions(original, target, roles)
    assert kinds == ("FusedMatMul", "Gemm") and len(calls) == 6
    assert gaps == refreshed == [True]
    assert not any(
        node.op_type in {"FusedMatMul", "Gemm"} for node in virtual.graph.node
    )
    assert _type(virtual.graph.output[0]).shape == (3, 64)
    assert [
        node.SerializeToString()
        for node in virtual.graph.node
        if node.op_type == "Cast"
    ] == [
        node.SerializeToString() for node in target.graph.node if node.op_type == "Cast"
    ]
    assert all(model.SerializeToString() == saved for model, saved in calls)
    assert (original.SerializeToString(), target.SerializeToString()) == before


def _integer_branch_pair(count=4):
    controls = {
        "zero": TensorData(6, (), struct.pack("<i", 0)),
        "extent": TensorData(6, (), struct.pack("<i", 8)),
        "upper": TensorData(6, (), struct.pack("<i", 7)),
        "flat_shape": TensorData(7, (2,), struct.pack("<2q", 3, 2)),
        "view_shape": TensorData(7, (3,), struct.pack("<3q", 3, 2, 1)),
        "axis2": TensorData(7, (1,), struct.pack("<q", 2)),
        "axis3": TensorData(7, (1,), struct.pack("<q", 3)),
        "fill": TensorData(1, (), bytes(4)),
        "fw": TensorData(1, (4, 4), bytes(64)),
        "fb": TensorData(1, (4,), bytes(16)),
    }
    types = {
        "indices": ValueType(7, (3, 2)),
        "cast": ValueType(6, (3, 2)),
        "fx": ValueType(1, (3, 4)),
        "fmat": ValueType(1, (3, 4)),
        "fout": ValueType(1, (3, 4)),
    }
    cast = _node(1999, "Cast", ("indices",), ("cast",), {"to": Attribute("INT", 6)})
    source_nodes, target_nodes, endpoints, outputs = [cast], [deepcopy(cast)], [], []
    for branch in range(count):
        prefix, tag = f"b{branch}_", 2000 + 20 * branch

        def n(offset, op, args, result, attrs=None):
            return _node(tag + offset, op, args, (prefix + result,), attrs)

        triple = [
            n(0, "Less", ("cast", "zero"), "negative"),
            n(1, "Add", ("cast", "extent"), "shifted"),
            n(2, "Where", (prefix + "negative", prefix + "shifted", "cast"), "normal"),
        ]
        views = [
            n(3, "Reshape", (prefix + "normal", "view_shape"), "view"),
            n(4, "Expand", (prefix + "view", "view_shape"), "expanded"),
            n(5, "Reshape", (prefix + "expanded", "flat_shape"), "flat"),
        ]
        tail = [
            n(6, "Less", (prefix + "flat", "zero"), "low"),
            n(7, "Greater", (prefix + "flat", "upper"), "high"),
            n(8, "Or", (prefix + "low", prefix + "high"), "invalid"),
            n(9, "Max", (prefix + "flat", "zero"), "positive"),
            n(10, "Min", (prefix + "positive", "upper"), "clamped"),
            n(11, "Unsqueeze", (prefix + "invalid", "axis2"), "mask3"),
        ]
        rank4 = branch % 2 == 0
        if rank4:
            tail.append(n(12, "Unsqueeze", (prefix + "mask3", "axis3"), "mask4"))
        table_shape = (8, 4, 4) if rank4 else (8, 4)
        controls[prefix + "table"] = TensorData(
            1,
            table_shape,
            struct.pack("<f", float(branch + 1)) * (128 if rank4 else 32),
        )
        result_shape = (3, 2, 4, 4) if rank4 else (3, 2, 4)
        gather = n(
            13,
            "Gather",
            (prefix + "table", prefix + "clamped"),
            "gather",
            {"axis": Attribute("INT", 0)},
        )
        mask = prefix + ("mask4" if rank4 else "mask3")
        where = n(14, "Where", (mask, "fill", prefix + "gather"), "out")
        source_nodes.extend(triple + views + tail + [gather, where])
        for suffix in ("shifted", "normal", "flat", "positive", "clamped"):
            types[prefix + suffix] = ValueType(6, (3, 2))
        for suffix in ("negative", "low", "high", "invalid"):
            types[prefix + suffix] = ValueType(9, (3, 2))
        types[prefix + "view"] = types[prefix + "expanded"] = ValueType(6, (3, 2, 1))
        types[prefix + "mask3"] = ValueType(9, (3, 2, 1))
        if rank4:
            types[prefix + "mask4"] = ValueType(9, (3, 2, 1, 1))
        types[prefix + "gather"] = types[prefix + "out"] = ValueType(1, result_shape)
        if branch == 0:
            shared_view = n(3, "Reshape", (prefix + "normal", "flat_shape"), "flat")
            target_nodes.extend(deepcopy(triple + [shared_view] + tail))
        actual_gather, actual_where = deepcopy(gather), deepcopy(where)
        actual_gather.inputs = (prefix + "table", "b0_clamped")
        actual_where.inputs = (
            "b0_mask4" if rank4 else "b0_mask3",
            "fill",
            prefix + "gather",
        )
        endpoints.extend((actual_gather, actual_where))
        outputs.append(prefix + "out")
    matrix = _node(1900, "MatMul", ("fx", "fw"), ("fmat",))
    add = _node(1901, "Add", ("fmat", "fb"), ("fout",))
    source_nodes.extend((matrix, add))
    target_nodes.extend(endpoints)
    target_nodes.append(
        _node(
            1900,
            "Gemm",
            ("fx", "fw", "fb"),
            ("fout",),
            {
                "alpha": Attribute("FLOAT", 1.0),
                "beta": Attribute("FLOAT", 1.0),
                "transA": Attribute("INT", 0),
                "transB": Attribute("INT", 0),
            },
        )
    )
    source = Graph(
        "source_integer",
        ("indices", "fx"),
        (*outputs, "fout"),
        dict(types),
        dict(controls),
        source_nodes,
    )
    target = Graph(
        "target_integer",
        source.inputs,
        source.outputs,
        dict(types),
        dict(controls),
        target_nodes,
    )
    roles = {node.name: (node.op, node.domain, "root") for node in source_nodes}
    return source, target, roles


def _assert_integer_restoration(source, target, virtual, roles):
    registry = Registry(roles)
    left, right = View(source, (), registry, set()), View(virtual, (), registry, set())
    expected = sorted(
        (registry.tag(node), port) for node, port in left.consumers["cast"]
    )
    assert (
        sorted((registry.tag(node), port) for node, port in right.consumers["cast"])
        == expected
    )
    assert [node for node in virtual.nodes if node.op == "Cast"] == [target.nodes[0]]
    actual = {
        registry.tag(node): node
        for node in virtual.nodes
        if node.op in {"Gather", "Where"}
        and (node.op == "Gather" or node.outputs[0].endswith("_out"))
    }
    for node in target.nodes:
        if (
            node.op not in {"Gather", "Where"}
            or node.op == "Where"
            and not node.outputs[0].endswith("_out")
        ):
            continue
        preserved = actual[registry.tag(node)]
        port = 1 if node.op == "Gather" else 0
        assert (
            preserved.attributes == node.attributes
            and preserved.outputs == node.outputs
        )
        assert all(
            a == b
            for index, (a, b) in enumerate(
                zip(preserved.inputs, node.inputs, strict=True)
            )
            if index != port
        )
    live = set(virtual.outputs)
    for node in reversed(virtual.nodes):
        if any(output in live for output in node.outputs):
            live.update(node.inputs)
    added = [
        node
        for node in virtual.nodes
        if node.name.startswith("__jax2onnx_virtual_integer_")
    ]
    assert added and all(node.outputs[0] in live for node in added)
    assert all(
        node.op != "Cast" and right.value_type(node.outputs[0]).dtype in {6, 7, 9}
        for node in added
    )
    defined = set(virtual.inputs) | set(virtual.initializers)
    for node in virtual.nodes:
        assert all(name in defined for name in node.inputs)
        defined.update(node.outputs)


@pytest.mark.parametrize("count", [2, 4])
def test_shared_integer_cast_branches_restore_both_live_endpoint_families(count):
    source, target, roles = _integer_branch_pair(count)
    before = deepcopy((source, target))
    virtual, kinds = expand_proven_fusions(source, target, roles)
    assert kinds == ("Gemm",)
    _assert_integer_restoration(source, target, virtual, roles)
    assert (source, target) == before


@pytest.mark.parametrize(
    "defect",
    [
        "constant",
        "constant-shape",
        "operand-order",
        "where-arm",
        "comparison",
        "attribute",
        "cast-root",
        "cast-to",
        "cast-saturate",
        "cast-round",
        "cast-reference",
        "cast-unknown",
        "cast-float",
        "ambiguous-cast",
        "ambiguous-endpoint",
        "swapped-gather",
        "missing-gather",
        "extra-gather",
        "missing-mask",
        "extra-mask",
        "wrong-mask",
        "mask-axis",
        "data",
        "fill",
        "axis",
        "expand",
        "permutation",
        "reshape",
        "type",
        "shape",
        "output",
        "capture",
        "external",
        "float",
        "cast",
    ],
)
def test_integer_branch_proof_rejects_hostile_closed_cones_atomically(defect):
    from jax2onnx._deployment_fusions import all_views, restore_integer_branches

    source, target, roles = _integer_branch_pair()
    by_output = {node.outputs[0]: node for node in target.nodes}
    if defect == "constant":
        target.initializers["extent"] = TensorData(6, (), struct.pack("<i", 9))
    elif defect == "constant-shape":
        target.initializers["extent"] = TensorData(6, (1,), struct.pack("<i", 8))
    elif defect == "operand-order":
        by_output["b0_shifted"].inputs = ("extent", "cast")
    elif defect == "where-arm":
        by_output["b0_normal"].inputs = ("b0_negative", "cast", "b0_shifted")
    elif defect == "comparison":
        by_output["b0_negative"].op = "Greater"
    elif defect == "attribute":
        by_output["b0_shifted"].attributes["unproved"] = Attribute("INT", 1)
    elif defect == "cast-root":
        target.inputs += ("other_indices",)
        target.types["other_indices"] = ValueType(7, (3, 2))
        target.nodes[0].inputs = ("other_indices",)
    elif defect == "cast-to":
        target.nodes[0].attributes["to"] = Attribute("INT", 7)
    elif defect == "cast-saturate":
        target.nodes[0].attributes["saturate"] = Attribute("INT", 0)
    elif defect == "cast-round":
        target.nodes[0].attributes["round_mode"] = Attribute("STRING", b"down")
    elif defect == "cast-reference":
        target.nodes[0].attributes["to"] = Attribute("INT", 6, "external_to")
    elif defect == "cast-unknown":
        target.nodes[0].attributes["unknown"] = Attribute("INT", 1)
    elif defect == "cast-float":
        source.types["indices"] = target.types["indices"] = ValueType(1, (3, 2))
    elif defect == "ambiguous-cast":
        duplicate = deepcopy(source.nodes[0])
        duplicate.outputs = ("other_cast",)
        source.nodes.append(duplicate)
        source.types["other_cast"] = source.types["cast"]
    elif defect == "ambiguous-endpoint":
        duplicate = deepcopy(by_output["b1_gather"])
        duplicate.outputs = ("duplicate",)
        target.nodes.append(duplicate)
        target.types["duplicate"] = target.types["b1_gather"]
    elif defect == "swapped-gather":
        a, b = by_output["b1_gather"], by_output["b3_gather"]
        a.name, b.name = b.name, a.name
    elif defect == "missing-gather":
        target.nodes.remove(by_output["b1_gather"])
        by_output["b1_out"].inputs = ("b0_mask3", "fill", "b3_gather")
    elif defect == "extra-gather":
        node = _node(
            2990,
            "Gather",
            ("b1_table", "b0_clamped"),
            ("extra",),
            {"axis": Attribute("INT", 0)},
        )
        target.nodes.append(node)
        roles[node.name] = (node.op, node.domain, "root")
        target.types["extra"] = target.types["b1_gather"]
    elif defect == "missing-mask":
        target.nodes.remove(by_output["b1_out"])
        target.outputs = tuple(name for name in target.outputs if name != "b1_out")
    elif defect == "extra-mask":
        node = _node(2993, "Where", ("b0_mask3", "fill", "b1_gather"), ("extra_mask",))
        target.nodes.append(node)
        roles[node.name] = (node.op, node.domain, "root")
        target.types["extra_mask"] = target.types["b1_out"]
    elif defect == "wrong-mask":
        by_output["b1_out"].inputs = ("b0_mask4", "fill", "b1_gather")
    elif defect == "mask-axis":
        target.initializers["axis2"] = TensorData(7, (1,), struct.pack("<q", 1))
    elif defect == "data":
        by_output["b1_gather"].inputs = ("b3_table", "b0_clamped")
    elif defect == "fill":
        target.initializers["fill"] = TensorData(1, (), struct.pack("<f", 1))
    elif defect == "axis":
        by_output["b1_gather"].attributes["axis"] = Attribute("INT", 1)
    elif defect == "expand":
        node = next(node for node in source.nodes if node.op == "Expand")
        source.initializers["replication_shape"] = TensorData(
            7, (3,), struct.pack("<3q", 3, 2, 2)
        )
        node.inputs = ("b0_view", "replication_shape")
        source.types["b0_expanded"] = ValueType(6, (3, 2, 2))
    elif defect == "permutation":
        node = next(node for node in source.nodes if node.op == "Expand")
        node.op, node.inputs = "Transpose", (node.inputs[0],)
        node.attributes = {"perm": Attribute("INTS", (1, 0, 2))}
    elif defect == "reshape":
        target.initializers["flat_shape"] = TensorData(
            7, (2,), struct.pack("<2q", 2, 3)
        )
    elif defect == "type":
        target.types["b0_clamped"] = ValueType(7, (3, 2))
    elif defect == "shape":
        target.types["b0_clamped"] = ValueType(6, (2, 3))
    elif defect == "output":
        target.outputs += ("b0_clamped",)
    elif defect == "capture":
        child = Graph("capture", (), ("b0_clamped",), {}, {}, [])
        node = _node(
            2991,
            "If",
            ("b0_negative",),
            ("captured",),
            {
                "then_branch": Attribute("GRAPH", child),
                "else_branch": Attribute("GRAPH", deepcopy(child)),
            },
        )
        target.nodes.append(node)
        roles[node.name] = (node.op, node.domain, "root")
    else:
        op = {"external": "Neg", "float": "Mul", "cast": "Cast"}[defect]
        node = _node(
            2992,
            op,
            ("b0_clamped",),
            ("unexpected",),
            {"to": Attribute("INT", 6)} if op == "Cast" else {},
        )
        target.nodes.append(node)
        roles[node.name] = (node.op, node.domain, "root")
        target.types["unexpected"] = ValueType(1 if defect == "float" else 6, (3, 2))
    registry = Registry(roles)
    original = deepcopy((source, target))
    sources = all_views(View(source, (), registry, set()))
    with pytest.raises(FusionUnfinished):
        restore_integer_branches(sources, target, registry, set())
    assert (source, target) == original


def _double_integer_branch_pair():
    source, target, roles = _integer_branch_pair(2)
    second_source, second_target, second_roles = _integer_branch_pair(2)

    def append(owner, other):
        rename = {
            name: "second_" + name
            for name in (*other.inputs, *other.initializers, *other.types)
        }
        owner.inputs += tuple(rename[name] for name in other.inputs)
        owner.outputs += tuple(rename[name] for name in other.outputs)
        owner.initializers.update(
            {rename[name]: value for name, value in other.initializers.items()}
        )
        owner.types.update({rename[name]: value for name, value in other.types.items()})
        for node in other.nodes:
            node.name = f"jax2onnx_src_{int(node.name[-6:]) + 2000:06d}"
            node.inputs = tuple(rename[name] for name in node.inputs)
            node.outputs = tuple(rename[name] for name in node.outputs)
            owner.nodes.append(node)

    append(source, second_source)
    append(target, second_target)
    roles.update(
        {
            f"jax2onnx_src_{int(name[-6:]) + 2000:06d}": value
            for name, value in second_roles.items()
        }
    )
    return source, target, roles


def test_integer_branch_all_root_plans_fail_without_partial_first_commit():
    from jax2onnx._deployment_fusions import all_views, restore_integer_branches

    source, target, roles = _double_integer_branch_pair()
    target.initializers["second_extent"] = TensorData(6, (), struct.pack("<i", 9))
    before = deepcopy((source, target))
    registry = Registry(roles)
    with pytest.raises(FusionUnfinished):
        restore_integer_branches(
            all_views(View(source, (), registry, set())), target, registry, set()
        )
    assert (source, target) == before


def test_integer_branch_multiple_valid_roots_commit_together():
    from jax2onnx._deployment_fusions import all_views, restore_integer_branches

    source, target, roles = _double_integer_branch_pair()
    before = deepcopy(source)
    registry = Registry(roles)
    restore_integer_branches(
        all_views(View(source, (), registry, set())), target, registry, set()
    )
    view = View(target, (), registry, set())
    for root in ("cast", "second_cast"):
        assert len(view.consumers[root]) == 6
    assert source == before


def test_unrelated_single_integer_cast_keeps_existing_no_cse_behavior():
    from jax2onnx._deployment_fusions import all_views, restore_integer_branches

    source, target, roles = _integer_branch_pair(1)
    target.types["b0_flat"] = ValueType(None, None)
    before = deepcopy(target)
    registry = Registry(roles)
    restore_integer_branches(
        all_views(View(source, (), registry, set())), target, registry, set()
    )
    assert target == before


def test_integer_root_explicit_default_schema_preserves_the_actual_cast():
    source, target, roles = _integer_branch_pair(2)
    target.nodes[0].attributes.update(
        {"saturate": Attribute("INT", 1), "round_mode": Attribute("STRING", b"up")}
    )
    before = deepcopy((source, target))
    virtual, kinds = expand_proven_fusions(source, target, roles)
    assert kinds == ("Gemm",)
    _assert_integer_restoration(source, target, virtual, roles)
    assert (source, target) == before


def test_unchanged_same_role_inlined_cast_calls_do_not_need_new_shape_proofs():
    from jax2onnx._deployment_fusions import all_views, restore_integer_branches

    source, _, roles = _double_integer_branch_pair()
    for node in source.nodes:
        if node.outputs[0].startswith("second_"):
            node.name = f"jax2onnx_src_{int(node.name[-6:]) - 2000:06d}"
    source.types["indices"] = source.types["second_indices"] = ValueType(7, None)
    target = deepcopy(source)
    before = deepcopy((source, target))
    registry = Registry(roles)
    restore_integer_branches(
        all_views(View(source, (), registry, set())), target, registry, set()
    )
    assert (source, target) == before


def test_integer_target_constant_closure_and_unrelated_untagged_node_are_preserved():
    source, target, roles = _integer_branch_pair(2)
    axes = target.initializers.pop("axis2")
    control = Node(
        "optimizer_axes",
        "",
        "Constant",
        "",
        (),
        ("axis2",),
        {"value": Attribute("TENSOR", axes)},
    )
    target.types["axis2"] = ValueType(7, (1,))
    position = next(
        index for index, node in enumerate(target.nodes) if node.op == "Unsqueeze"
    )
    target.nodes.insert(position, control)
    unrelated = Node(
        "optimizer_unused_shape",
        "",
        "Constant",
        "",
        (),
        ("unrelated_shape",),
        {"value": Attribute("TENSOR", TensorData(7, (1,), struct.pack("<q", 17)))},
    )
    target.nodes.append(unrelated)
    target.types["unrelated_shape"] = ValueType(7, (1,))
    before = deepcopy((source, target))
    virtual, kinds = expand_proven_fusions(source, target, roles)
    assert kinds == ("Gemm",)
    _assert_integer_restoration(source, target, virtual, roles)
    assert any(
        node.name == unrelated.name and node == unrelated for node in virtual.nodes
    )
    assert any(
        node.op == "Constant" and node.name.startswith("__jax2onnx_virtual_integer_")
        for node in virtual.nodes
    )
    assert (source, target) == before


def test_actual_integer_cse_restoration_passes_unchanged_final_boundary():
    import onnx
    from jax2onnx._deployment_fusion_bridge import (
        _graph,
        contains_fusions,
        prove_deployment_fusions,
    )
    from jax2onnx.converter.source_deployment import (
        DeploymentRequirements,
        ort_cpu_deployment_target,
    )
    from jax2onnx.precision import _deployment_graph_boundary

    source, target, roles = _integer_branch_pair()
    helper = onnx.helper

    def model(graph):
        return helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        node.op,
                        list(node.inputs),
                        list(node.outputs),
                        name=node.name,
                        domain=node.domain,
                        overload=node.overload,
                        **{key: item.value for key, item in node.attributes.items()},
                    )
                    for node in graph.nodes
                ],
                graph.name,
                [
                    helper.make_tensor_value_info(
                        name, graph.types[name].dtype, graph.types[name].shape
                    )
                    for name in graph.inputs
                ],
                [
                    helper.make_tensor_value_info(
                        name, graph.types[name].dtype, graph.types[name].shape
                    )
                    for name in graph.outputs
                ],
                initializer=[
                    helper.make_tensor(
                        name, value.dtype, value.shape, value.data, raw=True
                    )
                    for name, value in graph.initializers.items()
                ],
            ),
            opset_imports=[helper.make_opsetid("", 26)],
        )

    original, optimized = model(source), model(target)
    optimized.graph.node[0].attribute.extend(
        (
            helper.make_attribute("saturate", 1),
            helper.make_attribute("round_mode", "up"),
        )
    )
    before = original.SerializeToString(), optimized.SerializeToString()
    assert contains_fusions(optimized)
    virtual, kinds = prove_deployment_fusions(original, optimized, roles)
    assert kinds == ("Gemm",)
    inferred_source = onnx.shape_inference.infer_shapes(
        original, strict_mode=True, data_prop=True
    )
    inferred_target = onnx.shape_inference.infer_shapes(
        optimized, strict_mode=True, data_prop=True
    )
    _assert_integer_restoration(
        _graph(inferred_source.graph, {"": 26}),
        _graph(inferred_target.graph, {"": 26}),
        _graph(virtual.graph, {"": 26}),
        roles,
    )
    assert [
        node.SerializeToString()
        for node in virtual.graph.node
        if node.op_type == "Cast"
    ] == [
        node.SerializeToString()
        for node in optimized.graph.node
        if node.op_type == "Cast"
    ]
    requirements = DeploymentRequirements(
        ort_cpu_deployment_target("1.31.0", "cpython-312"), ()
    )
    assert (
        _deployment_graph_boundary(original, optimized, requirements)
        == "matmul-gemm-quickgelu-and-original-casts-v2"
    )
    assert (original.SerializeToString(), optimized.SerializeToString()) == before
