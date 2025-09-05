# --- tiny fakes to avoid pulling real onnx_ir in unit tests ---

# import the functions to test
from jax2onnx.converter2.ir_optimizations import (
    _is_elem,
    _get_perm_attr,
    _perms_compose_identity,
    _has_input_name_or_obj,
    _count_consumers,
    _find_next_consumer_idx,
)


class V:
    def __init__(self, name=None):
        self.name = name


class N:
    def __init__(self, op, inputs=(), outputs=(), attributes=()):
        self.op_type = op
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.attributes = list(attributes)


class Attr:
    def __init__(self, name, ints):
        self.name = name
        self.ints = list(ints)


def test_is_elem_lower_and_mixed():
    assert _is_elem("Relu")
    assert _is_elem("relu")
    assert _is_elem("Cast")
    assert _is_elem("castlike")
    assert not _is_elem("AveragePool")


def test_get_perm_attr_and_identity():
    t1 = N("Transpose", attributes=[Attr("perm", [0, 3, 1, 2])])
    t2 = N("Transpose", attributes=[Attr("perm", [0, 2, 3, 1])])
    p1 = _get_perm_attr(t1)
    p2 = _get_perm_attr(t2)
    assert p1 == [0, 3, 1, 2] and p2 == [0, 2, 3, 1]
    assert _perms_compose_identity(p1, p2)


def test_match_by_name_or_obj():
    a = V("a")
    b = V("b")
    n = N("Relu", inputs=[a])
    assert _has_input_name_or_obj(n, "a", None)
    assert _has_input_name_or_obj(n, None, a)
    assert not _has_input_name_or_obj(n, "b", None)
    assert not _has_input_name_or_obj(n, None, b)


def test_consumer_scan():
    v = V("x")
    nodes = [N("Transpose", outputs=[v]), N("Something"), N("Relu", inputs=[v])]
    assert _find_next_consumer_idx(nodes, 0, "x", v) == 2
    assert _count_consumers(nodes, "x", v) == 1
