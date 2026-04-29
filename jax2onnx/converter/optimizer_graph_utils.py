# jax2onnx/converter/optimizer_graph_utils.py

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple, TypeAlias, Union, cast

import onnx_ir as ir

NodeList: TypeAlias = List[ir.Node]
NodeSeq: TypeAlias = Sequence[ir.Node]
ValueList: TypeAlias = List[ir.Value]
ValueSeq: TypeAlias = Sequence[ir.Value]


def _v_name(v: Union[ir.Value, str, None]) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v or None
    if isinstance(v, ir.Value):
        name = v.name
        return name or None
    return None


def _value_identity(
    value_or_name: Union[ir.Value, str, None],
) -> Tuple[Optional[ir.Value], Optional[str]]:
    if value_or_name is None:
        return None, None
    if isinstance(value_or_name, ir.Value):
        return value_or_name, _v_name(value_or_name)
    if isinstance(value_or_name, str):
        return None, value_or_name or None
    return None, None


def _node_outputs(n: ir.Node) -> ValueList:
    return list(cast(ValueSeq, n.outputs))


def _node_output(n: ir.Node) -> Optional[ir.Value]:
    outs = _node_outputs(n)
    return outs[0] if outs else None


def _node_inputs(n: ir.Node) -> ValueList:
    return list(cast(ValueSeq, n.inputs))


def _set_node_inputs(n: ir.Node, new_ins: Sequence[ir.Value]) -> None:
    for idx, val in enumerate(new_ins):
        n.replace_input_with(idx, val)


def _has_input_name_or_obj(
    node: ir.Node, name: Optional[str], obj: Optional[ir.Value]
) -> bool:
    """
    Return True if 'node' has an input that matches either the given name
    (by .name on Value or string equality) or the given object identity.
    """
    if not isinstance(node, ir.Node):
        return False

    ins = _node_inputs(node)
    if obj is not None:
        for iv in ins:
            if iv is obj:
                return True

    ref_value, ref_name = _value_identity(obj)
    target_name = name or ref_name

    for iv in ins:
        if ref_value is not None and iv is ref_value:
            return True
        if target_name:
            ivn = _v_name(iv)
            if ivn == target_name:
                return True
    return False


def _consumer_nodes(
    nodes: Sequence[ir.Node], value_or_name: Union[ir.Value, str, None]
) -> List[ir.Node]:
    """
    Return consumer nodes for a value, preferring IR APIs with name-based fallback.
    """
    if value_or_name is None:
        return []
    current_node_ids: Set[int] = {id(node) for node in nodes}
    if isinstance(value_or_name, ir.Value):
        consumers = value_or_name.consumers()
        if consumers:
            try:
                if all(isinstance(c, ir.Node) for c in consumers):
                    filtered = [c for c in consumers if id(c) in current_node_ids]
                    if filtered:
                        return list(filtered)
            except Exception:
                # Fall back to name-based scan below.
                pass

    ref_value, ref_name = _value_identity(value_or_name)
    target_name = ref_name
    if target_name is None and isinstance(value_or_name, ir.Value):
        target_name = _v_name(value_or_name)

    obj_arg = ref_value
    if obj_arg is None and isinstance(value_or_name, ir.Value):
        obj_arg = value_or_name

    found: List[ir.Node] = []
    for node in nodes:
        if _has_input_name_or_obj(node, target_name, obj_arg):
            found.append(node)
    return found


def _producer_node(
    nodes: Sequence[ir.Node], value_or_name: Union[ir.Value, str, None]
) -> Optional[ir.Node]:
    """
    Return the producer node for a value/name, preferring IR APIs with fallback scans.
    """
    if value_or_name is None:
        return None
    current_node_ids: Set[int] = {id(node) for node in nodes}
    if isinstance(value_or_name, ir.Value):
        prod = value_or_name.producer()
        if isinstance(prod, ir.Node) and id(prod) in current_node_ids:
            return prod

    ref_value, ref_name = _value_identity(value_or_name)
    target_name = ref_name
    if target_name is None and isinstance(value_or_name, ir.Value):
        target_name = _v_name(value_or_name)

    for node in nodes:
        for ov in _node_outputs(node):
            if ref_value is not None and ov is ref_value:
                return node
            if target_name and _v_name(ov) == target_name:
                return node
    return None
