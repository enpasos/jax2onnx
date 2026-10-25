# jax2onnx/plugins/_source_structure.py

"""Finite plugin-owned contracts for direct SOURCE elementary expressions.

These rules describe graph semantics, not runtime accuracy. More complicated
lowerings need their own compositional rules; no model/example name participates
in rule selection. In particular, attention is not recognized as a fused pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.source_structure import (
    EquationRule,
    LiteralMatcher,
    LiteralOperand,
    RuleOperand,
    StructureMismatch,
    TensorContract,
)

_DIRECT: Final = {
    "add": ("Add", 2),
    "sub": ("Sub", 2),
    "mul": ("Mul", 2),
    "div": ("Div", 2),
    "neg": ("Neg", 1),
    "exp": ("Exp", 1),
    "sin": ("Sin", 1),
    "cos": ("Cos", 1),
    "sqrt": ("Sqrt", 1),
}
_FLOATS: Final = frozenset({ir.DataType.FLOAT, ir.DataType.DOUBLE})
_PRIMITIVES: Final = {
    "add": jax.lax.add_p,
    "sub": jax.lax.sub_p,
    "mul": jax.lax.mul_p,
    "div": jax.lax.div_p,
    "neg": jax.lax.neg_p,
    "exp": jax.lax.exp_p,
    "sin": jax.lax.sin_p,
    "cos": jax.lax.cos_p,
    "sqrt": jax.lax.sqrt_p,
    "transpose": jax.lax.transpose_p,
    "stop_gradient": jax.lax.stop_gradient_p,
    "convert_element_type": jax.lax.convert_element_type_p,
}


@dataclass(frozen=True)
class _DirectRule:
    op_type: str
    attributes: tuple[tuple[str, int | tuple[int, ...]], ...] = ()
    identity_alias: bool = False

    def check(
        self,
        graph: ir.Graph,
        inputs: tuple[RuleOperand, ...],
        outputs: tuple[ir.Value, ...],
        match_literal: LiteralMatcher,
    ) -> tuple[ir.Node, ...]:
        if len(outputs) != 1:
            raise StructureMismatch("direct-result-arity-differs")
        result = outputs[0]
        node = result.producer()
        if self.identity_alias and len(inputs) == 1:
            operand = inputs[0]
            if isinstance(operand, ir.Value):
                if result is operand:
                    return ()
            elif node is None or node.op_type == "Constant":
                # A literal no-op can legally reuse an equivalent immutable leaf.
                match_literal(result, operand)
                return ()
        if (
            node is None
            or node.graph is not graph
            or not any(candidate is node for candidate in graph)
            or node.domain
            or node.overload
            or node.op_type != self.op_type
            or len(node.outputs) != 1
            or node.outputs[0] is not result
        ):
            raise StructureMismatch("direct-operator-differs")
        if len(node.inputs) != len(inputs):
            raise StructureMismatch("direct-input-edge-differs")
        if set(node.attributes) != {name for name, _ in self.attributes}:
            raise StructureMismatch("direct-attribute-set-differs")
        for name, expected in self.attributes:
            attribute = node.attributes[name]
            if isinstance(expected, tuple):
                matches = (
                    attribute.type is ir.AttributeType.INTS
                    and tuple(attribute.as_ints()) == expected
                )
            else:
                matches = (
                    attribute.type is ir.AttributeType.INT
                    and attribute.as_int() == expected
                )
            if not matches:
                raise StructureMismatch("direct-attribute-differs")
        for actual, expected_operand in zip(node.inputs, inputs, strict=True):
            if isinstance(expected_operand, LiteralOperand):
                match_literal(actual, expected_operand)
            elif actual is not expected_operand:
                raise StructureMismatch("direct-input-edge-differs")
        return (node,)


def freeze_source_structure_rule(
    eqn: JaxprEqn,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> EquationRule | None:
    """Freeze expected semantics from the real equation before graph emission."""
    primitive = eqn.primitive
    if primitive is None:
        return None
    name = primitive.name
    if _PRIMITIVES.get(name) is not primitive:
        return None
    if (
        eqn.effects
        or len(outputs) != 1
        or any(
            value.encoding != "identity" or value.dtype not in _FLOATS
            for value in (*inputs, *outputs)
        )
    ):
        return None
    tokens = [
        value.symbol_token
        for value in (*inputs, *outputs)
        if value.symbol_token is not None
    ]
    if tokens and any(token is not tokens[0] for token in tokens):
        return None
    # Empty-output representations and FP64 cosine's half-angle decomposition
    # require compositional contracts beyond these direct operator rules.
    if any(0 in value.shape for value in outputs) or (
        name == "cos" and outputs[0].dtype == ir.DataType.DOUBLE
    ):
        return None
    params = dict(eqn.params)
    if name in _DIRECT:
        operation, arity = _DIRECT[name]
        if name in {"exp", "sin", "cos", "sqrt"} and params.get("accuracy") is None:
            params.pop("accuracy", None)
        if (
            params
            or len(inputs) != arity
            or any(value.dtype != outputs[0].dtype for value in inputs)
        ):
            return None
        symbolic_inputs = [value for value in inputs if value.symbol_token is not None]
        if bool(symbolic_inputs) != (outputs[0].symbol_token is not None):
            return None
        if symbolic_inputs:
            if any(
                value.symbol_axis is None
                or len(outputs[0].shape) - len(value.shape) + value.symbol_axis
                != outputs[0].symbol_axis
                for value in symbolic_inputs
            ):
                return None
        return _DirectRule(operation)
    if name == "stop_gradient":
        if (
            params
            or len(inputs) != 1
            or inputs[0].dtype != outputs[0].dtype
            or not inputs[0].same_shape(outputs[0])
        ):
            return None
        return _DirectRule("Identity", identity_alias=True)
    if name == "transpose":
        if set(params) != {"permutation"} or len(inputs) != 1:
            return None
        permutation = params["permutation"]
        if (
            not isinstance(permutation, (tuple, list))
            or any(type(axis) is not int for axis in permutation)
            or sorted(permutation) != list(range(len(inputs[0].shape)))
            or tuple(inputs[0].shape[axis] for axis in permutation) != outputs[0].shape
            or inputs[0].dtype != outputs[0].dtype
            or inputs[0].symbol_token is not outputs[0].symbol_token
            or outputs[0].symbol_axis
            != (
                tuple(permutation).index(inputs[0].symbol_axis)
                if inputs[0].symbol_axis is not None
                else None
            )
        ):
            return None
        return _DirectRule(
            "Transpose",
            (("perm", tuple(permutation)),),
            identity_alias=tuple(permutation) == tuple(range(len(permutation))),
        )
    if name == "convert_element_type":
        if (
            len(inputs) != 1
            or set(params) - {"new_dtype", "weak_type", "sharding"}
            or params.get("sharding") is not None
            or "new_dtype" not in params
            or np.dtype(params["new_dtype"]).name != outputs[0].logical_dtype
            or params.get("weak_type", False) != outputs[0].weak_type
            or not inputs[0].same_shape(outputs[0])
        ):
            return None
        return _DirectRule(
            "Cast",
            (("to", int(outputs[0].dtype.value)),),
            identity_alias=inputs[0].dtype == outputs[0].dtype,
        )
    return None
