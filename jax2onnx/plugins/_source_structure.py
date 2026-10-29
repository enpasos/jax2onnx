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
_INTEGERS: Final = frozenset({ir.DataType.INT32, ir.DataType.INT64})
_PREDICATES: Final = {
    jax.lax.lt_p: "Less",
    jax.lax.le_p: "LessOrEqual",
    jax.lax.gt_p: "Greater",
    jax.lax.ge_p: "GreaterOrEqual",
    jax.lax.eq_p: "Equal",
}
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
    "lt": jax.lax.lt_p,
    "le": jax.lax.le_p,
    "gt": jax.lax.gt_p,
    "ge": jax.lax.ge_p,
    "eq": jax.lax.eq_p,
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
    from jax2onnx.plugins._source_function_structure import (
        FunctionRule,
        original_function_body,
    )

    body = original_function_body(eqn)
    if body is not None:
        return FunctionRule(body)
    from jax2onnx.plugins.jax.core._source_inline_structure import freeze_inline_rule

    inline = freeze_inline_rule(eqn, inputs, outputs)
    if inline is not None:
        return inline
    from jax2onnx.plugins._source_control_flow_structure import freeze_control_flow_rule

    control = freeze_control_flow_rule(eqn, inputs, outputs)
    if control is not None:
        return control
    from jax2onnx.plugins.jax.lax._source_loop_structure import freeze_loop_rule

    loop = freeze_loop_rule(eqn, inputs, outputs)
    if loop is not None:
        return loop
    if primitive is jax.lax.dot_general_p:
        from jax2onnx.plugins.jax.lax._source_dot_structure import (
            freeze_dot_structure_rule,
        )

        return freeze_dot_structure_rule(eqn, inputs, outputs)
    if (
        primitive is jax.lax.reduce_sum_p
        or primitive is jax.lax.reduce_max_p
        or primitive is jax.lax.reduce_min_p
    ):
        from jax2onnx.plugins.jax.lax._source_reduce_structure import (
            freeze_reduction_structure_rule,
        )

        return freeze_reduction_structure_rule(eqn, inputs, outputs)
    if primitive is jax.lax.reshape_p or primitive is jax.lax.broadcast_in_dim_p:
        from jax2onnx.plugins.jax.lax._source_shape_structure import (
            freeze_shape_structure_rule,
        )

        return freeze_shape_structure_rule(eqn, inputs, outputs)
    name = primitive.name
    if _PRIMITIVES.get(name) is not primitive:
        return None
    # Scalar numeric conditions and counters have their own exact contracts.
    # Boolean arithmetic and general integer expressions remain separate rules.
    scalar_numeric = (
        not eqn.effects
        and type(eqn.params) is dict
        and not eqn.params
        and len(inputs) == 2
        and len(outputs) == 1
        and all(
            value.encoding == "identity"
            and value.symbol_token is None
            and value.shape == value.logical_shape == ()
            for value in (*inputs, *outputs)
        )
        and inputs[0].dtype == inputs[1].dtype
        and inputs[0].logical_dtype == inputs[1].logical_dtype
    )
    if scalar_numeric:
        operation = _PREDICATES.get(primitive)
        if (
            operation is not None
            and inputs[0].dtype in _FLOATS | _INTEGERS
            and outputs[0].dtype is ir.DataType.BOOL
            and outputs[0].logical_dtype == "bool"
        ):
            return _DirectRule(operation)
        if (
            primitive in (jax.lax.add_p, jax.lax.sub_p)
            and inputs[0].dtype in _INTEGERS
            and outputs[0].dtype == inputs[0].dtype
            and outputs[0].logical_dtype == inputs[0].logical_dtype
        ):
            return _DirectRule("Add" if primitive is jax.lax.add_p else "Sub")
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
        # Native newer JAX spells the unchanged default explicitly. Typed
        # output multiplication and every additional setting remain separate
        # unreviewed source operations, even when their endpoint dtypes match.
        if name == "mul" and "out_dtype" in params and params["out_dtype"] is None:
            params.pop("out_dtype")
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


def freeze_source_semantic_key(
    eqn: JaxprEqn,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> tuple[object, ...] | None:
    """Plugin-owned source keys, selected by exact native primitive identity."""
    primitive = eqn.primitive
    if primitive is None:
        return None
    primitive_name = primitive.name
    if primitive is jax.lax.convert_element_type_p:
        from jax2onnx.plugins.jax._source_f64_f16_structure import (
            freeze_f64_f16_semantic_key,
        )

        cast_semantics = freeze_f64_f16_semantic_key(eqn, inputs, outputs)
        if cast_semantics is not None:
            return cast_semantics
    from jax2onnx.plugins._source_function_structure import (
        FunctionRule,
    )
    from jax2onnx.plugins._source_control_flow_structure import (
        BooleanIndexRule,
        IfRule,
        LiteralIfRule,
    )
    from jax2onnx.plugins.jax.lax._source_dot_structure import _DotRule
    from jax2onnx.plugins.jax.lax._source_reduce_structure import _ReductionRule
    from jax2onnx.plugins.jax.lax._source_shape_structure import _ShapeRule
    from jax2onnx.plugins.jax.lax._source_loop_structure import ScanRule, WhileRule

    from jax2onnx.plugins.jax.core._source_inline_structure import InlineRule

    rule = freeze_source_structure_rule(eqn, inputs, outputs)
    if type(rule) is InlineRule:
        # The original frozen child remains the sole body-content authority.
        return ("inline-jit", rule.inline, rule.keep_unused)
    if type(rule) is BooleanIndexRule:
        return ("boolean-index",)
    if type(rule) is IfRule:
        return ("dynamic-if",)
    if type(rule) is LiteralIfRule:
        return ("literal-if", rule.index, rule.selector)
    if type(rule) is WhileRule:
        return ("while", rule.condition_constants, rule.body_constants, rule.carries)
    if type(rule) is ScanRule:
        return (
            "scan",
            rule.constants,
            rule.carries,
            rule.sequences,
            rule.results,
            rule.length,
            rule.unroll,
            rule.linear,
            rule.split_transpose,
            rule.partitions,
        )
    if type(rule) is FunctionRule:
        # FrozenEquation.children is the sole child-body content authority.
        # Repeating the entire body in this key doubles hash/equality traversal
        # at every nested call, even for a one-call-per-level chain.
        return ("function",)
    if type(rule) is _DotRule and eqn.primitive is jax.lax.dot_general_p:
        return ("dot_general", rule.axes, rule.precision, rule.preferred)
    if type(rule) is _ReductionRule and (
        eqn.primitive is jax.lax.reduce_sum_p
        or eqn.primitive is jax.lax.reduce_max_p
        or eqn.primitive is jax.lax.reduce_min_p
    ):
        return ("reduction", rule.operation, rule.axes)
    if type(rule) is _ShapeRule and (
        eqn.primitive is jax.lax.reshape_p
        or eqn.primitive is jax.lax.broadcast_in_dim_p
    ):
        return ("shape", rule.operation, rule.permutation, rule.broadcast_dimensions)
    if type(rule) is _DirectRule and _PRIMITIVES.get(primitive_name) is primitive:
        return (
            "direct",
            primitive_name,
            rule.op_type,
            rule.attributes,
            rule.identity_alias,
        )
    return None
