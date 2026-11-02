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

from jax2onnx._compat.jax import JaxprEqn, Literal, ShapedArray, Var
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
    "bitcast_convert_type": jax.lax.bitcast_convert_type_p,
    "lt": jax.lax.lt_p,
    "le": jax.lax.le_p,
    "gt": jax.lax.gt_p,
    "ge": jax.lax.ge_p,
    "eq": jax.lax.eq_p,
    "ne": jax.lax.ne_p,
    "is_finite": jax.lax.is_finite_p,
    "not": jax.lax.not_p,
    "and": jax.lax.and_p,
    "or": jax.lax.or_p,
    "xor": jax.lax.xor_p,
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


@dataclass(frozen=True)
class _PredicateRule:
    """The two closed predicate compositions, with typed Boolean intermediates."""

    operation: str
    output: TensorContract

    def check(
        self,
        graph: ir.Graph,
        inputs: tuple[RuleOperand, ...],
        outputs: tuple[ir.Value, ...],
        match_literal: LiteralMatcher,
    ) -> tuple[ir.Node, ...]:
        arity = 2 if self.operation == "ne" else 1
        if len(inputs) != arity or len(outputs) != 1:
            raise StructureMismatch("predicate-boundary-arity-differs")
        result = outputs[0]
        if not self.output.matches(result):
            raise StructureMismatch("predicate-output-type-or-shape-differs")
        negation = result.producer()
        if negation is None or len(negation.inputs) != 1 or negation.inputs[0] is None:
            raise StructureMismatch("predicate-negation-input-differs")
        inner = negation.inputs[0]
        assert inner is not None
        _DirectRule("Not").check(graph, (inner,), outputs, match_literal)
        if not self.output.matches(inner):
            raise StructureMismatch("predicate-intermediate-type-or-shape-differs")
        if self.operation == "ne":
            equality = _DirectRule("Equal").check(
                graph, inputs, (inner,), match_literal
            )
            return (*equality, negation)
        if self.operation != "is_finite":
            raise StructureMismatch("predicate-operation-differs")
        disjunction = inner.producer()
        if (
            disjunction is None
            or len(disjunction.inputs) != 2
            or any(value is None for value in disjunction.inputs)
        ):
            raise StructureMismatch("predicate-classification-inputs-differ")
        is_inf, is_nan = disjunction.inputs
        assert is_inf is not None and is_nan is not None
        _DirectRule("Or").check(graph, (is_inf, is_nan), (inner,), match_literal)
        if not self.output.matches(is_inf) or not self.output.matches(is_nan):
            raise StructureMismatch("predicate-intermediate-type-or-shape-differs")
        inf_nodes = _DirectRule("IsInf").check(graph, inputs, (is_inf,), match_literal)
        nan_nodes = _DirectRule("IsNaN").check(graph, inputs, (is_nan,), match_literal)
        return (*inf_nodes, *nan_nodes, disjunction, negation)


def _freeze_predicate_rule(
    eqn: JaxprEqn,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> EquationRule | None:
    """Freeze only canonical static ordinary predicates, before any IR exists."""
    if type(eqn) is not JaxprEqn or not any(
        eqn.primitive is primitive
        for primitive in (*_PREDICATES, jax.lax.ne_p, jax.lax.is_finite_p)
    ):
        return None
    finite = eqn.primitive is jax.lax.is_finite_p
    arity = 1 if finite else 2
    if (
        eqn.effects
        or type(eqn.params) is not dict
        or eqn.params
        or len(inputs) != arity
        or len(outputs) != 1
        or len(eqn.invars) != arity
        or len(eqn.outvars) != 1
    ):
        return None
    ordinary = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    }
    for value in (*inputs, *outputs):
        if (
            type(value) is not TensorContract
            or type(value.logical_dtype) is not str
            or value.logical_dtype not in ordinary
            or value.dtype != ir.DataType.from_numpy(np.dtype(value.logical_dtype))
            or value.encoding != "identity"
            or type(value.shape) is not tuple
            or type(value.logical_shape) is not tuple
            or value.shape != value.logical_shape
            or any(
                type(dim) is not int or not 0 <= dim <= 2**63 - 1 for dim in value.shape
            )
            or value.symbol_token is not None
            or value.symbol_axis is not None
            or value.logical_symbol_axis is not None
            or type(value.weak_type) is not bool
        ):
            return None
    for var, value in zip(
        (*eqn.invars, *eqn.outvars), (*inputs, *outputs), strict=True
    ):
        if type(var) not in {Var, Literal}:
            return None
        aval = var.aval
        if type(aval) is not ShapedArray or type(aval.weak_type) is not bool:
            return None
        if (
            not isinstance(aval.dtype, np.dtype)
            or aval.dtype.name != value.logical_dtype
            or type(aval.shape) is not tuple
            or any(type(dim) is not int for dim in aval.shape)
            or aval.shape != value.logical_shape
            or aval.weak_type != value.weak_type
        ):
            return None
    target = outputs[0]
    if (
        target.logical_dtype != "bool"
        or target.dtype is not ir.DataType.BOOL
        or target.weak_type != all(value.weak_type for value in inputs)
        or any(value.logical_dtype != inputs[0].logical_dtype for value in inputs)
    ):
        return None
    if finite:
        if inputs[0].dtype not in _FLOATS or inputs[0].shape != target.shape:
            return None
        return _PredicateRule("is_finite", target)
    if inputs[0].logical_dtype == "bool" and eqn.primitive not in (
        jax.lax.eq_p,
        jax.lax.ne_p,
    ):
        # ONNX ordered predicates do not give this rule Boolean-input authority.
        return None
    ranks = {len(value.shape) for value in inputs if value.shape}
    if len(ranks) > 1:
        return None
    rank = max((len(value.shape) for value in inputs), default=0)
    shape = []
    for axis in range(rank):
        dimensions = {
            value.shape[axis]
            for value in inputs
            if value.shape and value.shape[axis] != 1
        }
        if len(dimensions) > 1:
            return None
        shape.append(next(iter(dimensions), 1))
    if tuple(shape) != target.shape:
        return None
    if eqn.primitive is jax.lax.ne_p:
        return _PredicateRule("ne", target)
    operation = _PREDICATES.get(eqn.primitive)
    return _DirectRule(operation) if operation is not None else None


def freeze_source_structure_rule(
    eqn: JaxprEqn,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> EquationRule | None:
    """Freeze expected semantics from the real equation before graph emission."""
    if type(eqn) is not JaxprEqn:
        return None
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
        or primitive is jax.lax.reduce_and_p
        or primitive is jax.lax.reduce_or_p
        or primitive is jax.lax.reduce_xor_p
    ):
        from jax2onnx.plugins.jax.lax._source_reduce_structure import (
            freeze_reduction_structure_rule,
        )

        return freeze_reduction_structure_rule(eqn, inputs, outputs)
    if (
        primitive is jax.lax.reshape_p
        or primitive is jax.lax.broadcast_in_dim_p
        or primitive is jax.lax.slice_p
    ):
        from jax2onnx.plugins.jax.lax._source_shape_structure import (
            freeze_shape_structure_rule,
        )

        return freeze_shape_structure_rule(eqn, inputs, outputs)
    name = primitive.name
    if _PRIMITIVES.get(name) is not primitive:
        return None
    if primitive in _PREDICATES or primitive in (jax.lax.ne_p, jax.lax.is_finite_p):
        return _freeze_predicate_rule(eqn, inputs, outputs)
    if primitive is jax.lax.bitcast_convert_type_p:
        if (
            eqn.effects
            or type(eqn.params) is not dict
            or set(eqn.params) != {"new_dtype"}
            or len(inputs) != 1
            or len(outputs) != 1
        ):
            return None
        source, target = inputs[0], outputs[0]
        try:
            source_dtype = np.dtype(source.logical_dtype)
            target_dtype = np.dtype(target.logical_dtype)
            source_ir_dtype = ir.DataType.from_numpy(source_dtype)
            target_ir_dtype = ir.DataType.from_numpy(target_dtype)
        except (TypeError, ValueError):
            return None
        if (
            any(
                value.encoding != "identity"
                or value.shape != value.logical_shape
                or value.symbol_axis != value.logical_symbol_axis
                for value in (source, target)
            )
            or source_ir_dtype != source.dtype
            or target_ir_dtype != target.dtype
            or not (
                source_dtype.kind in "iuf"
                or source_dtype.name == "bfloat16"
                or source_dtype == target_dtype == np.dtype(np.bool_)
            )
            or not (
                target_dtype.kind in "iuf"
                or target_dtype.name == "bfloat16"
                or source_dtype == target_dtype == np.dtype(np.bool_)
            )
            or source_dtype.itemsize != target_dtype.itemsize
            or not isinstance(eqn.params["new_dtype"], np.dtype)
            or eqn.params["new_dtype"] != target_dtype
            or not source.same_shape(target)
            or target.weak_type is not False
        ):
            return None
        if source_dtype == target_dtype:
            return _DirectRule("Identity", identity_alias=True)
        return _DirectRule("BitCast", (("to", int(target.dtype.value)),))
    discrete_operations = {
        jax.lax.not_p: ("Not", "BitwiseNot", 1),
        jax.lax.and_p: ("And", "BitwiseAnd", 2),
        jax.lax.or_p: ("Or", "BitwiseOr", 2),
        jax.lax.xor_p: ("Xor", "BitwiseXor", 2),
    }
    if (
        primitive is jax.lax.convert_element_type_p
        and len(inputs) == len(outputs) == 1
        and (inputs[0].logical_dtype, outputs[0].logical_dtype) == ("int32", "float32")
    ):
        # This is the original typed Cast, including legitimate source rounding.
        # Sample-bound losslessness is a separate numerical applicability proof.
        source, target = inputs[0], outputs[0]
        params = eqn.params
        if (
            eqn.effects
            or type(params) is not dict
            or set(params)
            not in ({"new_dtype", "weak_type"}, {"new_dtype", "weak_type", "sharding"})
            or params.get("sharding") is not None
            or not isinstance(params["new_dtype"], np.dtype)
            or params["new_dtype"] != np.dtype("float32")
            or type(params["weak_type"]) is not bool
            or params["weak_type"] != target.weak_type
            or len(eqn.invars) != 1
            or len(eqn.outvars) != 1
            or source.shape != target.shape
            or any(
                value.encoding != "identity"
                or value.shape != value.logical_shape
                or value.dtype != ir.DataType.from_numpy(np.dtype(value.logical_dtype))
                or value.symbol_token is not None
                or value.symbol_axis is not None
                or value.logical_symbol_axis is not None
                or type(value.weak_type) is not bool
                or any(
                    type(dim) is not int or not 0 <= dim <= 2**63 - 1
                    for dim in value.shape
                )
                for value in (source, target)
            )
        ):
            return None
        for var, value in zip(
            (*eqn.invars, *eqn.outvars), (source, target), strict=True
        ):
            if (
                type(var) not in {Var, Literal}
                or type(var.aval) is not ShapedArray
                or type(var.aval.weak_type) is not bool
            ):
                return None
            if (
                not isinstance(var.aval.dtype, np.dtype)
                or var.aval.dtype.name != value.logical_dtype
                or tuple(var.aval.shape) != value.logical_shape
                or var.aval.weak_type != value.weak_type
            ):
                return None
        return _DirectRule("Cast", (("to", int(ir.DataType.FLOAT.value)),))
    discrete = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    static_discrete = bool(outputs) and all(
        value.logical_dtype in discrete
        and value.dtype == ir.DataType.from_numpy(np.dtype(value.logical_dtype))
        and value.encoding == "identity"
        and value.shape == value.logical_shape
        and all(type(dim) is int and 0 <= dim <= 2**63 - 1 for dim in value.shape)
        and value.symbol_token is None
        and value.symbol_axis is None
        and value.logical_symbol_axis is None
        and type(value.weak_type) is bool
        for value in (*inputs, *outputs)
    )
    if static_discrete:
        if len(eqn.invars) != len(inputs) or len(eqn.outvars) != len(outputs):
            return None
        for var, value in zip(
            (*eqn.invars, *eqn.outvars), (*inputs, *outputs), strict=True
        ):
            if (
                type(var) not in {Var, Literal}
                or type(var.aval) is not ShapedArray
                or type(var.aval.weak_type) is not bool
                or not isinstance(var.aval.dtype, np.dtype)
                or var.aval.dtype.name != value.logical_dtype
                or tuple(var.aval.shape) != value.logical_shape
                or var.aval.weak_type != value.weak_type
            ):
                return None
    if primitive in discrete_operations:
        boolean_op, integer_op, arity = discrete_operations[primitive]
        if (
            not static_discrete
            or eqn.effects
            or type(eqn.params) is not dict
            or eqn.params
            or len(inputs) != arity
            or len(outputs) != 1
        ):
            return None
        target = outputs[0]
        if any(value.logical_dtype != target.logical_dtype for value in inputs):
            return None
        ranks = {len(value.shape) for value in inputs if value.shape}
        if len(ranks) > 1:
            return None
        rank = max((len(value.shape) for value in inputs), default=0)
        shape = []
        for axis in range(rank):
            dimensions = {
                value.shape[axis]
                for value in inputs
                if value.shape and value.shape[axis] != 1
            }
            if len(dimensions) > 1:
                return None
            shape.append(next(iter(dimensions), 1))
        if tuple(shape) != target.shape or target.weak_type != all(
            value.weak_type for value in inputs
        ):
            return None
        return _DirectRule(boolean_op if target.logical_dtype == "bool" else integer_op)
    if static_discrete and primitive in (
        jax.lax.convert_element_type_p,
        jax.lax.transpose_p,
        jax.lax.stop_gradient_p,
    ):
        if (
            eqn.effects
            or type(eqn.params) is not dict
            or len(inputs) != 1
            or len(outputs) != 1
        ):
            return None
        source, target = inputs[0], outputs[0]
        params = eqn.params
        if primitive is jax.lax.convert_element_type_p:
            if (
                set(params)
                not in (
                    {"new_dtype", "weak_type"},
                    {"new_dtype", "weak_type", "sharding"},
                )
                or params.get("sharding") is not None
                or not isinstance(params["new_dtype"], np.dtype)
                or params["new_dtype"].name != target.logical_dtype
                or type(params["weak_type"]) is not bool
                or params["weak_type"] != target.weak_type
                or source.shape != target.shape
            ):
                return None
            return _DirectRule(
                "Cast",
                (("to", int(target.dtype.value)),),
                identity_alias=source.dtype == target.dtype,
            )
        if (
            source.logical_dtype != target.logical_dtype
            or source.weak_type != target.weak_type
        ):
            return None
        if primitive is jax.lax.stop_gradient_p:
            if params or source.shape != target.shape:
                return None
            return _DirectRule("Identity", identity_alias=True)
        permutation = params.get("permutation")
        if (
            set(params) != {"permutation"}
            or type(permutation) is not tuple
            or any(type(axis) is not int for axis in permutation)
            or sorted(permutation) != list(range(len(source.shape)))
            or tuple(source.shape[axis] for axis in permutation) != target.shape
        ):
            return None
        return _DirectRule(
            "Transpose",
            (("perm", permutation),),
            identity_alias=permutation == tuple(range(len(permutation))),
        )
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
    if type(eqn) is not JaxprEqn:
        return None
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
    from jax2onnx.plugins.jax.lax._source_reduce_structure import (
        _BooleanReductionRule,
        _ReductionRule,
    )
    from jax2onnx.plugins.jax.lax._source_shape_structure import _ShapeRule, _SliceRule
    from jax2onnx.plugins.jax.lax._source_loop_structure import ScanRule, WhileRule

    from jax2onnx.plugins.jax.core._source_inline_structure import InlineRule

    rule = freeze_source_structure_rule(eqn, inputs, outputs)
    if type(rule) is InlineRule:
        # The original frozen child remains the sole body-content authority.
        if rule.early_inline:
            return ("inline-jit-native-early", rule.keep_unused)
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
    if type(rule) is _BooleanReductionRule and (
        primitive is jax.lax.reduce_and_p
        or primitive is jax.lax.reduce_or_p
        or primitive is jax.lax.reduce_xor_p
    ):
        return (
            "boolean-reduction",
            rule.mode,
            rule.axes,
            rule.reduction_count,
            rule.empty_identity,
        )
    if type(rule) is _PredicateRule and (
        primitive is jax.lax.ne_p or primitive is jax.lax.is_finite_p
    ):
        return ("predicate", rule.operation)
    if type(rule) is _ShapeRule and (
        eqn.primitive is jax.lax.reshape_p
        or eqn.primitive is jax.lax.broadcast_in_dim_p
    ):
        return ("shape", rule.operation, rule.permutation, rule.broadcast_dimensions)
    if type(rule) is _SliceRule and eqn.primitive is jax.lax.slice_p:
        return ("slice", rule.starts, rule.limits, rule.axes, rule.steps)
    if type(rule) is _DirectRule and _PRIMITIVES.get(primitive_name) is primitive:
        return (
            "direct",
            primitive_name,
            rule.op_type,
            rule.attributes,
            rule.identity_alias,
        )
    return None
