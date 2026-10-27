# jax2onnx/converter/_source_regions.py

"""Immutable original typed regions, independent of lowering receipts and names."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, Literal
from .source_representation import representation_for_aval
from .source_structure import (
    ConstantContract,
    StructureUnavailable,
    TensorContract,
    _ShapeFreezer,
)


@dataclass(frozen=True)
class FrozenUse:
    kind: str
    index: int
    port: int
    tensor: TensorContract | None
    literal: ConstantContract | None = None


@dataclass(frozen=True)
class FrozenEquation:
    inputs: tuple[FrozenUse, ...]
    outputs: tuple[TensorContract | None, ...]
    semantics: tuple[object, ...] | None
    children: tuple[tuple[str, FrozenBody], ...] = ()


@dataclass(frozen=True)
class FrozenBody:
    inputs: tuple[TensorContract | None, ...]
    constants: tuple[tuple[TensorContract | None, ConstantContract | None], ...]
    equations: tuple[FrozenEquation, ...]
    outputs: tuple[FrozenUse, ...]
    gaps: tuple[str, ...] = ()

    @property
    def expected_equations(self) -> int | None:
        if "expected-region-total-unknown" in self.gaps:
            return None
        children = tuple(
            child.expected_equations
            for equation in self.equations
            for _, child in equation.children
        )
        if any(count is None for count in children):
            return None
        return len(self.equations) + sum(
            count for count in children if count is not None
        )

    def permuted_inputs(self, order: tuple[int, ...]) -> FrozenBody:
        if (
            type(order) is not tuple
            or any(type(port) is not int for port in order)
            or sorted(order) != list(range(len(self.inputs)))
        ):
            raise StructureUnavailable("function-original-port-permutation-unavailable")
        inverse = {old: new for new, old in enumerate(order)}

        def use(value: FrozenUse) -> FrozenUse:
            return (
                replace(value, index=inverse[value.index])
                if value.kind == "input"
                else value
            )

        return replace(
            self,
            inputs=tuple(self.inputs[index] for index in order),
            equations=tuple(
                replace(equation, inputs=tuple(use(value) for value in equation.inputs))
                for equation in self.equations
            ),
            outputs=tuple(use(value) for value in self.outputs),
        )


def _unhandled_native_body(params: object) -> bool:
    """Passively detect visible unaccounted bodies, never inspect foreign leaves.

    A container subclass, cycle or exhausted finite budget is unavailable
    accounting. No foreign traversal hook, thunk or body execution participates.
    """
    if type(params) is not dict:
        return True
    pending: list[tuple[object, frozenset[int]]] = [(params, frozenset())]
    remaining = 4096
    while pending:
        value, ancestors = pending.pop()
        remaining -= 1
        if remaining < 0:
            return True
        if type(value) is Jaxpr or type(value) is ClosedJaxpr:
            return True
        if type(value) is dict:
            if len(value) * 2 > remaining - len(pending):
                return True
            children = tuple(item for pair in value.items() for item in pair)
        elif type(value) is tuple or type(value) is list:
            if len(value) > remaining - len(pending):
                return True
            children = tuple(item for item in value)
        elif issubclass(type(value), (dict, tuple, list)):
            # Builtin subtype detection invokes no object-provided hooks.
            return True
        else:
            continue
        if id(value) in ancestors or len(ancestors) >= 64:
            return True
        descendant_ancestors = ancestors | {id(value)}
        pending.extend((child, descendant_ancestors) for child in children)
    return False


def declare_body(jaxpr: Any) -> FrozenBody:
    """Count visible native regions without observing a tensor or semantic rule.

    Placeholders are unavailable proofs. They independently preserve the exact
    expected tree if the subsequent shape/storage/rule observation fails.
    Malformed declarations explicitly make the total unknown, rather than
    presenting the remaining visible prefix as the complete denominator.
    """
    import jax
    from jax2onnx.plugins._source_function_structure import declared_function_body

    active: set[int] = set()

    def walk(native: Any) -> FrozenBody:
        body = getattr(native, "jaxpr", native)
        gaps = {"source-semantic-freeze-unavailable"}
        if id(body) in active:
            return FrozenBody((), (), (), (), ("expected-region-total-unknown",))
        active.add(id(body))
        try:
            equations = []
            for equation in body.eqns:
                children = []
                try:
                    function = declared_function_body(equation)
                    if function is not None:
                        children.append(("function", function))
                    elif equation.primitive is jax.lax.cond_p:
                        branches = equation.params.get("branches")
                        if type(branches) is not tuple or len(branches) != 2:
                            raise StructureUnavailable("branch-declaration-unavailable")
                        for role, branch in zip(
                            ("false", "true"), branches, strict=True
                        ):
                            children.append((role, walk(branch)))
                    elif equation.primitive is jax.lax.while_p:
                        for role, key in (
                            ("initial-condition", "cond_jaxpr"),
                            ("body", "body_jaxpr"),
                            ("next-condition", "cond_jaxpr"),
                        ):
                            children.append((role, walk(equation.params[key])))
                    elif _unhandled_native_body(equation.params):
                        gaps.add("expected-region-total-unknown")
                except Exception:
                    gaps.add("expected-region-total-unknown")
                equations.append(FrozenEquation((), (), None, tuple(children)))
            return FrozenBody((), (), tuple(equations), (), tuple(sorted(gaps)))
        finally:
            active.remove(id(body))

    try:
        return walk(jaxpr)
    except Exception:
        return FrozenBody((), (), (), (), ("expected-region-total-unknown",))


def freeze_body(
    jaxpr: Jaxpr,
    constants: tuple[object, ...] = (),
    *,
    constant_contracts: tuple[ConstantContract, ...] | None = None,
) -> FrozenBody:
    """Observe an existing trace once; retain no native SSA/shape/array owners."""
    import jax
    from jax2onnx.plugins._source_structure import freeze_source_semantic_key
    from jax2onnx.plugins._source_function_structure import original_function_body

    gaps: set[str] = set()
    shapes = _ShapeFreezer()

    def tensor(var: Any) -> TensorContract | None:
        try:
            result = TensorContract.from_aval(var.aval, shapes)
            if result.symbol_token is not None:
                raise StructureUnavailable("cross-region-symbolic-contract-unavailable")
            return result
        except Exception as exc:
            gaps.add(
                str(exc)
                if isinstance(exc, StructureUnavailable)
                else type(exc).__name__
            )
            return None

    def storage(var: Any, value: object) -> ConstantContract | None:
        try:
            representation = representation_for_aval(var.aval)
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.asarray(value, dtype=var.aval.dtype)
                encoded = representation.encode_array(array)
            return ConstantContract.from_array(encoded)
        except Exception as exc:
            gaps.add("source-storage:" + type(exc).__name__)
            return None

    bindings: dict[int, FrozenUse] = {}
    inputs = tuple(tensor(var) for var in jaxpr.invars)
    for index, (var, contract) in enumerate(zip(jaxpr.invars, inputs, strict=True)):
        bindings[id(var)] = FrozenUse("input", index, 0, contract)
    frozen_constants = []
    if constant_contracts is not None and len(constant_contracts) != len(
        jaxpr.constvars
    ):
        gaps.add("source-constant-count-unavailable")
    if constant_contracts is None and len(constants) != len(jaxpr.constvars):
        gaps.add("source-constant-count-unavailable")
    for index, var in enumerate(jaxpr.constvars):
        contract = tensor(var)
        payload = (
            constant_contracts[index]
            if constant_contracts is not None and index < len(constant_contracts)
            else storage(var, constants[index]) if index < len(constants) else None
        )
        if payload is None:
            gaps.add("source-constant-storage-unavailable")
        frozen_constants.append((contract, payload))
        bindings[id(var)] = FrozenUse("constant", index, 0, contract)

    def use(var: Any) -> FrozenUse:
        if isinstance(var, Literal):
            return FrozenUse("literal", 0, 0, tensor(var), storage(var, var.val))
        result = bindings.get(id(var))
        if result is None:
            gaps.add("source-ssa-use-unavailable")
            return FrozenUse("unavailable", 0, 0, tensor(var))
        return result

    def closed(value: Any) -> FrozenBody:
        body = getattr(value, "jaxpr", value)
        return freeze_body(body, tuple(getattr(value, "consts", ())))

    def required_tensor(value: TensorContract | None) -> TensorContract:
        if value is None:
            raise StructureUnavailable("source-tensor-contract-unavailable")
        return value

    equations = []
    for index, equation in enumerate(jaxpr.eqns):
        operands = tuple(use(var) for var in equation.invars)
        results = tuple(tensor(var) for var in equation.outvars)
        children: list[tuple[str, FrozenBody]] = []
        function = original_function_body(equation)
        if function is not None:
            children.append(("function", function))
        else:
            if equation.primitive is jax.lax.cond_p:
                branches = equation.params.get("branches")
                if isinstance(branches, tuple) and len(branches) == 2:
                    children.extend(
                        (role, closed(branch))
                        for role, branch in zip(
                            ("false", "true"), branches, strict=True
                        )
                    )
            elif equation.primitive is jax.lax.while_p:
                for role, key in (
                    ("initial-condition", "cond_jaxpr"),
                    ("body", "body_jaxpr"),
                    ("next-condition", "cond_jaxpr"),
                ):
                    if key in equation.params:
                        children.append((role, closed(equation.params[key])))
                gaps.add("while-attachment-contract-unavailable")
            elif _unhandled_native_body(equation.params):
                gaps.add("expected-region-total-unknown")
        semantics = None
        if (
            not equation.effects
            and all(value.tensor is not None for value in operands)
            and all(value is not None for value in results)
        ):
            try:
                semantics = freeze_source_semantic_key(
                    equation,
                    tuple(required_tensor(value.tensor) for value in operands),
                    tuple(required_tensor(value) for value in results),
                )
            except Exception as exc:
                gaps.add("source-semantic-key:" + type(exc).__name__)
        if semantics is None:
            gaps.add("source-semantic-contract-unavailable")
        if equation.primitive is jax.lax.cond_p and semantics != ("dynamic-if",):
            gaps.add("if-attachment-contract-unavailable")
        for _, child in children:
            gaps.update(child.gaps)
        equations.append(FrozenEquation(operands, results, semantics, tuple(children)))
        for port, (var, contract) in enumerate(
            zip(equation.outvars, results, strict=True)
        ):
            bindings[id(var)] = FrozenUse("equation", index, port, contract)
    if jaxpr.effects:
        gaps.add("source-effects-contract-unavailable")
    return FrozenBody(
        inputs,
        tuple(frozen_constants),
        tuple(equations),
        tuple(use(var) for var in jaxpr.outvars),
        tuple(sorted(gaps)),
    )


def freeze_closed_body(closed: Any) -> FrozenBody:
    """Bound freezer failures without changing the original trace's exception path."""
    declaration = declare_body(closed)
    try:
        body = freeze_body(closed.jaxpr, tuple(closed.consts))
        if declaration.expected_equations is None:
            return replace(
                body, gaps=tuple(sorted({*body.gaps, "expected-region-total-unknown"}))
            )
        if body.expected_equations != declaration.expected_equations:
            return replace(
                declaration,
                gaps=(*declaration.gaps, "source-region-declaration-differs"),
            )
        return body
    except Exception as exc:
        # No second trace/eval_shape fallback is permitted.
        return replace(
            declaration,
            gaps=tuple(
                sorted({*declaration.gaps, "original-body:" + type(exc).__name__})
            ),
        )
