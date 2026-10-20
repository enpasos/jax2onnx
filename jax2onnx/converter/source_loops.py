# jax2onnx/converter/source_loops.py

"""Typed native loop carriers for the shared private SOURCE export path.

Only existing static tensor storage and existing forward-scan/while lowerings
are admitted. Every actual child scope retains the product leaf validator;
carrier eligibility does not establish target execution or numerical agreement.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING

import jax
import numpy as np

from jax2onnx._compat.jax import (
    ClosedJaxpr,
    native_scan_flat_tree_partitions,
    scan_arity,
)

if TYPE_CHECKING:
    from jax2onnx.converter.source_profile import _Site


def _signature(variables: Any, site: _Site) -> tuple[Any, ...]:
    from jax2onnx.converter.source_profile import _signature as signature

    result = signature(variables, site)
    site.require(
        all(value.symbol is None and value.dtype != "key<fry>" for value, _ in result),
        "loop symbolic or typed-key values are outside this static carrier contract",
        "static Boolean/integer/floating/complex tensor storage; symbolic and logical-key loop transport require separate implementation evidence",
    )
    return result


def _closed_body(closed: Any, site: _Site, ancestors: frozenset[int]) -> Any:
    site.require(
        isinstance(closed, ClosedJaxpr),
        "loop body is not an actual native closed Jaxpr",
        "a concrete native closed body with explicit constant payloads",
        category="lowering-defect",
    )
    body = closed.jaxpr  # JAX 0.11 returns self; do not recursively unwrap it.
    site.require(
        id(body) not in ancestors and len(ancestors) < 32,
        "cyclic or excessively nested loop body",
        "an acyclic bounded tree of native scopes",
        category="lowering-defect",
    )
    _signature((*body.invars, *body.constvars, *body.outvars), site)
    site.require(
        len(closed.consts) == len(body.constvars),
        "loop constant payload count disagrees with its bindings",
        "one unchanged concrete payload per closed constant variable",
        category="lowering-defect",
    )
    for index, (var, value) in enumerate(
        zip(body.constvars, closed.consts, strict=True)
    ):
        site.require(
            isinstance(
                value, (np.ndarray, np.generic, jax.Array, bool, int, float, complex)
            ),
            f"loop constant {index} is not an array or scalar payload",
            "a concrete numeric payload without conversion hooks",
            category="lowering-defect",
        )
        array = np.asarray(
            value
        )  # Never recanonicalize stored payloads with jnp.asarray.
        site.require(
            array.dtype == np.dtype(var.aval.dtype)
            and array.shape == tuple(var.aval.shape),
            f"loop constant {index} payload dtype or shape disagrees with its aval",
            "the exact constant payload dtype, shape and bytes",
            category="lowering-defect",
        )
    return body


def _recurse(body: Any, site: _Site, ancestors: frozenset[int], role: str) -> None:
    from jax2onnx.converter.source_profile import SourcePrecisionError, _validate_scope

    try:
        _validate_scope(body, site.opset, ancestors)
    except SourcePrecisionError as exc:
        raise exc.with_context(f"{site.location}/{role}") from exc


def _native_results(eqn: Any, site: _Site) -> None:
    """Preserve native result weakness without strengthening carry invariants."""
    primitive = jax.lax.scan_p if site.primitive == "scan" else jax.lax.while_p
    site.require(
        eqn.primitive is primitive,
        "loop equation does not use the actual native primitive",
        "the installed JAX scan/while primitive, never an object-provided evaluator",
        category="lowering-defect",
    )
    try:
        inferred, effects = primitive.abstract_eval(
            *(var.aval for var in eqn.invars), **eqn.params
        )
    except (AssertionError, TypeError, ValueError) as exc:
        site.fail(
            f"native loop abstract signature is malformed: {exc}",
            "native attributes and typed body roles consistent with the actual operands",
            category="lowering-defect",
        )
    if not isinstance(inferred, (tuple, list)):
        site.fail(
            "native loop abstract evaluation did not return multiple result avals",
            "the actual native scan/while multiple-result signature",
            category="lowering-defect",
        )
    site.require(
        not effects
        and _signature([SimpleNamespace(aval=aval) for aval in inferred], site)
        == _signature(eqn.outvars, site),
        "loop result dtype, shape or weakness differs from native abstract evaluation",
        "the exact installed native result avals, including version-specific weakness",
        category="lowering-defect",
    )


def validate_loop_equation(eqn: Any, site: _Site, ancestors: frozenset[int]) -> None:
    from jax2onnx.converter.source_profile import _params

    inputs, outputs = _signature(eqn.invars, site), _signature(eqn.outvars, site)
    site.require(
        site.opset in ({23, 26} if site.primitive == "scan" else {18, 23, 26}),
        "loop carrier opset is outside the audited existing routes",
        "forward scan at opset23/26 or while at opset18/23/26, with every child operation retaining its version checks",
    )
    if site.primitive == "while":
        fields = {"cond_jaxpr", "body_jaxpr", "cond_nconsts", "body_nconsts"}
        _params(eqn, fields, site, required=fields)
        nc, nb = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
        site.require(
            type(nc) is int
            and type(nb) is int
            and nc >= 0
            and nb >= 0
            and nc + nb < len(inputs),
            "while constant counts or carry arity are malformed",
            "nonnegative native integer constant counts and at least one carried value",
            category="lowering-defect",
        )
        condition = _closed_body(
            eqn.params["cond_jaxpr"],
            replace(site, location=f"{site.location}/while-cond"),
            ancestors,
        )
        body = _closed_body(
            eqn.params["body_jaxpr"],
            replace(site, location=f"{site.location}/while-body"),
            ancestors,
        )
        carry = inputs[nc + nb :]
        condition_inputs = _signature(condition.invars, site)
        condition_outputs = _signature(condition.outvars, site)
        body_inputs, body_outputs = _signature(body.invars, site), _signature(
            body.outvars, site
        )
        site.require(
            condition_inputs == inputs[:nc] + carry
            and body_inputs == inputs[nc : nc + nb] + carry
            and len(body_outputs) == len(outputs) == len(carry)
            and tuple(value for value, _ in body_outputs)
            == tuple(value for value, _ in carry),
            "while condition/body/carry signatures disagree",
            "exact constant/carry bindings and unchanged logical carry dtype/shape; native result weakness is checked separately",
            category="lowering-defect",
        )
        site.require(
            len(condition_outputs) == 1 and condition_outputs[0][0].dtype == "bool",
            "while condition must return exactly one Boolean predicate",
            "one native scalar or existing batched Boolean predicate",
            category="lowering-defect",
        )
        predicate = condition_outputs[0][0].shape
        site.require(
            not predicate
            or all(value.shape[: len(predicate)] == predicate for value, _ in carry),
            "batched while predicate is not a prefix of every logical carry",
            "matching logical batch-prefix extents before physical complex channel axes",
            category="lowering-defect",
        )
        _recurse(condition, site, ancestors, "while-cond")
        _recurse(body, site, ancestors, "while-body")
    else:
        common = {"jaxpr", "reverse", "length", "unroll"}
        modern = {"ft_in", "ft_out"}
        legacy = {"num_consts", "num_carry"}
        fields = common | (modern if "ft_in" in eqn.params else legacy)
        _params(eqn, fields, site, required=fields)
        site.require(
            type(eqn.params["reverse"]) is bool
            and type(eqn.params["length"]) is int
            and eqn.params["length"] >= 0
            and type(eqn.params["unroll"]) is int
            and eqn.params["unroll"] >= 0,
            "scan direction, length or unroll metadata is malformed",
            "a native Boolean direction and nonnegative native integer length/unroll",
            category="lowering-defect",
        )
        site.require(
            not eqn.params["reverse"],
            "reverse scan is not implemented by the existing exporter",
            "the existing forward-scan lowering; reverse remains an exporter implementation gap",
        )
        body = _closed_body(
            eqn.params["jaxpr"],
            replace(site, location=f"{site.location}/scan-body"),
            ancestors,
        )
        if "ft_in" not in eqn.params:
            site.require(
                all(
                    type(eqn.params[key]) is int and eqn.params[key] >= 0
                    for key in legacy
                ),
                "scan constant/carry counts are malformed",
                "nonnegative native integer const/carry counts without coercion",
                category="lowering-defect",
            )
        partitions = native_scan_flat_tree_partitions(eqn.params)
        if partitions is not None:
            incoming, outgoing, supported = partitions
            nc, nk, nx = incoming
            site.require(
                sum(incoming) == len(body.invars)
                and outgoing == (nk, len(body.outvars) - nk),
                "scan flat-tree input/output role partitions are malformed",
                "actual native three-way const/carry/xs and two-way carry/ys flat trees",
                category="lowering-defect",
            )
            site.require(
                supported,
                "native top-level filtered/pytree scan input descriptor needs lowerer support",
                "the existing top-level input tuple partition; preserve this transformed-input-descriptor implementation gap until the shared scan lowerer supports it",
            )
        else:
            nc, nk, nx = scan_arity(eqn.params, len(body.invars))
        step_inputs, step_outputs = _signature(body.invars, site), _signature(
            body.outvars, site
        )
        length = eqn.params["length"]
        xs = inputs[nc + nk :]
        site.require(
            len(inputs) == nc + nk + nx
            and len(step_outputs) >= nk
            and len(outputs) == len(step_outputs)
            and all(value.shape and value.shape[0] == length for value, _ in xs),
            "scan operand/result counts or sequence extents disagree",
            "matching const/carry/xs/ys counts and every xs leading extent equal to native length",
            category="lowering-defect",
        )
        expected_inputs = inputs[: nc + nk] + tuple(
            (replace(value, shape=value.shape[1:]), weak) for value, weak in xs
        )
        site.require(
            step_inputs == expected_inputs
            and tuple(value for value, _ in step_outputs[:nk])
            == tuple(value for value, _ in inputs[nc : nc + nk]),
            "scan constant/carry/element signatures disagree",
            "exact body operand bindings and unchanged carry dtype/shape; preserve native result weakness",
            category="lowering-defect",
        )
        _recurse(body, site, ancestors, "scan-body")  # Also inspect a zero-trip body.
    site.require(
        all(eqn.params.get(key) is None for key in ("sharding", "out_sharding")),
        "loop placement is outside the audited native carrier",
        "unspecified loop placement",
    )
    _native_results(eqn, site)
