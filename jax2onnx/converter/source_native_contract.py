# jax2onnx/converter/source_native_contract.py

"""Strict native result inference after explicit SOURCE owner contracts.

This shared check does not grant primitive, parameter, body or lowerer admission.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, TYPE_CHECKING

from jax2onnx._compat.jax import Primitive, infer_native_abstract_results

if TYPE_CHECKING:
    from jax2onnx.converter.source_profile import _Site


def validate_native_results(
    eqn: Any, site: _Site, *, primitive: Primitive, parameters: set[str]
) -> None:
    """Compare strict native avals/effects without replacing semantic gates."""
    from jax2onnx.converter.source_profile import _signature

    site.require(
        eqn.primitive is primitive,
        "equation does not use the selected canonical native primitive",
        "the actual native identity selected by the owning contract",
        category="lowering-defect",
    )
    site.require(
        type(eqn.params) is dict and set(eqn.params) <= parameters,
        "native signature has unaudited parameters",
        "only the explicit owning contract's parameter schema",
    )
    # Validate original weakness, shape/scope and dtype before native inference.
    _signature(eqn.invars, site)
    declared = _signature(eqn.outvars, site)
    try:
        inferred, effects = infer_native_abstract_results(eqn, primitive)
    except (AssertionError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        site.fail(
            f"native abstract signature is malformed or unsupported: {exc}",
            "the selected installed native evaluator with original supported avals, attributes and recorded context",
            category="lowering-defect",
        )
    actual = _signature([SimpleNamespace(aval=aval) for aval in inferred], site)
    site.require(
        not eqn.effects and not effects,
        "declared or inferred native equation effects are not empty",
        "the owning pure contract with both recomputed and declared empty effects",
        category="lowering-defect",
    )
    site.require(
        actual == declared
        and all(
            (expected.symbol is None and observed.symbol is None)
            or (
                expected.symbol is not None
                and observed.symbol is not None
                and expected.symbol[0] == observed.symbol[0]
                and expected.symbol[1] is observed.symbol[1]
            )
            for (expected, _), (observed, _) in zip(actual, declared, strict=True)
        ),
        "result count, dtype, shape, weakness or symbolic scope differs from native inference",
        "the exact installed native result signatures, including Boolean weakness and native symbol identity",
        category="lowering-defect",
    )
