# jax2onnx/converter/source_split.py

"""Finite canonical native Split transport, retaining the actual input scope."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any, TYPE_CHECKING

from jax2onnx._compat.jax import (
    is_unpartitioned_prng_placement,
    native_split_primitive,
)

if TYPE_CHECKING:
    from jax2onnx.converter.source_profile import _Site, _Tensor
    from jax2onnx.converter.typing_support import LoweringContextProtocol


_INT64_MAX: int = (1 << 63) - 1


def validate_native_split(
    eqn: Any,
    inputs: Sequence[_Tensor],
    outputs: Sequence[_Tensor],
    formal_symbols: set[tuple[str, object | None]],
    site: _Site,
) -> None:
    from jax2onnx.converter.source_profile import (
        _REAL_VALUES,
        _arity,
        _integer_tuple,
        _same_shape,
        _validate_opset16_equation,
    )

    site.require(
        eqn.primitive is native_split_primitive(),
        "Split equation does not use the canonical native primitive",
        "the actual installed native lax.split_p, without a synthetic jnp owner",
        category="lowering-defect",
    )
    _arity(eqn, 1, len(eqn.outvars), site)
    site.require(
        not eqn.effects
        and type(eqn.params) is dict
        and set(eqn.params) == {"axis", "sizes"},
        "native Split effects or parameter schema differs",
        "one pure native input and only the original axis/sizes parameters",
        category="lowering-defect",
    )
    source = inputs[0]
    axis = eqn.params["axis"]
    site.require(
        type(axis) is int and 0 <= axis < len(source.shape),
        "native Split axis differs from its canonical rank contract",
        "one canonical nonnegative integer axis within the native input rank",
        category="lowering-defect",
    )
    sizes = _integer_tuple(eqn.params["sizes"], site, "native split sizes")
    site.require(
        bool(sizes) and len(sizes) == len(outputs),
        "native Split result count differs from its sizes",
        "one output per nonempty native sizes tuple",
        category="lowering-defect",
    )
    site.require(
        type(source.shape[axis]) is int,
        "native Split of the symbolic extent needs a separate size adapter",
        "a static split axis with any actual B retained on its original other axis",
    )
    site.require(
        all(0 <= size <= _INT64_MAX for size in sizes)
        and sum(sizes) == source.shape[axis],
        "native Split sizes differ from the original axis extent",
        "nonnegative INT64 sizes whose exact sum equals the static split axis",
        category="lowering-defect",
    )
    site.require(
        source.dtype in _REAL_VALUES,
        "native complex Split needs its own physical admission",
        "ordinary Boolean/integer/floating transport; complex tensor owners remain separate work",
    )
    _validate_opset16_equation(list(inputs), list(outputs), site)
    for variable in (*eqn.invars, *eqn.outvars):
        site.require(
            is_unpartitioned_prng_placement(
                variable.aval.sharding, len(variable.aval.shape)
            ),
            "native Split placement needs an exact adapter",
            "ordinary absent/empty unpartitioned native placement",
        )
    for target, size, variable in zip(outputs, sizes, eqn.outvars, strict=True):
        site.require(
            target.dtype == source.dtype
            and variable.aval.weak_type is eqn.invars[0].aval.weak_type,
            "native Split changes dtype or weakness",
            "every result retains the original data dtype and weakness",
            category="lowering-defect",
        )
        site.require(
            target.symbol == source.symbol
            and (target.symbol is None or target.symbol in formal_symbols),
            "native Split result batch scope differs from actual formals",
            "the original atomic native B and its exact formal scope",
        )
        expected = list(source.shape)
        expected[axis] = size
        _same_shape(target.shape, tuple(expected), site)
    for value in (*inputs, *outputs):
        site.require(
            all(not isinstance(dim, int) or dim <= _INT64_MAX for dim in value.shape)
            and prod(dim for dim in value.shape if isinstance(dim, int)) <= _INT64_MAX,
            "native Split shape exceeds signed-int64 tensor limits",
            "representable native extents and each static tensor size factor",
        )


def require_native_split_emission(eqn: Any, ctx: LoweringContextProtocol) -> None:
    """Recheck native identity and the complete data/size contract before emission."""
    from jax2onnx.converter.source_profile import _Site, _tensor

    site = _Site("native Split emission", ctx.opset, eqn.primitive.name)
    inputs = [_tensor(variable, site) for variable in eqn.invars]
    outputs = [_tensor(variable, site) for variable in eqn.outvars]
    formals = {
        value.symbol for value in (*inputs, *outputs) if value.symbol is not None
    }
    validate_native_split(eqn, inputs, outputs, formals, site)
