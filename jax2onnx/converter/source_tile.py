# jax2onnx/converter/source_tile.py

"""Canonical native Tile semantics, before the shared existing Tile lowerer."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any, TYPE_CHECKING

from jax2onnx._compat.jax import (
    is_unpartitioned_prng_placement,
    native_tile_primitive,
)
from jax2onnx.converter.source_representation import native_shape_contract

if TYPE_CHECKING:
    from jax2onnx.converter.source_profile import _Site, _Tensor
    from jax2onnx.converter.typing_support import LoweringContextProtocol


_INT64_MAX: int = (1 << 63) - 1


def validate_native_tile(
    eqn: Any,
    inputs: Sequence[_Tensor],
    outputs: Sequence[_Tensor],
    formal_symbols: set[tuple[str, object | None]],
    site: _Site,
) -> None:
    from jax2onnx.converter.source_profile import (
        _REAL_VALUES,
        _arity,
        _same_shape,
        _validate_opset16_equation,
    )

    native = native_tile_primitive()
    site.require(
        native is not None and eqn.primitive is native,
        "Tile equation does not use the canonical native primitive",
        "the actual installed native tile_p; older native decompositions keep their own owners",
        category="lowering-defect",
    )
    _arity(eqn, 1, 1, site)
    site.require(
        not eqn.effects and type(eqn.params) is dict and set(eqn.params) == {"reps"},
        "native Tile effects or parameter schema differs",
        "one pure native operand/result and only its original reps tuple",
        category="lowering-defect",
    )
    source, target = inputs[0], outputs[0]
    reps = eqn.params["reps"]
    site.require(
        type(reps) is tuple and len(reps) == len(source.shape) == len(target.shape),
        "native Tile repeat rank differs from the operand/result",
        "a rank-matched native reps tuple",
        category="lowering-defect",
    )
    site.require(
        source.dtype in _REAL_VALUES,
        "native complex Tile needs its own admission on both native schemas",
        "ordinary Boolean/integer/floating transport; existing complex tensor owners remain separate work",
    )
    site.require(
        source.dtype == target.dtype
        and eqn.invars[0].aval.weak_type is eqn.outvars[0].aval.weak_type,
        "native Tile changes dtype or weakness",
        "the original ordinary Boolean/integer/floating dtype and weakness",
        category="lowering-defect",
    )
    _validate_opset16_equation(list(inputs), list(outputs), site)
    for variable in (*eqn.invars, *eqn.outvars):
        site.require(
            is_unpartitioned_prng_placement(
                variable.aval.sharding, len(variable.aval.shape)
            ),
            "native Tile placement needs an exact adapter",
            "ordinary absent/empty unpartitioned native placement",
        )
    try:
        repeated = native_shape_contract(reps)
    except ValueError as exc:
        site.fail("native Tile repeats are outside the atomic shape contract", str(exc))
    for shape in (source.shape, repeated.shape, target.shape):
        site.require(
            all(not isinstance(dim, int) or dim <= _INT64_MAX for dim in shape)
            and prod(dim for dim in shape if isinstance(dim, int)) <= _INT64_MAX,
            "native Tile shape or repeats exceed signed-int64 tensor limits",
            "representable extents/repeats and each static size factor",
        )
    symbols = [
        symbol
        for symbol in (source.symbol, repeated.symbol, target.symbol)
        if symbol is not None
    ]
    site.require(
        all(symbol in formal_symbols for symbol in symbols)
        and all(
            symbol[0] == symbols[0][0] and symbol[1] is symbols[0][1]
            for symbol in symbols
        ),
        "native Tile repeat/result batch scope differs from actual formals",
        "one actual atomic B and its original native scope",
    )
    expected: list[int | str] = []
    for extent, repeat in zip(source.shape, repeated.shape, strict=True):
        if isinstance(extent, str):
            site.require(
                type(repeat) is int and repeat == 1,
                "native Tile derives B squared or another symbolic multiple",
                "an existing B repeated once",
            )
            expected.append(extent)
        elif isinstance(repeat, str):
            site.require(
                extent == 1,
                "native Tile derives an affine symbolic extent",
                "a static singleton repeated by the original B",
            )
            expected.append(repeat)
        else:
            size = extent * repeat
            site.require(
                0 <= size <= _INT64_MAX,
                "native Tile extent exceeds signed-int64 tensor limits",
                "native nonnegative extents representable by ONNX INT64 shapes",
            )
            expected.append(size)
    site.require(
        prod(dim for dim in expected if isinstance(dim, int)) <= _INT64_MAX,
        "native Tile tensor size exceeds signed-int64 limits",
        "a representable static size or static factor of the actual B extent",
    )
    _same_shape(target.shape, tuple(expected), site)


def require_native_tile_emission(eqn: Any, ctx: LoweringContextProtocol) -> None:
    """Recheck owner/type semantics and require actual symbolic repeat origins."""
    from jax2onnx.converter.source_profile import _Site, _tensor

    site = _Site("native Tile emission", ctx.opset, eqn.primitive.name)
    inputs = [_tensor(variable, site) for variable in eqn.invars]
    outputs = [_tensor(variable, site) for variable in eqn.outvars]
    symbols = {
        value.symbol for value in (*inputs, *outputs) if value.symbol is not None
    }
    validate_native_tile(eqn, inputs, outputs, symbols, site)
    repeat_axis = native_shape_contract(eqn.params["reps"]).symbol_axis
    if repeat_axis is not None:
        site.require(
            ctx.get_symbolic_dim_origin(eqn.params["reps"][repeat_axis]) is not None,
            "native Tile symbolic repeat has no actual input origin",
            "the existing native B origin in the enclosing graph/Function context",
            category="lowering-defect",
        )
