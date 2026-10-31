# jax2onnx/converter/source_top_k.py

"""Canonical TopK with native word order and an explicitly selected representation."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any, TYPE_CHECKING

from jax2onnx._compat.jax import (
    is_unpartitioned_prng_placement,
    native_top_k_parameter_names,
    native_top_k_primitive,
)

if TYPE_CHECKING:
    from jax2onnx.converter.source_profile import _Site, _Tensor
    from jax2onnx.converter.typing_support import NativeLoweringContext


_INT64_MAX: int = (1 << 63) - 1
_TOP_K_INTEGERS: frozenset[str] = frozenset(
    {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
)
_TOP_K_FLOATS: frozenset[str] = frozenset({"float16", "bfloat16", "float32", "float64"})


def native_top_k_result_is_empty(shape: Sequence[object], k: int) -> bool:
    """Prove emptiness without comparing a native symbolic dimension."""
    return k == 0 or any(
        type(dimension) is int and dimension == 0 for dimension in shape
    )


def validate_native_top_k(
    eqn: Any,
    inputs: Sequence[_Tensor],
    outputs: Sequence[_Tensor],
    formal_symbols: set[tuple[str, object | None]],
    site: _Site,
) -> None:
    from jax2onnx.converter.source_profile import _arity, _same_shape

    site.require(
        eqn.primitive is native_top_k_primitive(),
        "TopK equation does not use the canonical native primitive",
        "the frozen installed lax.top_k_p and its defining alias",
        category="lowering-defect",
    )
    _arity(eqn, 1, 2, site)
    schema = native_top_k_parameter_names()
    site.require(
        not eqn.effects and type(eqn.params) is dict and set(eqn.params) == schema,
        "native TopK effects or parameter schema differs",
        "the exact installed k/axis schema, with is_stable only on its native owner",
        category="lowering-defect",
    )
    if "is_stable" in schema:
        site.require(
            type(eqn.params["is_stable"]) is bool,
            "native TopK stability has a malformed type",
            "the actual native Boolean stability parameter",
            category="lowering-defect",
        )
    source = inputs[0]
    values, indices = outputs
    site.require(
        source.dtype in _TOP_K_FLOATS | _TOP_K_INTEGERS,
        "native TopK dtype needs another exact key adapter",
        "FP16/BF16/FP32/FP64 words or exact integer keys, including full UINT64 words",
    )
    axis, k = eqn.params["axis"], eqn.params["k"]
    site.require(
        type(axis) is int and 0 <= axis < len(source.shape),
        "native TopK axis differs from its canonical rank contract",
        "one canonical nonnegative integer axis within a non-scalar input",
        category="lowering-defect",
    )
    site.require(
        type(k) is int and k >= 0,
        "native TopK k differs from its integer contract",
        "one nonnegative integer k, without Boolean or symbolic sizes",
        category="lowering-defect",
    )
    site.require(
        all(not isinstance(dim, int) or dim >= 0 for dim in source.shape),
        "native TopK shape has a negative static extent",
        "nonnegative native static dimensions",
        category="lowering-defect",
    )
    empty = native_top_k_result_is_empty(source.shape, k)
    native_extent = source.shape[axis]
    if type(native_extent) is int:
        extent = native_extent
        site.require(
            k <= extent and (empty or extent <= (1 << 31)),
            "native TopK k or extent differs from its index contract",
            "k <= static extent; a nonempty selection additionally requires extent <= 2**31 for native INT32 indices",
            category="lowering-defect",
        )
    else:
        site.require(
            k == 0
            and type(native_extent) is str
            and source.symbol is not None
            and source.symbol_axis == axis
            and native_extent == source.symbol[0]
            and source.symbol in formal_symbols,
            "TopK of a symbolic extent needs its own index-bound proof",
            "k=0 on the actual atomic native B in its real formal scope; positive-k symbolic selection remains unfinished lowering work",
        )
    site.require(
        values.dtype == source.dtype
        and indices.dtype == "int32"
        and all(
            variable.aval.weak_type is eqn.invars[0].aval.weak_type
            for variable in eqn.outvars
        ),
        "native TopK result dtype or weakness differs",
        "original selected values and native INT32 indices with unchanged weakness",
        category="lowering-defect",
    )
    expected = list(source.shape)
    expected[axis] = k
    removes_symbol = source.symbol is not None and source.symbol_axis == axis
    expected_symbol = None if removes_symbol else source.symbol
    expected_symbol_axis = None if removes_symbol else source.symbol_axis
    for target in outputs:
        _same_shape(target.shape, tuple(expected), site)
        site.require(
            target.symbol == expected_symbol
            and target.symbol_axis == expected_symbol_axis
            and (target.symbol is None or target.symbol in formal_symbols),
            "native TopK result batch scope differs from actual formals",
            "the exact retained atomic B/scope/axis, or no output symbol when k=0 replaces its selected axis",
        )
    for variable in (*eqn.invars, *eqn.outvars):
        site.require(
            is_unpartitioned_prng_placement(
                variable.aval.sharding, len(variable.aval.shape)
            ),
            "native TopK placement needs an exact adapter",
            "ordinary absent/empty unpartitioned native placement",
        )
    for value in (*inputs, *outputs):
        site.require(
            all(not isinstance(dim, int) or dim <= _INT64_MAX for dim in value.shape)
            and prod(dim for dim in value.shape if isinstance(dim, int)) <= _INT64_MAX,
            "native TopK shape exceeds signed-int64 tensor limits",
            "representable native dimensions and each static tensor-size factor",
        )
    if (
        not empty
        and (source.dtype in _TOP_K_FLOATS or source.dtype == "uint64")
        and site.opset < 26
    ):
        site.fail(
            "exact native word TopK ordering requires ONNX BitCast at opset 26",
            "lossless FP16/BF16/FP32/FP64 total-order and full UINT64 keys, preserving original selected words",
            "select an explicit opset=26 target whose ONNX checker and runtime provider support BitCast26, INT64 TopK and GatherElements; this request is never upgraded automatically",
            category="representation-limit",
        )


def require_native_top_k_emission(eqn: Any, ctx: NativeLoweringContext) -> None:
    """Recheck the canonical owner and complete shape/type contract at emission."""
    from jax2onnx.converter.source_profile import _Site, _tensor

    site = _Site("native TopK emission", ctx.opset, eqn.primitive.name)
    inputs = [_tensor(variable, site) for variable in eqn.invars]
    outputs = [_tensor(variable, site) for variable in eqn.outvars]
    formals = {
        value.symbol for value in (*inputs, *outputs) if value.symbol is not None
    }
    validate_native_top_k(eqn, inputs, outputs, formals, site)
