# jax2onnx/converter/source_scientific.py

"""Finite native scatter and pure rematerialization contracts for SOURCE.

These contracts preserve the program's native typed bodies. Eligibility at the
original scientific opset21 establishes neither CPU kernels nor numerical parity.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax

from jax2onnx._compat.jax import Jaxpr, native_remat_primitive
from jax2onnx.converter.source_native_contract import validate_native_results

if TYPE_CHECKING:
    from jax2onnx.converter.source_profile import _Site


def validate_scatter_equation(eqn: Any, site: _Site, ancestors: frozenset[int]) -> None:
    """Validate the full static windows already emitted by the shared lowerer."""
    from jax2onnx.converter.source_profile import (
        _arity,
        _axes,
        _params,
        _signature,
        _validate_scope,
    )

    primitives = {
        "scatter": jax.lax.scatter_p,
        "scatter-add": jax.lax.scatter_add_p,
        "scatter-mul": jax.lax.scatter_mul_p,
    }
    site.require(
        site.opset == 21,
        "scatter opset is outside the audited scientific route",
        "the original opset21 ScatterND representation",
    )
    site.require(
        eqn.primitive is primitives.get(site.primitive or ""),
        "scatter equation does not use the actual native primitive",
        "the installed native set/add/multiply scatter primitive",
        category="lowering-defect",
    )
    _arity(eqn, 3, 1, site)
    fields = {
        "dimension_numbers",
        "indices_are_sorted",
        "unique_indices",
        "mode",
        "update_jaxpr",
        "update_consts",
    }
    _params(eqn, fields, site, required=fields)
    signatures = _signature((*eqn.invars, *eqn.outvars), site)
    operand, indices, updates, target = (value for value, _ in signatures)
    site.require(
        all(value.symbol is None for value, _ in signatures)
        and operand.dtype in {"float32", "float64"}
        and target.dtype == updates.dtype == operand.dtype
        and target.shape == operand.shape
        and all(
            type(dim) is int and dim > 0
            for value, _ in signatures
            for dim in value.shape
        ),
        "scatter storage is outside the static scientific contract",
        "positive static FP32/FP64 operand/update windows and unchanged result dtype/shape",
    )
    site.require(
        indices.dtype in {"int32", "int64"} and bool(indices.shape),
        "scatter indices lack the native typed vector axis",
        "unchanged INT32/INT64 indices with an explicit final coordinate axis",
        category="lowering-defect",
    )
    params = eqn.params
    site.require(
        all(
            type(params[key]) is bool
            for key in ("indices_are_sorted", "unique_indices")
        )
        and all(params.get(key) is None for key in ("sharding", "out_sharding")),
        "scatter sorted/unique hints or placement are malformed",
        "native Boolean hints and unspecified placement",
        category="lowering-defect",
    )
    site.require(
        type(params["mode"]) is jax.lax.GatherScatterMode
        and params["mode"]
        in {
            jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
            jax.lax.GatherScatterMode.CLIP,
            jax.lax.GatherScatterMode.FILL_OR_DROP,
        },
        "scatter mode is outside the existing window emitter",
        "the native promise-in-bounds, clip or fill/drop enum",
    )
    site.require(
        site.primitive != "scatter" or params["unique_indices"],
        "overlapping set-scatter ordering is outside the audited representation",
        "native unique_indices=True for assignment; reductions retain their native hints",
    )
    dimensions = params["dimension_numbers"]
    site.require(
        type(dimensions) is jax.lax.ScatterDimensionNumbers,
        "scatter dimension numbers are not the native descriptor",
        "the installed native ScatterDimensionNumbers",
        category="lowering-defect",
    )
    rank = len(operand.shape)
    window = _axes(
        dimensions.update_window_dims, len(updates.shape), site, "update window axes"
    )
    inserted = _axes(
        dimensions.inserted_window_dims, rank, site, "inserted window axes"
    )
    mapped = _axes(
        dimensions.scatter_dims_to_operand_dims, rank, site, "scatter operand axes"
    )
    site.require(
        not dimensions.operand_batching_dims
        and not dimensions.scatter_indices_batching_dims,
        "scatter batching is outside the existing full-window emitter",
        "ordinary unbatched scatter dimensions; explicit native tensor batch axes remain windows",
    )
    site.require(
        bool(mapped)
        and indices.shape[-1] == len(mapped)
        and len(window) + len(inserted) == rank
        and tuple(sorted(window)) == window
        and tuple(sorted(inserted)) == inserted,
        "scatter window/coordinate dimensions disagree",
        "canonical window and inserted dimensions covering the operand and one coordinate per mapped axis",
        category="lowering-defect",
    )
    operand_window = tuple(axis for axis in range(rank) if axis not in inserted)
    update_batches = tuple(
        updates.shape[axis] for axis in range(len(updates.shape)) if axis not in window
    )
    site.require(
        update_batches == indices.shape[:-1]
        and all(
            updates.shape[u] <= operand.shape[o]
            for o, u in zip(operand_window, window, strict=True)
        ),
        "scatter update window or index-batch extents disagree",
        "matching native index/update batch extents and update windows bounded by their actual operand axes",
        category="lowering-defect",
    )
    site.require(
        type(params["update_consts"]) is tuple and not params["update_consts"],
        "scatter reducer captures are outside the native scalar contract",
        "the actual empty update_consts tuple",
        category="lowering-defect",
    )
    reducer = params["update_jaxpr"]
    if site.primitive == "scatter":
        site.require(
            reducer is None,
            "set-scatter has a foreign reducer body",
            "native assignment with update_jaxpr=None",
            category="lowering-defect",
        )
    else:
        operation = jax.lax.add_p if site.primitive == "scatter-add" else jax.lax.mul_p
        site.require(
            isinstance(reducer, Jaxpr)
            and not reducer.constvars
            and not getattr(reducer, "consts", ())
            and not reducer.effects
            and len(reducer.invars) == 2
            and len(reducer.outvars) == len(reducer.eqns) == 1,
            "scatter reducer is not the native pure scalar body",
            "one two-input add/multiply equation, one result and no captures/effects",
            category="lowering-defect",
        )
        scalar = _signature((*reducer.invars, *reducer.outvars), site)
        inner = reducer.eqns[0]
        site.require(
            all(
                value.dtype == operand.dtype and value.shape == () and weak is False
                for value, weak in scalar
            )
            and inner.primitive is operation
            and tuple(inner.invars) == tuple(reducer.invars)
            and tuple(inner.outvars) == tuple(reducer.outvars)
            and not inner.effects
            and (
                not inner.params
                or (
                    operation is jax.lax.mul_p
                    and set(inner.params) == {"out_dtype"}
                    and inner.params["out_dtype"] is None
                )
            ),
            "scatter reducer operation or scalar dtype/shape/weakness differs",
            "the exact native strong scalar add/multiply body in the original data dtype",
            category="lowering-defect",
        )
        _validate_scope(reducer, site.opset, ancestors, site.deployment_target)
    validate_native_results(
        eqn,
        site,
        primitive=primitives[site.primitive or ""],
        parameters=fields | {"sharding", "out_sharding"},
    )


def validate_remat_equation(eqn: Any, site: _Site, ancestors: frozenset[int]) -> None:
    """Inline only an actual pure forward native remat with exact typed ports."""
    from jax2onnx.converter.source_profile import (
        SourcePrecisionError,
        _params,
        _signature,
        _validate_scope,
    )

    site.require(
        site.opset in {21, 23, 26},
        "rematerialization opset is outside the audited forward route",
        "pure forward remat at21/23/26 with recursive operation-specific eligibility",
    )
    site.require(
        eqn.primitive is native_remat_primitive(),
        "rematerialization does not use the actual native primitive",
        "the installed native remat2 primitive without replacement by name",
        category="lowering-defect",
    )
    fields = {"jaxpr", "prevent_cse", "differentiated", "policy"}
    _params(eqn, fields, site, required=fields)
    params = eqn.params
    site.require(
        type(params["prevent_cse"]) is bool
        and params["differentiated"] is False
        and params["policy"] is None
        and all(params.get(key) is None for key in ("sharding", "out_sharding")),
        "rematerialization flags or policy are outside the pure forward contract",
        "native Boolean prevent_cse, differentiated=False, policy=None and unspecified placement",
    )
    body = params["jaxpr"]
    site.require(
        isinstance(body, Jaxpr)
        and not body.constvars
        and not getattr(body, "consts", ())
        and not body.effects,
        "rematerialization body is not a native pure converted-capture Jaxpr",
        "the original concrete forward body with captures converted to explicit operands and empty effects",
        category="lowering-defect",
    )
    incoming, outgoing = _signature(eqn.invars, site), _signature(eqn.outvars, site)
    site.require(
        incoming == _signature(body.invars, site)
        and outgoing == _signature(body.outvars, site)
        and all(
            value.symbol is None and value.dtype != "key<fry>"
            for value, _ in (*incoming, *outgoing)
        ),
        "rematerialization outer/body signatures differ or use unaudited storage",
        "exact static operand/output dtype, shape, weakness and capture relationships",
        category="lowering-defect",
    )
    try:
        _validate_scope(body, site.opset, ancestors, site.deployment_target)
    except SourcePrecisionError as exc:
        raise exc.with_context(f"{site.location}/remat2 forward") from exc
    validate_native_results(
        eqn,
        site,
        primitive=native_remat_primitive(),
        parameters=fields | {"sharding", "out_sharding"},
    )
