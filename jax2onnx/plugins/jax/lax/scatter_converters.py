# jax2onnx/plugins/jax/lax/scatter_converters.py
from __future__ import annotations

from typing import Any, Optional, Tuple
from jax.lax import ScatterDimensionNumbers

from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

from .scatter_utils import prepare_and_emit_scatternd  # reuse helpers

# ---------- small normalizers (dimension_numbers / mode) ----------


def _normalize_dnums(dnums_like: Any) -> ScatterDimensionNumbers:
    """
    Accept either a real ScatterDimensionNumbers or a lightweight object/dict
    with the same fields and return a ScatterDimensionNumbers.
    """
    if isinstance(dnums_like, ScatterDimensionNumbers):
        return dnums_like
    # object with attributes

    def _get(name):
        if isinstance(dnums_like, dict):
            return tuple(dnums_like.get(name, ()))
        return tuple(getattr(dnums_like, name, ()))

    return ScatterDimensionNumbers(
        update_window_dims=_get("update_window_dims"),
        inserted_window_dims=_get("inserted_window_dims"),
        scatter_dims_to_operand_dims=_get("scatter_dims_to_operand_dims"),
        operand_batching_dims=_get("operand_batching_dims"),
        scatter_indices_batching_dims=_get("scatter_indices_batching_dims"),
    )


def _normalize_mode(params: dict) -> Optional[Any]:
    """
    Try a few common keys; returns a GatherScatterMode or a string or None.
    """
    for k in ("mode", "scatter_mode", "gather_scatter_mode"):
        if k in params and params[k] is not None:
            return params[k]
    return None


# ---------- shared implementation ----------


def _convert_scatter_common(
    s: "Jaxpr2OnnxConverter",
    *,
    operand_v: Any,
    indices_v: Any,
    updates_v: Any,
    params: dict,
    reduction: str,
    onnx_out_var: Any,
) -> str:
    """
    - Normalizes dnums & mode
    - Calls prepare_and_emit_scatternd(...)
    - Forces the produced ONNX output name to match the framework’s desired name
    """
    # Params are typically on eqn.params; keep this tolerant
    dnums_like = params.get("dimension_numbers")
    if dnums_like is None:
        raise ValueError("scatter converters: missing 'dimension_numbers' in params.")
    dnums = _normalize_dnums(dnums_like)

    mode = _normalize_mode(params)

    # Make the ONNX output name identical to the JAX var’s chosen name
    out_name = s.get_name(onnx_out_var)

    return prepare_and_emit_scatternd(
        s,
        operand_v=operand_v,
        indices_v=indices_v,
        updates_v=updates_v,
        dimension_numbers=dnums,
        scatter_mode=mode,
        reduction=reduction,
        out_name=out_name,
    )


# ---------- 4 public converters (wire these in your registry) ----------


def convert_lax_scatter(
    s: "Jaxpr2OnnxConverter",
    eqn: Any,
    invars: Tuple[Any, Any, Any],
    outvars: Tuple[Any],
) -> str:
    operand_v, indices_v, updates_v = invars
    return _convert_scatter_common(
        s,
        operand_v=operand_v,
        indices_v=indices_v,
        updates_v=updates_v,
        params=getattr(eqn, "params", {}),
        reduction="none",  # replace semantics
        onnx_out_var=outvars[0],
    )


def convert_lax_scatter_add(
    s: "Jaxpr2OnnxConverter",
    eqn: Any,
    invars: Tuple[Any, Any, Any],
    outvars: Tuple[Any],
) -> str:
    operand_v, indices_v, updates_v = invars
    return _convert_scatter_common(
        s,
        operand_v=operand_v,
        indices_v=indices_v,
        updates_v=updates_v,
        params=getattr(eqn, "params", {}),
        reduction="add",
        onnx_out_var=outvars[0],
    )


def convert_lax_scatter_min(
    s: "Jaxpr2OnnxConverter",
    eqn: Any,
    invars: Tuple[Any, Any, Any],
    outvars: Tuple[Any],
) -> str:
    operand_v, indices_v, updates_v = invars
    return _convert_scatter_common(
        s,
        operand_v=operand_v,
        indices_v=indices_v,
        updates_v=updates_v,
        params=getattr(eqn, "params", {}),
        reduction="min",
        onnx_out_var=outvars[0],
    )


def convert_lax_scatter_max(
    s: "Jaxpr2OnnxConverter",
    eqn: Any,
    invars: Tuple[Any, Any, Any],
    outvars: Tuple[Any],
) -> str:
    operand_v, indices_v, updates_v = invars
    return _convert_scatter_common(
        s,
        operand_v=operand_v,
        indices_v=indices_v,
        updates_v=updates_v,
        params=getattr(eqn, "params", {}),
        reduction="max",
        onnx_out_var=outvars[0],
    )


def convert_lax_scatter_mul(
    s: "Jaxpr2OnnxConverter",
    eqn: Any,
    invars: Tuple[Any, Any, Any],
    outvars: Tuple[Any],
) -> str:
    """
    Style B converter for lax.scatter_mul → ONNX ScatterND(reduction='mul').
    Mirrors convert_lax_scatter_{add,min,max}.
    """
    operand_v, indices_v, updates_v = invars
    return _convert_scatter_common(
        s,
        operand_v=operand_v,
        indices_v=indices_v,
        updates_v=updates_v,
        params=getattr(eqn, "params", {}),
        reduction="mul",
        onnx_out_var=outvars[0],
    )
