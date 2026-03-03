# jax2onnx/plugins/jax/lax/reduce.py

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np
import jax.numpy as jnp

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._reduce_utils import (
    lower_boolean_reduction,
    lower_reduction,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _canonical_reducer_name(name: str | None) -> str | None:
    if name is None:
        return None

    base = name.split(".")[-1]
    alias_map = {
        "maximum": "max",
        "minimum": "min",
        "logical_and": "and",
        "logical_or": "or",
        "logical_xor": "xor",
    }
    return alias_map.get(base, base)


def _extract_reducer_primitive_name(params: dict[str, Any]) -> str | None:
    jaxpr_obj = params.get("jaxpr") or params.get("computation")
    jaxpr = getattr(jaxpr_obj, "jaxpr", jaxpr_obj)
    eqns = tuple(getattr(jaxpr, "eqns", ()) or ())
    outvars = tuple(getattr(jaxpr, "outvars", ()) or ())
    if len(eqns) != 1 or len(outvars) != 1:
        return None
    inner = eqns[0]
    inner_outvars = tuple(getattr(inner, "outvars", ()) or ())
    if len(inner_outvars) != 1 or inner_outvars[0] != outvars[0]:
        return None
    primitive = getattr(inner, "primitive", None)
    return _canonical_reducer_name(getattr(primitive, "name", None))


def _extract_scalar_literal(var: Any) -> np.ndarray | None:
    if hasattr(var, "val"):
        arr = np.asarray(getattr(var, "val"))
        if arr.ndim == 0:
            return cast(np.ndarray, arr)
    return None


def _is_identity_init(reducer: str, init_arr: np.ndarray, dtype: np.dtype) -> bool:
    val: Any = init_arr.item()
    if reducer == "add":
        return bool(np.asarray(val == 0))
    if reducer == "mul":
        return bool(np.asarray(val == 1))
    if reducer == "max":
        if np.issubdtype(dtype, np.floating):
            return bool(np.isneginf(val))
        if np.issubdtype(dtype, np.integer):
            return int(val) == np.iinfo(dtype).min
        return False
    if reducer == "min":
        if np.issubdtype(dtype, np.floating):
            return bool(np.isposinf(val))
        if np.issubdtype(dtype, np.integer):
            return int(val) == np.iinfo(dtype).max
        return False
    if reducer == "and":
        if np.issubdtype(dtype, np.bool_):
            return bool(val is True)
        if np.issubdtype(dtype, np.integer):
            return int(val) == np.iinfo(dtype).max
        return False
    if reducer in {"or", "xor"}:
        if np.issubdtype(dtype, np.bool_):
            return bool(val is False)
        if np.issubdtype(dtype, np.integer):
            return int(val) == 0
        return False
    return False


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce.html",
    onnx=[
        {
            "component": "ReduceMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
        },
        {
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="reduce",
    testcases=[
        {
            "testcase": "reduce_max_lambda",
            "callable": lambda x: jax.lax.reduce(
                x,
                -np.inf,
                lambda a, b: jnp.maximum(a, b),
                (1,),
            ),
            "input_values": [
                np.asarray(
                    [[1.0, -2.0, 3.0, 0.5], [0.0, 5.0, 2.0, -1.0]], dtype=np.float32
                )
            ],
            "post_check_onnx_graph": EG(
                ["ReduceMax"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_min_lambda",
            "callable": lambda x: jax.lax.reduce(
                x,
                np.inf,
                lambda a, b: jnp.minimum(a, b),
                (0,),
            ),
            "input_values": [
                np.asarray(
                    [[1.0, -2.0, 3.0], [0.0, 5.0, 2.0], [-1.0, 4.0, 9.0]],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["ReduceMin"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReducePlugin(PrimitiveLeafPlugin):
    """Lower selected ``lax.reduce`` reducers to ONNX reductions."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        params = dict(getattr(eqn, "params", {}) or {})
        reducer = _extract_reducer_primitive_name(params)
        if reducer not in {"add", "mul", "max", "min", "and", "or", "xor"}:
            raise NotImplementedError(
                "reduce currently supports reducer primitives: add/mul/max/min/and/or/xor"
            )

        if len(eqn.invars) != 2:
            raise NotImplementedError(
                "reduce currently supports a single operand and scalar init value"
            )
        init_lit = _extract_scalar_literal(eqn.invars[1])
        if init_lit is None:
            raise NotImplementedError(
                "reduce currently requires a scalar literal init value"
            )

        operand_dtype = np.dtype(getattr(getattr(eqn.invars[0], "aval", None), "dtype"))
        if not _is_identity_init(reducer, init_lit, operand_dtype):
            raise NotImplementedError(
                f"reduce with reducer='{reducer}' currently requires its identity init value"
            )

        dimensions = tuple(int(d) for d in (params.get("dimensions", ()) or ()))
        proxy_eqn = SimpleNamespace(
            invars=[eqn.invars[0]],
            outvars=eqn.outvars,
            params={"axes": dimensions, "keepdims": False},
        )

        if reducer == "add":
            lower_reduction(
                ctx, proxy_eqn, op_type="ReduceSum", allow_dtype_param=False
            )
            return
        if reducer == "mul":
            lower_reduction(
                ctx, proxy_eqn, op_type="ReduceProd", allow_dtype_param=False
            )
            return
        if reducer == "max":
            lower_reduction(
                ctx, proxy_eqn, op_type="ReduceMax", allow_dtype_param=False
            )
            return
        if reducer == "min":
            lower_reduction(
                ctx, proxy_eqn, op_type="ReduceMin", allow_dtype_param=False
            )
            return
        if reducer == "and":
            lower_boolean_reduction(ctx, proxy_eqn, mode="reduce_and")
            return
        if reducer == "or":
            lower_boolean_reduction(ctx, proxy_eqn, mode="reduce_or")
            return
        lower_boolean_reduction(ctx, proxy_eqn, mode="reduce_xor")
