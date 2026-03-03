# jax2onnx/plugins/jax/lax/schur.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _stamp_like(value: Any, ref: Any) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.schur_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.schur.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="schur",
    testcases=[
        {
            "testcase": "schur_1x1_default",
            "callable": lambda x: jax.lax.linalg.schur(x),
            "input_values": [np.asarray([[3.0]], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Identity"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "schur_1x1_no_vectors",
            "callable": lambda x: jax.lax.linalg.schur(x, compute_schur_vectors=False),
            "input_values": [np.asarray([[-1.5]], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Identity"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SchurPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.schur`` for static square ``1x1`` matrices."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        params = dict(getattr(eqn, "params", {}) or {})
        compute_vecs = bool(params.get("compute_schur_vectors", True))
        select_callable = params.get("select_callable", None)
        if select_callable is not None:
            raise NotImplementedError("schur select_callable is not supported yet")

        (x_var,) = eqn.invars
        outvars = list(eqn.outvars)
        expected_n = 2 if compute_vecs else 1
        if len(outvars) != expected_n:
            raise NotImplementedError(
                f"schur output arity mismatch: expected {expected_n}, got {len(outvars)}"
            )

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("schur_in"))
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if len(x_shape) != 2:
            raise NotImplementedError("schur currently supports rank-2 inputs only")
        n_rows_raw, n_cols_raw = x_shape
        if not isinstance(n_rows_raw, (int, np.integer)) or not isinstance(
            n_cols_raw, (int, np.integer)
        ):
            raise NotImplementedError("schur requires static matrix dimensions")
        n_rows = int(n_rows_raw)
        n_cols = int(n_cols_raw)
        if n_rows != n_cols:
            raise ValueError("schur requires square matrices")
        if n_rows != 1:
            raise NotImplementedError(
                "schur currently supports only 1x1 matrices; larger decompositions are pending"
            )

        t_var = outvars[0]
        t_spec = ctx.get_value_for_var(t_var, name_hint=ctx.fresh_name("schur_t"))
        t_name = getattr(t_spec, "name", None) or ctx.fresh_name("schur_t")
        t_out = ctx.builder.Identity(x, _outputs=[t_name])
        _stamp_like(t_out, t_spec if getattr(t_spec, "type", None) else x)
        if getattr(t_spec, "shape", None) is not None:
            t_out.shape = t_spec.shape
        ctx.bind_value_for_var(t_var, t_out)

        if not compute_vecs:
            return

        q_var = outvars[1]
        q_spec = ctx.get_value_for_var(q_var, name_hint=ctx.fresh_name("schur_q"))
        q_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(q_var, "aval", None), "dtype", np.float32)
        )
        q_const = ctx.bind_const_for_var(object(), np.asarray([[1]], dtype=q_dtype))
        q_name = getattr(q_spec, "name", None) or ctx.fresh_name("schur_q")
        q_out = ctx.builder.Identity(q_const, _outputs=[q_name])
        _stamp_like(q_out, q_spec if getattr(q_spec, "type", None) else q_const)
        if getattr(q_spec, "shape", None) is not None:
            q_out.shape = q_spec.shape
        ctx.bind_value_for_var(q_var, q_out)
