# file: jax2onnx/plugins/jax/lax/select.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import jax.numpy as jnp
from jax import lax
from jax.extend.core import Var
from onnx import helper

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.select")

# ---------------------------------------------------------------------------
# Registry metadata
# ---------------------------------------------------------------------------


@register_primitive(
    jaxpr_primitive="select",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.select.html",
    onnx=[
        {"component": "Where", "doc": "https://onnx.ai/onnx/operators/onnx__Where.html"}
    ],
    since="v0.7.1",
    context="primitives.lax",
    component="select",
    # Testcases *must* satisfy native JAX shape rules.
    testcases=[
        {
            "testcase": "select_simple",
            "callable": lambda c, x, y: lax.select(c, x, y),
            "input_shapes": [(3,), (3,), (3,)],
            "input_dtypes": [jnp.bool_, jnp.float32, jnp.float32],
            "expected_output_shapes": [(3,)],
        },
        {
            # Mask  (B, H, T, T)  – mask already expanded
            # Scores(B, H, T, T)
            # Else   (B, H, T, T) – full tensor, **not** scalar
            "testcase": "select_mask_scores_tensor_else",
            "callable": lambda m, s, z: lax.select(m, s, z),
            "input_shapes": [
                ("B", 12, "T", "T"),
                ("B", 12, "T", "T"),
                ("B", 12, "T", "T"),
            ],
            "input_dtypes": [jnp.bool_, jnp.float32, jnp.float32],
            "expected_output_shapes": [("B", 12, "T", "T")],
        },
    ],
)
class SelectPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.select`` (boolean 2‑case) to ONNX `Where`."""

    # ---------------------------------------------------------------------
    # ONNX lowering
    # ---------------------------------------------------------------------
    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        """
        Map
            out = lax.select(pred, on_true, on_false)
        to
            out = Where(pred, on_true, on_false)
        Assumes `pred`, `on_true`, `on_false` already share the same shape.
        """
        cond_v, x_v, y_v = node_inputs
        out_v = node_outputs[0]

        cond_name = s.get_name(cond_v)
        x_name = s.get_name(x_v)
        y_name = s.get_name(y_v)
        out_name = s.get_name(out_v)

        # Emit ONNX node
        s.add_node(
            helper.make_node(
                "Where", inputs=[cond_name, x_name, y_name], outputs=[out_name]
            )
        )
        s.add_shape_info(out_name, out_v.aval.shape, out_v.aval.dtype)

    # ---------------------------------------------------------------------
    # No runtime patching – native JAX rules apply
    # ---------------------------------------------------------------------
    @staticmethod
    def patch_info():
        """
        No patches are needed: we rely on users/tests providing inputs that
        already satisfy `lax.select`'s shape requirements.
        """
        return None
