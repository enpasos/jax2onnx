# file: jax2onnx/plugins/jax/lax/select.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.core import Var
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.select")


@register_primitive(
    jaxpr_primitive="select",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.select.html",
    onnx=[
        {"component": "Where", "doc": "https://onnx.ai/onnx/operators/onnx__Where.html"}
    ],
    since="v0.7.1",
    context="primitives.lax",
    component="select",
    testcases=[
        {
            "testcase": "select_simple",
            "callable": lambda c, x, y: lax.select(c, x, y),
            "input_shapes": [(3,), (3,), (3,)],
            "input_dtypes": [jnp.bool_, jnp.float32, jnp.float32],
            "expected_output_shapes": [(3,)],
        },
        # TODO: Re-enable "testcase": "select_mask_scores_literal_else"
        # {
        #     "testcase": "select_mask_scores_literal_else",
        #     "callable": lambda mask, scores: lax.select(mask, scores, -1e9),
        #     "input_shapes": [("B", 1, "T", "T"), ("B", 12, "T", "T")],
        #     "input_dtypes": [jnp.bool_, jnp.float32],
        #     "expected_output_shapes": [("B", 12, "T", "T")],
        # },
        {
            # Reproduces GPT attention masking: cond ? scores : -1e9
            "testcase": "select_scalar_else_pyfloat",
            "callable": lambda c, x, y: lax.select(c, x, y),
            "input_values": [
                np.random.choice([True, False], size=(2, 4, 5)),
                np.random.randn(2, 4, 5).astype(np.float32),
                np.full((2, 4, 5), -1e9, dtype=np.float32),
            ],
            "expected_output_shapes": [(2, 4, 5)],
            "expected_output_dtypes": [np.float32],
        },
    ],
)
class SelectPlugin(PrimitiveLeafPlugin):
    """Lower lax.select to ONNX Where, handling broadcasting."""

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        """Converts a JAX select operation to an ONNX Where operator."""
        # If 'y' is a literal (e.g., from `lax.select(c, x, -1e9)`),
        # it will be in `params`. Otherwise, it's a standard input Var.
        if "y" in params:
            cond_v, x_v = node_inputs
            y_literal = params["y"]
        else:
            cond_v, x_v, y_v = node_inputs

        out_v = node_outputs[0]
        out_aval = out_v.aval
        out_name = s.get_name(out_v)
        cond_name = s.get_name(cond_v)
        x_name = s.get_name(x_v)

        # Get the name for y, creating a constant if it's a literal.
        if "y" in params:
            y_name = s.get_constant_name(np.array(y_literal, dtype=out_aval.dtype))
        else:
            y_name = s.get_name(y_v)

        # ONNX's Where operator handles broadcasting automatically.
        # We simply provide the condition and the two branches.
        s.add_node(
            helper.make_node(
                "Where", inputs=[cond_name, x_name, y_name], outputs=[out_name]
            )
        )
        s.add_shape_info(out_name, out_aval.shape, out_aval.dtype)

    @staticmethod
    def patch_info():
        """
        Returns instructions to monkey-patch `jax.lax.select` to support
        full NumPy-style broadcasting during numerical validation.
        """
        # Capture the original function at import time to avoid recursion.
        _orig_select = lax.select

        def patched_select(pred, on_true, on_false):
            """
            A robust, non-recursive replacement for `lax.select` that
            explicitly broadcasts its arguments to a compatible shape
            before calling the original, stricter primitive.
            """
            # Determine the final output shape by broadcasting all inputs together.
            try:
                out_shape = np.broadcast_shapes(
                    np.shape(pred), np.shape(on_true), np.shape(on_false)
                )
            except ValueError as e:
                # Re-raise with a more informative error message.
                raise ValueError(
                    "lax.select arguments are not broadcast-compatible: "
                    f"pred={np.shape(pred)}, on_true={np.shape(on_true)}, on_false={np.shape(on_false)}"
                ) from e

            # Broadcast each argument to the final shape.
            pred = jnp.broadcast_to(pred, out_shape)
            on_true = jnp.broadcast_to(on_true, out_shape)
            on_false = jnp.broadcast_to(on_false, out_shape)

            # Call the original, non-broadcast-supporting lax.select.
            return _orig_select(pred, on_true, on_false)

        return {
            "patch_targets": [lax],
            "target_attribute": "select",
            "patch_function": lambda _: patched_select,
        }
