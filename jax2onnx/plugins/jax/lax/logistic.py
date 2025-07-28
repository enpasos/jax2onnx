# jax2onnx/plugins/jax/lax/logistic.py

"""ONNX plugin for the JAX `lax.logistic` primitive."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.logistic_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.logistic.html",
    onnx=[
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        }
    ],
    since="v0.7.4",
    context="primitives.lax",
    component="logistic",
    testcases=[
        {
            "testcase": "lax_logistic_basic",
            "callable": jax.lax.logistic,
            "input_shapes": [(3, 4)],
        },
    ],
)
class LogisticPlugin(PrimitiveLeafPlugin):
    """Plugin for converting `jax.lax.logistic` to ONNX `Sigmoid`."""

    def to_onnx(
        self,
        conv: "Jaxpr2OnnxConverter",
        invars: List[jax.core.Var],
        outvars: List[jax.core.Var],
        params: Dict[str, Any],
    ):
        """Lower the `logistic` primitive to an ONNX `Sigmoid` node."""
        # The logistic function is a direct 1-to-1 mapping to the Sigmoid operator.
        inp_name = conv.get_name(invars[0])
        out_name = conv.get_name(outvars[0])

        sigmoid_node = helper.make_node(
            "Sigmoid",
            inputs=[inp_name],
            outputs=[out_name],
            name=conv.get_unique_name("lax_logistic"),
        )
        conv.add_node(sigmoid_node)
