# jax2onnx/plugins/jax/lax/split.py

"""ONNX plugin for the JAX `lax.split` primitive."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.split_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.split.html",
    onnx=[{"component": "Split", "doc": "https://onnx.ai/onnx/operators/onnx__Split.html"}],
    since="v0.7.2",
    context="primitives.lax",
    component="split",
    testcases=[
        {
            "testcase": "lax_split_equal_parts",
            # FIX: Use jnp.split, which correctly lowers to the lax.split primitive
            # with the 'sizes' parameter calculated. This also matches the GRUCell use case.
            "callable": lambda x: jnp.split(x, 2, axis=1),
            "input_shapes": [(4, 6)],
            "opset_version": 13,
        },
        {
            "testcase": "lax_split_unequal_parts",
            "callable": lambda x: jnp.split(x, [2, 5], axis=1),
            "input_shapes": [(4, 9)],
            "opset_version": 13,
        },
    ],
)
class LaxSplitPlugin(PrimitiveLeafPlugin):
    """Plugin for converting `jax.lax.split` to ONNX `Split`."""

    def to_onnx(
        self,
        conv: "Jaxpr2OnnxConverter",
        invars: List[jax.core.Var],
        outvars: List[jax.core.Var],
        params: Dict[str, Any],
    ):
        """Lower the `split` primitive to an ONNX `Split` node."""
        axis: int = params["axis"]
        sizes: list[int] = params["sizes"]

        inp_name = conv.get_name(invars[0])
        out_names = [conv.get_name(v) for v in outvars]

        # FIX: Use the correct converter method 'get_constant_name' to create the tensor.
        split_tensor_name = conv.get_constant_name(np.array(sizes, dtype=np.int64))
        node_inputs = [inp_name, split_tensor_name]

        split_node = helper.make_node(
            "Split",
            inputs=node_inputs,
            outputs=out_names,
            name=conv.get_unique_name("lax_split"),
            axis=axis,
        )
        conv.add_node(split_node)