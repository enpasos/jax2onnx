# jax2onnx/plugins/equinox/eqx/nn/identity.py
"""
ONNX plugin for equinox.nn.Identity.

This plugin provides a direct mapping from `equinox.nn.Identity` to the
corresponding ONNX `Identity` operator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import equinox as eqx
from jax import core
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# -----------------------------------------------------------------------------
# 1. Primitive Definition
# -----------------------------------------------------------------------------
eqx.nn.identity_p = Primitive("eqx.nn.identity")
eqx.nn.identity_p.multiple_results = False


# -----------------------------------------------------------------------------
# 2. Plugin Registration and Definition
# -----------------------------------------------------------------------------
@register_primitive(
    jaxpr_primitive=eqx.nn.identity_p.name,
    jax_doc="https://docs.kidger.site/equinox/api/nn/linear/#equinox.nn.Identity",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.7.1",
    context="primitives.eqx",
    component="identity",
    testcases=[
        {
            "testcase": "eqx_identity_static",
            "callable": eqx.nn.Identity(),
            "input_shapes": [(10, 20)],
            "post_check_onnx_graph": lambda m: (
                any(node.op_type == "Identity" for node in m.graph.node)
            ),
        },
        {
            "testcase": "eqx_identity_symbolic_batch",
            "callable": eqx.nn.Identity(),
            "input_shapes": [("B", 32)],
            "post_check_onnx_graph": lambda m: (
                any(node.op_type == "Identity" for node in m.graph.node)
            ),
        },
    ],
)
class EqxIdentityPlugin(PrimitiveLeafPlugin):
    """Convert **equinox.nn.Identity** to an ONNX Identity operator."""

    _ORIGINAL_IDENTITY_CALL: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(x: core.ShapedArray) -> core.ShapedArray:
        """The abstract_eval rule for Identity is trivial: output is same as input."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: list,
        node_outputs: list,
        params: dict,
    ):
        """Maps the primitive to an ONNX Identity node."""
        # There is only one input and one output
        x_var = node_inputs[0]
        y_var = node_outputs[0]

        x_name = s.get_name(x_var)
        y_name = s.get_name(y_var)

        # Create the ONNX Identity node
        s.add_node(
            helper.make_node(
                "Identity",
                inputs=[x_name],
                outputs=[y_name],
                name=s.get_unique_name("identity"),
            )
        )
        # The output shape and type are identical to the input
        s.add_shape_info(y_name, x_var.aval.shape, x_var.aval.dtype)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return a patched version of __call__ that uses the primitive."""
        EqxIdentityPlugin._ORIGINAL_IDENTITY_CALL = orig_fn

        def patched_call(self, x, *, key=None):
            # Bind the input to our custom primitive.
            # The original __call__ signature includes an optional `key`.
            return eqx.nn.identity_p.bind(x)

        return patched_call

    @staticmethod
    def patch_info() -> dict:
        """Specifies the target for monkey-patching."""
        return {
            "patch_targets": [eqx.nn.Identity],
            "patch_function": EqxIdentityPlugin.get_monkey_patch,
            "target_attribute": "__call__",
        }


# -----------------------------------------------------------------------------
# 3. Register Primitive Rules
# -----------------------------------------------------------------------------
# Link the primitive to its abstract evaluation implementation.
eqx.nn.identity_p.def_abstract_eval(EqxIdentityPlugin.abstract_eval)


def _eqx_identity_batching_rule(batched_args, batch_dims, **_):
    """
    The batching rule for an identity operation.
    The result of vmap(identity)(x) is just x itself.
    """
    (x,) = batched_args
    (bd,) = batch_dims
    return x, bd


# Register the batching rule for the primitive.
batching.primitive_batchers[eqx.nn.identity_p] = _eqx_identity_batching_rule
