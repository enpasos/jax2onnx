# file: jax2onnx/plugins/jax/debug.py

from typing import TYPE_CHECKING

# Instead of importing, we create a Primitive object with the known name.
from jax.extend.core import Primitive
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Create a representation of the internal 'debug_callback' primitive.
debug_callback_p = Primitive("debug_callback")


@register_primitive(
    jaxpr_primitive=debug_callback_p.name,
    since="v0.2.0",
    context="primitives.debug",
    component="debug_callback",
)
class DebugCallbackPlugin(PrimitiveLeafPlugin):
    """
    Handles jax.debug.print and other debug callbacks during conversion.
    The ONNX standard has no equivalent for a debug callback, so this
    handler effectively removes it from the graph by doing nothing.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """
        This primitive is for host-side debugging and has no ONNX equivalent.
        It has no outputs, so we can simply ignore it during conversion.
        """
        pass
