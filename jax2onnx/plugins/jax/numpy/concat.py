from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
from typing import TYPE_CHECKING
from jax2onnx.plugin_system import register_primitive, PrimitiveLeafPlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the Concat primitive
jnp.concat_p = Primitive("jnp.concat")
jnp.concat_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=jnp.concat_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concat.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="concat",
    testcases=[
        {
            "testcase": "concat",
            "callable": lambda a, b: jnp.concat((a, b), axis=0),
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class ConcatPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.concatenate to ONNX.  Note:  jax.numpy.concat
    is an alias for jax.numpy.concatenate.
    """

    @staticmethod
    def abstract_eval(*arrays, axis):
        """Abstract evaluation function for Concat."""
        base_shape = list(arrays[0].shape)
        total_dim = sum(a.shape[axis] for a in arrays)
        base_shape[axis] = total_dim
        return core.ShapedArray(tuple(base_shape), arrays[0].dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of Concat to ONNX format."""
        # Expect node_inputs: a list of arrays to concatenate.
        axis = params.get("axis", 0)
        input_names = [s.get_name(var) for var in node_inputs]
        output_name = s.get_name(node_outputs[0])

        concat_node = helper.make_node(
            "Concat",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("concat"),
            axis=axis,
        )
        s.add_node(concat_node)

    @staticmethod
    def _concat(arrays, axis):
        """Defines the primitive binding for Concat."""
        return jnp.concat_p.bind(*arrays, axis=axis)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Concat."""

        def patched_concat(arrays, axis):
            return ConcatPlugin._concat(arrays, axis)

        return patched_concat

    @staticmethod
    def patch_info():
        """Provides patching information for Concat."""
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: ConcatPlugin.get_monkey_patch(),
            "target_attribute": "concatenate",  # Correct attribute name
        }


# Register abstract evaluation function
jnp.concat_p.def_abstract_eval(ConcatPlugin.abstract_eval)
