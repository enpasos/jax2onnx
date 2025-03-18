from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the Concat primitive
jnp.concat_p = Primitive("jnp.concat")


def get_primitive():
    """Returns the jnp.concat primitive."""
    return jnp.concat_p


def concat_abstract_eval(*arrays, axis):
    """Abstract evaluation function for Concat."""
    base_shape = list(arrays[0].shape)
    total_dim = sum(a.shape[axis] for a in arrays)
    base_shape[axis] = total_dim
    return core.ShapedArray(tuple(base_shape), arrays[0].dtype)


# Register abstract evaluation function
jnp.concat_p.def_abstract_eval(concat_abstract_eval)


def concat(arrays, axis):
    """Defines the primitive binding for Concat."""
    return jnp.concat_p.bind(*arrays, axis=axis)


def patch_info():
    """Provides patching information for Concat."""
    return {
        "patch_targets": [jnp],
        "patch_function": lambda _: concat,
        "target_attribute": "concat",
    }


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_concat(node_inputs, node_outputs, params):
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

    return handle_concat


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "jnp.concat",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concat.html",
        "onnx": [
            {
                "component": "Concat",
                "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.jnp",
        "testcases": [
            {
                "testcase": "concat",
                "callable": lambda a, b: jnp.concat((a, b), axis=0),
                "input_shapes": [(3,), (3,)],
            }
        ],
    }
