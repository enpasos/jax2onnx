from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define a new primitive for concatenation.
jnp.concat_p = Primitive("jnp.concat")


def get_primitive():
    return jnp.concat_p


def _concat_abstract_eval(arrays, axis):
    # arrays is a tuple of ShapedArray objects.
    arrays = list(arrays)
    base = list(arrays[0].shape)
    # Here we assume that all inputs match except along the concatenation axis.
    total = 0
    for a in arrays:
        total += a.shape[axis]
    base[axis] = total
    return core.ShapedArray(tuple(base), arrays[0].dtype)


def _get_monkey_patch():
    def concat(arrays, axis):
        # Define an abstract evaluation function capturing the axis.
        def abstract_eval(*arrays, axis=axis):
            return _concat_abstract_eval(arrays, axis)

        jnp.concat_p.multiple_results = False
        jnp.concat_p.def_abstract_eval(abstract_eval)
        # Bind each array as an individual argument, along with the axis.
        return jnp.concat_p.bind(*arrays, axis=axis)

    return concat


@contextlib.contextmanager
def temporary_patch():
    original_fn = jnp.concat
    jnp.concat = _get_monkey_patch()
    try:
        yield
    finally:
        jnp.concat = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_concat(node_inputs, node_outputs, params):
        # Expect node_inputs: a list of arrays to concatenate.
        axis = params.get("axis")
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
