from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define a new primitive for addition.
jnp.add_p = Primitive("jnp.add")


def get_primitive():
    return jnp.add_p


def _get_monkey_patch():
    def add(x, y):
        def add_abstract_eval(x, y):
            # For simplicity, assume x and y have the same shape.
            # (In general, jax.numpy.add supports broadcasting.)
            return core.ShapedArray(x.shape, x.dtype)

        jnp.add_p.multiple_results = False
        jnp.add_p.def_abstract_eval(add_abstract_eval)
        return jnp.add_p.bind(x, y)

    return add


@contextlib.contextmanager
def temporary_patch():
    # Save the original function from jax.numpy.
    original_fn = jnp.add
    jnp.add = _get_monkey_patch()
    try:
        yield
    finally:
        jnp.add = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_add(node_inputs, node_outputs, params):
        # Expect node_inputs: [x, y]
        x_var = node_inputs[0]
        y_var = node_inputs[1]
        output_var = node_outputs[0]

        x_name = s.get_name(x_var)
        y_name = s.get_name(y_var)
        output_name = s.get_name(output_var)

        add_node = helper.make_node(
            "Add",
            inputs=[x_name, y_name],
            outputs=[output_name],
            name=s.get_unique_name("add"),
        )
        s.add_node(add_node)

    return handle_add


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "jnp.add",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.add.html",
        "onnx": [
            {
                "component": "Add",
                "doc": "https://onnx.ai/onnx/operators/onnx__Add.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.jnp",
        "testcases": [
            {
                "testcase": "add",
                "callable": lambda x, y: jnp.add(x, y),
                "input_shapes": [(3,), (3,)],
            }
        ],
    }
