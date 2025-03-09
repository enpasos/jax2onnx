import contextlib
from typing import TYPE_CHECKING

import jax
from flax import nnx
from jax.core import Primitive
from onnx import helper
from jax import numpy as jnp
from jax import core
import numpy as np

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the new primitive for dropout.
dropout_p = Primitive("dropout")


def get_primitive():
    return dropout_p


def _get_monkey_patch():
    # Define the function that sets up the binding.
    def dropout(x, rate, deterministic):
        def dropout_abstract_eval(x, rate, deterministic):
            # Dropout does not change the shape.
            return core.ShapedArray(x.shape, x.dtype)

        dropout_p.multiple_results = False
        dropout_p.def_abstract_eval(dropout_abstract_eval)
        return dropout_p.bind(x, rate=rate, deterministic=deterministic)

    # The patched __call__ method extracts parameters from the instance.
    def patched_dropout_call(self, x, deterministic=None):
        # Use provided 'deterministic' flag if given; otherwise use self.deterministic.
        det = deterministic if deterministic is not None else self.deterministic
        return dropout(x, self.rate, det)

    return patched_dropout_call


@contextlib.contextmanager
def temporary_patch():
    original_call = nnx.Dropout.__call__
    nnx.Dropout.__call__ = _get_monkey_patch()
    try:
        yield
    finally:
        nnx.Dropout.__call__ = original_call


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_dropout(node_inputs, node_outputs, params):
        # Expect node_inputs: [x]
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])
        # Retrieve dropout parameters.
        rate = params.get("rate", 0.0)
        deterministic = params.get("deterministic", True)
        # For ONNX Dropout, in inference mode (deterministic True) the ratio should be 0.
        ratio = 0.0 if deterministic else rate
        # Only include the "ratio" attribute if it's nonzero,
        # because some operator versions don't expect it.
        dropout_attrs = {}
        if ratio != 0.0:
            dropout_attrs["ratio"] = ratio
        dropout_node = helper.make_node(
            "Dropout",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("dropout"),
            **dropout_attrs,
        )
        s.add_node(dropout_node)
        # Dropout does not change the shape.
        s.add_shape_info(output_name, node_inputs[0].aval.shape)

    return handle_dropout


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "dropout",
        "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/dropout.html",
        "onnx": [
            {
                "component": "Dropout",
                "doc": "https://onnx.ai/onnx/operators/onnx__Dropout.html",
            }
        ],
        "since": "v0.1.0",
        "testcases": [
            {
                "testcase": "dropout_inference",
                "callable": nnx.Dropout(rate=0.5, deterministic=True, rngs=nnx.Rngs(0)),
                "input_shapes": [(5, 10)],
            },
            {
                "testcase": "dropout_training",
                "callable": nnx.Dropout(
                    rate=0.5, deterministic=False, rngs=nnx.Rngs(0)
                ),
                "input_shapes": [(5, 10)],
            },
        ],
    }
