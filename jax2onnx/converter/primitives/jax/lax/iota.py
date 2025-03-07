import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.iota_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_iota(node_inputs, node_outputs, params):
        """Handle JAX iota primitive."""
        output_name = s.get_var_name(node_outputs[0])
        dtype = params["dtype"]
        if dtype != jnp.int32:
            raise NotImplementedError("iota only implemented for int32")
        shape = params["shape"]
        L = shape[0]
        start_name = s.get_constant_name(jnp.array(0, dtype=jnp.int32))
        end_name = s.get_constant_name(jnp.array(L, dtype=jnp.int32))
        step_name = s.get_constant_name(jnp.array(1, dtype=jnp.int32))
        node = helper.make_node(
            "Range",
            inputs=[start_name, end_name, step_name],
            outputs=[output_name],
            name=s.get_unique_name("iota"),
        )
        s.add_node(node)

    return _handle_iota
