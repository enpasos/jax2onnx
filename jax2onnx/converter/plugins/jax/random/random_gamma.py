import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.random.random_gamma_p


def gamma(key, alpha):
    d = alpha - 1 / 3
    c = 1 / jnp.sqrt(9 * d)
    z = jax.random.normal(key, alpha.shape)
    v = (1 + c * z) ** 3
    u = jax.random.uniform(key, alpha.shape)
    x = d * v
    acceptance = (v > 0) & (jnp.log(u) < (0.5 * z**2 + d - d * v + d * jnp.log(v)))
    # Re-sample for rejected values
    z = jax.random.normal(key, alpha.shape)
    v = (1 + c * z) ** 3
    x = jnp.where(acceptance, x, d * v)
    # Clip when alpha is zero
    x = jnp.where(alpha == 0, 0.0, x)
    return x


def gamma_log(key, alpha):
    return jnp.log(gamma(key, alpha))


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_random_gamma(node_inputs, node_outputs, params):
        """
        Handle JAX random_gamma primitive by constructing a subgraph for gamma sampling.

        The implementation uses a sub-converter to trace the gamma function (or its logarithm)
        and then connects the resulting subgraph to the main graph via Identity nodes.

        Note: This implementation assumes that your converter instance (s) provides methods
        such as create_subconverter(), add_nodes(), add_initializers(), adjust_name_counter(),
        get_name(), get_var_name(), get_constant_name(), get_unique_name(), and add_node().
        """
        # Assume the second input holds the alpha parameter shape information.
        shape = node_inputs[1].aval.shape
        key = jax.random.PRNGKey(0)
        alpha = jnp.zeros(shape)

        # Create a sub-converter to trace the gamma sampling subgraph.
        # (Ensure that your converter supports creating a subconverter.)
        subconverter = s.create_subconverter()

        if params.get("log_space", False):
            subconverter.trace_jaxpr(gamma_log, (key, alpha))
        else:
            subconverter.trace_jaxpr(gamma, (key, alpha))

        # Connect each subgraph input to the corresponding outer input using Identity nodes.
        for outer_var, inner_tensor in zip(node_inputs, subconverter.builder.inputs):
            outer_name = s.get_name(outer_var)
            inner_name = inner_tensor.name
            id_node = helper.make_node(
                "Identity",
                inputs=[outer_name],
                outputs=[inner_name],
                name=s.get_unique_name("gamma_input"),
            )
            s.add_node(id_node)

        # Append the subgraph's nodes and initializers to the main graph.
        s.add_nodes(subconverter.builder.nodes)
        s.add_initializers(subconverter.builder.initializers)
        s.adjust_name_counter(subconverter.builder.name_counter)

        # Connect subgraph outputs back to the main graph using Identity nodes.
        for outer_var, inner_tensor in zip(node_outputs, subconverter.builder.outputs):
            outer_name = s.get_name(outer_var)
            inner_name = inner_tensor.name
            id_node = helper.make_node(
                "Identity",
                inputs=[inner_name],
                outputs=[outer_name],
                name=s.get_unique_name("gamma_output"),
            )
            s.add_node(id_node)

    return _handle_random_gamma


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""

    return {
        "jaxpr_primitive": "random_gamma",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.random.gamma.html",
        "onnx": [],
        "since": "v0.2.0",
        "context": "plugins.random",
        "testcases": [
            # {
            #     "testcase": "random_gamma_test1",
            #     "callable": lambda alpha: jax.random.gamma(jax.random.PRNGKey(0), alpha),
            #     "input_shapes": [(3,)],
            # }
        ],
    }
