import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING
from onnx import helper
from jax2onnx.converter.name_generator import GlobalNameCounter, UniqueNameGenerator
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import register_primitive, PrimitiveLeafPlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


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


@register_primitive(
    jaxpr_primitive=jax.random.random_gamma_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.random.gamma.html",
    onnx=[],
    since="v0.2.0",
    context="primitives.random",
    testcases=[
        # {
        #     "testcase": "random_gamma_test1",
        #     "callable": lambda alpha: jax.random.gamma(jax.random.PRNGKey(0), alpha),
        #     "input_shapes": [(3,)],
        # }
    ],
)
class RandomGammaPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.random.random_gamma to ONNX."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX random_gamma primitive by constructing a subgraph for gamma sampling."""
        shape = node_inputs[1].aval.shape
        key = jax.random.PRNGKey(0)
        alpha = jnp.zeros(shape)

        builder = OnnxBuilder(GlobalNameCounter(), UniqueNameGenerator())
        subconverter = Jaxpr2OnnxConverter(builder)

        if params.get("log_space", False):
            subconverter.trace_jaxpr(gamma_log, (key, alpha))
        else:
            subconverter.trace_jaxpr(gamma, (key, alpha))

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

        s.builder.nodes.extend(subconverter.builder.nodes)
        s.builder.initializers.extend(subconverter.builder.initializers)
        s.builder.name_counter = subconverter.builder.name_counter

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
