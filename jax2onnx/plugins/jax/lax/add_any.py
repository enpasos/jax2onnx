from typing import TYPE_CHECKING
import jax

# Reuse the existing Add implementation; this file just registers the alias name.
from jax2onnx.plugins.jax.lax.add import AddPlugin
from jax2onnx.plugin_system import register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    # Internal JAX primitive used by AD rules to sum tangents of different dtypes.
    jaxpr_primitive="add_any",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.add.html",
    onnx=[{"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"}],
    since="v0.8.0",
    context="primitives.lax",
    component="add_any",
    testcases=[
        {
            "testcase": "add_any_via_jvp_on_mul",
            # JVP of f(a,b)=a*b is: t = (ta*b) + (a*tb)  → emits 'add_any'
            # We return the tangent (index 1 of jvp's output tuple).
            "callable": lambda x1, x2: jax.jvp(lambda a, b: a * b, (x1, x2), (x1, x2))[
                1
            ],
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class AddAnyPlugin(AddPlugin):
    """Alias for JAX's internal 'add_any' primitive; identical lowering to Add."""

    pass
