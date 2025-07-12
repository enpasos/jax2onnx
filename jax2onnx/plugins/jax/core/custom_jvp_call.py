# file: jax2onnx/plugins/jax/core/custom_jvp_call.py
from typing import TYPE_CHECKING
import jax
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


# ---------- tiny custom-JVP function used by the testcase ----------
@jax.custom_jvp
def square(x):
    return x * x


@square.defjvp
def square_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return square(x), 2 * x * t


# -------------------------------------------------------------------


@register_primitive(
    jaxpr_primitive="custom_jvp_call",
    jax_doc="Generic passthrough for custom JVP calls",
    onnx=[{"component": "CustomJvp", "doc": ""}],
    context="primitives.core",
    component="custom_jvp_generic",
    since="v0.7.1",
    testcases=[
        {
            "testcase": "custom_jvp_square",
            "callable": lambda x: square(x),
            "input_shapes": [(3,)],
        },
    ],
)
class GenericCustomJvpPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(*xs, **params):
        return params["out_abstract_vals"]

    def to_onnx(self, s, node_inputs, node_outputs, params):
        # 1) peel apart the ClosedJaxpr
        closed = params["call_jaxpr"]
        sub_jaxpr = closed.jaxpr
        sub_consts = getattr(closed, "consts", [])

        # 2) hook up sub-invars → outer inputs
        for sub_var, outer_var in zip(sub_jaxpr.invars, node_inputs):
            s.var_to_name[sub_var] = s.get_name(outer_var)

        # 3) inline the sub-Jaxpr
        s._process_jaxpr(sub_jaxpr, sub_consts)

        # 4) hook up sub-outvars → outer outputs
        for sub_var, outer_var in zip(sub_jaxpr.outvars, node_outputs):
            s.var_to_name[outer_var] = s.get_name(sub_var)
