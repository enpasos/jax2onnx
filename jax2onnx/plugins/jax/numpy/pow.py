# file: jax2onnx/plugins/jax/numpy/pow.py


from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define custom primitives we bind to via monkey patching.
jnp.power_p = Primitive("jnp.power")
jnp.power_p.multiple_results = False
jnp.pow_p = Primitive("jnp.pow")
jnp.pow_p.multiple_results = False


def _broadcast_shape_like_add(
    x_shape: Sequence[Any], y_shape: Sequence[Any]
) -> tuple[Any, ...]:
    """Minimal, tracer-friendly broadcast shape (same logic used in jnp.add plugin)."""
    if len(x_shape) != len(y_shape):
        raise ValueError("x and y must have the same rank for simple broadcast here.")
    out: list[Any] = []
    for xs, ys in zip(x_shape, y_shape):
        # If either is symbolic/-1, pick the other (optimistic)
        if xs == -1 or ys == -1:
            out.append(xs if ys == -1 else ys)
        elif xs != ys and xs != 1 and ys != 1:
            raise ValueError(f"Shapes {x_shape} and {y_shape} are not broadcastable.")
        else:
            out.append(xs if ys == 1 else ys)
    return tuple(out)


class _PowBase(PrimitiveLeafPlugin):
    """Shared ONNX lowering for jnp.power/jnp.pow."""

    @staticmethod
    def abstract_eval(x: core.ShapedArray, y: core.ShapedArray) -> core.ShapedArray:
        # Keep dtype = base dtype (matches JAX's typical float behavior).
        out_shape = _broadcast_shape_like_add(x.shape, y.shape)
        return core.ShapedArray(out_shape, x.dtype)

    def to_onnx(
        self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params
    ) -> None:
        base_v, exp_v = node_inputs
        out_v = node_outputs[0]

        base_name = s.get_name(base_v)
        exp_name = s.get_name(exp_v)
        out_name = s.get_name(out_v)

        base_dt = np.dtype(base_v.aval.dtype)
        exp_dt = np.dtype(exp_v.aval.dtype)

        # Cast exponent to base dtype to satisfy ONNX Pow input requirements
        if exp_dt != base_dt:
            cast_out = s.get_unique_name("jnp_pow_exp_cast_out")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[exp_name],
                    outputs=[cast_out],
                    name=s.get_unique_name("jnp_pow_exp_cast"),
                    to=int(s.builder._numpy_dtype_to_onnx(base_dt)),
                )
            )
            # Ensure builder knows the cast tensor's shape/dtype (important in subgraphs)
            s.add_shape_info(cast_out, exp_v.aval.shape, base_dt)
            exp_name = cast_out

        s.add_node(
            helper.make_node(
                "Pow",
                inputs=[base_name, exp_name],
                outputs=[out_name],
                name=s.get_unique_name("jnp_pow"),
            )
        )
        s.add_shape_info(out_name, out_v.aval.shape, base_dt)

    # ----- monkey patch helpers (bind fixed-arity primitive) -----
    # (Pattern: like jnp.add)
    @staticmethod
    def _bind_primitive(prim: Primitive, x, y):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return prim.bind(x, y)


@register_primitive(
    jaxpr_primitive=jnp.power_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.power.html",
    onnx=[{"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"}],
    since="v0.8.1",
    context="primitives.jnp",
    component="power",
    testcases=[
        {
            "testcase": "pow_jnp_power",
            "callable": lambda x1, x2: jnp.power(x1, x2),
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class JnpPowerPlugin(_PowBase):
    """Patch jnp.power → custom primitive 'jnp.power' and lower to ONNX Pow."""

    @staticmethod
    def _power(x, y):
        """Defines the primitive binding for jnp.power."""
        return _PowBase._bind_primitive(jnp.power_p, x, y)

    @staticmethod
    def get_monkey_patch():
        def patched_power(x, y):
            return JnpPowerPlugin._power(x, y)

        return patched_power

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: JnpPowerPlugin.get_monkey_patch(),
            "target_attribute": "power",
        }


@register_primitive(
    jaxpr_primitive=jnp.pow_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.power.html",
    onnx=[{"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"}],
    since="v0.8.1",
    context="primitives.jnp",
    component="pow",
    testcases=[
        {
            "testcase": "pow_jnp_pow",
            "callable": lambda x1, x2: jnp.pow(x1, x2),
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class JnpPowAliasPlugin(_PowBase):
    """
    Patch jnp.pow (alias of jnp.power) → custom primitive 'jnp.pow' and lower to ONNX Pow.
    If a JAX version lacks jnp.pow, this patch will fail at import time — which would
    surface quickly when running tests; in that case, drop this class and rely on 'power'.
    """

    @staticmethod
    def _pow(x, y):
        """Defines the primitive binding for jnp.pow."""
        return _PowBase._bind_primitive(jnp.pow_p, x, y)

    @staticmethod
    def get_monkey_patch():
        def patched_pow(x, y):
            return JnpPowAliasPlugin._pow(x, y)

        return patched_pow

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: JnpPowAliasPlugin.get_monkey_patch(),
            "target_attribute": "pow",
        }

    # Optional: keep abstract eval identical to base
    @staticmethod
    def abstract_eval(x: core.ShapedArray, y: core.ShapedArray) -> core.ShapedArray:
        return _PowBase.abstract_eval(x, y)


#
# Register abstract evaluation functions (pattern: like jnp.add)
#
jnp.power_p.def_abstract_eval(JnpPowerPlugin.abstract_eval)
jnp.pow_p.def_abstract_eval(JnpPowAliasPlugin.abstract_eval)
