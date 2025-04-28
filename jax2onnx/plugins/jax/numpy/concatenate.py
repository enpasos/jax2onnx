# --- Imports ---------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, Sequence

import jax
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.patched_callable_wrapper import PatchedCallableWrapper

import logging

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.concatenate")

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# ---------------------------------------------------------------------------
#  Primitive definition
# ---------------------------------------------------------------------------
if not hasattr(jnp, "concatenate_p"):
    jnp.concatenate_p = Primitive("jnp.concatenate")
    jnp.concatenate_p.multiple_results = False


# ---------------------------------------------------------------------------
#  Plugin
# ---------------------------------------------------------------------------
@register_primitive(
    jaxpr_primitive=jnp.concatenate_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concatenate.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="concatenate",
    testcases=[
        {
            "testcase": "concatenate",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=0),
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "concatenate_abstract_middle_dim",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=1),
            "input_shapes": [("B", 1, 8), ("B", 10, 8)],
            "expected_output_shapes": [("B", 11, 8)],
        },
    ],
)
class ConcatenatePlugin(PrimitiveLeafPlugin):
    """
    Symbolic-shape aware converter for `jax.numpy.concatenate`.
    Its `abstract_eval` rule defers shape inference to **jax.eval_shape**,
    which is safe to call while the outer `jax.make_jaxpr` trace is live.
    """

    # Will be filled the first time we patch `jnp.concatenate`
    _ORIGINAL_CONCATENATE: Callable | None = None

    # ---------------------------------------------------------------------
    #  abstract_eval  (⇐ **now uses jax.eval_shape**)
    # ---------------------------------------------------------------------
    @staticmethod
    def abstract_eval(*avals: core.ShapedArray, axis: int):
        logger.debug("ConcatenatePlugin.abstract_eval – start")

        # ---- sanity checks ------------------------------------------------
        if not avals:
            raise ValueError("concatenate expects at least one input")
        if not all(isinstance(a, core.ShapedArray) for a in avals):
            raise TypeError(
                "All inputs to concatenate must be ShapedArray, got "
                f"{[type(a) for a in avals]}"
            )
        if not isinstance(axis, int):
            raise TypeError(f"`axis` must be an int, got {type(axis)}")

        # ---- original function reference ---------------------------------
        orig = ConcatenatePlugin._ORIGINAL_CONCATENATE
        if orig is None:
            raise RuntimeError("Original jnp.concatenate was not captured.")

        # ---- ShapeDtypeStruct specs --------------------------------------
        specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]

        # ---- helper that calls the *un-patched* concatenate --------------
        def _helper(*xs):
            return orig(xs, axis=axis)

        # ---- delegate to jax.eval_shape ----------------------------------
        try:
            result_spec = jax.eval_shape(_helper, *specs)
            result_spec = jax.tree_util.tree_leaves(result_spec)[0]  # single tensor
        except Exception as exc:
            logger.error("jax.eval_shape failed inside abstract_eval", exc_info=True)
            raise

        logger.debug(
            "abstract_eval result: shape=%s dtype=%s",
            result_spec.shape,
            result_spec.dtype,
        )
        return core.ShapedArray(result_spec.shape, result_spec.dtype)

    # ---------------------------------------------------------------------
    #  to_onnx – unchanged
    # ---------------------------------------------------------------------
    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        axis = int(params.get("axis", 0))
        node = helper.make_node(
            "Concat",
            inputs=[s.get_name(v) for v in node_inputs],
            outputs=[s.get_name(node_outputs[0])],
            name=s.get_unique_name("concat"),
            axis=axis,
        )
        s.add_node(node)

        out_aval = node_outputs[0].aval
        s.add_shape_info(
            s.get_name(node_outputs[0]), tuple(out_aval.shape), out_aval.dtype
        )

    # ---------------------------------------------------------------------
    #  patch_info – capture original fn & inject wrapper
    # ---------------------------------------------------------------------
    @staticmethod
    def patch_info() -> dict[str, Any]:
        def _creator(orig_fn: Callable):
            logger.info("Storing original jnp.concatenate reference")
            ConcatenatePlugin._ORIGINAL_CONCATENATE = orig_fn
            return PatchedCallableWrapper(orig_fn, jnp.concatenate_p)

        return {
            "patch_targets": [jnp],
            "patch_function": _creator,
            "target_attribute": "concatenate",
        }


# Register the rule with the primitive
jnp.concatenate_p.def_abstract_eval(ConcatenatePlugin.abstract_eval)
