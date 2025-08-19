# file: jax2onnx/plugins/jax/numpy/concatenate.py

# --- Imports ---------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, Sequence

import jax
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive
import numpy as np
from onnx import helper
from onnx import mapping as onnx_mapping

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.patched_callable_wrapper import PatchedCallableWrapper

import logging

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.concatenate")

_SENTINEL = -1  # what `jnp.tile` produces for “unknown” sizes


if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


def concat_dynamic_tile_func(x):
    # x : (B, N, D)
    D = x.shape[2]  # 256 (concrete) in the testcase
    token = jnp.zeros((1, 1, D), dtype=x.dtype)  # (1, 1, D)

    # broadcast_to accepts symbolic sizes, so we can use `x.shape[0]`
    tiled_token = jnp.broadcast_to(token, (x.shape[0], 1, D))  # (B, 1, D)

    return jnp.concatenate([tiled_token, x], axis=1)  # (B, 1+N, D)


def concat_mixed_dtypes_noargs():
    """
    Regression repro: concatenating int32 and float32 without inputs.
    Used to fail ORT load with:
      Type Error: Type parameter (T) of Concat bound to different types
    Our post-build sanitizer should cast to a common type so it loads & runs.
    """
    a = jnp.array([1, 2, 3], dtype=jnp.int32)
    b = jnp.array([1.1, 2.2, 3.3], dtype=jnp.float32)
    return jnp.concatenate((a, b), axis=0)


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
            "testcase": "concatenate_mixed_dtypes_noargs",
            "callable": concat_mixed_dtypes_noargs,
            "input_shapes": [],
            "check_onnx_load": True,
        },
        {
            "testcase": "concatenate_abstract_middle_dim",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=1),
            "input_shapes": [("B", 1, 8), ("B", 10, 8)],
            "expected_output_shapes": [("B", 11, 8)],
        },
        {
            "testcase": "concatenate_tile_and_symbolic",
            "callable": concat_dynamic_tile_func,
            "input_shapes": [("B", 49, 256)],  # Matches failing ConcatClsToken
            "expected_output_shapes": [("B", 50, 256)],  # 1 + 49 = 50
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
            result_spec = jax.tree_util.tree_leaves(result_spec)[0]
            return core.ShapedArray(result_spec.shape, result_spec.dtype)
        except Exception as exc:
            logger.debug("eval_shape failed, using manual rule: %s", exc)
            shape = ConcatenatePlugin._manual_shape(avals, axis=axis)
            dtype = jax.dtypes.result_type(*[a.dtype for a in avals])
            return core.ShapedArray(shape, dtype)

    # ---------------------------------------------------------------------
    #  to_onnx – unchanged
    # ---------------------------------------------------------------------
    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        axis = int(params.get("axis", 0))

        # Decide the final dtype for Concat:
        # Use JAX's promoted output dtype from abstract eval. Do NOT override here,
        # or we may disagree with previously-registered ValueInfo dtypes.
        out_aval = node_outputs[0].aval
        final_jax_dtype = out_aval.dtype
        final_np_dtype = np.dtype(final_jax_dtype)
        final_onnx_dtype = onnx_mapping.NP_TYPE_TO_TENSOR_TYPE[final_np_dtype]

        # Ensure all inputs match the final dtype.
        # If the final dtype is floating, cast *every* input to it. This is robust
        # against upstream float literals becoming f64 initializers when x64 is enabled.
        concat_inputs: list[str] = []
        if jnp.issubdtype(final_jax_dtype, jnp.floating):
            for vin in node_inputs:
                in_name = s.get_name(vin)
                cast_out = s.get_unique_name("cast_for_concat")
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[in_name],
                    outputs=[cast_out],
                    name=s.get_unique_name("cast_to_concat"),
                    to=final_onnx_dtype,
                )
                s.add_node(cast_node)
                s.add_shape_info(cast_out, tuple(vin.aval.shape), final_jax_dtype)
                concat_inputs.append(cast_out)
        else:
            # For non-floating outputs, only cast when the dtype disagrees.
            for vin in node_inputs:
                in_name = s.get_name(vin)
                in_dtype = vin.aval.dtype
                if np.dtype(in_dtype) != final_np_dtype:
                    cast_out = s.get_unique_name("cast_for_concat")
                    cast_node = helper.make_node(
                        "Cast",
                        inputs=[in_name],
                        outputs=[cast_out],
                        name=s.get_unique_name("cast_to_concat"),
                        to=final_onnx_dtype,
                    )
                    s.add_node(cast_node)
                    s.add_shape_info(cast_out, tuple(vin.aval.shape), final_jax_dtype)
                    concat_inputs.append(cast_out)
                else:
                    concat_inputs.append(in_name)

        # Now create the Concat with uniform dtypes
        node = helper.make_node(
            "Concat",
            inputs=concat_inputs,
            outputs=[s.get_name(node_outputs[0])],
            name=s.get_unique_name("concat"),
            axis=axis,
        )
        s.add_node(node)

        # Record output shape & final dtype
        s.add_shape_info(
            s.get_name(node_outputs[0]), tuple(out_aval.shape), final_jax_dtype
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

    @staticmethod
    def _manual_shape(avals: Sequence[core.ShapedArray], *, axis: int):
        """Light‑weight concatenate shape rule that tolerates the -1 sentinel."""
        rank = len(avals[0].shape)
        out: list[Any] = []  # Add type annotation to match expected list type

        for d in range(rank):
            if d == axis:
                # sum along the concat axis, ignoring sentinels
                sizes = [a.shape[d] for a in avals]
                int_total = sum(
                    s for s in sizes if isinstance(s, int) and s != _SENTINEL
                )
                sym_sizes = [
                    s for s in sizes if not isinstance(s, int) or s == _SENTINEL
                ]
                if sym_sizes:
                    # any symbolic → keep the symbolic one (they must all agree)
                    out.append(sym_sizes[0])
                else:
                    out.append(int_total)
            else:
                # all other axes must agree up to broadcasting of 1 / sentinel
                size_set = {
                    s for s in (a.shape[d] for a in avals) if s not in (1, _SENTINEL)
                }
                if len(size_set) > 1:
                    raise TypeError("non‑concat dims disagree: " + str(size_set))
                out.append(next(iter(size_set)) if size_set else avals[0].shape[d])
        return tuple(out)


# Register the rule with the primitive
jnp.concatenate_p.def_abstract_eval(ConcatenatePlugin.abstract_eval)
