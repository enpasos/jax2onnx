# file: jax2onnx/plugins/jax/numpy/where.py


from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive, Var
import numpy as np
from onnx import helper

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.where")

# Define the primitive for jnp.where
jnp.where_p = Primitive("jnp.where")
jnp.where_p.multiple_results = False


# Example definition (ensure it's globally accessible for the test generator):
def create_problematic_where_sequence(cond_input, data_input):
    scalar_true_val = jnp.array(1.0, dtype=data_input.dtype)
    scalar_false_val = jnp.array(0.0, dtype=data_input.dtype)
    where_output = jnp.where(cond_input, scalar_true_val, scalar_false_val)
    processed_data = data_input * where_output
    return processed_data


@register_primitive(
    jaxpr_primitive=jnp.where_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html",
    onnx=[
        {"component": "Where", "doc": "https://onnx.ai/onnx/operators/onnx__Where.html"}
    ],
    since="v0.5.2",
    context="primitives.jnp",
    component="where",
    testcases=[
        {
            # Reproduce GPT causal‐attention mask: broadcast mask vs. scores, with a scalar "else"… currently unhandled.
            "testcase": "where_gpt_mask_scores_literal_else",
            "callable": lambda mask, scores: jnp.where(mask, scores, -1e9),
            "input_shapes": [
                ("B", 1, "T", "T"),  # the causal mask
                ("B", 12, "T", "T"),  # the attention scores
            ],
            "input_dtypes": [jnp.bool_, jnp.float32],
            "expected_output_shapes": [
                ("B", 12, "T", "T"),
            ],
        },
        {
            "testcase": "where_simple",
            "callable": lambda c, x, y: jnp.where(c, x, y),
            "input_shapes": [(3,), (3,), (3,)],
            "expected_output_shapes": [(3,)],
        },
        {
            "testcase": "where_broadcast",
            "callable": lambda c, x, y: jnp.where(c[:, None], x, y),
            "input_shapes": [(4,), (4, 5), (4, 5)],
            "expected_output_shapes": [(4, 5)],
        },
        {
            "testcase": "where_multidim_condition_scalar_branches_broadcast",
            "callable": lambda c, t, f: jnp.where(c, t, f),
            "input_shapes": [(201, 1, 1), (), ()],
        },
        {
            "testcase": "where_multidim_condition_scalar_branches_broadcast",
            "callable": lambda c, t, f: jnp.where(c, t, f),
            "input_shapes": [(201, 1, 1), (), ()],
        },
        {
            "testcase": "where_A",
            "callable": create_problematic_where_sequence,
            "input_values": [
                np.random.choice([True, False], size=(201, 1, 1)),
                np.random.rand(201, 1, 201).astype(np.float32),
            ],
            "expected_output_shapes": [(201, 1, 201)],
            "expected_output_dtypes": [np.float32],
        },
        {
            "testcase": "where_B",
            "callable": create_problematic_where_sequence,
            "input_values": [
                np.random.choice([True, False], size=(201, 1, 1)),
                np.random.rand(201, 1, 201).astype(np.int32),
            ],
            "expected_output_shapes": [(201, 1, 201)],
            "expected_output_dtypes": [np.int32],
        },
        {
            # Fails exactly like GPT causal attention: bool mask, float scores, scalar else
            "testcase": "where_gpt_mask_scores_scalar_else",
            "callable": lambda mask, scores: jnp.where(mask, scores, -1e9),
            "input_shapes": [("B", 1, "T", "T"), ("B", 12, "T", "T")],
            "input_dtypes": [jnp.bool_, jnp.float32],
            "expected_output_shapes": [("B", 12, "T", "T")],
        },
        {
            "testcase": "where_int_condition_cast",
            "callable": lambda c_int, x, y: jnp.where(c_int, x, y),
            "input_shapes": [(3,), (3,), (3,)],
            "input_dtypes": [np.int32, np.float32, np.float32],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [np.float32],
        },
        {
            "testcase": "where_literal_else_pyfloat",
            "callable": lambda cond, x: jnp.where(cond, x, -1e9),
            "input_shapes": [(4, 4), (4, 4)],
            "expected_output_shapes": [(4, 4)],
            "expected_output_dtypes": [np.float32],
        },
        {
            "testcase": "where_jax_int_literals_broadcast_f64_mode",
            "callable": lambda c, x_scalar_py, y_scalar_py: jnp.where(
                c,
                jnp.array(x_scalar_py, dtype=jnp.int64),
                jnp.array(y_scalar_py, dtype=jnp.int64),
            ),
            "input_values": [
                np.array([[True], [False], [True]], dtype=np.bool_),
                1,  # Python int for x
                0,  # Python int for y
            ],
            "expected_output_shapes": [(3, 1)],
            "expected_output_dtypes": [np.int64],
            "run_only_f64_variant": True,
        },
        {
            # Regression: mismatch between true/false branch dtypes (f64 vs i32).
            # Mirrors:
            #   jax.config.update("jax_enable_x64", True)
            #   def mismatch_where(x):
            #       cond = x > 0
            #       return jnp.where(cond, x, jnp.array([1, 2, 3], dtype=jnp.int32))
            #   x = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float64)
            "testcase": "where_dtype_mismatch_f64_vs_i32_promote",
            "callable": lambda x: jnp.where(
                x > 0, x, jnp.array([1, 2, 3], dtype=jnp.int32)
            ),
            "input_values": [np.array([1.0, -2.0, 3.0], dtype=np.float64)],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [
                np.float64
            ],  # promote(int32, float64) -> float64
            "run_only_f64_variant": True,
        },
        {
            "testcase": "where_simple",
            "callable": lambda x, y: jnp.where(x > 0, x, y),
            "input_values": [
                jnp.array([-1.0, 1.0, 0.0], dtype=jnp.float32),
                jnp.array([10.0, 11.0, 12.0], dtype=jnp.float32),
            ],
        },
    ],
)
class WherePlugin(PrimitiveLeafPlugin):
    """Lower `jnp.where` to ONNX Where operator."""

    @staticmethod
    def abstract_eval(
        cond_av: core.AbstractValue,
        x_av: core.AbstractValue,
        y_av: core.AbstractValue,
        **kwargs,
    ) -> core.AbstractValue:
        # All inputs must be ShapedArrays
        if not all(isinstance(av, core.ShapedArray) for av in (cond_av, x_av, y_av)):
            raise TypeError("All inputs to jnp.where must be ShapedArrays.")

        # Determine promoted dtype
        promoted_dtype = jnp.promote_types(x_av.dtype, y_av.dtype)
        # Broadcast shapes via JAX's own broadcast_shapes (handles symbolic dims)
        from jax import numpy as _jnp

        output_shape = _jnp.broadcast_shapes(cond_av.shape, x_av.shape, y_av.shape)

        return core.ShapedArray(output_shape, promoted_dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        # Map inputs
        cond_v, x_v, y_v = node_inputs
        cond_name = s.get_name(cond_v)
        x_name = s.get_name(x_v)
        y_name = s.get_name(y_v)
        out_v = node_outputs[0]
        out_name = s.get_name(out_v)

        # --- Ensure condition is BOOL for ONNX ---
        from onnx import TensorProto

        cond_dtype = getattr(cond_v.aval, "dtype", None)
        if cond_dtype is not None and np.dtype(cond_dtype) != np.bool_:
            cond_cast_name = s.builder.get_unique_name("where_cond_cast")
            s.builder.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[cond_name],
                    outputs=[cond_cast_name],
                    to=TensorProto.BOOL,
                    name=s.builder.get_unique_name("cast_where_cond"),
                )
            )
            s.add_shape_info(cond_cast_name, cond_v.aval.shape, np.bool_)
            cond_name = cond_cast_name

        # --- Harmonize X and Y dtypes (numpy-style promotion) ---
        # Minimal, regression-safe fix: only add Cast nodes when needed.
        x_dtype = getattr(x_v.aval, "dtype", None)
        y_dtype = getattr(y_v.aval, "dtype", None)
        target_dtype = None
        if x_dtype is not None and y_dtype is not None:
            # Use NumPy's promotion so float64 vs int32 -> float64, etc.
            target_dtype = np.promote_types(np.dtype(x_dtype), np.dtype(y_dtype))

            def _maybe_cast(inp_name: str, aval, tag: str) -> str:
                cur = np.dtype(getattr(aval, "dtype", target_dtype))
                if cur == target_dtype:
                    return inp_name
                casted = s.builder.get_unique_name(
                    f"where_{tag}_cast_{target_dtype.name}"
                )
                s.builder.add_node(
                    helper.make_node(
                        "Cast",
                        inputs=[inp_name],
                        outputs=[casted],
                        to=int(s.builder._numpy_dtype_to_onnx(target_dtype)),
                        name=s.builder.get_unique_name(f"cast_where_{tag}"),
                    )
                )
                s.add_shape_info(casted, getattr(aval, "shape", ()), target_dtype)
                return casted

            x_name = _maybe_cast(x_name, x_v.aval, "x")
            y_name = _maybe_cast(y_name, y_v.aval, "y")
        # (If either dtype is unknown, defer to abstract_eval result.)

        # Create ONNX Where node
        node = helper.make_node(
            "Where",
            inputs=[cond_name, x_name, y_name],
            outputs=[out_name],
            name=s.builder.get_unique_name("WhereOp"),
        )
        s.add_node(node)
        # Record the promoted dtype if we computed one; otherwise use abstract dtype.
        out_dtype = (
            target_dtype
            if target_dtype is not None
            else getattr(out_v.aval, "dtype", None)
        )
        if out_dtype is None:
            out_dtype = (
                getattr(x_v.aval, "dtype", None)
                or getattr(y_v.aval, "dtype", None)
                or np.float32
            )
        s.add_shape_info(out_name, out_v.aval.shape, out_dtype)

    @staticmethod
    def patch_info():
        # Monkey-patch jnp.where and lax.select to emit our primitive in the jaxpr
        def patched_where(cond, x=None, y=None):
            if x is None or y is None:
                raise NotImplementedError(
                    "Only `jnp.where(cond, x, y)` is supported for ONNX conversion."
                )
            return jnp.where_p.bind(cond, x, y)

        return {
            "patch_targets": [jnp],
            "target_attribute": "where",
            "patch_function": lambda orig: patched_where,
        }


# Bind abstract evaluation
jnp.where_p.def_abstract_eval(WherePlugin.abstract_eval)
