# jax2onnx/plugins/jax/numpy/clip.py
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive, Var
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# Define a dedicated primitive for jnp.clip and patch jnp.clip to emit it
jnp.clip_p = Primitive("jnp.clip")
jnp.clip_p.multiple_results = False


def _np_dtype(x) -> np.dtype:
    return x if isinstance(x, np.dtype) else np.dtype(x)


def _cast_to(
    s: "Jaxpr2OnnxConverter",
    name: str,
    cur_dt: np.dtype,
    tgt_dt: np.dtype,
    *,
    ctx: str,
    shape_hint: tuple[Any, ...],
) -> str:
    """Insert Cast(name -> tgt_dt) if needed, keep original shape."""
    if cur_dt == tgt_dt:
        return name
    out = s.builder.get_unique_name(f"{ctx}_cast")
    s.add_node(
        helper.make_node(
            "Cast",
            inputs=[name],
            outputs=[out],
            to=int(s.builder._numpy_dtype_to_onnx(tgt_dt)),
            name=s.builder.get_unique_name(f"{ctx}_Cast"),
        )
    )
    s.add_shape_info(out, shape_hint, tgt_dt)
    return out


def _dtype_min_max(dtype: np.dtype) -> tuple[Any, Any]:
    """Return the 'ignore' bounds for None: (-inf/+inf for floats, min/max for ints, False/True for bool)."""
    if np.issubdtype(dtype, np.floating):
        return -jnp.inf, jnp.inf
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    if dtype == np.bool_:
        return False, True
    # Fallback: treat like floats
    return -jnp.inf, jnp.inf


@register_primitive(
    jaxpr_primitive=jnp.clip_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.clip.html",
    onnx=[
        {"component": "Max", "doc": "https://onnx.ai/onnx/operators/onnx__Max.html"},
        {"component": "Min", "doc": "https://onnx.ai/onnx/operators/onnx__Min.html"},
    ],
    since="v0.7.5",
    context="primitives.jnp",
    component="clip",
    testcases=[
        # int32 input, Python-int bounds → keep dtype=int32
        {
            "testcase": "clip_i32_scalar_bounds",
            "callable": lambda x: jnp.clip(x, 0, 4),
            "input_values": [np.array([-3, 1, 9, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
        },
        # float32 input, Python-float bounds in f64-export mode should NOT upcast result
        {
            "testcase": "clip_f32_scalar_bounds_no_upcast_f64_mode",
            "callable": lambda x: jnp.clip(x, -1.5, 2.5),
            "input_values": [np.array([-2.0, 0.5, 3.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "run_only_f64_variant": True,
        },
        # Only upper bound (min=None)
        {
            "testcase": "clip_only_upper",
            "callable": lambda x: jnp.clip(x, None, 1.0),
            "input_values": [np.array([-2.0, 0.5, 3.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
        },
        # Only lower bound (max=None)
        {
            "testcase": "clip_only_lower",
            "callable": lambda x: jnp.clip(x, -1, None),
            "input_values": [np.array([-5, -1, 0, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
        },
        # Broadcasted bounds (result has shape of x)
        {
            "testcase": "clip_broadcast_bounds",
            "callable": lambda x, lo, hi: jnp.clip(x, lo, hi),
            "input_values": [
                np.array(
                    [[-2.0, -0.5, 3.0], [1.0, 2.0, 5.0]], dtype=np.float64
                ),  # x  (2,3)
                np.array([[-1.0, 0.0, 0.0]], dtype=np.float64),  # lo (1,3)
                np.array(
                    [
                        [1.5],
                    ],
                    dtype=np.float64,
                ),  # hi (1,1)->(2,1)
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
    ],
)
class ClipPlugin(PrimitiveLeafPlugin):
    """Lower `jnp.clip(a, a_min, a_max)` using ONNX Max/Min, casting bounds to `a.dtype`."""

    @staticmethod
    def abstract_eval(
        x_av: core.ShapedArray,
        min_av: core.ShapedArray,
        max_av: core.ShapedArray,
        **params,
    ) -> core.ShapedArray:
        # Result shape and dtype follow x (NumPy semantics).
        return core.ShapedArray(x_av.shape, x_av.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        v_x, v_min, v_max = node_inputs
        v_out = node_outputs[0]

        x_name = s.get_name(v_x)
        min_name = s.get_name(v_min)
        max_name = s.get_name(v_max)
        out_name = s.get_name(v_out)

        x_dt = _np_dtype(v_x.aval.dtype)
        mn_dt = _np_dtype(v_min.aval.dtype)
        mx_dt = _np_dtype(v_max.aval.dtype)

        # Cast bounds to x dtype to avoid ONNX type-parameter mismatches
        min_name = _cast_to(
            s, min_name, mn_dt, x_dt, ctx="ClipMin", shape_hint=v_min.aval.shape
        )
        max_name = _cast_to(
            s, max_name, mx_dt, x_dt, ctx="ClipMax", shape_hint=v_max.aval.shape
        )

        # y = max(x, min)
        y_name = s.builder.get_unique_name("clip_lowered_max")
        s.add_node(
            helper.make_node(
                "Max",
                inputs=[x_name, min_name],
                outputs=[y_name],
                name=s.builder.get_unique_name("Max"),
            )
        )
        s.add_shape_info(y_name, v_x.aval.shape, x_dt)  # shape of x

        # z = min(y, max)
        s.add_node(
            helper.make_node(
                "Min",
                inputs=[y_name, max_name],
                outputs=[out_name],
                name=s.builder.get_unique_name("Min"),
            )
        )
        s.add_shape_info(out_name, v_x.aval.shape, x_dt)

    @staticmethod
    def patch_info():
        def patched_clip(a, a_min=None, a_max=None):
            # Match JAX/NumPy behavior: result dtype == a.dtype, bounds coerced to a.dtype
            x = jnp.asarray(a)
            dt = x.dtype

            lo_default, hi_default = _dtype_min_max(np.dtype(dt))

            if a_min is None:
                lo = jnp.asarray(lo_default, dtype=dt)
            else:
                lo = jnp.asarray(a_min, dtype=dt)

            if a_max is None:
                hi = jnp.asarray(hi_default, dtype=dt)
            else:
                hi = jnp.asarray(a_max, dtype=dt)

            # Important: always bind three args (fixed arity primitive)
            return jnp.clip_p.bind(x, lo, hi)

        return {
            "patch_targets": [jnp],
            "target_attribute": "clip",
            "patch_function": lambda orig: patched_clip,
        }


# Bind abstract evaluation
jnp.clip_p.def_abstract_eval(ClipPlugin.abstract_eval)
