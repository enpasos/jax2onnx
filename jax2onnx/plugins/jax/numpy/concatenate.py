# file: jax2onnx/plugins/jax/numpy/concatenate.py

# --- Imports ---------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, Iterable, Sequence

import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.patched_callable_wrapper import PatchedCallableWrapper

import logging

from jax2onnx.plugins.jax.lax.mul import _np_dt

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
        },
        {
            "component": "Cast",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html",
        },
    ],
    since="v0.2.0",
    context="primitives.jnp",
    component="concatenate",
    testcases=[
        {
            "testcase": "concatenate_basic",
            "callable": lambda a, b: jnp.concatenate([a, b], axis=0),
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "concatenate_mixed_dtypes",
            "callable": lambda a, b: jnp.concatenate([a, b], axis=0),
            "input_shapes": [(3,), (3,)],
            "input_dtypes": [np.float32, np.int32],
        },
        {
            "testcase": "concatenate_with_explicit_dtype",
            "callable": lambda a, b: jnp.concatenate([a, b], axis=0, dtype=np.float64),
            "input_shapes": [(3,), (3,)],
            "input_dtypes": [np.float32, np.int32],
        },
        {
            "testcase": "concatenate_with_explicit_dtype_casts_inputs",
            "callable": lambda a, b: jnp.concatenate([a, b], axis=1, dtype=jnp.float32),
            # Two int32 inputs of shape (5, 1) concatenated along axis=1 -> (5, 2)
            "input_shapes": [(5, 1), (5, 1)],
            "input_dtypes": [np.int32, np.int32],
            "expected_output_shapes": [(5, 2)],
            "run_only_f64_variant": True,
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
    @staticmethod
    def abstract_eval(*xs, axis=0, dtype=None, **params):
        # Normalize inputs: either (xs) or a single list/tuple
        arrays: Iterable[core.ShapedArray]
        arrays = xs[0] if len(xs) == 1 and isinstance(xs[0], (list, tuple)) else xs

        rank = len(arrays[0].shape)
        ax = axis if axis >= 0 else axis + rank

        # Output shape: sum along concat axis, keep others
        out_shape = list(arrays[0].shape)
        out_shape[ax] = sum(int(a.shape[ax]) for a in arrays)

        # Target dtype: explicit dtype wins; otherwise JAX-style promotion
        if dtype is not None:
            out_dtype = _np_dt(dtype)
        else:
            out_dtype = arrays[0].dtype
            for a in arrays[1:]:
                out_dtype = np.promote_types(out_dtype, a.dtype)

        return core.ShapedArray(tuple(out_shape), out_dtype)

    # ---------------------------------------------------------------------
    #  to_onnx – unchanged
    # ---------------------------------------------------------------------
    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        # Inputs may be given as a list (from the wrapper) or as varargs
        in_vals = (
            list(node_inputs[0])
            if isinstance(node_inputs[0], (list, tuple))
            else list(node_inputs)
        )

        axis = int(params.get("axis", 0))
        dtype_param = params.get("dtype", None)

        names = [s.get_name(v) for v in in_vals]
        avals = [v.aval for v in in_vals]
        dtypes = [_np_dt(v.aval.dtype) for v in in_vals]

        # Decide target dtype
        if dtype_param is not None:
            tgt_dt = _np_dt(dtype_param)
        else:
            tgt_dt = dtypes[0]
            for dt in dtypes[1:]:
                tgt_dt = np.promote_types(tgt_dt, dt)

        # Normalize axis to non-negative
        rank = len(avals[0].shape)
        ax = axis if axis >= 0 else axis + rank

        # Cast inputs as needed to satisfy ONNX Concat type constraints
        casted_names: list[str] = []
        for name, dt, aval in zip(names, dtypes, avals):
            if _np_dt(dt) != tgt_dt:
                cast_out = s.builder.get_unique_name("Concat_cast")
                s.add_node(
                    helper.make_node(
                        "Cast",
                        [name],
                        [cast_out],
                        to=int(s.builder._numpy_dtype_to_onnx(tgt_dt)),
                        name=s.builder.get_unique_name("Cast"),
                    )
                )
                s.add_shape_info(cast_out, aval.shape, tgt_dt)
                casted_names.append(cast_out)
            else:
                casted_names.append(name)

        out_name = s.get_name(node_outputs[0])
        s.add_node(
            helper.make_node(
                "Concat",
                casted_names,
                [out_name],
                axis=ax,
                name=s.builder.get_unique_name("Concat"),
            )
        )

        # Output shape & dtype
        out_shape = list(avals[0].shape)
        out_shape[ax] = sum(a.shape[ax] for a in avals)
        s.add_shape_info(out_name, tuple(out_shape), tgt_dt)

    # ---------------------------------------------------------------------
    #  patch_info – capture original fn & inject wrapper
    # ---------------------------------------------------------------------
    @staticmethod
    def patch_info() -> dict[str, Any]:
        def _creator(orig_fn: Callable):
            logger.info("Storing original jnp.concatenate reference")
            _ORIGINAL_JNP_CONCATENATE = orig_fn
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
