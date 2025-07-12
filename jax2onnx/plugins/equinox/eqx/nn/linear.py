# jax2onnx/plugins/equinox/eqx/nn/linear.py
"""
ONNX plugin for **equinox.nn.Linear** that supports symbolic-batch dimensions
and high-rank inputs.

Key differences vs. the Flax/NNX version
----------------------------------------
* Equinox stores parameters directly (`self.weight`, `self.bias`) – no `.value`.
* Weight has shape ``(out_features, in_features)`` so ONNX ``Gemm`` is emitted
  with ``transB = 1`` instead of transposing beforehand.
* The monkey-patch targets **eqx.nn.Linear** and binds through
  ``eqx.nn.linear_p``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable

import equinox as eqx
import jax
import numpy as np
from jax import core
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # only for type checkers & IDEs
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.equinox.eqx.nn.linear")

# --------------------------------------------------------------------------
# Example modules for testcases: seed once at import time, so __init__ PRNG
# happens now, not inside the traced function.
_eqx_linear_symbolic_mod = eqx.nn.Linear(128, 64, key=jax.random.PRNGKey(0))
_eqx_linear_highrank_mod = eqx.nn.Linear(128, 64, key=jax.random.PRNGKey(0))
# --------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. Primitive -----------------------------------------------------------------
# -----------------------------------------------------------------------------
eqx.nn.linear_p = Primitive("eqx.nn.linear")
eqx.nn.linear_p.multiple_results = False


# -----------------------------------------------------------------------------
# 2. Plugin registration -------------------------------------------------------
# -----------------------------------------------------------------------------
@register_primitive(
    jaxpr_primitive=eqx.nn.linear_p.name,
    jax_doc="https://docs.kidger.site/equinox/api/eqx/nn/linear/",
    onnx=[
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
    ],
    since="v0.7.2",
    context="primitives.eqx",
    component="linear",
    testcases=[
        # TODO: enable testcases
        # {
        #     "testcase": "eqx_linear_symbolic_batch",
        #     "callable": lambda x, _mod=_eqx_linear_symbolic_mod: jax.vmap(_mod)(x),
        #     "input_shapes": [("B", 128)],
        #     "post_check_onnx_graph": lambda m: (
        #         any(node.op_type == "Gemm" for node in m.graph.node)
        #     ),
        # },
        # {
        #     "testcase": "eqx_linear_high_rank",
        #     "callable": lambda x, _mod=_eqx_linear_highrank_mod: jax.vmap(_mod)(x),
        #     "input_shapes": [(32, 10, 128)],
        #     "post_check_onnx_graph": lambda m: (
        #         any(node.op_type == "Gemm" for node in m.graph.node)
        #     ),
        # },
    ],
)
class EqxLinearPlugin(PrimitiveLeafPlugin):
    """Convert **equinox.nn.Linear** to ONNX (symbolic-dim aware)."""

    # ------------------------------------------------------------------
    # keep a reference to the pristine implementation
    # ------------------------------------------------------------------
    _ORIGINAL_LINEAR_CALL: Callable | None = None

    # ------------------------------------------------------------------
    # helper ------------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_graph_input(s: "Jaxpr2OnnxConverter", name: str, var) -> None:
        """Make *name* a graph input if it is not a constant/initializer."""
        if name in s.name_to_const:  # → will be an initializer
            return
        if any(inp.name == name for inp in s.builder.inputs):
            return
        dtype_enum = s.builder._numpy_dtype_to_onnx(var.aval.dtype)
        value_info = helper.make_tensor_value_info(
            name,
            dtype_enum,
            [d if isinstance(d, int) else None for d in var.aval.shape],
        )
        s.builder.inputs.append(value_info)

    # ------------------------------------------------------------------
    # abstract-eval -----------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def abstract_eval(
        x: core.ShapedArray,
        weight: core.ShapedArray,
        bias: core.ShapedArray,
    ):
        """
        Symbolic-shape rule **delegating** to the untouched
        `equinox.nn.Linear.__call__`.
        """
        if EqxLinearPlugin._ORIGINAL_LINEAR_CALL is None:
            raise RuntimeError("Original eqx.nn.Linear.__call__ not stored.")

        # lightweight ShapeDtypeStruct shells
        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        w_spec = jax.ShapeDtypeStruct(weight.shape, weight.dtype)
        b_spec = jax.ShapeDtypeStruct(bias.shape, bias.dtype)

        def _helper(xv, wv, bv):
            """Call the un-patched Linear with a dummy module."""

            dummy = SimpleNamespace(
                weight=wv,
                bias=bv,
                use_bias=bv is not None,
            )
            return EqxLinearPlugin._ORIGINAL_LINEAR_CALL(dummy, xv)

        try:
            out = jax.eval_shape(_helper, x_spec, w_spec, b_spec)
            out = jax.tree_util.tree_leaves(out)[0]
            return core.ShapedArray(out.shape, out.dtype)
        except Exception:  # fallback – handle flattening
            need_flat = x.ndim > 2
            out_features, _ = weight.shape
            if need_flat:
                out_shape = (x.shape[0], out_features)
            else:
                out_shape = (*x.shape[:-1], out_features)
            return core.ShapedArray(out_shape, x.dtype)

    # ------------------------------------------------------------------
    # ONNX lowering -----------------------------------------------------
    # ------------------------------------------------------------------
    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        x_var, w_var, b_var = node_inputs
        y_var = node_outputs[0]

        x_name = s.get_name(x_var)
        w_name = s.get_name(w_var)
        b_name = s.get_name(b_var)
        y_name = s.get_name(y_var)

        # make sure they are graph inputs (important for constants)
        for n, v in [(x_name, x_var), (w_name, w_var), (b_name, b_var)]:
            self._ensure_graph_input(s, n, v)

        x_shape = x_var.aval.shape
        out_shape = y_var.aval.shape
        dtype = x_var.aval.dtype

        in_features = w_var.aval.shape[1]
        out_features = w_var.aval.shape[0]
        batch_dims = x_shape[:-1]
        need_flatten = len(x_shape) > 2

        # -- Step 1: flatten input if needed ---------------------------------
        if need_flatten:
            flat_name = s.get_unique_name("x2d")
            reshape_shape = [-1, in_features]
            shape_const = s.get_constant_name(np.array(reshape_shape, np.int64))

            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[x_name, shape_const],
                    outputs=[flat_name],
                    name=s.get_unique_name("reshape_flatten"),
                )
            )
            x_name = flat_name
            s.add_shape_info(x_name, tuple(reshape_shape), dtype)

        # -- Step 2: Gemm  (note: transB = 1 !) ------------------------------
        gemm_out = s.get_unique_name("gemm_out")
        s.add_node(
            helper.make_node(
                "Gemm",
                inputs=[x_name, w_name, b_name],
                outputs=[gemm_out],
                name=s.get_unique_name("linear_gemm"),
                transB=1,  # weight is (out, in)
            )
        )
        s.add_shape_info(gemm_out, (-1, out_features), dtype)

        # -- Step 3: restore original shape if we flattened ------------------
        if need_flatten:
            target_shape = [
                (-1 if not isinstance(d, int) else d) for d in batch_dims
            ] + [out_features]
            shape_const = s.get_constant_name(np.array(target_shape, np.int64))

            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[gemm_out, shape_const],
                    outputs=[y_name],
                    name=s.get_unique_name("reshape_output"),
                )
            )
            s.add_shape_info(y_name, out_shape, dtype)
        else:
            s.var_to_name[y_var] = gemm_out
            s.add_shape_info(gemm_out, out_shape, dtype)

    # ------------------------------------------------------------------
    # monkey-patch ------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def get_monkey_patch(orig_fn):
        """
        Store the *original* implementation and return a patched version that
        routes calls through the new primitive.
        """
        EqxLinearPlugin._ORIGINAL_LINEAR_CALL = orig_fn

        def patched_call(self, x):
            return eqx.nn.linear_p.bind(x, self.weight, self.bias)

        return patched_call

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [eqx.nn.Linear],
            "patch_function": EqxLinearPlugin.get_monkey_patch,  # receives orig_fn
            "target_attribute": "__call__",
        }


# -----------------------------------------------------------------------------
# 3. Register abstract-eval ----------------------------------------------------
# -----------------------------------------------------------------------------
eqx.nn.linear_p.def_abstract_eval(EqxLinearPlugin.abstract_eval)


# ------------------------------------------------------------------
# 4.  Batching rule ------------------------------------------------
# ------------------------------------------------------------------
def _eqx_linear_batching_rule(batched_args, batch_dims, **_):
    """Batching rule for `eqx.nn.linear_p`."""
    x, weight, bias = batched_args
    x_bdim, w_bdim, b_bdim = batch_dims

    # For `vmap(model)(xs)`, only `xs` has a batch dimension.
    # The model parameters (weight, bias) are treated as constants w.r.t. `vmap`.
    if w_bdim is not None or b_bdim is not None:
        raise NotImplementedError(
            "Batching over `eqx.nn.Linear` parameters is not supported."
        )

    # The primitive is now applied to a batched `x`. The `to_onnx` implementation
    # will see the extra dimension on `x` and handle it by flattening/unflattening.
    out = eqx.nn.linear_p.bind(x, weight, bias)

    # The output has a batch dimension at the same axis as the input.
    return out, x_bdim


# Register the batching rule for our primitive
batching.primitive_batchers[eqx.nn.linear_p] = _eqx_linear_batching_rule
