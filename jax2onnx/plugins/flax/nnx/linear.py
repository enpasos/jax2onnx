# jax2onnx/plugins/flax/nnx/linear.py
"""
ONNX plugin for **flax.nnx.Linear** that supports symbolic batch dimensions and
high‑rank inputs.

Fix for missing graph‑input error
---------------------------------
* After renaming the three logical inputs to ``x``, ``kernel`` and ``bias`` we
  must *also* register them as **graph inputs** in the ``OnnxBuilder``.  Merely
  attaching value‑info is not enough – ONNX requires that every node input be a
  graph input, an initializer or the output of another node.
* Helper ``_ensure_graph_input`` adds the appropriate tensor‑value‑info entry
  unless the name already refers to a constant initializer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import logging
import numpy as np
import jax
import jax.numpy as jnp
from jax import core, lax
from flax import nnx
from jax.extend.core import Primitive
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # only for static analysis / IDEs
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.flax.nnx.linear")

# -----------------------------------------------------------------------------
# 1.  Primitive ----------------------------------------------------------------
# -----------------------------------------------------------------------------
nnx.linear_p = Primitive("nnx.linear")
nnx.linear_p.multiple_results = False


# -----------------------------------------------------------------------------
# 2.  Plugin registration ------------------------------------------------------
# -----------------------------------------------------------------------------
@register_primitive(
    jaxpr_primitive=nnx.linear_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html",
    onnx=[
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="linear",
    testcases=[
        {
            "testcase": "linear_symbolic_batch",
            "callable": nnx.Linear(128, 64, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 128)],
        },
        {
            "testcase": "linear_high_rank",
            "callable": nnx.Linear(128, 64, rngs=nnx.Rngs(0)),
            "input_shapes": [(32, 10, 128)],
        },
    ],
)
class LinearPlugin(PrimitiveLeafPlugin):
    """Convert **flax.nnx.Linear** to ONNX (symbolic‑dim aware)."""

    # ------------------------------------------------------------------
    # helper ------------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_graph_input(s: "Jaxpr2OnnxConverter", name: str, var) -> None:
        """Make *name* a graph input if it is not a constant/initializer."""
        if name in s.name_to_const:
            # constant → will become an initializer, nothing to do
            return
        # Avoid duplicate inputs
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
    # abstract‑eval -----------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def abstract_eval(
        x: core.ShapedArray,
        kernel: core.ShapedArray,
        bias: core.ShapedArray,
        dimension_numbers=None,
    ):
        """Shape inference via :pyfunc:`jax.eval_shape`."""
        if dimension_numbers is None:
            lhs, rhs = ((x.ndim - 1,), (0,))
            dimension_numbers = ((lhs, rhs), ((), ()))

        def _linear(lhs, w, b):
            return lax.dot_general(lhs, w, dimension_numbers) + b

        out_aval = jax.eval_shape(
            _linear,
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(kernel.shape, kernel.dtype),
            jax.ShapeDtypeStruct(bias.shape, bias.dtype),
        )
        return core.ShapedArray(out_aval.shape, out_aval.dtype)

    # ------------------------------------------------------------------
    # ONNX lowering -----------------------------------------------------
    # ------------------------------------------------------------------
    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: list[Any],
        node_outputs: list[Any],
        params: dict[str, Any],
    ) -> None:
        x_var, k_var, b_var = node_inputs
        y_var = node_outputs[0]

        # -------------------------- nice names ---------------------------
        # Store *old* names so we can detect if a rename happened.
        old_x_name = s.get_name(x_var)
        old_k_name = s.get_name(k_var)
        old_b_name = s.get_name(b_var)

        s.set_var_name(x_var, "x")
        s.set_var_name(k_var, "kernel")
        s.set_var_name(b_var, "bias")

        x_name = s.get_name(x_var)
        k_name = s.get_name(k_var)
        b_name = s.get_name(b_var)
        y_name = s.get_name(y_var)

        # Register as graph inputs (builder complains otherwise)
        for name, var in ((x_name, x_var), (k_name, k_var), (b_name, b_var)):
            self._ensure_graph_input(s, name, var)
            s.add_shape_info(name, tuple(var.aval.shape), var.aval.dtype)

        # --------------------- flatten leading batch dims ---------------
        x_shape = x_var.aval.shape
        batch_dims, in_features = x_shape[:-1], x_shape[-1]

        if len(batch_dims) > 1:
            flat_shape = np.array([-1, in_features], dtype=np.int64)
            flat_shape_name = s.get_constant_name(flat_shape)

            x2d_name = s.get_unique_name("x2d")
            reshape_in = helper.make_node(
                "Reshape",
                inputs=[x_name, flat_shape_name],
                outputs=[x2d_name],
                name=s.get_unique_name("reshape_in"),
                allowzero=0,
            )
            s.add_node(reshape_in)
            s.add_shape_info(x2d_name, (-1, in_features), x_var.aval.dtype)
        else:
            x2d_name = x_name  # already 2‑D

        # -------------------- weight handling ---------------------------
        if k_name in s.name_to_const:
            weight_name = s.get_constant_name(s.name_to_const[k_name])
        else:
            weight_name = k_name

        # -------------------- GEMM --------------------------------------
        gemm_out = s.get_unique_name("gemm_out")
        gemm_node = helper.make_node(
            "Gemm",
            inputs=[x2d_name, weight_name, b_name],
            outputs=[gemm_out],
            name=s.get_unique_name("Gemm"),
        )
        s.add_node(gemm_node)

        out_features = k_var.aval.shape[1]
        s.add_shape_info(gemm_out, (-1, out_features), x_var.aval.dtype)

        # -------------------- restore batch dims ------------------------
        if len(batch_dims) > 1:
            final_shape = np.array(batch_dims + (out_features,), dtype=np.int64)
            final_shape_name = s.get_constant_name(final_shape)

            reshape_out = helper.make_node(
                "Reshape",
                inputs=[gemm_out, final_shape_name],
                outputs=[y_name],
                name=s.get_unique_name("reshape_out"),
                allowzero=0,
            )
            s.add_node(reshape_out)
            s.add_shape_info(
                y_name, tuple(batch_dims + (out_features,)), x_var.aval.dtype
            )
        else:
            s.var_to_name[y_var] = gemm_out
            batch = batch_dims[0] if batch_dims else 1
            s.add_shape_info(gemm_out, (batch, out_features), x_var.aval.dtype)

    # ------------------------------------------------------------------
    # monkey‑patch -------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def get_monkey_patch():
        def patched_call(self, x):
            dn = (((x.ndim - 1,), (0,)), ((), ()))  # last dim ⋅ first dim
            return nnx.linear_p.bind(
                x, self.kernel.value, self.bias.value, dimension_numbers=dn
            )

        return patched_call

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.Linear],
            "patch_function": lambda _: LinearPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# -----------------------------------------------------------------------------
# 3.  Register abstract‑eval ---------------------------------------------------
# -----------------------------------------------------------------------------
nnx.linear_p.def_abstract_eval(LinearPlugin.abstract_eval)
