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

from typing import TYPE_CHECKING, Callable
import logging
import numpy as np
import jax
from jax import core, lax
from flax import nnx
from jax.extend.core import Primitive
from onnx import helper

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
    # keep a reference to the pristine implementation
    # ------------------------------------------------------------------
    _ORIGINAL_LINEAR_CALL: Callable | None = None

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
        """
        Symbolic-shape rule **delegating** to the untouched
        `flax.nnx.Linear.__call__`.
        """
        if LinearPlugin._ORIGINAL_LINEAR_CALL is None:
            raise RuntimeError("Original nnx.Linear.__call__ has not been stored.")

        # default dimension_numbers (last-dim ⋅ first-dim) if None
        if dimension_numbers is None:
            lhs, rhs = ((x.ndim - 1,), (0,))
            dimension_numbers = ((lhs, rhs), ((), ()))

        # prepare ShapeDtypeStruct shells
        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        k_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        b_spec = jax.ShapeDtypeStruct(bias.shape, bias.dtype)

        def _helper(xv, kv, bv):
            """Call the *un-patched* Linear with a lightweight dummy."""
            # emulate the minimal attribute set Linear.__call__ touches
            from types import SimpleNamespace

            def promote_dtype(args, dtype=None):  # noqa: ANN001
                return args  # no casting in abstract mode

            def dot_general(x, y, dimension_numbers=None, precision=None, **kwargs):
                # Use JAX's dot_general directly for shape computation
                # Ignore precision and other args that may be passed
                return lax.dot_general(x, y, dimension_numbers)

            dummy = SimpleNamespace(
                kernel=SimpleNamespace(value=kv),
                bias=SimpleNamespace(value=bv),
                use_bias=bv is not None,
                axis=-1,
                in_features=kv.shape[0],
                out_features=kv.shape[1],
                promote_dtype=promote_dtype,
                dtype=x.dtype,
                dot_general=dot_general,
                precision=None,  # Add missing precision attribute
            )
            return LinearPlugin._ORIGINAL_LINEAR_CALL(dummy, xv)

        # -- first choice: let the *real* implementation decide -------------
        try:
            out = jax.eval_shape(_helper, x_spec, k_spec, b_spec)
            out = jax.tree_util.tree_leaves(out)[0]
            return core.ShapedArray(out.shape, out.dtype)
        except Exception:  # -- contracting-dim mismatch
            # Fallback: if we would need to flatten, the resulting tensor is
            #           (batch, kernel_out); else keep original rank.
            need_flat = (kernel.shape[0] != x.shape[-1]) or (x.ndim > 2)
            if need_flat:
                out_shape = (x.shape[0], kernel.shape[1])
            else:
                out_shape = (*x.shape[:-1], kernel.shape[1])
            return core.ShapedArray(out_shape, x.dtype)

    # ------------------------------------------------------------------
    # ONNX lowering -----------------------------------------------------
    # ------------------------------------------------------------------
    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        x_var, kernel_var, bias_var = node_inputs
        y_var = node_outputs[0]

        x_name = s.get_name(x_var)
        kernel_name = s.get_name(kernel_var)
        bias_name = s.get_name(bias_var)

        x_shape = x_var.aval.shape
        out_shape = y_var.aval.shape
        dtype = x_var.aval.dtype

        in_features = kernel_var.aval.shape[0]
        out_features = kernel_var.aval.shape[1]
        batch_dims = x_shape[:-1]

        need_flatten = len(x_shape) > 2

        # Step 1: Flatten input if needed
        if need_flatten:
            flat_name = s.get_unique_name("x2d")
            reshape_shape = [-1, in_features]
            shape_const = s.get_constant_name(np.array(reshape_shape, dtype=np.int64))

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

        # Step 2: Linear layer → Gemm
        gemm_out = s.get_unique_name("gemm_out")
        s.add_node(
            helper.make_node(
                "Gemm",
                inputs=[x_name, kernel_name, bias_name],
                outputs=[gemm_out],
                name=s.get_unique_name("linear_gemm"),
            )
        )
        s.add_shape_info(gemm_out, (-1, out_features), dtype)

        # Step 3: Restore original shape if input was flattened
        if need_flatten:
            target_shape = []
            for d in batch_dims:
                target_shape.append(-1 if not isinstance(d, int) else d)
            target_shape.append(out_features)

            shape_const = s.get_constant_name(np.array(target_shape, dtype=np.int64))
            output_name = s.get_name(y_var)

            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[gemm_out, shape_const],
                    outputs=[output_name],
                    name=s.get_unique_name("reshape_output"),
                )
            )
            s.add_shape_info(output_name, out_shape, dtype)
        else:
            output_name = s.get_name(y_var)
            s.var_to_name[y_var] = gemm_out
            s.add_shape_info(gemm_out, out_shape, dtype)

    # ------------------------------------------------------------------
    # monkey‑patch -------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def get_monkey_patch(orig_fn):
        """
        Store *orig_fn* for `abstract_eval` and return the patched
        implementation that routes through the new primitive.
        """
        LinearPlugin._ORIGINAL_LINEAR_CALL = orig_fn

        def patched_call(self, x):
            dn = (((x.ndim - 1,), (0,)), ((), ()))  # last dim ⋅ first dim
            # Extract kernel and bias directly from the module's parameters
            kernel = self.kernel.value
            bias = self.bias.value
            return nnx.linear_p.bind(x, kernel, bias, dimension_numbers=dn)

        return patched_call

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.Linear],
            "patch_function": LinearPlugin.get_monkey_patch,  # receives orig_fn
            "target_attribute": "__call__",
        }


# -----------------------------------------------------------------------------
# 3.  Register abstract‑eval ---------------------------------------------------
# -----------------------------------------------------------------------------
nnx.linear_p.def_abstract_eval(LinearPlugin.abstract_eval)
