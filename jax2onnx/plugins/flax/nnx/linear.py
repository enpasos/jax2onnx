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

from typing import TYPE_CHECKING, Callable, Any, Sequence
import logging
import numpy as np
import onnx
import jax
import jax.numpy as jnp
from jax import core, lax
from flax import nnx
from jax.extend.core import Primitive
from onnx import helper, numpy_helper

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.flax.nnx.linear_general import _shape_of


def _static_dim(dim):
    """Return an int when the dimension is statically known."""
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    try:
        return int(dim)  # handles numpy scalars that advertise __int__
    except Exception:
        return None


def _update_symbolic_shape(converter, name: str, shape: tuple[Any, ...]) -> None:
    if not hasattr(converter, "symbolic_shapes"):
        return
    if hasattr(converter, "_dim_to_symbol_safe"):
        sanitized = tuple(
            converter._dim_to_symbol_safe(d) if d is not None else None for d in shape
        )
    else:
        sanitized = shape
    converter.symbolic_shapes[name] = sanitized


def _tensor_as_array(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == name:
            for attr in node.attribute:
                if (
                    attr.name == "value"
                    and attr.type == onnx.AttributeProto.TENSOR
                    and attr.t is not None
                ):
                    return numpy_helper.to_array(attr.t)
    return None


def _first_flatten_shape(model):
    for node in model.graph.node:
        if node.op_type == "Reshape" and len(node.input) >= 2:
            arr = _tensor_as_array(model, node.input[1])
            if arr is not None and arr.ndim == 1 and arr.size == 2 and arr[0] == -1:
                return arr
    return None


def _gemm_weight_shape(model):
    for node in model.graph.node:
        if node.op_type == "Gemm" and len(node.input) >= 2:
            arr = _tensor_as_array(model, node.input[1])
            if arr is not None:
                return arr.shape
    return None


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
            "input_shapes": [("B", 10, 128)],
        },
        {
            "testcase": "linear_no_bias",
            "callable": nnx.Linear(128, 64, use_bias=False, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 128)],
        },
        {
            "testcase": "linear_high_rank_no_bias",
            "callable": nnx.Linear(128, 64, use_bias=False, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 10, 128)],
        },
        {
            "testcase": "linear_merge_symbolic_dim",
            "callable": nnx.Linear(128, 64, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 10, 128)],  # B is symbolic
            "run_only_dynamic": True,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: (
                # input  B×10×128
                _shape_of(m.graph.input, "var_0") == ("B", 10, 128)
                # after flatten via reshape constant [-1, 128]
                and (lambda vec: vec is not None and vec.size == 2 and vec[1] == 128)(
                    _first_flatten_shape(m)
                )
                # gemm weight   128×64 → ensures output width 64
                and (lambda shp: shp is not None and len(shp) == 2 and shp[1] == 64)(
                    _gemm_weight_shape(m)
                )
                # final output   B×10×64
                and _shape_of(m.graph.output, "var_3") == ("B", 10, 64)
            ),
        },
    ],
)
class LinearPlugin(PrimitiveLeafPlugin):
    """Convert **flax.nnx.Linear** to ONNX (symbolic‑dim aware)."""

    _ORIGINAL_LINEAR_CALL: Callable | None = None

    @staticmethod
    def _ensure_graph_input(s: "Jaxpr2OnnxConverter", name: str, var) -> None:
        if name in s.name_to_const:
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

    @staticmethod
    def abstract_eval(
        x: core.ShapedArray,
        kernel: core.ShapedArray,
        bias: core.ShapedArray,
        *,
        use_bias: bool,
        dimension_numbers=None,
    ):
        if LinearPlugin._ORIGINAL_LINEAR_CALL is None:
            raise RuntimeError("Original nnx.Linear.__call__ has not been stored.")

        if dimension_numbers is None:
            lhs, rhs = ((x.ndim - 1,), (0,))
            dimension_numbers = ((lhs, rhs), ((), ()))

        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        k_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        b_spec = jax.ShapeDtypeStruct(bias.shape, bias.dtype) if use_bias else None

        def _helper(xv, kv, bv):
            from types import SimpleNamespace

            def promote_dtype(args, dtype=None):
                return args

            def dot_general(x, y, dimension_numbers=None, precision=None, **kwargs):
                return lax.dot_general(x, y, dimension_numbers)

            dummy = SimpleNamespace(
                kernel=SimpleNamespace(value=kv),
                bias=SimpleNamespace(value=bv) if use_bias else None,
                use_bias=use_bias,
                axis=-1,
                in_features=kv.shape[0],
                out_features=kv.shape[1],
                promote_dtype=promote_dtype,
                dtype=x.dtype,
                dot_general=dot_general,
                precision=None,
            )
            return LinearPlugin._ORIGINAL_LINEAR_CALL(dummy, xv)

        try:
            out = jax.eval_shape(_helper, x_spec, k_spec, b_spec)
            out = jax.tree_util.tree_leaves(out)[0]
            return core.ShapedArray(out.shape, out.dtype)
        except Exception:
            need_flat = (kernel.shape[0] != x.shape[-1]) or (x.ndim > 2)
            if need_flat:
                out_shape = (x.shape[0], kernel.shape[1])
            else:
                out_shape = (*x.shape[:-1], kernel.shape[1])
            return core.ShapedArray(out_shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        x_var, kernel_var, bias_var = node_inputs
        y_var = node_outputs[0]
        use_bias = params["use_bias"]

        x_name = s.get_name(x_var)
        kernel_name = s.get_name(kernel_var)
        bias_name = s.get_name(bias_var) if use_bias else ""

        x_shape = x_var.aval.shape
        out_shape = y_var.aval.shape
        dtype = x_var.aval.dtype

        kernel_shape_attr = getattr(kernel_var.aval, "shape", ())
        if isinstance(kernel_shape_attr, Sequence):
            kernel_shape: tuple[Any, ...] = tuple(kernel_shape_attr)
        else:
            kernel_shape = ()
        kernel_static = tuple(_static_dim(d) for d in kernel_shape)
        kernel_const = s.name_to_const.get(kernel_name)
        const_shape: tuple[int, ...] = ()
        if kernel_const is not None:
            if hasattr(kernel_const, "shape"):
                const_shape = tuple(int(v) for v in kernel_const.shape)
            elif hasattr(kernel_const, "dims"):
                const_shape = tuple(int(v) for v in kernel_const.dims)
        if const_shape:
            padded = list(kernel_static)
            if len(padded) < len(const_shape):
                padded.extend([None] * (len(const_shape) - len(padded)))
            kernel_static = tuple(
                padded[i] if padded[i] is not None else const_shape[i]
                for i in range(len(const_shape))
            )
        logger = logging.getLogger("jax2onnx.plugins.flax.nnx.linear")
        logger.debug(
            "linear.merge: kernel_shape=%s kernel_static=%s const_shape=%s",
            kernel_shape,
            kernel_static,
            const_shape,
        )

        in_features_static = kernel_static[0] if len(kernel_static) > 0 else None
        out_features_static = kernel_static[1] if len(kernel_static) > 1 else None
        in_features = (
            in_features_static
            if in_features_static is not None
            else (kernel_shape[0] if kernel_shape else None)
        )
        out_features = (
            out_features_static
            if out_features_static is not None
            else (kernel_shape[1] if len(kernel_shape) > 1 else None)
        )

        need_flatten = len(x_shape) > 2
        if need_flatten:
            flat_name = s.get_unique_name("x2d")
            reshape_shape = s.get_constant_name(
                np.array([-1, in_features], dtype=np.int64)
            )
            s.add_node(
                helper.make_node("Reshape", [x_name, reshape_shape], [flat_name])
            )
            flat_shape = (
                None,
                in_features_static if in_features_static is not None else None,
            )
            _update_symbolic_shape(s, flat_name, flat_shape)
            s.add_shape_info(flat_name, flat_shape, dtype)
            x_name = flat_name

        gemm_out = s.get_unique_name("gemm_out")
        gemm_inputs = [x_name, kernel_name]
        if use_bias:
            gemm_inputs.append(bias_name)

        s.add_node(helper.make_node("Gemm", gemm_inputs, [gemm_out]))

        # After flattening an (unknown) batch we want “?” not "unk__0"
        batch_dim_gemm = None if need_flatten else (x_shape[0] if x_shape else None)
        gemm_shape = (
            batch_dim_gemm,
            out_features_static if out_features_static is not None else None,
        )
        _update_symbolic_shape(s, gemm_out, gemm_shape)
        s.add_shape_info(gemm_out, gemm_shape, dtype)

        if need_flatten:
            original_x_shape = s.get_unique_name("original_x_shape")
            s.add_node(
                helper.make_node("Shape", [s.get_name(x_var)], [original_x_shape])
            )
            s.add_shape_info(original_x_shape, (len(x_shape),), np.int64)

            batch_dims_shape = s.get_unique_name("batch_dims_shape")
            rank = len(x_shape)
            starts = s.get_constant_name(np.array([0], dtype=np.int64))
            ends = s.get_constant_name(np.array([rank - 1], dtype=np.int64))
            s.add_node(
                helper.make_node(
                    "Slice", [original_x_shape, starts, ends], [batch_dims_shape]
                )
            )
            s.add_shape_info(batch_dims_shape, (rank - 1,), np.int64)

            out_feat_int = (
                out_features_static if out_features_static is not None else out_features
            )
            out_features_const = s.get_constant_name(
                np.array([out_feat_int], dtype=np.int64)
            )

            final_shape = s.get_unique_name("final_shape")
            s.add_node(
                helper.make_node(
                    "Concat",
                    [batch_dims_shape, out_features_const],
                    [final_shape],
                    axis=0,
                )
            )
            s.add_shape_info(final_shape, (rank,), np.int64)

            output_name = s.get_name(y_var)
            s.add_node(
                helper.make_node("Reshape", [gemm_out, final_shape], [output_name])
            )
            _update_symbolic_shape(s, output_name, out_shape)
            s.add_shape_info(output_name, out_shape, dtype)
        else:
            s.var_to_name[y_var] = gemm_out

    @staticmethod
    def get_monkey_patch(orig_fn):
        LinearPlugin._ORIGINAL_LINEAR_CALL = orig_fn

        def patched_call(self, x):
            dn = (((x.ndim - 1,), (0,)), ((), ()))
            kernel = self.kernel.value
            use_bias = self.bias is not None
            if use_bias:
                bias = self.bias.value
            else:
                bias = jnp.zeros((), dtype=x.dtype)

            return nnx.linear_p.bind(
                x, kernel, bias, use_bias=use_bias, dimension_numbers=dn
            )

        return patched_call

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.Linear],
            "patch_function": LinearPlugin.get_monkey_patch,
            "target_attribute": "__call__",
        }


nnx.linear_p.def_abstract_eval(LinearPlugin.abstract_eval)
