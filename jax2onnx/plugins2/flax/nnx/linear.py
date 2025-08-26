# file: jax2onnx/plugins2/flax/nnx/linear.py

from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from contextlib import contextmanager
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive  
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


@register_primitive(
    jaxpr_primitive="nnx.linear",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html",
    onnx=[
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {"component": "Reshape", "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html"},
        {"component": "Shape", "doc": "https://onnx.ai/onnx/operators/onnx__Shape.html"},
        {"component": "Slice", "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html"},
        {"component": "Concat", "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html"},
        {"component": "CastLike", "doc": "https://onnx.ai/onnx/operators/onnx__CastLike.html"},
    ],
    since="v0.1.0",
    context="primitives2.nnx",
    component="linear",
    testcases=[
        {"testcase": "linear_symbolic_batch", "callable": nnx.Linear(128, 64, rngs=nnx.Rngs(0)),
         "input_shapes": [("B", 128)], "use_onnx_ir": True},
        {"testcase": "linear_high_rank", "callable": nnx.Linear(128, 64, rngs=nnx.Rngs(0)),
         "input_shapes": [("B", 10, 128)], "use_onnx_ir": True},
        {"testcase": "linear_no_bias", "callable": nnx.Linear(128, 64, use_bias=False, rngs=nnx.Rngs(0)),
         "input_shapes": [("B", 128)], "use_onnx_ir": True},
        {"testcase": "linear_high_rank_no_bias", "callable": nnx.Linear(128, 64, use_bias=False, rngs=nnx.Rngs(0)),
         "input_shapes": [("B", 10, 128)], "use_onnx_ir": True},
        {"testcase": "linear_merge_symbolic_dim", "callable": nnx.Linear(128, 64, rngs=nnx.Rngs(0)),
         "input_shapes": [("B", 10, 128)], "run_only_dynamic": True, "run_only_f32_variant": True,
         "use_onnx_ir": True},
    ],
)
class LinearPlugin(PrimitiveLeafPlugin):
    """
    IR-only plugin for flax.nnx.Linear:
      optional Reshape → Gemm → optional Reshape-back.
    The nnx.linear primitive is installed *scoped* during tracing only.
    """
    # Private primitive for this world (no import-time global assignment)
    _PRIM = Primitive("nnx.linear")
    _PRIM.multiple_results = False

    _ORIGINAL_LINEAR_CALL: Callable | None = None
    _ABSTRACT_EVAL_BOUND: bool = False  # bound lazily once

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(x, kernel, bias, *, use_bias: bool, dimension_numbers=None):
        # If we don't have the original __call__, fall back to shape math.
        if LinearPlugin._ORIGINAL_LINEAR_CALL is None:
            if dimension_numbers is None:
                lhs, rhs = ((x.ndim - 1,), (0,))
                dimension_numbers = ((lhs, rhs), ((), ()))
            k0, k1 = kernel.shape
            need_flat = (k0 != x.shape[-1]) or (x.ndim > 2)
            out_shape = (x.shape[0], k1) if need_flat else (*x.shape[:-1], k1)
            return jax.core.ShapedArray(out_shape, x.dtype)

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
            def dot_general(a, b, dimension_numbers=None, precision=None, **kwargs):
                return jax.lax.dot_general(a, b, dimension_numbers)
            dummy = SimpleNamespace(
                kernel=SimpleNamespace(value=kv),
                bias=SimpleNamespace(value=bv) if use_bias else None,
                use_bias=use_bias,
                axis=-1,
                in_features=kv.shape[0],
                out_features=kv.shape[1],
                promote_dtype=promote_dtype,
                dtype=xv.dtype,
                dot_general=dot_general,
                precision=None,
            )
            return LinearPlugin._ORIGINAL_LINEAR_CALL(dummy, xv)

        out = jax.eval_shape(_helper, x_spec, k_spec, b_spec)
        out = jax.tree_util.tree_leaves(out)[0]
        return jax.core.ShapedArray(out.shape, out.dtype)

    # ---------- lowering (IR) ----------
 

    def lower(self, ctx: "IRBuildContext", eqn):
        x_var, kernel_var, bias_var = eqn.invars
        out_var = eqn.outvars[0]
        use_bias = bool(eqn.params["use_bias"])

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        k_val = ctx.get_value_for_var(kernel_var, name_hint=ctx.fresh_name("kernel"))
        if use_bias:
            b_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("bias"))

        # Unify dtypes: cast *parameters* to the input dtype (mirrors JAX promotion).
        # NOTE: previously the CastLike ended up applied to the INPUT (down-casting
        # fp64 to fp32). Do the opposite: cast params to x's dtype.
        def _cast_param_to_input(val, name_hint):
            x_dtype = getattr(x_val.type, "dtype", None)
            v_dtype = getattr(val.type, "dtype", None)
            if x_dtype is not None and v_dtype is not None and v_dtype != x_dtype:
                out = ir.Value(
                    name=ctx.fresh_name(name_hint),
                    type=ir.TensorType(x_dtype),
                    shape=val.shape,
                )
                ctx.add_node(ir.Node(
                    op_type="CastLike", domain="",
                    inputs=[val, x_val],  # cast VAL to like X
                    outputs=[out],
                    name=ctx.fresh_name("CastLike"),
                ))
                return out
            return val
        k_val = _cast_param_to_input(k_val, "kernel_cast")
        if use_bias:
            b_val = _cast_param_to_input(b_val, "bias_cast")

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        k_shape = tuple(getattr(getattr(kernel_var, "aval", None), "shape", ()))
        in_features = int(k_shape[0])
        out_features = int(k_shape[1])
        need_flatten = len(x_shape) > 2

        gemm_in = x_val
        if need_flatten:
            rs_val = ir.Value(
                name=ctx.fresh_name("x2d_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((2,)),
                const_value=ir.tensor(np.asarray([-1, in_features], dtype=np.int64)),
            )
            ctx._initializers.append(rs_val)
            x2d = ir.Value(
                name=ctx.fresh_name("x2d"),
                type=x_val.type,
                # batch is unknown; let IR carry an unknown dim
                shape=ir.Shape((None, in_features)),
            )
            ctx.add_node(ir.Node(
                op_type="Reshape", domain="",
                inputs=[x_val, rs_val], outputs=[x2d],
                name=ctx.fresh_name("Reshape")))
            gemm_in = x2d

        # Gemm: if we will reshape later, write into a fresh temp Value
        if need_flatten:
            gemm_out = ir.Value(
                name=ctx.fresh_name("gemm_out"),
                type=x_val.type,  # output dtype = input dtype (params are cast to it)
                shape=ir.Shape((None, out_features)),
            )
        else:
            gemm_out = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("out"))
        inputs = [gemm_in, k_val] + ([b_val] if use_bias else [])
        ctx.add_node(ir.Node(
            op_type="Gemm", domain="",
            inputs=inputs, outputs=[gemm_out],
            name=ctx.fresh_name("Gemm"),
            attributes=[
                ir.Attr("alpha",  ir.AttributeType.FLOAT, 1.0),
                ir.Attr("beta",   ir.AttributeType.FLOAT, 1.0),
                ir.Attr("transA", ir.AttributeType.INT,   0),
                ir.Attr("transB", ir.AttributeType.INT,   0),
            ],
        ))

        # Reshape back if needed: final_shape = x.shape[:-1] ++ [out_features]
        if need_flatten:
            shp = ir.Value(
                name=ctx.fresh_name("x_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((len(x_shape),)),
            )
            ctx.add_node(ir.Node(
                op_type="Shape", domain="",
                inputs=[x_val], outputs=[shp],
                name=ctx.fresh_name("Shape")))

            starts = ir.Value(name=ctx.fresh_name("slice_starts"),
                              type=ir.TensorType(ir.DataType.INT64),
                              shape=ir.Shape((1,)),
                              const_value=ir.tensor(np.array([0], dtype=np.int64)))
            ends = ir.Value(name=ctx.fresh_name("slice_ends"),
                            type=ir.TensorType(ir.DataType.INT64),
                            shape=ir.Shape((1,)),
                            const_value=ir.tensor(np.array([len(x_shape)-1], dtype=np.int64)))
            ctx._initializers.extend([starts, ends])

            batch_dims = ir.Value(
                name=ctx.fresh_name("batch_dims"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((len(x_shape)-1,)),
            )
            ctx.add_node(ir.Node(
                op_type="Slice", domain="",
                inputs=[shp, starts, ends], outputs=[batch_dims],
                name=ctx.fresh_name("Slice")))

            of = ir.Value(name=ctx.fresh_name("out_features_c"),
                          type=ir.TensorType(ir.DataType.INT64),
                          shape=ir.Shape((1,)),
                          const_value=ir.tensor(np.array([out_features], dtype=np.int64)))
            ctx._initializers.append(of)

            final_shape = ir.Value(
                name=ctx.fresh_name("final_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((len(x_shape),)),
            )
            ctx.add_node(ir.Node(
                op_type="Concat",
                domain="",
                inputs=[batch_dims, of],
                outputs=[final_shape],
                name=ctx.fresh_name("Concat"),
                attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
            ))

            reshaped_out = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("out"))
            ctx.add_node(ir.Node(
                op_type="Reshape", domain="",
                inputs=[gemm_out, final_shape], outputs=[reshaped_out],
                name=ctx.fresh_name("Reshape")))

    # ---------- monkey-patch helper (single, non-duplicated) ----------
    @staticmethod
    def get_monkey_patch(orig_fn):
        LinearPlugin._ORIGINAL_LINEAR_CALL = orig_fn
        prim = LinearPlugin._PRIM

        def patched(self, x):
            dn = (((x.ndim - 1,), (0,)), ((), ()))
            kernel = self.kernel.value
            use_bias = self.bias is not None
            bias = self.bias.value if use_bias else jnp.zeros((), dtype=x.dtype)
            return prim.bind(x, kernel, bias, use_bias=use_bias, dimension_numbers=dn)

        return patched

    # ---------- conversion-time world binding (scoped) ----------
    @staticmethod
    @contextmanager
    def world_activation():
        """
        Temporarily:
          • assign flax.nnx.linear_p to this plugin's private Primitive
          • ensure abstract_eval is registered on that Primitive
          • monkey-patch nnx.Linear.__call__ to bind that Primitive
        """
        prev_prim = getattr(nnx, "linear_p", None)
        prev_call = getattr(nnx.Linear, "__call__")

        nnx.linear_p = LinearPlugin._PRIM
        if not LinearPlugin._ABSTRACT_EVAL_BOUND:
            LinearPlugin._PRIM.def_abstract_eval(LinearPlugin.abstract_eval)
            LinearPlugin._ABSTRACT_EVAL_BOUND = True

        setattr(nnx.Linear, "__call__", LinearPlugin.get_monkey_patch(prev_call))
        try:
            yield
        finally:
            setattr(nnx.Linear, "__call__", prev_call)
            if prev_prim is None:
                try:
                    delattr(nnx, "linear_p")
                except Exception:
                    pass
            else:
                setattr(nnx, "linear_p", prev_prim)
