# file: jax2onnx/plugins2/flax/nnx/linear.py

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2._utils import cast_param_like
from jax2onnx.plugins2._ir_shapes import (
    _stamp_type_and_shape,
    _is_static_int,
    _dim_label_from_value_or_aval,
    _ensure_value_info,
    is_shape_all_unknown,
)
from jax2onnx.plugins2._post_check_onnx_graph import expect_graph

if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore

# ------------------------------------------------------------------
# Graph-pattern expectations used by tests
# ------------------------------------------------------------------
# Basic presence of a single Gemm (no flatten/reshape path needed).
EXPECT_GEMM_ONLY = expect_graph(["Gemm"], match="contains")
# Static flatten path: Reshape -> Gemm -> Reshape (no dynamic shape ops).
EXPECT_RGR = expect_graph(["^Reshape->Gemm->Reshape$"], match="exact")
# Dynamic flatten path: input Reshape to Gemm, and separate dynamic-shape chain
# (Shape->Slice->Concat) that feeds the final Reshape's shape, plus Gemm->Reshape.
EXPECT_DYNAMIC_RGR = expect_graph(
    [
        "^Reshape->Gemm->Reshape$",
        "^Shape->Slice->Concat->Reshape$",
    ],
    mode="all",
    match="exact",
)


@register_primitive(
    jaxpr_primitive="nnx.linear",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html",
    onnx=[
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Shape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Shape.html",
        },
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {
            "component": "CastLike",
            "doc": "https://onnx.ai/onnx/operators/onnx__CastLike.html",
        },
    ],
    since="v0.1.0",
    context="primitives2.nnx",
    component="linear",
    testcases=[
        {
            "testcase": "linear_symbolic_batch",
            "callable_factory": lambda dtype: nnx.Linear(
                128, 64, dtype=dtype, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 128)],
            "use_onnx_ir": True,
            "expected_output_shapes": [("B", 64)],
            "post_check_onnx_graph": EXPECT_GEMM_ONLY,
        },
        {
            "testcase": "linear_high_rank",
            "callable_factory": lambda dtype: nnx.Linear(
                128, 64, dtype=dtype, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 10, 128)],
            "run_only_dynamic": True,
            "use_onnx_ir": True,
            "expected_output_shapes": [("B", 10, 64)],
            "post_check_onnx_graph": EXPECT_DYNAMIC_RGR,
        },
        {
            "testcase": "linear_high_rank_static",
            "callable_factory": lambda dtype: nnx.Linear(
                128, 64, dtype=dtype, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(3, 10, 128)],
            "use_onnx_ir": True,
            "expected_output_shapes": [(3, 10, 64)],
            "post_check_onnx_graph": EXPECT_RGR,
        },
        {
            "testcase": "linear_no_bias",
            "callable_factory": lambda dtype: nnx.Linear(
                128, 64, use_bias=False, dtype=dtype, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 128)],
            "use_onnx_ir": True,
            "expected_output_shapes": [("B", 64)],
            "post_check_onnx_graph": EXPECT_GEMM_ONLY,
        },
        {
            "testcase": "linear_high_rank_no_bias",
            "callable_factory": lambda dtype: nnx.Linear(
                128, 64, use_bias=False, dtype=dtype, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 10, 128)],
            "use_onnx_ir": True,
            "run_only_dynamic": True,
            "expected_output_shapes": [("B", 10, 64)],
            "post_check_onnx_graph": EXPECT_DYNAMIC_RGR,
        },
        {
            "testcase": "linear_high_rank_no_bias",
            "callable_factory": lambda dtype: nnx.Linear(
                128, 64, use_bias=False, dtype=dtype, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 10, 128)],
            "use_onnx_ir": True,
            "expected_output_shapes": [(2, 10, 64)],
            "post_check_onnx_graph": EXPECT_RGR,
        },
        {
            "testcase": "linear_merge_symbolic_dim",
            "callable_factory": lambda dtype: nnx.Linear(
                128, 64, dtype=dtype, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 10, 128)],
            "run_only_dynamic": True,
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "expected_output_shapes": [("B", 10, 64)],
            "post_check_onnx_graph": EXPECT_DYNAMIC_RGR,
        },
    ],
)
class LinearPlugin(PrimitiveLeafPlugin):
    # Private primitive for this world (no import-time global assignment)
    _PRIM: ClassVar[Primitive] = Primitive("nnx.linear")
    _PRIM.multiple_results = False
    _ORIGINAL_LINEAR_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

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

        # Values
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        k_val = ctx.get_value_for_var(kernel_var, name_hint=ctx.fresh_name("kernel"))
        if use_bias:
            b_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("bias"))

        # Preserve original input meta shape on graph.input if the binder left it unknown
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if is_shape_all_unknown(getattr(x_val, "shape", None)):
            if any(d is not None for d in x_shape):
                _stamp_type_and_shape(x_val, x_shape)

        # Cast parameters AFTER shaping/promotion decisions (promote params -> input dtype)
        k_val = cast_param_like(ctx, k_val, x_val, "kernel_cast")
        if use_bias:
            b_val = cast_param_like(ctx, b_val, x_val, "bias_cast")

        k_shape = tuple(getattr(getattr(kernel_var, "aval", None), "shape", ()))
        in_features = int(k_shape[0])
        out_features = int(k_shape[1])

        # Flatten if rank > 2
        need_flatten = len(x_shape) > 2
        gemm_in = x_val
        if need_flatten:
            x2d_shape_c = ir.Value(
                name=ctx.fresh_name("x2d_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((2,)),
                const_value=ir.tensor(np.asarray([-1, in_features], dtype=np.int64)),
            )
            ctx._initializers.append(x2d_shape_c)
            x2d = ir.Value(
                name=ctx.fresh_name("input_reshape"),
                type=x_val.type,
                shape=ir.Shape((None, in_features)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Reshape",
                    domain="",
                    inputs=[x_val, x2d_shape_c],
                    outputs=[x2d],
                    name=ctx.fresh_name("Reshape"),
                )
            )
            gemm_in = x2d

        # Gemm
        if need_flatten:
            gemm_out = ir.Value(
                name=ctx.fresh_name("gemm_output"),
                type=x_val.type,
                shape=ir.Shape((None, out_features)),
            )
        else:
            gemm_out = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("out"))
            # Preserve symbolic batch labels on the direct Gemm output
            x_batch_idx = list(range(max(len(x_shape) - 1, 0)))
            y_meta = tuple(
                [_dim_label_from_value_or_aval(x_val, x_shape, i) for i in x_batch_idx]
                + [int(out_features)]
            )
            _stamp_type_and_shape(gemm_out, y_meta)

        inputs = [gemm_in, k_val] + ([b_val] if use_bias else [])
        ctx.add_node(
            ir.Node(
                op_type="Gemm",
                domain="",
                inputs=inputs,
                outputs=[gemm_out],
                name=ctx.fresh_name("Gemm"),
                attributes=[
                    ir.Attr("alpha", ir.AttributeType.FLOAT, 1.0),
                    # If there's no bias input, make beta=0.0 for strictness.
                    ir.Attr(
                        "beta", ir.AttributeType.FLOAT, 0.0 if not use_bias else 1.0
                    ),
                    ir.Attr("transA", ir.AttributeType.INT, 0),
                    ir.Attr("transB", ir.AttributeType.INT, 0),
                ],
            )
        )

        # Reshape back if needed: final_shape = x.shape[:-1] ++ [out_features]
        if need_flatten:
            x_batch_idx = list(range(max(len(x_shape) - 1, 0)))
            batch_dim_vals = [x_shape[i] for i in x_batch_idx]
            all_batch_static = all(_is_static_int(d) for d in batch_dim_vals)

            if all_batch_static:
                final_vals = [int(d) for d in batch_dim_vals] + [int(out_features)]
                final_shape_c = ir.Value(
                    name=ctx.fresh_name("final_shape_c"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((len(final_vals),)),
                    const_value=ir.tensor(np.array(final_vals, dtype=np.int64)),
                )
                ctx._initializers.append(final_shape_c)

                y_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("out"))
                y_meta = tuple(
                    [
                        _dim_label_from_value_or_aval(x_val, x_shape, i)
                        for i in x_batch_idx
                    ]
                    + [int(out_features)]
                )
                _stamp_type_and_shape(y_val, y_meta)
                ctx.add_node(
                    ir.Node(
                        op_type="Reshape",
                        domain="",
                        inputs=[gemm_out, final_shape_c],
                        outputs=[y_val],
                        name=ctx.fresh_name("Reshape"),
                    )
                )
                _stamp_type_and_shape(y_val, y_meta)
                _ensure_value_info(ctx, y_val)
            else:
                shp = ir.Value(
                    name=ctx.fresh_name("x_shape"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((len(x_shape),)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Shape",
                        domain="",
                        inputs=[x_val],
                        outputs=[shp],
                        name=ctx.fresh_name("Shape"),
                    )
                )
                starts = ir.Value(
                    name=ctx.fresh_name("slice_starts"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                    const_value=ir.tensor(np.array([0], dtype=np.int64)),
                )
                ends = ir.Value(
                    name=ctx.fresh_name("slice_ends"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                    const_value=ir.tensor(np.array([len(x_shape) - 1], dtype=np.int64)),
                )
                ctx._initializers.extend([starts, ends])
                axes_val = ir.Value(
                    name=ctx.fresh_name("slice_axes"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                    const_value=ir.tensor(np.array([0], dtype=np.int64)),
                )
                ctx._initializers.append(axes_val)
                # Missing output placeholder for Slice → define it before the node.
                batch_dims = ir.Value(
                    name=ctx.fresh_name("batch_dims"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((len(x_shape) - 1,)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Slice",
                        domain="",
                        inputs=[shp, starts, ends, axes_val],
                        outputs=[batch_dims],
                        name=ctx.fresh_name("Slice"),
                    )
                )
                of = ir.Value(
                    name=ctx.fresh_name("out_features_c"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                    const_value=ir.tensor(np.array([out_features], dtype=np.int64)),
                )
                ctx._initializers.append(of)
                final_shape = ir.Value(
                    name=ctx.fresh_name("final_shape"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((len(x_shape),)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Concat",
                        domain="",
                        inputs=[batch_dims, of],
                        outputs=[final_shape],
                        name=ctx.fresh_name("Concat"),
                        attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
                    )
                )
                y_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("out"))
                y_meta = tuple(
                    [
                        _dim_label_from_value_or_aval(x_val, x_shape, i)
                        for i in x_batch_idx
                    ]
                    + [int(out_features)]
                )
                _stamp_type_and_shape(y_val, y_meta)
                ctx.add_node(
                    ir.Node(
                        op_type="Reshape",
                        domain="",
                        inputs=[gemm_out, final_shape],
                        outputs=[y_val],
                        name=ctx.fresh_name("Reshape"),
                    )
                )
                _stamp_type_and_shape(y_val, y_meta)
                _ensure_value_info(ctx, y_val)

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

    @classmethod
    def binding_specs(cls):
        """Patch bindings while active."""
        return [
            AssignSpec("flax.nnx", "linear_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx.Linear",
                attr="__call__",
                make_value=lambda orig: cls.get_monkey_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@LinearPlugin._PRIM.def_impl
def _impl(x, kernel, bias, *, use_bias, dimension_numbers):
    y = jax.lax.dot_general(x, kernel, dimension_numbers=dimension_numbers)
    if use_bias and bias is not None:
        y = y + bias
    return y
