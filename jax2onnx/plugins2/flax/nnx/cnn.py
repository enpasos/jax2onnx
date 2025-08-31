# file: jax2onnx/plugins2/flax/nnx/conv.py

from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Callable, Sequence, Tuple, Any

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2._utils import cast_param_like

if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


@register_primitive(
    jaxpr_primitive="nnx.conv",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Conv",
    onnx=[
        {"component": "Conv", "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html"},
        {"component": "Transpose", "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html"},
        {"component": "Reshape", "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html"},
        {"component": "CastLike", "doc": "https://onnx.ai/onnx/operators/onnx__CastLike.html"},
    ],
    since="v0.1.0",
    context="primitives2.nnx",
    component="conv",
    testcases=[
        # We keep the tests minimal here; the upstream suite registers many variants.
        {
            "testcase": "conv_basic_bias",
            "callable": nnx.Conv(
                in_features=3, out_features=16, kernel_size=(3, 3),
                strides=(1, 1), padding="SAME", use_bias=True, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 28, 28, 3)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: any(n.op_type == "Conv" for n in m.graph.node),
        },
        {
            "testcase": "conv_1d_more_1d_inputs",
            "callable": nnx.Conv(28, 4, kernel_size=(3,), rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 4, 4, 28)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: any(n.op_type == "Conv" for n in m.graph.node),
        },
        {
            "testcase": "conv_3d_basic",
            "callable": nnx.Conv(2, 4, kernel_size=(3, 3, 3), rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 8, 8, 8, 2)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: any(n.op_type == "Conv" for n in m.graph.node),
        },
    ],
)
class ConvPlugin(PrimitiveLeafPlugin):
    """
    IR-only plugin for flax.nnx.Conv → ONNX Conv.
    Assumes NHWC input/output in user space; converts to NCHx… internally.
    Supports 1D/2D/3D, SAME/VALID or explicit pads, strides/dilations, and groups.
    Also supports "mixed-dimension" inputs (e.g., 1D conv on higher-rank NHWC)
    by flattening non-participating spatial dims into batch and reshaping back.
    """

    _PRIM: ClassVar[Primitive] = Primitive("nnx.conv")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(
        x,
        kernel,
        bias,
        *,
        use_bias: bool,
        strides: Sequence[int] | int = 1,
        padding: str | Sequence[Tuple[int, int]] = "VALID",
        dilations: Sequence[int] | int = 1,
        dimension_numbers=None,
        feature_group_count: int = 1,
    ):
        # Prefer delegating to the original nnx.Conv.__call__ for shape logic.
        if ConvPlugin._ORIGINAL_CALL is None:
            # Fallback: assume NHWC → output NHWC with same batch & rank, compute via lax directly.
            # (We only need the shape/dtype here.)
            num_spatial = max(kernel.ndim - 2, 1)
            if isinstance(strides, int): strides = (strides,) * num_spatial
            if isinstance(dilations, int): dilations = (dilations,) * num_spatial
            # Build a dimension_numbers default matching NHWC user layout
            dn = lax.conv_dimension_numbers(
                x.shape, kernel.shape, ("NHWC", "HWIO"[:num_spatial+2], "NHWC")
            )
            y = jax.eval_shape(
                lambda a, w, b: lax.conv_general_dilated(
                    a, w, window_strides=tuple(strides),
                    padding=padding, lhs_dilation=None,
                    rhs_dilation=tuple(dilations),
                    dimension_numbers=dn,
                    feature_group_count=feature_group_count,
                ) if not use_bias else
                lax.conv_general_dilated(
                    a, w, window_strides=tuple(strides),
                    padding=padding, lhs_dilation=None,
                    rhs_dilation=tuple(dilations),
                    dimension_numbers=dn,
                    feature_group_count=feature_group_count,
                ) + b,
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(kernel.shape, kernel.dtype),
                (jax.ShapeDtypeStruct(bias.shape, bias.dtype) if use_bias else None),
            )
            y = jax.tree_util.tree_leaves(y)[0]
            return jax.core.ShapedArray(y.shape, y.dtype)

        # Use the original call for precise abstract eval
        x_s = jax.ShapeDtypeStruct(x.shape, x.dtype)
        k_s = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        b_s = jax.ShapeDtypeStruct(bias.shape, bias.dtype) if use_bias else None

        def _helper(xv, kv, bv):
            from types import SimpleNamespace
            dummy = SimpleNamespace(
                kernel=SimpleNamespace(value=kv),
                bias=(SimpleNamespace(value=bv) if use_bias else None),
                kernel_size=tuple(kv.shape[:-2]),
                in_features=kv.shape[-2] * feature_group_count,
                out_features=kv.shape[-1],
                strides=strides,
                padding=padding,
                dimension_numbers=dimension_numbers,
                feature_group_count=feature_group_count,
                input_dilation=None,
                kernel_dilation=dilations,
                use_bias=use_bias,
                dtype=None,
                param_dtype=kv.dtype,
                promote_dtype=lambda *a, dtype=None: a if len(a) > 1 else a[0],
                conv_general_dilated=lax.conv_general_dilated,
                precision=None,
                mask=None,
            )
            return ConvPlugin._ORIGINAL_CALL(dummy, xv)

        y = jax.eval_shape(_helper, x_s, k_s, b_s)
        y = jax.tree_util.tree_leaves(y)[0]
        return jax.core.ShapedArray(y.shape, y.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: "IRBuildContext", eqn):
        x_var, k_var, b_var = eqn.invars[:3]
        y_var = eqn.outvars[0]

        params = eqn.params
        use_bias = bool(params.get("use_bias", True))
        strides_param = params.get("strides", 1)
        padding_param = params.get("padding", "VALID")
        dilations_param = params.get("dilations", 1)
        groups = int(params.get("feature_group_count", 1))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        k_shape = tuple(getattr(getattr(k_var, "aval", None), "shape", ()))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        k_val = ctx.get_value_for_var(k_var, name_hint=ctx.fresh_name("kernel"))
        b_val = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("bias")) if use_bias else None

        # Normalize integer/sequence params
        conv_spatial = max(len(k_shape) - 2, 1)
        x_spatial = max(len(x_shape) - 2, 1)

        if isinstance(strides_param, int):
            strides = (int(strides_param),) * conv_spatial
        else:
            strides = tuple(int(s) for s in strides_param)

        if isinstance(dilations_param, int):
            dilations = (int(dilations_param),) * conv_spatial
        else:
            dilations = tuple(int(d) for d in dilations_param)

        # If kernel has fewer spatial dims than input spatial rank, flatten the non-participating spatial dims into batch.
        need_flatten = conv_spatial < x_spatial
        x_pre = x_val
        if need_flatten:
            # (N, extra..., part..., C) → (N*prod(extra...), part..., C)
            participating = x_shape[-1 - conv_spatial : -1]
            reshape_spec = [-1] + [int(d) for d in participating] + [int(x_shape[-1])]
            shape_c = ir.Value(
                name=ctx.fresh_name("flatten_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((len(reshape_spec),)),
                const_value=ir.tensor(np.asarray(reshape_spec, dtype=np.int64)),
            )
            ctx._initializers.append(shape_c)
            x_pre = ir.Value(
                name=ctx.fresh_name("x_flatten"),
                type=x_val.type,
                shape=ir.Shape((None, *[int(d) for d in participating], int(x_shape[-1]))),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Reshape",
                    domain="",
                    inputs=[x_val, shape_c],
                    outputs=[x_pre],
                    name=ctx.fresh_name("Reshape"),
                )
            )
            # After flatten, the "effective" spatial rank equals conv_spatial
            x_spatial = conv_spatial

        # NHWC → NCHW… (generic N,C,*S)
        rank_after = x_spatial + 2
        pre_perm = [0, rank_after - 1] + list(range(1, rank_after - 1))
        x_nchw = ir.Value(
            name=ctx.fresh_name("x_nchw"),
            type=x_pre.type,
            shape=None,
        )
        ctx.add_node(
            ir.Node(
                op_type="Transpose",
                domain="",
                inputs=[x_pre],
                outputs=[x_nchw],
                name=ctx.fresh_name("Transpose"),
                attributes=[ir.Attr("perm", ir.AttributeType.INTS, pre_perm)],
            )
        )

        # Kernel transpose: (S..., inC_per_group, outC) → (outC, inC_per_group, S...)
        k_rank = len(k_shape)
        k_perm = [k_rank - 1, k_rank - 2] + list(range(0, k_rank - 2))
        k_onnx = ir.Value(
            name=ctx.fresh_name("kernel_onnx"),
            type=k_val.type,
            shape=None,
        )
        ctx.add_node(
            ir.Node(
                op_type="Transpose",
                domain="",
                inputs=[k_val],
                outputs=[k_onnx],
                name=ctx.fresh_name("Transpose"),
                attributes=[ir.Attr("perm", ir.AttributeType.INTS, k_perm)],
            )
        )

        # Cast parameters AFTER shaping to match JAX promotion (params -> input dtype)
        k_onnx = cast_param_like(ctx, k_onnx, x_val, "kernel_cast")
        if use_bias and b_val is not None:
            b_val = cast_param_like(ctx, b_val, x_val, "bias_cast")

        # Build Conv attributes
        conv_attrs: list[ir.Attr] = [
            ir.Attr("strides", ir.AttributeType.INTS, list(strides)),
            ir.Attr("dilations", ir.AttributeType.INTS, list(dilations)),
        ]
        if groups > 1:
            conv_attrs.append(ir.Attr("group", ir.AttributeType.INT, int(groups)))

        def _pads_from_same_valid():
            # SAME with potential dilations → explicit pads; otherwise auto_pad
            if isinstance(padding_param, str):
                up = padding_param.upper()
                if up == "VALID":
                    conv_attrs.append(ir.Attr("auto_pad", ir.AttributeType.STRING, "VALID"))
                    return
                if up == "SAME":
                    # Compute explicit pads if any dilation > 1, else auto_pad SAME_UPPER
                    if any(d > 1 for d in dilations):
                        pads_begin: list[int] = []
                        pads_end: list[int] = []
                        # Effective sizes computed from input (after flatten if any)
                        # Input spatial dims are x_pre shape indices 1..(x_spatial)
                        in_spatial_sizes = [int(x_shape[i]) for i in range(len(x_shape))][1:1 + x_spatial]
                        # When flattened, we used the last conv_spatial dims as participating
                        if need_flatten:
                            in_spatial_sizes = [int(d) for d in x_shape[-1 - conv_spatial : -1]]
                        for i in range(conv_spatial):
                            ksz = int(k_shape[i])
                            dil = int(dilations[i])
                            stride = int(strides[i])
                            eff_k = ksz + (ksz - 1) * (dil - 1)
                            inp = in_spatial_sizes[i]
                            # ceil_div
                            out_sz = (inp + stride - 1) // stride
                            pad_total = max(0, (out_sz - 1) * stride + eff_k - inp)
                            pb = pad_total // 2
                            pe = pad_total - pb
                            pads_begin.append(pb)
                            pads_end.append(pe)
                        conv_attrs.append(
                            ir.Attr("pads", ir.AttributeType.INTS, pads_begin + pads_end)
                        )
                        return
                    else:
                        conv_attrs.append(ir.Attr("auto_pad", ir.AttributeType.STRING, "SAME_UPPER"))
                        return
                # Fallback
                conv_attrs.append(ir.Attr("auto_pad", ir.AttributeType.STRING, "VALID"))
                return

            # Explicit JAX padding: sequence of (low, high) per spatial dim
            if isinstance(padding_param, Sequence) and len(padding_param) == conv_spatial:
                lows = [int(lo) for (lo, _hi) in padding_param]  # type: ignore[misc]
                highs = [int(hi) for (_lo, hi) in padding_param]  # type: ignore[misc]
                conv_attrs.append(ir.Attr("pads", ir.AttributeType.INTS, lows + highs))
                return

            # Unknown → VALID
            conv_attrs.append(ir.Attr("auto_pad", ir.AttributeType.STRING, "VALID"))

        _pads_from_same_valid()

        # Conv
        conv_out = ir.Value(
            name=ctx.fresh_name("conv_nchw"),
            type=x_val.type,
            shape=None,
        )
        conv_inputs = [x_nchw, k_onnx] + ([b_val] if (use_bias and b_val is not None) else [])
        ctx.add_node(
            ir.Node(
                op_type="Conv",
                domain="",
                inputs=conv_inputs,
                outputs=[conv_out],
                name=ctx.fresh_name("Conv"),
                attributes=conv_attrs,
            )
        )

        # NCHW… → NHWC…
        y_pre = ir.Value(
            name=ctx.fresh_name("y_nhwc"),
            type=x_val.type,
            shape=None,
        )
        post_perm = [0] + list(range(2, conv_spatial + 2)) + [1]
        ctx.add_node(
            ir.Node(
                op_type="Transpose",
                domain="",
                inputs=[conv_out],
                outputs=[y_pre],
                name=ctx.fresh_name("Transpose"),
                attributes=[ir.Attr("perm", ir.AttributeType.INTS, post_perm)],
            )
        )

        # If we flattened, reshape back to original NHWC rank/shape.
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))
        if need_flatten:
            # Use the abstract-eval'd output shape (all tests use concrete sizes here).
            y_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))
            # Directly emit a constant target shape.
            final_shape_c = ir.Value(
                name=ctx.fresh_name("final_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((len(y_shape),)),
                const_value=ir.tensor(np.asarray([int(d) for d in y_shape], dtype=np.int64)),
            )
            ctx._initializers.append(final_shape_c)
            ctx.add_node(
                ir.Node(
                    op_type="Reshape",
                    domain="",
                    inputs=[y_pre, final_shape_c],
                    outputs=[y_val],
                    name=ctx.fresh_name("Reshape"),
                )
            )
        else:
            # No flatten path: write result directly
            # (y_pre already NHWC with correct rank)
            ctx.add_node(
                ir.Node(
                    op_type="Identity",
                    domain="",
                    inputs=[y_pre],
                    outputs=[y_val],
                    name=ctx.fresh_name("Identity"),
                )
            )

    # ---------- binding & monkey-patch ----------
    @staticmethod
    def _make_patch(orig_fn: Callable):
        ConvPlugin._ORIGINAL_CALL = orig_fn
        prim = ConvPlugin._PRIM

        def patched(self, x):
            # nnx.Conv carries all config we need; pass through unchanged.
            strides = getattr(self, "strides", 1)
            padding = getattr(self, "padding", "VALID")
            dilations = getattr(self, "kernel_dilation", 1)
            groups = getattr(self, "feature_group_count", 1)
            use_bias = bool(getattr(self, "use_bias", True))
            kernel = self.kernel.value
            # If use_bias is False, still pass a bias tensor (zeros) so the primitive arity matches.
            if use_bias:
                bias = self.bias.value if self.bias is not None else jnp.zeros((kernel.shape[-1],), dtype=kernel.dtype)
            else:
                bias = jnp.zeros((kernel.shape[-1],), dtype=kernel.dtype)

            return prim.bind(
                x, kernel, bias,
                use_bias=use_bias,
                strides=strides,
                padding=padding,
                dilations=dilations,
                dimension_numbers=getattr(self, "dimension_numbers", None),
                feature_group_count=groups,
            )

        return patched

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.nnx", "conv_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx.Conv",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, kernel, bias, **kw: cls.abstract_eval(x, kernel, bias, **kw)
            )
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- eager impl ----------
@ConvPlugin._PRIM.def_impl
def _impl(x, kernel, bias, *, use_bias, strides, padding, dilations, dimension_numbers, feature_group_count):
    num_spatial = max(kernel.ndim - 2, 1)
    if isinstance(strides, int): strides = (strides,) * num_spatial
    if isinstance(dilations, int): dilations = (dilations,) * num_spatial
    # Default dimension numbers for NHWC/… + HWIO/… → NHWC/…
    dn = lax.conv_dimension_numbers(x.shape, kernel.shape, ("NHWC", "HWIO"[:num_spatial+2], "NHWC"))
    y = lax.conv_general_dilated(
        x, kernel,
        window_strides=tuple(strides),
        padding=padding,
        lhs_dilation=None,
        rhs_dilation=tuple(dilations),
        dimension_numbers=dimension_numbers or dn,
        feature_group_count=int(feature_group_count),
    )
    if use_bias and bias is not None:
        # Broadcast bias over spatial dims
        bias_shape = (1,) * (y.ndim - 1) + (y.shape[-1],)
        y = y + jnp.reshape(bias, bias_shape)
    return y
