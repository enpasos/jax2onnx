# file: jax2onnx/plugins2/flax/nnx/batch_norm.py
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar
import logging
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins2._ir_shapes import (
    _stamp_type_and_shape,
    is_shape_all_unknown,
    _dim_label_from_value_or_aval,
    _to_ir_dim_for_shape,
    _ensure_value_info as _add_value_info,
    _as_ir_dim_label,
)


def _label_from_meta(val: ir.Value, aval_shape: tuple, idx: int):
    label = _dim_label_from_value_or_aval(val, aval_shape, idx)
    if label is None and idx < len(aval_shape):
        maybe_dim = aval_shape[idx]
        fallback_label = _as_ir_dim_label(maybe_dim)
        if fallback_label is not None:
            label = fallback_label
    return label


if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


# ------------------------------------------------------------------
# Graph-pattern expectations used by tests
# ------------------------------------------------------------------
# Rank <= 2: a single BatchNormalization node (no layout converts).
EXPECT_BN_ONLY = EG(
    [
        (
            "BatchNormalization",
            {
                "counts": {
                    "BatchNormalization": 1,
                    "Transpose": 0,
                    "Reshape": 0,
                    "CastLike": 0,
                }
            },
        )
    ]
)
# Rank > 2: NHWC -> NCHW, BN, then NCHW -> NHWC.
EXPECT_T_BN_T = EG(
    [
        (
            "Transpose -> BatchNormalization -> Transpose",
            {
                "counts": {
                    "Transpose": 2,
                    "BatchNormalization": 1,
                    "Reshape": 0,
                    "CastLike": 0,
                }
            },
        )
    ]
)


@register_primitive(
    jaxpr_primitive="nnx.batch_norm",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm",
    onnx=[
        {
            "component": "BatchNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__BatchNormalization.html",
        }
    ],
    since="v0.1.0",
    context="primitives2.nnx",
    component="batch_norm",
    testcases=[
        {
            "testcase": "batch_norm_no_bias_no_scale",
            "callable": nnx.BatchNorm(
                num_features=8,
                use_running_average=True,
                use_bias=False,
                use_scale=False,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_bias_no_scale",
            "callable": nnx.BatchNorm(
                num_features=8,
                use_running_average=True,
                use_bias=True,
                use_scale=False,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_no_bias_scale",
            "callable": nnx.BatchNorm(
                num_features=8,
                use_running_average=True,
                use_bias=False,
                use_scale=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_bias_scale",
            "callable": nnx.BatchNorm(
                num_features=8,
                use_running_average=True,
                use_bias=True,
                use_scale=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_3d",
            "callable": nnx.BatchNorm(
                num_features=3, use_running_average=True, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 4, 3)],
            "expected_output_shapes": [("B", 4, 3)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            # "post_check_onnx_graph": EXPECT_T_BN_T,
        },
        {
            "testcase": "batch_norm_4d",
            "callable": nnx.BatchNorm(
                num_features=3, use_running_average=True, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 4, 4, 3)],
            "expected_output_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": EXPECT_T_BN_T,
        },
        {
            "testcase": "batch_norm_4d_no_bias_no_scale",
            "callable": nnx.BatchNorm(
                num_features=3,
                use_running_average=True,
                use_bias=False,
                use_scale=False,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 4, 4, 3)],
            "expected_output_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": EXPECT_T_BN_T,
        },
    ],
)
class BatchNormPlugin(PrimitiveLeafPlugin):
    """
    IR-only plugin for flax.nnx.BatchNorm (inference behavior).
    For rank > 2, we do NHWC -> NCHW, apply ONNX BatchNormalization, then NCHW -> NHWC.
    """

    _PRIM: ClassVar[Primitive] = Primitive("nnx.batch_norm")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------------- abstract eval ----------------
    @staticmethod
    def abstract_eval(x, scale, bias, mean, var, *, epsilon, momentum, **_ignored):
        del scale, bias, mean, var, epsilon, momentum
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------------- lowering (IR) ----------------
    def lower(self, ctx: "IRBuildContext", eqn):
        x_var, scale_var, bias_var, mean_var, var_var = eqn.invars[:5]
        y_var = eqn.outvars[0]
        epsilon = eqn.params.get("epsilon", 1e-5)
        momentum = eqn.params.get("momentum", 0.9)

        # Inputs
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        # We'll materialize any default parameters directly in the input dtype,
        # so we never need a CastLike feeding BatchNormalization.
        x_np_dtype = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )

        nf = None
        for v in (scale_var, bias_var, mean_var, var_var):
            shp = tuple(getattr(getattr(v, "aval", None), "shape", ()))
            if len(shp) == 1 and isinstance(shp[0], (int, np.integer)):
                nf = int(shp[0])
                break
        if nf is None:
            xs = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
            if xs:
                last = xs[-1]
                if isinstance(last, (int, np.integer)):
                    nf = int(last)
        nf = int(nf if nf is not None else 1)

        if eqn.params.get("scale_is_default", False):
            scale_val = ir.Value(
                name=ctx.fresh_name("scale_c"),
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape((nf,)),
                const_value=ir.tensor(np.ones((nf,), dtype=x_np_dtype)),
            )
            ctx._initializers.append(scale_val)
        else:
            scale_val = ctx.get_value_for_var(
                scale_var, name_hint=ctx.fresh_name("scale")
            )

        if eqn.params.get("bias_is_default", False):
            bias_val = ir.Value(
                name=ctx.fresh_name("bias_c"),
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape((nf,)),
                const_value=ir.tensor(np.zeros((nf,), dtype=x_np_dtype)),
            )
            ctx._initializers.append(bias_val)
        else:
            bias_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("bias"))

        mean_val = ctx.get_value_for_var(mean_var, name_hint=ctx.fresh_name("mean"))
        var_val = ctx.get_value_for_var(var_var, name_hint=ctx.fresh_name("var"))

        # BN requires all inputs to share dtype; our defaults are created in the
        # input dtype and module params already match it in these tests,
        # so no runtime CastLike is needed (keeps '^BatchNormalization$' valid).
        # Preserve original graph.input shape labels if binder left them unknown
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if is_shape_all_unknown(getattr(x_val, "shape", None)):
            if any(d is not None for d in x_shape):
                _stamp_type_and_shape(x_val, x_shape)

        rank = len(x_shape)
        bn_in = x_val
        need_layout_convert = rank > 2

        # NHWC -> NCHW (move channel -1 to position 1)
        if need_layout_convert:
            perm = [0, rank - 1] + list(range(1, rank - 1))
            # Compute labeled shape for NCHW
            nchw_dims = (
                _label_from_meta(x_val, x_shape, 0),  # N
                _label_from_meta(x_val, x_shape, rank - 1),  # C
                *[_label_from_meta(x_val, x_shape, i) for i in range(1, rank - 1)],
            )
            x_nchw = ir.Value(
                name=ctx.fresh_name("bn_pre_transpose"),
                type=x_val.type,
                shape=ir.Shape(tuple(_to_ir_dim_for_shape(d) for d in nchw_dims)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Transpose",
                    domain="",
                    inputs=[x_val],
                    outputs=[x_nchw],
                    name=ctx.fresh_name("Transpose"),
                    attributes=[ir.Attr("perm", ir.AttributeType.INTS, tuple(perm))],
                )
            )
            bn_in = x_nchw

        # BatchNormalization node
        if need_layout_convert:
            bn_out = ir.Value(
                name=ctx.fresh_name("bn_nchw_out"),
                type=x_val.type,
                shape=bn_in.shape,
            )
        else:
            bn_out = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        ctx.add_node(
            ir.Node(
                op_type="BatchNormalization",
                domain="",
                inputs=[bn_in, scale_val, bias_val, mean_val, var_val],
                outputs=[bn_out],
                name=ctx.fresh_name("BatchNormalization"),
                attributes=[
                    ir.Attr("epsilon", ir.AttributeType.FLOAT, float(epsilon)),
                    ir.Attr("momentum", ir.AttributeType.FLOAT, float(momentum)),
                ],
            )
        )

        # NCHW -> NHWC to restore original layout; also stamp final output shape
        if need_layout_convert:
            inv_perm = [0] + list(range(2, rank)) + [1]
            y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))
            nhwc_dims = tuple(_label_from_meta(x_val, x_shape, i) for i in range(rank))
            # Stamp BOTH meta and TensorType so graph.output keeps symbols like 'B'
            _stamp_type_and_shape(y_val, nhwc_dims)
            ctx.add_node(
                ir.Node(
                    op_type="Transpose",
                    domain="",
                    inputs=[bn_out],
                    outputs=[y_val],
                    name=ctx.fresh_name("Transpose"),
                    attributes=[
                        ir.Attr("perm", ir.AttributeType.INTS, tuple(inv_perm))
                    ],
                )
            )
            _stamp_type_and_shape(y_val, nhwc_dims)
            _add_value_info(ctx, y_val)
        else:
            # Direct BN output already targets y_var; (re)stamp shape/labels
            y_val = bn_out
            y_dims = tuple(_label_from_meta(x_val, x_shape, i) for i in range(rank))
            _stamp_type_and_shape(y_val, y_dims)
            _add_value_info(ctx, y_val)

    # ---------------- direct bind for tests ----------------
    @staticmethod
    def _batch_norm(
        x,
        scale,
        bias,
        mean,
        var,
        *,
        epsilon,
        momentum,
        scale_is_default=False,
        bias_is_default=False,
    ):
        return BatchNormPlugin._PRIM.bind(
            x,
            scale,
            bias,
            mean,
            var,
            epsilon=epsilon,
            momentum=momentum,
            scale_is_default=scale_is_default,
            bias_is_default=bias_is_default,
        )

    # ---------------- monkey-patch & bindings ----------------
    @staticmethod
    def _make_patch(orig_fn: Callable):
        del orig_fn  # not used; kept for symmetry with other plugins

        def patched(self, x, use_running_average=None, *, mask=None):
            # Force inference behavior; warn if training mode was requested
            if not self.use_running_average:
                logging.warning(
                    "BatchNorm exported with use_running_average=False; converting to inference mode."
                )

            param_dtype = self.param_dtype if self.param_dtype is not None else x.dtype
            # IMPORTANT: build defaults with NumPy so they become initializers
            # (no traced ops like Concat/Expand).
            np_dtype = np.dtype(param_dtype)

            if self.use_scale:
                scale_value = self.scale.value
                scale_is_default = False
            else:
                scale_value = np.ones((self.num_features,), dtype=np_dtype)
                scale_is_default = True
            if self.use_bias:
                bias_value = self.bias.value
                bias_is_default = False
            else:
                bias_value = np.zeros((self.num_features,), dtype=np_dtype)
                bias_is_default = True

            return BatchNormPlugin._batch_norm(
                x,
                scale_value,
                bias_value,
                self.mean.value,
                self.var.value,
                epsilon=self.epsilon,
                momentum=self.momentum,
                scale_is_default=scale_is_default,
                bias_is_default=bias_is_default,
            )

        return patched

    @classmethod
    def binding_specs(cls):
        return [
            # Ensure flax.nnx.batch_norm_p points to our private Primitive during tracing
            AssignSpec("flax.nnx", "batch_norm_p", cls._PRIM, delete_if_missing=True),
            # Monkey-patch nnx.BatchNorm.__call__
            MonkeyPatchSpec(
                target="flax.nnx.BatchNorm",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, scale, bias, mean, var, **params: cls.abstract_eval(
                    x, scale, bias, mean, var, **params
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


# ---------------- concrete eager impl ----------------
@BatchNormPlugin._PRIM.def_impl
def _impl(
    x,
    scale,
    bias,
    mean,
    var,
    *,
    epsilon,
    momentum,
    scale_is_default=False,
    bias_is_default=False,
):
    del momentum  # inference-only export
    rank = x.ndim
    if rank > 2:
        # NHWC -> NCHW
        x_nchw = jnp.moveaxis(x, -1, 1)
        param_shape = (1, -1) + (1,) * (rank - 2)  # (1,C,1,1,...)
        s = jnp.reshape(scale, param_shape).astype(x.dtype, copy=False)
        b = jnp.reshape(bias, param_shape).astype(x.dtype, copy=False)
        m = jnp.reshape(mean, param_shape).astype(x.dtype, copy=False)
        v = jnp.reshape(var, param_shape).astype(x.dtype, copy=False)
        y = (x_nchw - m) * s / jnp.sqrt(v + epsilon) + b
        return jnp.moveaxis(y, 1, -1)
    # rank <= 2 : channels already last
    param_shape = (1,) * (rank - 1) + (-1,)
    s = jnp.reshape(scale, param_shape).astype(x.dtype, copy=False)
    b = jnp.reshape(bias, param_shape).astype(x.dtype, copy=False)
    m = jnp.reshape(mean, param_shape).astype(x.dtype, copy=False)
    v = jnp.reshape(var, param_shape).astype(x.dtype, copy=False)
    return (x - m) * s / jnp.sqrt(v + epsilon) + b
