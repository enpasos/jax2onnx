# jax2onnx/plugins/flax/nnx/batch_norm.py

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar, Final
import logging
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    is_shape_all_unknown,
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
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
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


def _require_builder(ctx):
    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError("IR build context missing builder")
    return builder


def _const_from_array(ctx, name_hint: str, arr: np.ndarray) -> ir.Value:
    builder = getattr(ctx, "builder", None)
    base_name = ctx.fresh_name(name_hint) if hasattr(ctx, "fresh_name") else name_hint
    np_arr = np.asarray(arr)

    builder_mode = (
        bool(getattr(builder, "_function_mode", False))
        if builder is not None
        else False
    )
    inside_function = bool(
        getattr(ctx, "_inside_function_scope", False)
        or getattr(ctx, "_function_mode", False)
    )

    if builder is not None and not inside_function and not builder_mode:
        add_initializer = getattr(builder, "add_initializer_from_array", None)
        if callable(add_initializer):
            return add_initializer(base_name, np_arr)

    value = ir.Value(
        name=base_name,
        type=ir.TensorType(_ir_dtype_from_numpy(np_arr.dtype)),
        shape=ir.Shape(tuple(int(d) for d in np_arr.shape)),
        const_value=ir.tensor(np_arr),
    )

    handler = getattr(ctx, "_handle_initializer_append", None)
    if callable(handler):
        handler(value)
        return value

    init_list = getattr(ctx, "_initializers", None)
    if init_list is not None and hasattr(init_list, "append"):
        init_list.append(value)
        return value

    if builder is not None:
        builder_inits = getattr(builder, "initializers", None)
        if isinstance(builder_inits, list):
            builder_inits.append(value)

    return value


# ------------------------------------------------------------------
# Graph-pattern expectations used by tests
# ------------------------------------------------------------------
# Rank <= 2: a single BatchNormalization node (no layout converts).
EXPECT_BN_ONLY: Final = EG(
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
EXPECT_T_BN_T: Final = EG(
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
    context="primitives.nnx",
    component="batch_norm",
    testcases=[
        {
            "testcase": "batch_norm_no_bias_no_scale",
            "callable": construct_and_call(
                nnx.BatchNorm,
                num_features=8,
                use_running_average=True,
                use_bias=False,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_bias_no_scale",
            "callable": construct_and_call(
                nnx.BatchNorm,
                num_features=8,
                use_running_average=True,
                use_bias=True,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_no_bias_scale",
            "callable": construct_and_call(
                nnx.BatchNorm,
                num_features=8,
                use_running_average=True,
                use_bias=False,
                use_scale=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_bias_scale",
            "callable": construct_and_call(
                nnx.BatchNorm,
                num_features=8,
                use_running_average=True,
                use_bias=True,
                use_scale=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8)],
            "expected_output_shapes": [("B", 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_BN_ONLY,
        },
        {
            "testcase": "batch_norm_3d",
            "callable": construct_and_call(
                nnx.BatchNorm,
                num_features=3,
                use_running_average=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 3)],
            "expected_output_shapes": [("B", 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_BN_T,
        },
        {
            "testcase": "batch_norm_4d",
            "callable": construct_and_call(
                nnx.BatchNorm,
                num_features=3,
                use_running_average=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 4, 3)],
            "expected_output_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_T_BN_T,
        },
        {
            "testcase": "batch_norm_4d_no_bias_no_scale",
            "callable": construct_and_call(
                nnx.BatchNorm,
                num_features=3,
                use_running_average=True,
                use_bias=False,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 4, 3)],
            "expected_output_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
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
        builder = _require_builder(ctx)
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
            scale_val = _const_from_array(
                ctx,
                "scale_c",
                np.ones((nf,), dtype=x_np_dtype),
            )
        else:
            scale_val = ctx.get_value_for_var(
                scale_var, name_hint=ctx.fresh_name("scale")
            )

        if eqn.params.get("bias_is_default", False):
            bias_val = _const_from_array(
                ctx,
                "bias_c",
                np.zeros((nf,), dtype=x_np_dtype),
            )
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

        x_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        bn_dtype = x_dtype if x_dtype is not None else _ir_dtype_from_numpy(x_np_dtype)

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
            x_nchw = builder.Transpose(
                x_val,
                perm=tuple(int(p) for p in perm),
                _outputs=[ctx.fresh_name("bn_pre_transpose")],
            )
            x_nchw.type = ir.TensorType(bn_dtype)
            _stamp_type_and_shape(x_nchw, nchw_dims)
            _ensure_value_metadata(ctx, x_nchw)
            bn_in = x_nchw

        # BatchNormalization node
        bn_out = builder.BatchNormalization(
            bn_in,
            scale_val,
            bias_val,
            mean_val,
            var_val,
            epsilon=float(epsilon),
            momentum=float(momentum),
            _outputs=[ctx.fresh_name("BatchNormalization")],
        )
        bn_out.type = ir.TensorType(bn_dtype)

        # NCHW -> NHWC to restore original layout; also stamp final output shape
        if need_layout_convert:
            inv_perm = [0] + list(range(2, rank)) + [1]
            y_val = builder.Transpose(
                bn_out,
                perm=tuple(int(p) for p in inv_perm),
                _outputs=[ctx.fresh_name("BatchNormOut")],
            )
            nhwc_dims = tuple(_label_from_meta(x_val, x_shape, i) for i in range(rank))
            # Stamp BOTH meta and TensorType so graph.output keeps symbols like 'B'
            _stamp_type_and_shape(y_val, nhwc_dims)
            y_val.type = ir.TensorType(bn_dtype)
            _ensure_value_metadata(ctx, y_val)
            final_value = y_val
        else:
            # Direct BN output already targets y_var; (re)stamp shape/labels
            final_value = bn_out
            y_dims = tuple(_label_from_meta(x_val, x_shape, i) for i in range(rank))
            _stamp_type_and_shape(final_value, y_dims)
            final_value.type = ir.TensorType(bn_dtype)
            _ensure_value_metadata(ctx, final_value)

        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(y_var, final_value)

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


def _ir_dtype_from_numpy(dt: np.dtype) -> ir.DataType:
    dt = np.dtype(dt)
    if dt == np.dtype("float32"):
        return ir.DataType.FLOAT
    if dt == np.dtype("float64"):
        return ir.DataType.DOUBLE
    if dt == np.dtype("int64"):
        return ir.DataType.INT64
    if dt == np.dtype("int32"):
        return ir.DataType.INT32
    if dt == np.dtype("bool"):
        return ir.DataType.BOOL
    return ir.DataType.FLOAT
