# jax2onnx/plugins/flax/nnx/rms_norm.py

from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar, Final, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.extend.core import Primitive

import onnx_ir as ir
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore

RMS_NORM_PRIM: Final[Primitive] = Primitive("nnx.rms_norm")
RMS_NORM_PRIM.multiple_results = False


EXPECT_RMS_NORM_GRAPH: Final = EG(
    [
        (
            "RMSNormalization",
            {
                "counts": {"RMSNormalization": 1},
            },
        ),
        (
            "Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul",
            {
                "counts": {
                    "Pow": 1,
                    "ReduceMean": 1,
                    "Add": 1,
                    "Sqrt": 1,
                    "Div": 1,
                    "Mul": 1,
                }
            },
        ),
    ],
    mode="any",
)


def _require_builder(ctx: Any):
    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError("IR build context missing builder")
    return builder


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


def _const_from_array(ctx: Any, name_hint: str, arr: np.ndarray) -> ir.Value:
    builder = getattr(ctx, "builder", None)
    base_name = ctx.fresh_name(name_hint) if hasattr(ctx, "fresh_name") else name_hint
    np_arr = np.asarray(arr)

    inside_function = bool(
        getattr(ctx, "_inside_function_scope", False)
        or getattr(ctx, "_function_mode", False)
    )
    builder_mode = (
        bool(getattr(builder, "_function_mode", False))
        if builder is not None
        else False
    )

    if builder is not None and not inside_function and not builder_mode:
        add_init = getattr(builder, "add_initializer_from_array", None)
        if callable(add_init):
            return add_init(base_name, np_arr)

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


@register_primitive(
    jaxpr_primitive=RMS_NORM_PRIM.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.RMSNorm",
    onnx=[
        {
            "component": "RMSNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__RMSNormalization.html",
        }
    ],
    since="v0.2.0",
    context="primitives.nnx",
    component="rms_norm",
    testcases=[
        {
            "testcase": "rms_norm_basic",
            "callable": construct_and_call(
                nnx.RMSNorm,
                num_features=6,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 6)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_RMS_NORM_GRAPH,
        },
        {
            "testcase": "rms_norm_use_scale_false",
            "callable": construct_and_call(
                nnx.RMSNorm,
                num_features=6,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 6)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_RMS_NORM_GRAPH,
        },
        {
            "testcase": "rms_norm_4d_dynamic",
            "callable": construct_and_call(
                nnx.RMSNorm,
                num_features=3,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_RMS_NORM_GRAPH,
        },
        {
            "testcase": "rms_norm_4d_dynamic_no_scale",
            "callable": construct_and_call(
                nnx.RMSNorm,
                num_features=3,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_RMS_NORM_GRAPH,
        },
    ],
)
class RMSNormPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.nnx.RMSNorm → ONNX RMSNormalization."""

    _PRIM: ClassVar[Primitive] = RMS_NORM_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------------- abstract eval ----------------
    @staticmethod
    def abstract_eval(x, scale, *, epsilon, axis):
        del scale, epsilon, axis
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------------- lowering ----------------
    def lower(self, ctx: "IRBuildContext", eqn):
        x_var, scale_var = eqn.invars[:2]
        y_var = eqn.outvars[0]

        params = dict(getattr(eqn, "params", {}) or {})
        epsilon = float(params.get("epsilon", 1e-5))
        axis = int(params.get("axis", -1))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        scale_val = ctx.get_value_for_var(scale_var, name_hint=ctx.fresh_name("scale"))

        scale_val = cast_param_like(ctx, scale_val, x_val, name_hint="rms_scale_cast")

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        if rank == 0:
            raise ValueError("RMSNorm requires tensor inputs")
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError("axis out of range for RMSNorm")

        builder = _require_builder(ctx)
        opset = None
        if builder is not None:
            opset = getattr(builder, "opset", None)
            if opset is None:
                opset = getattr(builder, "opset_version", None)
            if opset is None:
                imports = getattr(builder, "opset_imports", {})
                if isinstance(imports, dict):
                    opset = imports.get("", None)
        opset = int(opset) if opset is not None else 0

        dims = tuple(
            _dim_label_from_value_or_aval(x_val, x_shape, i) for i in range(rank)
        )

        x_np_dtype = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )
        x_ir_dtype = getattr(getattr(x_val, "type", None), "dtype", ir.DataType.FLOAT)

        if opset >= 23 and hasattr(builder, "RMSNormalization"):
            y_val = builder.RMSNormalization(
                x_val,
                scale_val,
                axis=int(axis),
                epsilon=float(epsilon),
                _outputs=[ctx.fresh_name("RMSNorm")],
            )
            x_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
            if x_dtype is not None:
                y_val.type = ir.TensorType(x_dtype)
            _stamp_type_and_shape(y_val, dims)
            _ensure_value_metadata(ctx, y_val)
            bind_value = getattr(ctx, "bind_value_for_var", None)
            if not callable(bind_value):
                raise AttributeError("IR build context missing bind_value_for_var")
            bind_value(y_var, y_val)
            return

        two_const = _const_from_array(ctx, "two", np.asarray(2.0, dtype=x_np_dtype))
        eps_const = _const_from_array(ctx, "eps", np.asarray(epsilon, dtype=x_np_dtype))
        axes_const = _const_from_array(
            ctx, "axes", np.asarray([int(axis)], dtype=np.int64)
        )

        pow_out = builder.Pow(
            x_val,
            two_const,
            _outputs=[ctx.fresh_name("rms_pow")],
        )
        pow_out.type = ir.TensorType(x_ir_dtype)
        _stamp_type_and_shape(pow_out, dims)

        mean_shape = list(x_shape)
        if axis < len(mean_shape):
            mean_shape[axis] = 1
        mean_dims = tuple(mean_shape)

        mean_out = builder.ReduceMean(
            pow_out,
            axes_const,
            keepdims=1,
            _outputs=[ctx.fresh_name("rms_mean")],
        )
        mean_out.type = ir.TensorType(x_ir_dtype)
        _stamp_type_and_shape(mean_out, mean_dims)

        add_out = builder.Add(
            mean_out,
            eps_const,
            _outputs=[ctx.fresh_name("rms_add")],
        )
        add_out.type = ir.TensorType(x_ir_dtype)
        _stamp_type_and_shape(add_out, mean_dims)

        sqrt_out = builder.Sqrt(
            add_out,
            _outputs=[ctx.fresh_name("rms_sqrt")],
        )
        sqrt_out.type = ir.TensorType(x_ir_dtype)
        _stamp_type_and_shape(sqrt_out, mean_dims)

        div_out = builder.Div(
            x_val,
            sqrt_out,
            _outputs=[ctx.fresh_name("rms_div")],
        )
        div_out.type = ir.TensorType(x_ir_dtype)
        _stamp_type_and_shape(div_out, dims)

        y_val = builder.Mul(
            div_out,
            scale_val,
            _outputs=[ctx.fresh_name("rms_out")],
        )
        y_val.type = ir.TensorType(x_ir_dtype)
        _stamp_type_and_shape(y_val, dims)
        _ensure_value_metadata(ctx, y_val)

        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(y_var, y_val)

    # ---------------- monkey patch & binding ----------------
    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.nnx", "rms_norm_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(nnx.RMSNorm, "__call__", cls._patch_call),
        ]

    @staticmethod
    def _prepare_scale(scale_obj, size: int, dtype):
        if scale_obj is None:
            return jnp.ones((size,), dtype=dtype)
        arr = jnp.asarray(scale_obj, dtype=dtype)
        if arr.size != size:
            return jnp.ones((size,), dtype=dtype)
        return jnp.reshape(arr, (size,))

    @classmethod
    def _patch_call(cls, orig):
        def wrapped(self: nnx.RMSNorm, x, mask=None):
            if mask is not None:
                return orig(self, x, mask=mask)

            param_dtype = getattr(self, "param_dtype", None) or x.dtype
            if x.dtype != param_dtype:
                x = x.astype(param_dtype)

            axis = getattr(self, "feature_axes", -1)
            if isinstance(axis, Sequence):
                axis = axis[0]
            axis = int(axis)

            feat_dim = x.shape[axis]
            if feat_dim is None:
                raise ValueError("RMSNorm requires a known feature dimension")

            scale_val = None
            if getattr(self, "use_scale", True):
                scale_field = getattr(self, "scale", None)
                if scale_field is not None:
                    scale_val = scale_field.value
            scale_vec = cls._prepare_scale(scale_val, feat_dim, param_dtype)

            return cls._PRIM.bind(
                x,
                scale_vec,
                epsilon=float(getattr(self, "epsilon", 1e-5)),
                axis=axis,
            )

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@RMSNormPlugin._PRIM.def_impl
def _impl_rms_norm(x, scale, *, epsilon: float, axis: int):
    axis_val = int(axis)
    if axis_val < 0:
        axis_val += x.ndim
    if axis_val < 0 or axis_val >= x.ndim:
        raise ValueError("axis out of range for RMSNorm")

    sq_mean = jnp.mean(jnp.square(x), axis=axis_val, keepdims=True)
    inv_rms = jnp.reciprocal(jnp.sqrt(sq_mean + epsilon))
    normed = x * inv_rms

    scale = jnp.asarray(scale, dtype=normed.dtype)
    bshape = [1] * normed.ndim
    bshape[axis_val] = scale.shape[0]
    scale = jnp.reshape(scale, bshape)

    return normed * scale
