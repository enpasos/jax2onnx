# file: jax2onnx/plugins/flax/nnx/rms_norm.py

from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

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
    _to_ir_dim_for_shape,
    _ensure_value_info as _add_value_info,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore

RMS_NORM_PRIM = Primitive("nnx.rms_norm")
RMS_NORM_PRIM.multiple_results = False


EXPECT_RMS_NORM_GRAPH = EG(
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


def _set_attrs(ctx: Any, node: ir.Node, attrs: dict[str, object]) -> None:
    setter = getattr(ctx, "set_node_attrs", None)
    if callable(setter):
        setter(node, attrs)


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
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        scale_val = cast_param_like(ctx, scale_val, x_val, name_hint="rms_scale_cast")

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)
        if rank == 0:
            raise ValueError("RMSNorm requires tensor inputs")
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError("axis out of range for RMSNorm")

        builder = getattr(ctx, "builder", None)
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

        if opset >= 23:
            rms = ir.Node(
                op_type="RMSNormalization",
                domain="",
                inputs=[x_val, scale_val],
                outputs=[y_val],
                name=ctx.fresh_name("RMSNorm"),
            )
            ctx.add_node(rms)
            _set_attrs(
                ctx,
                rms,
                {"axis": int(axis), "epsilon": float(epsilon)},
            )
            _stamp_type_and_shape(y_val, dims)
            _add_value_info(ctx, y_val)
            return

        x_np_dtype = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )
        x_ir_dtype = getattr(getattr(x_val, "type", None), "dtype", ir.DataType.FLOAT)

        builder = getattr(ctx, "builder", None)

        def _const(
            name: str,
            value: np.ndarray,
            *,
            np_dtype: np.dtype | None = None,
            ir_dtype=None,
        ) -> ir.Value:
            arr = np.asarray(value, dtype=np_dtype or x_np_dtype)
            if builder is not None and hasattr(builder, "add_initializer_from_array"):
                return builder.add_initializer_from_array(ctx.fresh_name(name), arr)
            val = ir.Value(
                name=ctx.fresh_name(name),
                type=ir.TensorType(ir_dtype or x_ir_dtype),
                shape=ir.Shape(tuple(int(d) for d in arr.shape)),
                const_value=ir.tensor(arr),
            )
            ctx._initializers.append(val)
            return val

        two_const = _const("two", np.array(2.0, dtype=x_np_dtype))
        eps_const = _const("eps", np.array(epsilon, dtype=x_np_dtype))
        axes_const = _const(
            "axes",
            np.array([int(axis)], dtype=np.int64),
            np_dtype=np.int64,
            ir_dtype=ir.DataType.INT64,
        )

        pow_out = ir.Value(
            name=ctx.fresh_name("rms_pow"),
            type=x_val.type,
            shape=x_val.shape,
        )
        pow_node = ir.Node(
            op_type="Pow",
            domain="",
            inputs=[x_val, two_const],
            outputs=[pow_out],
            name=ctx.fresh_name("Pow"),
        )
        ctx.add_node(pow_node)

        mean_shape = list(x_shape)
        if axis < len(mean_shape):
            mean_shape[axis] = 1
        mean_dims = tuple(mean_shape)
        mean_out = ir.Value(
            name=ctx.fresh_name("rms_mean"),
            type=x_val.type,
            shape=ir.Shape(tuple(_to_ir_dim_for_shape(d) for d in mean_dims)),
        )
        mean_node = ir.Node(
            op_type="ReduceMean",
            domain="",
            inputs=[pow_out, axes_const],
            outputs=[mean_out],
            name=ctx.fresh_name("ReduceMean"),
        )
        ctx.add_node(mean_node)
        _set_attrs(ctx, mean_node, {"keepdims": 1})
        _stamp_type_and_shape(mean_out, mean_dims)

        add_out = ir.Value(
            name=ctx.fresh_name("rms_add"),
            type=mean_out.type,
            shape=mean_out.shape,
        )
        add_node = ir.Node(
            op_type="Add",
            domain="",
            inputs=[mean_out, eps_const],
            outputs=[add_out],
            name=ctx.fresh_name("Add"),
        )
        ctx.add_node(add_node)

        sqrt_out = ir.Value(
            name=ctx.fresh_name("rms_sqrt"),
            type=mean_out.type,
            shape=mean_out.shape,
        )
        sqrt_node = ir.Node(
            op_type="Sqrt",
            domain="",
            inputs=[add_out],
            outputs=[sqrt_out],
            name=ctx.fresh_name("Sqrt"),
        )
        ctx.add_node(sqrt_node)
        _stamp_type_and_shape(sqrt_out, mean_dims)

        div_out = ir.Value(
            name=ctx.fresh_name("rms_div"),
            type=x_val.type,
            shape=x_val.shape,
        )
        div_node = ir.Node(
            op_type="Div",
            domain="",
            inputs=[x_val, sqrt_out],
            outputs=[div_out],
            name=ctx.fresh_name("Div"),
        )
        ctx.add_node(div_node)
        _stamp_type_and_shape(div_out, dims)

        mul_node = ir.Node(
            op_type="Mul",
            domain="",
            inputs=[div_out, scale_val],
            outputs=[y_val],
            name=ctx.fresh_name("Mul"),
        )
        ctx.add_node(mul_node)

        _stamp_type_and_shape(y_val, dims)
        _add_value_info(ctx, y_val)

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
