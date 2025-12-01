# jax2onnx/plugins/equinox/eqx/nn/rms_norm.py

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from equinox.nn._normalisation import sentinel as _SENTINEL
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _axes_for_tail(x_rank: int, tail_rank: int) -> tuple[int, ...]:
    axis_start = max(x_rank - tail_rank, 0)
    return tuple(range(axis_start, x_rank))


@register_primitive(
    jaxpr_primitive="eqx.nn.rms_norm",
    jax_doc="https://docs.kidger.site/equinox/api/nn/normalisation/#equinox.nn.RMSNorm",
    onnx=[
        {
            "component": "RMSNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__RMSNormalization.html",
        }
    ],
    since="v0.10.2",
    context="primitives.eqx",
    component="rms_norm",
    testcases=[
        {
            "testcase": "rms_norm_vector",
            "callable": eqx.nn.RMSNorm(16, eps=1e-5),
            "input_shapes": [(16,)],
            "post_check_onnx_graph": expect_graph(
                ["Div:16 -> Mul:16 -> Add:16 -> Identity:16"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "rms_norm_no_affine",
            "callable": eqx.nn.RMSNorm(8, use_weight=False, use_bias=False),
            "input_shapes": [(8,)],
            "post_check_onnx_graph": expect_graph(
                ["Div:8 -> Mul:8 -> Add:8 -> Identity:8"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class RMSNormPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.rms_norm")
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, scale, bias, *, epsilon, result_dtype):
        del scale, bias, epsilon
        return ShapedArray(x.shape, result_dtype)

    def lower(self, ctx, eqn):
        x_var, scale_var, bias_var = eqn.invars
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("rms_in"))
        scale_val = ctx.get_value_for_var(
            scale_var, name_hint=ctx.fresh_name("rms_scale")
        )
        bias_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("rms_bias"))

        scale_val = cast_param_like(ctx, scale_val, x_val, name_hint="rms_scale_cast")
        bias_val = cast_param_like(ctx, bias_val, x_val, name_hint="rms_bias_cast")

        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("rms_out"))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        scale_shape = tuple(getattr(getattr(scale_var, "aval", None), "shape", ()))

        axes = _axes_for_tail(len(x_shape), len(scale_shape))
        axes_val = _const_i64(ctx, np.asarray(axes, dtype=np.int64), "rms_axes")

        epsilon = float(eqn.params.get("epsilon", 1e-5))
        result_dtype = np.dtype(
            eqn.params.get("result_dtype", getattr(x_var.aval, "dtype", np.float32))
        )

        x_np_dtype = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )
        two_val = ctx.bind_const_for_var(object(), np.asarray(2.0, dtype=x_np_dtype))
        eps_val = ctx.bind_const_for_var(
            object(), np.asarray(epsilon, dtype=x_np_dtype)
        )

        builder = ctx.builder

        pow_out = builder.Pow(
            x_val,
            two_val,
            _outputs=[ctx.fresh_name("rms_pow")],
        )
        pow_out.type = ir.TensorType(getattr(x_val.type, "dtype", None))
        _stamp_type_and_shape(pow_out, x_shape)

        mean_dims = list(x_shape)
        for axis in axes:
            if axis < len(mean_dims):
                mean_dims[axis] = 1
        mean_dims_tuple = tuple(mean_dims)

        mean_out = builder.ReduceMean(
            pow_out,
            axes_val,
            keepdims=1,
            _outputs=[ctx.fresh_name("rms_mean")],
        )
        mean_out.type = ir.TensorType(getattr(x_val.type, "dtype", None))
        _stamp_type_and_shape(mean_out, mean_dims_tuple)

        add_out = builder.Add(
            mean_out,
            eps_val,
            _outputs=[ctx.fresh_name("rms_add")],
        )
        add_out.type = ir.TensorType(getattr(x_val.type, "dtype", None))
        _stamp_type_and_shape(add_out, mean_dims_tuple)

        sqrt_out = builder.Sqrt(
            add_out,
            _outputs=[ctx.fresh_name("rms_sqrt")],
        )
        sqrt_out.type = ir.TensorType(getattr(x_val.type, "dtype", None))
        _stamp_type_and_shape(sqrt_out, mean_dims_tuple)

        div_out = builder.Div(
            x_val,
            sqrt_out,
            _outputs=[ctx.fresh_name("rms_div")],
        )
        div_out.type = ir.TensorType(getattr(x_val.type, "dtype", None))
        _stamp_type_and_shape(div_out, x_shape)

        scaled = builder.Mul(
            div_out,
            scale_val,
            _outputs=[ctx.fresh_name("rms_scaled")],
        )
        scaled.type = ir.TensorType(getattr(x_val.type, "dtype", None))
        _stamp_type_and_shape(scaled, x_shape)

        affine = builder.Add(
            scaled,
            bias_val,
            _outputs=[ctx.fresh_name("rms_affine")],
        )
        affine.type = ir.TensorType(getattr(x_val.type, "dtype", None))
        _stamp_type_and_shape(affine, x_shape)

        if result_dtype != x_np_dtype:
            exemplar = ctx.bind_const_for_var(
                object(), np.zeros((), dtype=result_dtype)
            )
            result = cast_param_like(ctx, affine, exemplar, name_hint="rms_cast")
        else:
            result = affine

        stamped_dims = []
        for idx, dim in enumerate(x_shape):
            label = _dim_label_from_value_or_aval(x_val, x_shape, idx)
            stamped_dims.append(label if label is not None else dim)
        stamped_shape = tuple(stamped_dims)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("RMSNorm")
        if getattr(result, "name", None) != desired_name:
            prior_type = getattr(result, "type", None)
            result = builder.Identity(result, _outputs=[desired_name])
            if prior_type is not None:
                result.type = prior_type

        _stamp_type_and_shape(result, stamped_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("equinox.nn", "rms_norm_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.RMSNorm",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _patch_call(orig):
        def wrapped(self, x, state=_SENTINEL, *, key=None):
            del key
            if x.shape != self.shape:
                raise ValueError(
                    "`RMSNorm(shape)(x)` requires `x.shape == shape`; consider jax.vmap."
                )

            orig_dtype = getattr(x, "dtype", None) or jnp.result_type(x)
            compute_dtype = jnp.result_type(orig_dtype, jnp.float32)
            x_cast = jnp.asarray(x, dtype=compute_dtype)

            if getattr(self, "use_weight", True):
                weight = jnp.asarray(self.weight, dtype=compute_dtype)
            else:
                weight = jnp.ones(self.shape, dtype=compute_dtype)

            if getattr(self, "use_bias", True):
                bias = jnp.asarray(self.bias, dtype=compute_dtype)
            else:
                bias = jnp.zeros(self.shape, dtype=compute_dtype)

            out = RMSNormPlugin._PRIM.bind(
                x_cast,
                weight,
                bias,
                epsilon=float(self.eps),
                result_dtype=np.dtype(orig_dtype),
            )

            if state is _SENTINEL:
                return out
            return out, state

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@RMSNormPlugin._PRIM.def_impl
def _rms_norm_impl(x, scale, bias, *, epsilon, result_dtype):
    tail_rank = scale.ndim or 1
    axes = _axes_for_tail(x.ndim, tail_rank)
    sq_mean = jnp.mean(jnp.square(x), axis=axes, keepdims=True)
    inv_rms = jnp.reciprocal(jnp.sqrt(sq_mean + float(epsilon)))
    normed = x * inv_rms

    scale_arr = jnp.asarray(scale, dtype=normed.dtype)
    bias_arr = jnp.asarray(bias, dtype=normed.dtype)

    result = normed * scale_arr + bias_arr
    return jnp.asarray(result, dtype=result_dtype)


def _rms_norm_batch_rule(batched_args, batch_dims, *, epsilon, result_dtype):
    x, scale, bias = batched_args
    x_bdim, scale_bdim, bias_bdim = batch_dims
    if scale_bdim is not None or bias_bdim is not None:
        raise NotImplementedError("Batching over RMSNorm parameters is not supported.")
    result = RMSNormPlugin._PRIM.bind(
        x,
        scale,
        bias,
        epsilon=epsilon,
        result_dtype=result_dtype,
    )
    return result, x_bdim


batching.primitive_batchers[RMSNormPlugin._PRIM] = _rms_norm_batch_rule
