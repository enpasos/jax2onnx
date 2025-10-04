# jax2onnx/plugins/equinox/eqx/nn/layer_norm.py

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import onnx_ir as ir
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG2
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_EXPECT_LN = EG2(["LayerNormalization"], mode="all", search_functions=False)


@register_primitive(
    jaxpr_primitive="eqx.nn.layer_norm",
    jax_doc="https://docs.kidger.site/equinox/api/nn/normalisation/#equinox.nn.LayerNorm",
    onnx=[
        {
            "component": "LayerNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__LayerNormalization.html",
        }
    ],
    since="v0.8.0",
    context="primitives.eqx",
    component="layer_norm",
    testcases=[
        {
            "testcase": "layer_norm",
            "callable": eqx.nn.LayerNorm(32, eps=1e-5),
            "input_shapes": [(32,)],
            "post_check_onnx_graph": _EXPECT_LN,
        },
        {
            "testcase": "layer_norm_multiaxis",
            "callable": eqx.nn.LayerNorm((20, 32)),
            "input_shapes": [(20, 32)],
            "post_check_onnx_graph": _EXPECT_LN,
        },
        {
            "testcase": "batched_layer_norm",
            "callable": jax.vmap(eqx.nn.LayerNorm(32, eps=1e-5)),
            "input_shapes": [("B", 32)],
            "post_check_onnx_graph": _EXPECT_LN,
        },
        {
            "testcase": "layer_norm_no_bias_no_scale",
            "callable": eqx.nn.LayerNorm(32, use_bias=False, use_weight=False),
            "input_shapes": [(32,)],
            "post_check_onnx_graph": _EXPECT_LN,
        },
    ],
)
class LayerNormPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.layer_norm")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, scale, bias, *, epsilon):
        del scale, bias, epsilon
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        x_var, scale_var, bias_var = eqn.invars
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("ln_in"))
        scale_val = ctx.get_value_for_var(
            scale_var, name_hint=ctx.fresh_name("ln_scale")
        )
        bias_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("ln_bias"))
        scale_val = cast_param_like(ctx, scale_val, x_val, name_hint="ln_scale_cast")
        bias_val = cast_param_like(ctx, bias_val, x_val, name_hint="ln_bias_cast")

        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("ln_out"))
        node = ir.Node(
            op_type="LayerNormalization",
            domain="",
            inputs=[x_val, scale_val, bias_val],
            outputs=[out_val],
            name=ctx.fresh_name("LayerNorm"),
        )
        ctx.add_node(node)

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        scale_shape = tuple(getattr(getattr(scale_var, "aval", None), "shape", ()))
        axis = max(len(x_shape) - len(scale_shape), 0)
        epsilon = float(eqn.params.get("epsilon", 1e-5))
        ctx.set_node_attrs(node, {"axis": int(axis), "epsilon": epsilon})

        if x_shape:
            _stamp_type_and_shape(out_val, x_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("equinox.nn", "layer_norm_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.LayerNorm",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _patch_call(orig):
        def wrapped(self, x, state=None, *, key=None):
            del key
            if getattr(self, "shape", None) is not None and x.shape != self.shape:
                raise ValueError(
                    "`LayerNorm(shape)(x)` requires `x.shape == shape`; consider jax.vmap."
                )
            dtype = getattr(x, "dtype", None) or jnp.result_type(x)
            if getattr(self, "use_weight", True):
                scale = jnp.asarray(self.weight, dtype=dtype)
            else:
                scale = jnp.ones(self.shape, dtype=dtype)
            if getattr(self, "use_bias", True):
                bias = jnp.asarray(self.bias, dtype=dtype)
            else:
                bias = jnp.zeros(self.shape, dtype=dtype)
            out = LayerNormPlugin._PRIM.bind(x, scale, bias, epsilon=float(self.eps))
            return out if state is None else (out, state)

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, scale, bias, *, epsilon: cls.abstract_eval(
                    x, scale, bias, epsilon=epsilon
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


@LayerNormPlugin._PRIM.def_impl
def _layer_norm_impl(x, scale, bias, *, epsilon):
    x_arr = jnp.asarray(x)
    scale_arr = jnp.asarray(scale, dtype=x_arr.dtype)
    bias_arr = jnp.asarray(bias, dtype=x_arr.dtype)
    tail_ndim = scale_arr.ndim or 1
    axis0 = x_arr.ndim - tail_ndim
    if axis0 < 0:
        axis0 = 0
    axes = tuple(range(axis0, x_arr.ndim))
    mean = jnp.mean(x_arr, axis=axes, keepdims=True)
    var = jnp.var(x_arr, axis=axes, keepdims=True)
    norm = (x_arr - mean) / jnp.sqrt(var + float(epsilon))
    reshape_shape = (1,) * axis0 + scale_arr.shape
    scale_b = jnp.reshape(scale_arr, reshape_shape)
    bias_b = jnp.reshape(bias_arr, reshape_shape)
    return norm * scale_b + bias_b


def _layer_norm_batch_rule(batched_args, batch_dims, *, epsilon):
    x, scale, bias = batched_args
    x_bdim, scale_bdim, bias_bdim = batch_dims
    if scale_bdim is not None or bias_bdim is not None:
        raise NotImplementedError(
            "Batching over LayerNorm parameters is not supported."
        )
    out = LayerNormPlugin._PRIM.bind(x, scale, bias, epsilon=epsilon)
    return out, x_bdim


batching.primitive_batchers[LayerNormPlugin._PRIM] = _layer_norm_batch_rule
