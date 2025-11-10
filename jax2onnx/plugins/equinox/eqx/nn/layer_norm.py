# jax2onnx/plugins/equinox/eqx/nn/layer_norm.py

from __future__ import annotations

from typing import Any, Callable, ClassVar

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import onnx_ir as ir
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


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
            "post_check_onnx_graph": expect_graph(
                ["LayerNormalization:32"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "layer_norm_multiaxis",
            "callable": eqx.nn.LayerNorm((20, 32)),
            "input_shapes": [(20, 32)],
            "post_check_onnx_graph": expect_graph(
                ["LayerNormalization:20x32"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "batched_layer_norm",
            "callable": jax.vmap(eqx.nn.LayerNorm(32, eps=1e-5)),
            "input_shapes": [("B", 32)],
            "post_check_onnx_graph": expect_graph(
                ["LayerNormalization:Bx32"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "layer_norm_no_bias_no_scale",
            "callable": eqx.nn.LayerNorm(32, use_bias=False, use_weight=False),
            "input_shapes": [(32,)],
            "post_check_onnx_graph": expect_graph(
                ["LayerNormalization:32"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class LayerNormPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.layer_norm")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax_core.AbstractValue,
        scale: jax_core.AbstractValue,
        bias: jax_core.AbstractValue,
        *,
        epsilon: float,
    ) -> ShapedArray:
        del scale, bias, epsilon
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax_core.JaxprEqn) -> None:
        x_var, scale_var, bias_var = eqn.invars
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("ln_in"))
        scale_val = ctx.get_value_for_var(
            scale_var, name_hint=ctx.fresh_name("ln_scale")
        )
        bias_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("ln_bias"))
        scale_val = cast_param_like(ctx, scale_val, x_val, name_hint="ln_scale_cast")
        bias_val = cast_param_like(ctx, bias_val, x_val, name_hint="ln_bias_cast")

        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("ln_out"))
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        scale_shape = tuple(getattr(getattr(scale_var, "aval", None), "shape", ()))
        axis = max(len(x_shape) - len(scale_shape), 0)
        epsilon = float(eqn.params.get("epsilon", 1e-5))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("LayerNorm")
        result = ctx.builder.LayerNormalization(
            x_val,
            scale_val,
            bias_val,
            axis=int(axis),
            epsilon=epsilon,
            _outputs=[desired_name],
        )

        if x_shape:
            stamped_dims = []
            for idx, dim in enumerate(x_shape):
                label = _dim_label_from_value_or_aval(x_val, x_shape, idx)
                stamped_dims.append(label if label is not None else dim)
            _stamp_type_and_shape(result, tuple(stamped_dims))
        else:
            _stamp_type_and_shape(result, ())

        x_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if x_dtype is not None:
            result.type = ir.TensorType(x_dtype)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
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
    def _patch_call(
        orig: Callable[..., jax.Array] | None,
    ) -> Callable[..., Any]:
        def wrapped(
            self: eqx.nn.LayerNorm,
            x: jax.Array,
            state: Any = None,
            *,
            key: jax.Array | None = None,
        ) -> Any:
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
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, scale, bias, *, epsilon: cls.abstract_eval(
                    x, scale, bias, epsilon=epsilon
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


@LayerNormPlugin._PRIM.def_impl
def _layer_norm_impl(
    x: jax.Array,
    scale: jax.Array,
    bias: jax.Array,
    *,
    epsilon: float,
) -> jax.Array:
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


def _layer_norm_batch_rule(
    batched_args: tuple[jax.Array, jax.Array, jax.Array],
    batch_dims: tuple[int | None, int | None, int | None],
    *,
    epsilon: float,
) -> tuple[jax.Array, int | None]:
    x, scale, bias = batched_args
    x_bdim, scale_bdim, bias_bdim = batch_dims
    if scale_bdim is not None or bias_bdim is not None:
        raise NotImplementedError(
            "Batching over LayerNorm parameters is not supported."
        )
    out = LayerNormPlugin._PRIM.bind(x, scale, bias, epsilon=epsilon)
    return out, x_bdim


batching.primitive_batchers[LayerNormPlugin._PRIM] = _layer_norm_batch_rule
