# jax2onnx/plugins/flax/linen/layer_norm.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, List, Optional, Sequence, Tuple, cast

import jax.numpy as jnp
from flax import linen as nn
from jax.core import ShapedArray
from jax.extend.core import Primitive
import onnx_ir as ir

from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _stamp_type_and_shape,
    _ensure_value_metadata,
)


def _canonicalize_axes(ndim: int, axes: Sequence[int] | int) -> Tuple[int, ...]:
    if isinstance(axes, int):
        axes = (axes,)
    out = []
    for axis in axes:
        axis = int(axis)
        if axis < 0:
            axis += ndim
        out.append(axis)
    return tuple(out)


def _layer_norm_attr_check(
    model: Any,
    *,
    axis: Optional[int] = None,
    epsilon: Optional[float] = None,
) -> bool:
    graph = getattr(model, "graph", None)
    if graph is None:
        return False
    nodes: Sequence[Any] = getattr(graph, "node", [])
    ln_nodes = [n for n in nodes if getattr(n, "op_type", "") == "LayerNormalization"]
    if len(ln_nodes) != 1:
        return False
    ln = ln_nodes[0]
    attrs = (
        getattr(ln, "attribute", None)
        or getattr(ln, "attributes", None)
        or getattr(ln, "_attributes", None)
    )
    if attrs is None:
        return False

    def _attr_value(name: str) -> Optional[Any]:
        for attr in attrs:
            if getattr(attr, "name", "") != name:
                continue
            if hasattr(attr, "i"):
                try:
                    has_field = getattr(attr, "HasField", None)
                    if callable(has_field) and not has_field("i"):
                        pass
                    else:
                        return getattr(attr, "i")
                except Exception:
                    return getattr(attr, "i")
            if hasattr(attr, "f"):
                try:
                    has_field = getattr(attr, "HasField", None)
                    if callable(has_field) and not has_field("f"):
                        pass
                    else:
                        return getattr(attr, "f")
                except Exception:
                    return getattr(attr, "f")
            floats = getattr(attr, "floats", None)
            if floats:
                return list(floats)
            ints = getattr(attr, "ints", None)
            if ints:
                return list(ints)
            sval = getattr(attr, "s", None)
            if sval is not None:
                if isinstance(sval, bytes):
                    try:
                        return sval.decode("utf-8")
                    except Exception:
                        return sval
                return sval
            strings = getattr(attr, "strings", None)
            if strings:
                try:
                    return [
                        s.decode("utf-8") if isinstance(s, bytes) else s
                        for s in strings
                    ]
                except Exception:
                    return list(strings)
        return None

    if axis is not None:
        axis_attr = _attr_value("axis")
        try:
            if axis_attr is None:
                return False
            if isinstance(axis_attr, (list, tuple)):
                if not axis_attr:
                    return False
                axis_attr = axis_attr[0]
            if int(axis_attr) != int(axis):
                return False
        except Exception:
            return False
    if epsilon is not None:
        eps_attr = _attr_value("epsilon")
        try:
            if eps_attr is None:
                return False
            if isinstance(eps_attr, (list, tuple)):
                if not eps_attr:
                    return False
                eps_attr = eps_attr[0]
            if abs(float(eps_attr) - float(epsilon)) > 1e-9:
                return False
        except Exception:
            return False
    return True


LAYER_NORM_PRIM: Primitive = Primitive("linen.layer_norm")
LAYER_NORM_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=LAYER_NORM_PRIM.name,
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.LayerNorm",
    onnx=[
        {
            "component": "LayerNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__LayerNormalization.html",
        }
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="layer_norm",
    testcases=[
        {
            "testcase": "layer_norm",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.LayerNorm,
                input_shape=(1, 20, 32),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 20, 32)],
            "expected_output_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LayerNormalization:Bx20x32"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "layer_norm_no_bias_no_scale",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.LayerNorm,
                input_shape=(1, 20, 32),
                use_bias=False,
                use_scale=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 20, 32)],
            "expected_output_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LayerNormalization:Bx20x32"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "layer_norm_multiaxis",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.LayerNorm,
                input_shape=(1, 3, 3, 64),
                reduction_axes=(1, 2, 3),
                feature_axes=(1, 2, 3),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 3, 3, 64)],
            "expected_output_shapes": [("B", 3, 3, 64)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Reshape:Bx576 -> LayerNormalization:Bx576 -> Reshape:Bx3x3x64"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "layer_norm_default_epsilon",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.LayerNorm,
                input_shape=(1, 10, 3),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 10, 3)],
            "expected_output_shapes": [("B", 10, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: (
                _layer_norm_attr_check(m, axis=2, epsilon=1e-6)
            ),
        },
    ],
)
class LayerNormPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = LAYER_NORM_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None

    @staticmethod
    def abstract_eval(x, scale, bias, *, epsilon: float, axis: int):
        x_aval = x if isinstance(x, ShapedArray) else ShapedArray(x.shape, x.dtype)
        return ShapedArray(x_aval.shape, x_aval.dtype)

    def lower(self, ctx, eqn, params: dict[str, Any] | None = None):
        x_v = ctx.get_value_for_var(eqn.invars[0])
        scale_v = ctx.get_value_for_var(eqn.invars[1])
        bias_v = ctx.get_value_for_var(eqn.invars[2])

        scale_v = cast_param_like(ctx, scale_v, x_v, name_hint="ln_scale_cast")
        bias_v = cast_param_like(ctx, bias_v, x_v, name_hint="ln_bias_cast")

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder")

        p = params or getattr(eqn, "params", {}) or {}
        in_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
        rank = len(in_shape)
        axis = int(p.get("axis", -1))
        if axis < 0:
            axis += rank
        eps = float(p.get("epsilon", 1e-5))

        y_val = builder.LayerNormalization(
            x_v,
            scale_v,
            bias_v,
            axis=int(axis),
            epsilon=float(eps),
            _outputs=[ctx.fresh_name("LayerNorm")],
        )

        x_dtype = getattr(getattr(x_v, "type", None), "dtype", None)
        if x_dtype is not None:
            y_val.type = ir.TensorType(x_dtype)

        out_aval_shape = tuple(getattr(eqn.outvars[0].aval, "shape", ()))
        if out_aval_shape:
            dims: list[Any] = []
            for idx in range(len(out_aval_shape)):
                label = _dim_label_from_value_or_aval(x_v, in_shape, idx)
                dims.append(label if label is not None else out_aval_shape[idx])
            _stamp_type_and_shape(y_val, tuple(dims))
        _ensure_value_metadata(ctx, y_val)

        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(eqn.outvars[0], y_val)

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="flax.linen.LayerNorm",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _make_patch(orig):
        LayerNormPlugin._ORIGINAL_CALL = orig
        prim = LayerNormPlugin._PRIM

        def patched(self, x, *, mask=None):
            if mask is not None:
                return orig(self, x, mask=mask)
            if getattr(self, "axis_name", None) is not None:
                return orig(self, x, mask=mask)
            if getattr(self, "axis_index_groups", None) is not None:
                return orig(self, x, mask=mask)
            if not getattr(self, "use_fast_variance", True):
                return orig(self, x, mask=mask)

            scope = getattr(self, "scope", None)
            if scope is None or not hasattr(scope, "variables"):
                return orig(self, x, mask=mask)
            variables = scope.variables()
            params = variables.get("params", {})

            reduction_axes = getattr(self, "reduction_axes", -1)
            feature_axes = getattr(self, "feature_axes", -1)
            red_axes = _canonicalize_axes(x.ndim, reduction_axes)
            feat_axes = _canonicalize_axes(x.ndim, feature_axes)
            if tuple(sorted(red_axes)) != tuple(sorted(feat_axes)):
                return orig(self, x, mask=mask)

            axis0 = min(red_axes)
            expected_axes = tuple(range(axis0, x.ndim))
            if tuple(sorted(red_axes)) != expected_axes:
                return orig(self, x, mask=mask)

            tail_shape = tuple(x.shape[a] for a in red_axes)
            if getattr(self, "use_scale", True):
                scale_param = params.get("scale")
                if scale_param is None:
                    return orig(self, x, mask=mask)
                base_scale = (
                    jnp.reshape(scale_param, tail_shape)
                    if tuple(scale_param.shape) != tail_shape
                    else scale_param
                )
                base_scale = jnp.asarray(base_scale, dtype=x.dtype)
            else:
                base_scale = jnp.ones(tail_shape, dtype=x.dtype)

            if getattr(self, "use_bias", True):
                bias_param = params.get("bias")
                if bias_param is None:
                    return orig(self, x, mask=mask)
                base_bias = (
                    jnp.reshape(bias_param, tail_shape)
                    if tuple(bias_param.shape) != tail_shape
                    else bias_param
                )
                base_bias = jnp.asarray(base_bias, dtype=x.dtype)
            else:
                base_bias = jnp.zeros(tail_shape, dtype=x.dtype)

            eps = float(getattr(self, "epsilon", 1e-5))
            needs_flatten = (axis0 != x.ndim - 1) or (len(red_axes) > 1)
            if needs_flatten:
                orig_shape = x.shape
                x_flat = jnp.reshape(x, (*orig_shape[:axis0], -1))
                scale_vec = jnp.reshape(base_scale, (-1,))
                bias_vec = jnp.reshape(base_bias, (-1,))
                y_flat = prim.bind(
                    x_flat,
                    scale_vec,
                    bias_vec,
                    epsilon=eps,
                    axis=x_flat.ndim - 1,
                )
                return jnp.reshape(y_flat, orig_shape)

            return prim.bind(
                x,
                base_scale,
                base_bias,
                epsilon=eps,
                axis=axis0,
            )

        return patched


@LayerNormPlugin._PRIM.def_impl
def _ln_impl(x, scale, bias, *, epsilon: float, axis: int):
    axis_val: int = int(axis)
    ndim_val: int = int(cast(int, getattr(x, "ndim", 0)))
    normalized_axis: int
    if axis_val >= 0:
        normalized_axis = axis_val
    else:
        normalized_axis = int(ndim_val + axis_val)

    mean = jnp.mean(x, axis=normalized_axis, keepdims=True)
    mean2 = jnp.mean(jnp.square(x), axis=normalized_axis, keepdims=True)
    var = mean2 - jnp.square(mean)
    inv = jnp.reciprocal(jnp.sqrt(var + epsilon))

    rank_i: int = int(cast(int, getattr(x, "ndim", 0)))
    bshape: List[int] = [1] * rank_i
    bshape_idx: int = int(normalized_axis)
    bshape[bshape_idx] = int(x.shape[bshape_idx])
    try:
        scale_b = jnp.reshape(scale, bshape)
    except Exception:
        scale_b = scale
    try:
        bias_b = jnp.reshape(bias, bshape)
    except Exception:
        bias_b = bias

    return (x - mean) * inv * scale_b + bias_b
