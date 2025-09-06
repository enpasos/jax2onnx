# file: jax2onnx/plugins2/flax/nnx/layer_norm.py
from __future__ import annotations

from typing import Any, ClassVar
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.core import ShapedArray
from jax.extend.core import Primitive

import onnx_ir as ir
from jax2onnx.plugins2.plugin_system import (
    PrimitiveLeafPlugin,
    register_primitive,
)
from jax2onnx.plugins2._patching import MonkeyPatchSpec

LAYER_NORM_PRIM = Primitive("nnx.layer_norm")
LAYER_NORM_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=LAYER_NORM_PRIM.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm",
    onnx=[{
        "component": "LayerNormalization",
        "doc": "https://onnx.ai/onnx/operators/onnx__LayerNormalization.html",
    }],
    since="v0.1.0",
    context="primitives2.flax.nnx",
    component="layer_norm",
    testcases=[
        {
            "testcase": "layer_norm",
            "callable": nnx.LayerNorm(num_features=32, epsilon=1e-5, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "layer_norm_no_bias_no_scale",
            "callable": nnx.LayerNorm(32, use_bias=False, use_scale=False, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "layer_norm_bias_no_scale",
            "callable": nnx.LayerNorm(32, use_bias=True, use_scale=False, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "layer_norm_no_bias_scale",
            "callable": nnx.LayerNorm(32, use_bias=False, use_scale=True, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "layer_norm_bias_scale",
            "callable": nnx.LayerNorm(32, use_bias=True, use_scale=True, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "layer_norm_multiaxis",
            "callable": nnx.LayerNorm(
                3 * 3 * 64,
                reduction_axes=(1, 2, 3),
                feature_axes=(1, 2, 3),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 3, 3, 64)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "layer_norm_symbolic_batch",
            "callable": nnx.LayerNorm(num_features=16, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 8, 16)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: all(
                n.op_type not in ("Unsqueeze", "Reshape") for n in m.graph.node
            ),
            "use_onnx_ir": True,
        },
        {
            "testcase": "layer_norm_negative_axis_no_div",
            "callable": nnx.LayerNorm(
                num_features=32, epsilon=1e-5, reduction_axes=-1, feature_axes=-1, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 20, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: any(
                n.op_type == "LayerNormalization" for n in m.graph.node
            ) and all(n.op_type != "Div" for n in m.graph.node),
            "use_onnx_ir": True,
        },
    ],
)
class LayerNormPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = LAYER_NORM_PRIM

    @staticmethod
    def abstract_eval(x, scale, bias, *, epsilon: float, axis: int):
        x_aval = x if isinstance(x, ShapedArray) else ShapedArray(x.shape, x.dtype)
        return ShapedArray(x_aval.shape, x_aval.dtype)

    def lower(self, ctx, eqn, params: dict[str, Any] | None = None):
        """
        Emit a single LayerNormalization node.
        scale/bias are already shaped to X.shape[axis:] by the monkey-patch,
        so we don't need any Reshape or attributes here.
        """
        x_v     = ctx.get_value_for_var(eqn.invars[0])
        scale_v = ctx.get_value_for_var(eqn.invars[1])
        bias_v  = ctx.get_value_for_var(eqn.invars[2])
        y_v     = ctx.get_value_for_var(eqn.outvars[0])

        ln = ir.Node(
            op_type="LayerNormalization",
            domain="",
            inputs=[x_v, scale_v, bias_v],
            outputs=[y_v],
            name=ctx.builder.fresh_name("LayerNorm"),
        )
        ctx.add_node(ln)


    # ---- patch nnx.LayerNorm.__call__ to bind our primitive ----------------
    @classmethod
    def binding_specs(cls):
        return [MonkeyPatchSpec(nnx.LayerNorm, "__call__", cls._patch_call)]

    @staticmethod
    def _patch_call(orig):
        def wrapped(self: nnx.LayerNorm, x):
            # Prefer explicit reduction_axes, then feature_axes, else last dim
            if getattr(self, "reduction_axes", None) is not None:
                axes = self.reduction_axes
            elif getattr(self, "feature_axes", None) is not None:
                axes = self.feature_axes
            else:
                axes = -1

            if isinstance(axes, int):
                axes = (axes,)
            axes = tuple(a if a >= 0 else a + x.ndim for a in axes)
            axis0 = min(axes)

            param_dtype = getattr(self, "param_dtype", None) or x.dtype
            if x.dtype != param_dtype:
                x = x.astype(param_dtype)

            # Tail shape across all normalized axes (NOT including batch/seq)
            tail_shape = tuple(x.shape[a] for a in axes)
            # Build base scale/bias (matching tail dimensions)
            if getattr(self, "use_scale", False) and getattr(self, "scale", None) is not None:
                sv = self.scale.value
                base_scale = jnp.reshape(sv, tail_shape) if tuple(sv.shape) != tail_shape else sv
            else:
                base_scale = jnp.ones(tail_shape, dtype=param_dtype)
            if getattr(self, "use_bias", False) and getattr(self, "bias", None) is not None:
                bv = self.bias.value
                base_bias = jnp.reshape(bv, tail_shape) if tuple(bv.shape) != tail_shape else bv
            else:
                base_bias = jnp.zeros(tail_shape, dtype=param_dtype)

            # If we normalize the last dim only, we can bind directly.
            # Otherwise, flatten the tail so ONNX LN default (last dim) is correct,
            # bind with vector scale/bias, then reshape back to the original shape.
            needs_flatten = (axis0 != x.ndim - 1) or (len(axes) > 1)
            if not needs_flatten:
                return LAYER_NORM_PRIM.bind(
                    x,
                    base_scale,   # shape: (last_dim,)
                    base_bias,    # shape: (last_dim,)
                    epsilon=float(getattr(self, "epsilon", 1e-5)),
                    axis=int(axis0),
                )

            # Flatten tail dims to a single last dimension
            orig_shape = x.shape
            x_flat = jnp.reshape(x, (*orig_shape[:axis0], -1))
            scale_vec = jnp.reshape(base_scale, (-1,))
            bias_vec  = jnp.reshape(base_bias,  (-1,))

            y_flat = LAYER_NORM_PRIM.bind(
                x_flat,
                scale_vec,
                bias_vec,
                epsilon=float(getattr(self, "epsilon", 1e-5)),
                axis=x_flat.ndim - 1,   # last dimension
            )
            return jnp.reshape(y_flat, orig_shape)
        return wrapped


# Bind abstract eval on the primitive
LAYER_NORM_PRIM.def_abstract_eval(LayerNormPlugin.abstract_eval)