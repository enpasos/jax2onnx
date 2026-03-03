# jax2onnx/plugins/jax/numpy/atan2.py

from __future__ import annotations

from typing import Any, ClassVar, Final, cast

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.jax.lax.atan2 import Atan2Plugin as LaxAtan2Plugin
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_ATAN2_PRIM: Final = make_jnp_primitive("jax.numpy.atan2")


def _abstract_eval_via_orig_binary(
    prim: Any,
    func_name: str,
    x: core.AbstractValue,
    y: core.AbstractValue,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    y_shape = tuple(getattr(y, "shape", ()))
    x_dtype: np.dtype[Any] = np.dtype(getattr(x, "dtype", np.float32))
    y_dtype: np.dtype[Any] = np.dtype(getattr(y, "dtype", np.float32))
    x_spec = jax.ShapeDtypeStruct(x_shape, x_dtype)
    y_spec = jax.ShapeDtypeStruct(y_shape, y_dtype)
    orig = get_orig_impl(prim, func_name)
    out = jax.eval_shape(lambda a, b: orig(a, b), x_spec, y_spec)
    out_shape = tuple(getattr(out, "shape", ()))
    out_dtype = np.dtype(getattr(out, "dtype", x_dtype))
    return core.ShapedArray(out_shape, out_dtype)


@register_primitive(
    jaxpr_primitive=_ATAN2_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atan2.html",
    onnx=[
        {"component": "Atan", "doc": "https://onnx.ai/onnx/operators/onnx__Atan.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="atan2",
    testcases=[
        {
            "testcase": "jnp_atan2_quadrants",
            "callable": lambda x, y: jnp.atan2(x, y),
            "input_values": [
                np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Div:7 -> Atan:7"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_atan2_broadcast",
            "callable": lambda x, y: jnp.atan2(x, y),
            "input_shapes": [(2, 1), (1, 3)],
            "expected_output_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "atan2_vmap_batching",
            "callable": lambda x, y: jax.vmap(jnp.atan2)(x, y),
            "input_shapes": [(3, 4), (3, 4)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "atan2_grad_issue_batch_diff_rules",
            "callable": lambda x, y: jax.grad(lambda a, b: jnp.sum(jnp.atan2(a, b)))(
                x, y
            ),
            "input_shapes": [(2, 3), (2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpAtan2Plugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ATAN2_PRIM
    _FUNC_NAME: ClassVar[str] = "atan2"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        y: core.AbstractValue,
    ) -> core.ShapedArray:
        return _abstract_eval_via_orig_binary(
            JnpAtan2Plugin._PRIM,
            JnpAtan2Plugin._FUNC_NAME,
            x,
            y,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        LaxAtan2Plugin().lower(ctx, eqn)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = jnp_binding_specs(
            cls._PRIM, cls._FUNC_NAME
        )
        return specs


@JnpAtan2Plugin._PRIM.def_impl
def _atan2_impl(*args: object, **kwargs: object) -> object:
    orig = get_orig_impl(JnpAtan2Plugin._PRIM, JnpAtan2Plugin._FUNC_NAME)
    return orig(*args, **kwargs)


register_jvp_via_jax_jvp(JnpAtan2Plugin._PRIM, _atan2_impl)


def _atan2_batch_rule(
    args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
) -> tuple[Any, Any]:
    return cast(
        tuple[Any, Any],
        broadcast_batcher_compat(JnpAtan2Plugin._PRIM, args, dims, **params),
    )


batching.primitive_batchers[JnpAtan2Plugin._PRIM] = _atan2_batch_rule
