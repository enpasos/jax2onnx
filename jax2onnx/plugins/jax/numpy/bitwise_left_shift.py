# jax2onnx/plugins/jax/numpy/bitwise_left_shift.py

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
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.jax.numpy._common import (
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.jax.numpy.left_shift import (
    abstract_eval_via_orig_binary,
    lower_left_shift_core,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_BITWISE_LEFT_SHIFT_PRIM: Final = make_jnp_primitive("jax.numpy.bitwise_left_shift")


@register_primitive(
    jaxpr_primitive=_BITWISE_LEFT_SHIFT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bitwise_left_shift.html",
    onnx=[
        {
            "component": "BitShift",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitShift.html",
        },
    ],
    since="0.12.7",
    context="primitives.jnp",
    component="bitwise_left_shift",
    testcases=[
        {
            "testcase": "jnp_bitwise_left_shift_basic",
            "callable": lambda x, s: jnp.bitwise_left_shift(x, s),
            "input_values": [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([1, 2, 1], dtype=np.int32),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["BitShift:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_bitwise_left_shift_bool_bool",
            "callable": lambda x, s: jnp.bitwise_left_shift(x, s),
            "input_values": [
                np.array([True, False, True], dtype=np.bool_),
                np.array([True, False, True], dtype=np.bool_),
            ],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Cast:3 -> BitShift:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "bitwise_left_shift_vmap_batching",
            "callable": lambda x, s: jax.vmap(jnp.bitwise_left_shift)(x, s),
            "input_shapes": [(3, 4), (3, 4)],
            "input_dtypes": [np.int32, np.int32],
        },
    ],
)
class JnpBitwiseLeftShiftPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _BITWISE_LEFT_SHIFT_PRIM
    _FUNC_NAME: ClassVar[str] = "bitwise_left_shift"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        y: core.AbstractValue,
    ) -> core.ShapedArray:
        return abstract_eval_via_orig_binary(
            JnpBitwiseLeftShiftPlugin._PRIM,
            JnpBitwiseLeftShiftPlugin._FUNC_NAME,
            x,
            y,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        lower_left_shift_core(
            ctx,
            eqn,
            input_x_hint="jnp_bitwise_left_shift_x",
            input_y_hint="jnp_bitwise_left_shift_s",
            output_hint="jnp_bitwise_left_shift_out",
            cast_x_hint="jnp_bitwise_left_shift_x_cast",
            cast_y_hint="jnp_bitwise_left_shift_s_cast",
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return cast(
            list[AssignSpec | MonkeyPatchSpec],
            jnp_binding_specs(cls._PRIM, cls._FUNC_NAME),
        )


@JnpBitwiseLeftShiftPlugin._PRIM.def_impl
def _bitwise_left_shift_impl(*args: object, **kwargs: object) -> object:
    orig = get_orig_impl(
        JnpBitwiseLeftShiftPlugin._PRIM,
        JnpBitwiseLeftShiftPlugin._FUNC_NAME,
    )
    return orig(*args, **kwargs)


def _bitwise_left_shift_batch_rule(
    args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
) -> tuple[Any, Any]:
    return cast(
        tuple[Any, Any],
        broadcast_batcher_compat(
            JnpBitwiseLeftShiftPlugin._PRIM,
            args,
            dims,
            **params,
        ),
    )


batching.primitive_batchers[JnpBitwiseLeftShiftPlugin._PRIM] = (
    _bitwise_left_shift_batch_rule
)
