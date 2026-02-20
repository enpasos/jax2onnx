# jax2onnx/plugins/flax/nnx/prelu.py

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Final

import jax
from jax.interpreters import batching
import jax.numpy as jnp
from flax import nnx
from jax.extend.core import Primitive

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext


PRELU_PRIM: Final[Primitive] = Primitive("nnx.prelu")
PRELU_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=PRELU_PRIM.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.PReLU",
    onnx=[
        {"component": "PRelu", "doc": "https://onnx.ai/onnx/operators/onnx__PRelu.html"}
    ],
    since="0.12.1",
    context="primitives.nnx",
    component="prelu",
    testcases=[
        {
            "testcase": "prelu_default",
            "callable": construct_and_call(
                nnx.PReLU,
                param_dtype=with_requested_dtype(),
            ),
            "input_shapes": [(2, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["PRelu:2x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "prelu_custom_slope",
            "callable": construct_and_call(
                nnx.PReLU,
                negative_slope_init=0.2,
                param_dtype=with_requested_dtype(),
            ),
            "input_shapes": [("B", 3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["PRelu:Bx3x4"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class PReluPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for ``flax.nnx.PReLU`` → ONNX ``PRelu``."""

    _PRIM: ClassVar[Primitive] = PRELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue,
        negative_slope: jax.core.AbstractValue,
    ) -> jax.core.ShapedArray:
        del negative_slope
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRBuildContext", eqn: jax.core.JaxprEqn) -> None:
        x_var, slope_var = eqn.invars
        y_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("prelu_x"))
        slope_val = ctx.get_value_for_var(
            slope_var, name_hint=ctx.fresh_name("prelu_slope")
        )
        slope_val = cast_param_like(ctx, slope_val, x_val, name_hint="prelu_slope_cast")

        out_spec = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("prelu_out"))
        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("prelu_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            out_name = ctx.fresh_name("prelu_out")

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for PRelu lowering")

        result = builder.PRelu(
            x_val,
            slope_val,
            _outputs=[out_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif getattr(x_val, "type", None) is not None:
            result.type = x_val.type

        out_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(y_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("flax.nnx", "prelu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(nnx.PReLU, "__call__", cls._patch_call),
        ]

    @classmethod
    def _patch_call(cls, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(self: nnx.PReLU, inputs: Any) -> Any:
            slope_param = getattr(self, "negative_slope", None)
            if slope_param is None:
                return orig(self, inputs)
            slope_value = getattr(slope_param, "value", slope_param)
            input_dtype = getattr(inputs, "dtype", None)
            if input_dtype is not None:
                slope = jnp.asarray(slope_value, dtype=input_dtype)
            else:
                slope = jnp.asarray(slope_value)
            return cls._PRIM.bind(inputs, slope)

        return wrapped


@PReluPlugin._PRIM.def_impl
def _prelu_impl(inputs: Any, negative_slope: Any) -> Any:
    x = jnp.asarray(inputs)
    slope = jnp.asarray(negative_slope, dtype=x.dtype)
    return jnp.where(x >= 0, x, slope * x)


def _prelu_batch_rule(
    args: Sequence[Any],
    dims: Sequence[Any],
    **params: Any,
) -> Any:
    from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat

    return broadcast_batcher_compat(PReluPlugin._PRIM, args, dims, **params)


batching.primitive_batchers[PReluPlugin._PRIM] = _prelu_batch_rule
