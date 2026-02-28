# jax2onnx/plugins/equinox/eqx/nn/prelu.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, ClassVar, Final

import equinox as eqx
import jax.core as jax_core
import jax.numpy as jnp
import onnx_ir as ir
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


EQX_PRELU_PRIM: Final[Primitive] = Primitive("eqx.nn.prelu")
EQX_PRELU_PRIM.multiple_results = False

_EQX_PRELU_DEFAULT: Final[eqx.nn.PReLU] = eqx.nn.PReLU()
_EQX_PRELU_CHANNELWISE: Final[eqx.nn.PReLU] = eqx.nn.PReLU(
    init_alpha=jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32)
)


@register_primitive(
    jaxpr_primitive=EQX_PRELU_PRIM.name,
    jax_doc="https://docs.kidger.site/equinox/api/nn/activations/#equinox.nn.PReLU",
    onnx=[
        {"component": "PRelu", "doc": "https://onnx.ai/onnx/operators/onnx__PRelu.html"}
    ],
    since="0.12.5",
    context="primitives.eqx",
    component="prelu",
    testcases=[
        {
            "testcase": "eqx_prelu_default",
            "callable": _EQX_PRELU_DEFAULT,
            "input_shapes": [("B", 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["PRelu:Bx5"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_prelu_channelwise",
            "callable": _EQX_PRELU_CHANNELWISE,
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["PRelu:2x3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class PReluPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for ``equinox.nn.PReLU`` -> ONNX ``PRelu``."""

    _PRIM: ClassVar[Primitive] = EQX_PRELU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax_core.AbstractValue,
        negative_slope: jax_core.AbstractValue,
    ) -> jax_core.ShapedArray:
        del negative_slope
        return jax_core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax_core.JaxprEqn) -> None:
        x_var, slope_var = eqn.invars
        y_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("eqx_prelu_x"))
        slope_val = ctx.get_value_for_var(
            slope_var, name_hint=ctx.fresh_name("eqx_prelu_slope")
        )
        slope_val = cast_param_like(
            ctx, slope_val, x_val, name_hint="eqx_prelu_slope_cast"
        )

        out_spec = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("eqx_prelu_y"))
        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("eqx_prelu_y")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            out_name = ctx.fresh_name("eqx_prelu_y")

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for Eqx PReLU")

        result = builder.PRelu(
            x_val,
            slope_val,
            _outputs=[out_name],
        )

        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif getattr(x_val, "type", None) is not None:
            result.type = x_val.type
        else:
            x_dtype = getattr(getattr(x_var, "aval", None), "dtype", None)
            if x_dtype is not None:
                ir_dtype = ir.DataType.from_numpy(x_dtype)
                result.type = ir.TensorType(ir_dtype)

        out_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(y_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("equinox.nn", "prelu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.PReLU",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def _patch_call(cls, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(self: eqx.nn.PReLU, x: Any) -> Any:
            slope = jnp.asarray(self.negative_slope)
            x_dtype = getattr(x, "dtype", None)
            if x_dtype is not None:
                slope = jnp.asarray(slope, dtype=x_dtype)
            return cls._PRIM.bind(x, slope)

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
