# jax2onnx/plugins/equinox/eqx/nn/lambda.py

from __future__ import annotations

from functools import partial
from typing import Any, Callable, ClassVar, Final

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import onnx_ir as ir
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

_LAMBDA_IMPLS: Final[dict[str, Callable[[jax.Array], jax.Array]]] = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
}

_LAMBDA_ONNX_OPS: Final[dict[str, str]] = {
    "relu": "Relu",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
}


@register_primitive(
    jaxpr_primitive="eqx.nn.lambda",
    jax_doc="https://docs.kidger.site/equinox/api/nn/sequential/#equinox.nn.Lambda",
    onnx=[
        {"component": "Relu", "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html"},
        {"component": "Tanh", "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html"},
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        },
    ],
    since="0.12.3",
    context="primitives.eqx",
    component="lambda",
    testcases=[
        {
            "testcase": "eqx_lambda_relu",
            "callable": eqx.nn.Lambda(jax.nn.relu),
            "input_shapes": [(16,)],
            "post_check_onnx_graph": expect_graph(
                ["Relu:16"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_lambda_sigmoid",
            "callable": eqx.nn.Lambda(jax.nn.sigmoid),
            "input_shapes": [(8,)],
            "post_check_onnx_graph": expect_graph(
                ["Sigmoid:8"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class LambdaPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.lambda")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax_core.AbstractValue, *, op: str) -> jax_core.AbstractValue:
        if op not in _LAMBDA_IMPLS:
            raise NotImplementedError(f"Unsupported eqx.nn.Lambda op '{op}'.")
        return x

    def lower(self, ctx: LoweringContextProtocol, eqn: jax_core.JaxprEqn) -> None:
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for Eqx Lambda")

        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars
        op_name = str(eqn.params.get("op"))
        onnx_op = _LAMBDA_ONNX_OPS.get(op_name)
        if onnx_op is None:
            raise NotImplementedError(f"Unsupported eqx.nn.Lambda op '{op_name}'.")

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("eqx_lambda_x"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("eqx_lambda")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("eqx_lambda")
        method = getattr(builder, onnx_op, None)
        if callable(method):
            result = method(
                x_val,
                _outputs=[desired_name],
            )
        else:
            result = builder.op(  # pragma: no cover - fallback API
                onnx_op,
                [x_val],
                {},
                name=desired_name,
            )

        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            x_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
            if x_dtype is not None:
                result.type = ir.TensorType(x_dtype)

        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
            _stamp_type_and_shape(result, x_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("equinox.nn", "lambda_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.Lambda",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _resolve_op(fn: Any) -> str | None:
        if fn is jax.nn.relu:
            return "relu"
        if fn is jax.nn.sigmoid:
            return "sigmoid"
        if fn is jnp.tanh:
            return "tanh"

        if isinstance(fn, partial):
            base = fn.func
            if base is jax.nn.relu:
                return "relu"
            if base is jax.nn.sigmoid:
                return "sigmoid"
            if base is jnp.tanh:
                return "tanh"

        name = getattr(fn, "__name__", "")
        if name in _LAMBDA_IMPLS:
            return str(name)
        return None

    @staticmethod
    def _patch_call(
        orig: Callable[..., jax.Array] | None,
    ) -> Callable[..., jax.Array]:
        def wrapped(
            self: eqx.nn.Lambda,
            x: jax.Array,
            *,
            key: jax.Array | None = None,
        ) -> jax.Array:
            del key
            op = LambdaPlugin._resolve_op(self.fn)
            if op is None:
                if orig is not None:
                    return orig(self, x)
                return self.fn(x)
            return LambdaPlugin._PRIM.bind(x, op=op)

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(lambda x, *, op: cls.abstract_eval(x, op=op))
            cls._ABSTRACT_EVAL_BOUND = True


@LambdaPlugin._PRIM.def_impl
def _lambda_impl(x: jax.Array, *, op: str) -> jax.Array:
    fn = _LAMBDA_IMPLS.get(op)
    if fn is None:
        raise NotImplementedError(f"Unsupported eqx.nn.Lambda op '{op}'.")
    return fn(x)


def _lambda_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    *,
    op: str,
) -> tuple[jax.Array, int | None]:
    (x,) = batched_args
    (x_bdim,) = batch_dims
    out = LambdaPlugin._PRIM.bind(x, op=op)
    return out, x_bdim


batching.primitive_batchers[LambdaPlugin._PRIM] = _lambda_batch_rule
