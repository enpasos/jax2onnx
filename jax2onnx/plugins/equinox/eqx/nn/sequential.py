# jax2onnx/plugins/equinox/eqx/nn/sequential.py

from __future__ import annotations

from functools import partial
from typing import Any, Callable, ClassVar, Final

import equinox as eqx
from equinox.nn import _sequential as eqx_sequential
import jax
import jax.core as jax_core
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _identity_activation(x: jax.Array) -> jax.Array:
    return x


SEQUENTIAL_PRIM: Final[Primitive] = Primitive("eqx.nn.sequential")
SEQUENTIAL_PRIM.multiple_results = False

_SEQUENTIAL_IMPLS: Final[dict[str, Callable[[jax.Array], jax.Array]]] = {
    "identity": _identity_activation,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jnp.tanh,
}

_SEQUENTIAL_ONNX_OPS: Final[dict[str, str]] = {
    "identity": "Identity",
    "relu": "Relu",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
}

_EQX_SEQ_RELU_TANH: Final[eqx.nn.Sequential] = eqx.nn.Sequential(
    [eqx.nn.Lambda(jax.nn.relu), eqx.nn.Lambda(jnp.tanh)]
)
_EQX_SEQ_IDENTITY_SIGMOID: Final[eqx.nn.Sequential] = eqx.nn.Sequential(
    [eqx.nn.Identity(), eqx.nn.Lambda(jax.nn.sigmoid)]
)


def _resolve_activation_fn(fn: Any) -> str | None:
    if isinstance(fn, partial):
        if fn.args or fn.keywords:
            return None
        fn = fn.func

    if fn is jax.nn.relu:
        return "relu"
    if fn is jax.nn.sigmoid:
        return "sigmoid"
    if fn is jnp.tanh:
        return "tanh"

    name = getattr(fn, "__name__", "")
    if name in {"relu", "sigmoid", "tanh"}:
        return str(name)
    return None


def _resolve_layer_op(layer: Any) -> str | None:
    if isinstance(layer, eqx.nn.Identity):
        return "identity"
    if isinstance(layer, eqx.nn.Lambda):
        return _resolve_activation_fn(getattr(layer, "fn", None))
    return None


def _resolve_layer_ops(layers: Any) -> tuple[str, ...] | None:
    try:
        layer_seq = tuple(layers)
    except TypeError:
        return None

    ops: list[str] = []
    for layer in layer_seq:
        op = _resolve_layer_op(layer)
        if op is None:
            return None
        ops.append(op)
    return tuple(ops)


@register_primitive(
    jaxpr_primitive=SEQUENTIAL_PRIM.name,
    jax_doc="https://docs.kidger.site/equinox/api/nn/sequential/#equinox.nn.Sequential",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        },
        {"component": "Relu", "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html"},
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        },
        {"component": "Tanh", "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html"},
    ],
    since="0.12.5",
    context="primitives.eqx",
    component="sequential",
    testcases=[
        {
            "testcase": "eqx_sequential_relu_tanh",
            "callable": _EQX_SEQ_RELU_TANH,
            "input_shapes": [("B", 8)],
            "post_check_onnx_graph": EG(
                ["Relu:Bx8 -> Tanh:Bx8"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_sequential_identity_sigmoid",
            "callable": _EQX_SEQ_IDENTITY_SIGMOID,
            "input_shapes": [(2, 6)],
            "post_check_onnx_graph": EG(
                ["Identity:2x6 -> Sigmoid:2x6"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SequentialPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for stateless ``equinox.nn.Sequential`` activation chains."""

    _PRIM: ClassVar[Primitive] = SEQUENTIAL_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax_core.AbstractValue,
        *,
        ops: tuple[str, ...],
    ) -> jax_core.AbstractValue:
        for op in ops:
            if op not in _SEQUENTIAL_IMPLS:
                raise NotImplementedError(f"Unsupported eqx.nn.Sequential op '{op}'.")
        return x

    def lower(self, ctx: LoweringContextProtocol, eqn: jax_core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars
        ops = tuple(str(op) for op in eqn.params.get("ops", ()))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("eqx_seq_x"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("eqx_seq_out")
        )

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for Eqx Sequential")

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        x_type = getattr(x_val, "type", None)
        current = x_val

        if not ops:
            identity_name = getattr(out_spec, "name", None) or ctx.fresh_name(
                "eqx_seq_identity"
            )
            current = builder.Identity(
                current,
                _outputs=[identity_name],
            )
            if x_type is not None:
                current.type = x_type
            _stamp_type_and_shape(current, x_shape)
            _ensure_value_metadata(ctx, current)

        for idx, op in enumerate(ops):
            onnx_op = _SEQUENTIAL_ONNX_OPS.get(op)
            if onnx_op is None:
                raise NotImplementedError(f"Unsupported eqx.nn.Sequential op '{op}'.")
            method = getattr(builder, onnx_op, None)
            if not callable(method):
                raise AttributeError(f"IR builder missing {onnx_op} for Eqx Sequential")

            is_last = idx == len(ops) - 1
            node_name = (
                getattr(out_spec, "name", None)
                if is_last and getattr(out_spec, "name", None)
                else ctx.fresh_name(f"eqx_seq_{op}")
            )
            current = method(
                current,
                _outputs=[node_name],
            )

            if is_last and getattr(out_spec, "type", None) is not None:
                current.type = out_spec.type
            elif x_type is not None:
                current.type = x_type

            _stamp_type_and_shape(current, x_shape)
            _ensure_value_metadata(ctx, current)

        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))
        if getattr(out_spec, "shape", None) is not None:
            current.shape = out_spec.shape
        else:
            _stamp_type_and_shape(current, out_shape)
        _ensure_value_metadata(ctx, current)
        ctx.bind_value_for_var(out_var, current)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("equinox.nn", "sequential_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.Sequential",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def _patch_call(cls, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(
            self: eqx.nn.Sequential,
            x: Any,
            state: Any = eqx_sequential.sentinel,
            *,
            key: jax.Array | None = None,
        ) -> Any:
            if state is not eqx_sequential.sentinel or key is not None:
                return orig(self, x, state=state, key=key)

            ops = _resolve_layer_ops(getattr(self, "layers", ()))
            if ops is None:
                return orig(self, x, state=state, key=key)
            return cls._PRIM.bind(x, ops=ops)

        return wrapped


@SequentialPlugin._PRIM.def_impl
def _sequential_impl(x: Any, *, ops: tuple[str, ...]) -> Any:
    out = x
    for op in ops:
        impl = _SEQUENTIAL_IMPLS.get(op)
        if impl is None:
            raise NotImplementedError(f"Unsupported eqx.nn.Sequential op '{op}'.")
        out = impl(out)
    return out


def _sequential_batch_rule(
    batched_args: tuple[Any, ...],
    batch_dims: tuple[int | None, ...],
    *,
    ops: tuple[str, ...],
) -> Any:
    from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat

    return broadcast_batcher_compat(
        SequentialPlugin._PRIM,
        batched_args,
        batch_dims,
        ops=ops,
    )


batching.primitive_batchers[SequentialPlugin._PRIM] = _sequential_batch_rule
