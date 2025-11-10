# jax2onnx/plugins/equinox/eqx/nn/identity.py

from __future__ import annotations

from typing import Callable, ClassVar

import equinox as eqx
import jax
import jax.core as jax_core
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive="eqx.nn.identity",
    jax_doc="https://docs.kidger.site/equinox/api/nn/linear/#equinox.nn.Identity",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.8.0",
    context="primitives.eqx",
    component="identity",
    testcases=[
        {
            "testcase": "eqx_identity_static",
            "callable": eqx.nn.Identity(),
            "input_shapes": [(10, 20)],
            "post_check_onnx_graph": expect_graph(["Identity:10x20"]),
        },
        {
            "testcase": "eqx_identity_symbolic_batch",
            "callable": eqx.nn.Identity(),
            "input_shapes": [("B", 32)],
            "post_check_onnx_graph": expect_graph(["Identity:Bx32"]),
        },
    ],
)
class IdentityPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.identity")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: jax_core.AbstractValue) -> jax_core.AbstractValue:
        return x

    def lower(self, ctx: LoweringContextProtocol, eqn: jax_core.JaxprEqn) -> None:
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for Eqx Identity lowering"
            )

        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("identity_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("identity_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Identity")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Identity")

        identity_val = builder.Identity(
            x_val,
            _outputs=[desired_name],
        )

        if getattr(out_spec, "type", None) is not None:
            identity_val.type = out_spec.type
        elif getattr(x_val, "type", None) is not None:
            identity_val.type = x_val.type

        if getattr(out_spec, "shape", None) is not None:
            identity_val.shape = out_spec.shape
        else:
            x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
            if x_shape:
                _stamp_type_and_shape(identity_val, x_shape)
        _ensure_value_metadata(ctx, identity_val)
        ctx.bind_value_for_var(out_var, identity_val)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("equinox.nn", "identity_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.Identity",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _patch_call(
        orig: Callable[..., jax.Array] | None,
    ) -> Callable[[eqx.nn.Identity, jax.Array], jax.Array]:
        del orig

        def wrapped(
            self: eqx.nn.Identity, x: jax.Array, *, key: jax.Array | None = None
        ) -> jax.Array:
            del key
            return IdentityPlugin._PRIM.bind(x)

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(lambda x: cls.abstract_eval(x))
            cls._ABSTRACT_EVAL_BOUND = True


@IdentityPlugin._PRIM.def_impl
def _identity_impl(x: jax.Array) -> jax.Array:
    return x


def _identity_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
) -> tuple[jax.Array, int | None]:
    (x,) = batched_args
    (bd,) = batch_dims
    return IdentityPlugin._PRIM.bind(x), bd


batching.primitive_batchers[IdentityPlugin._PRIM] = _identity_batch_rule
