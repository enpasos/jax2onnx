from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import onnx_ir as ir
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG2
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_EXPECT_IDENTITY = EG2(["Identity"], mode="all", search_functions=False)


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
            "post_check_onnx_graph": _EXPECT_IDENTITY,
        },
        {
            "testcase": "eqx_identity_symbolic_batch",
            "callable": eqx.nn.Identity(),
            "input_shapes": [("B", 32)],
            "post_check_onnx_graph": _EXPECT_IDENTITY,
        },
    ],
)
class IdentityPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.identity")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x):
        return x

    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("identity_in"))
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("identity_out")
        )
        node = ir.Node(
            op_type="Identity",
            domain="",
            inputs=[x_val],
            outputs=[out_val],
            name=ctx.fresh_name("Identity"),
        )
        ctx.add_node(node)
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if x_shape:
            _stamp_type_and_shape(out_val, x_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
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
    def _patch_call(orig):
        del orig

        def wrapped(self, x, *, key=None):
            del key
            return IdentityPlugin._PRIM.bind(x)

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(lambda x: cls.abstract_eval(x))
            cls._ABSTRACT_EVAL_BOUND = True


@IdentityPlugin._PRIM.def_impl
def _identity_impl(x):
    return x


def _identity_batch_rule(batched_args, batch_dims):
    (x,) = batched_args
    (bd,) = batch_dims
    return IdentityPlugin._PRIM.bind(x), bd


batching.primitive_batchers[IdentityPlugin._PRIM] = _identity_batch_rule
