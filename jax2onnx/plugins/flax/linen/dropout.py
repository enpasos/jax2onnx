# jax2onnx/plugins/flax/linen/dropout.py

from __future__ import annotations

from typing import Callable, ClassVar

from flax import linen as nn
from jax.extend.core import Primitive

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.flax.nnx import dropout as nnx_dropout
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)


class _DropoutWithDefault(nn.Module):
    rate: float
    deterministic_default: bool = True

    @nn.compact
    def __call__(self, x, *, deterministic: bool | None = None):
        det = self.deterministic_default if deterministic is None else deterministic
        return nn.Dropout(rate=self.rate)(x, deterministic=det)


@register_primitive(
    jaxpr_primitive="linen.dropout",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.Dropout",
    onnx=[
        {
            "component": "Dropout",
            "doc": "https://onnx.ai/onnx/operators/onnx__Dropout.html",
        }
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="dropout",
    testcases=[
        {
            "testcase": "dropout_init_params",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Dropout,
                input_shape=(1, 10),
                dtype=with_requested_dtype(),
                rate=0.5,
                deterministic=True,
                rngs=with_rng_seed(5),
            ),
            "input_shapes": [("B", 10)],
            "post_check_onnx_graph": nnx_dropout.post_check_onnx_graph_init,
        },
        {
            "testcase": "dropout_call_params",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=_DropoutWithDefault,
                input_shape=(1, 10),
                dtype=with_requested_dtype(),
                rate=0.5,
                rngs=with_rng_seed(5),
            ),
            "input_shapes": [("B", 10)],
            "input_params": {"deterministic": True},
            "post_check_onnx_graph": nnx_dropout.post_check_onnx_graph,
        },
    ],
)
class DropoutPlugin(nnx_dropout.DropoutPlugin):
    """IR-only plugin for flax.linen.Dropout."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.dropout")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None

    @staticmethod
    def _dropout(x, deterministic, *, rate, call_time):
        return DropoutPlugin._PRIM.bind(
            x, deterministic, rate=rate, call_time=call_time
        )

    @staticmethod
    def _make_patch(orig_fn: Callable):
        DropoutPlugin._ORIGINAL_CALL = orig_fn
        prim = DropoutPlugin._PRIM

        def patched(self, x, deterministic=None, rng=None):
            try:
                det = nn.merge_param("deterministic", self.deterministic, deterministic)
            except ValueError:
                return orig_fn(self, x, deterministic=deterministic, rng=rng)
            call_time = deterministic is not None
            return prim.bind(x, det, rate=float(self.rate), call_time=call_time)

        return patched

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.linen", "dropout_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.linen.Dropout",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]


@DropoutPlugin._PRIM.def_impl
def _impl(x, deterministic, *, rate, call_time=False):
    del deterministic, rate, call_time
    return x
