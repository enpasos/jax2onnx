# jax2onnx/plugins/jax/numpy/pow.py

from __future__ import annotations

import dataclasses
import operator
from typing import Any, Callable, ClassVar, Final

import jax
from jax2onnx._compat.jax import (
    AbstractValue,
    JaxprEqn,
    Primitive,
    ShapedArray,
    batching,
)
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.lax.pow import lower_pow
from jax2onnx.plugins.jax.numpy._common import (
    abstract_eval_via_jax,
    get_orig_impl,
    jnp_binding_specs,
    make_jnp_primitive,
)
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_native_primitive,
    with_requested_dtype,
)


def _power_fixture(*, dtype: Any) -> Callable[[], jax.Array]:
    values = np.array([1.0, 2.0, 3.0], dtype=dtype)

    def call() -> jax.Array:
        return jnp.power(values, 2.0)

    return call


def _pow_fixture(*, dtype: Any) -> Callable[[], jax.Array]:
    values = np.array([1.0, 2.0, 3.0], dtype=dtype)

    def call() -> jax.Array:
        return jnp.pow(values, 3.0)

    return call


class _BaseJnpPow(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive]
    _FUNC_NAME: ClassVar[str]

    @classmethod
    def abstract_eval(cls, x: AbstractValue, y: AbstractValue) -> ShapedArray:
        original = get_orig_impl(cls._PRIM, cls._FUNC_NAME)
        return abstract_eval_via_jax(original, x, y)

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        lower_pow(ctx, eqn)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            (
                _index_concrete_exponent(spec)
                if isinstance(spec, MonkeyPatchSpec)
                else spec
            )
            for spec in jnp_binding_specs(cls._PRIM, cls._FUNC_NAME)
        ]


def _index_concrete_exponent(spec: MonkeyPatchSpec) -> MonkeyPatchSpec:
    """Bind a concrete integer exponent as a Python int, as ``jnp.power`` does.

    JAX lowers such exponents with ``lax.integer_pow``, which keeps the base
    dtype. Binding a typed scalar such as ``np.int64(2)`` instead would promote
    an int32 base to int64 under x64.
    """

    def make_value(orig: Any) -> Callable[[ArrayLike, ArrayLike], Any]:
        bound = spec.make_value(orig)

        def patched(x1: ArrayLike, x2: ArrayLike) -> Any:
            try:
                x2 = operator.index(x2)
            except TypeError:
                pass
            return bound(x1, x2)

        return patched

    return dataclasses.replace(spec, make_value=make_value)


_POWER_PRIM: Final = make_jnp_primitive("jax.numpy.power")


@register_native_primitive(
    jaxpr_primitive=_POWER_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.power.html",
    onnx=[{"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"}],
    since="0.8.0",
    context="primitives.jnp",
    component="power",
    testcases=[
        {
            "testcase": "jnp_power_vector",
            "callable": lambda x, y: jnp.power(x, y),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Pow:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pow_jnp_power",
            "callable": construct_and_call(
                _power_fixture, dtype=with_requested_dtype()
            ),
            "input_values": [],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [with_requested_dtype()],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 2.0}},
                        "path": "Pow:3",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            # A weakly typed int base takes the exponent's float dtype, as in JAX.
            "testcase": "jnp_power_int_base_float_exponent",
            "callable": lambda x: jnp.power(10000, x),
            "input_values": [np.array([0.0, -0.25, -0.5, -0.75], dtype=np.float32)],
            "expected_output_dtypes": [with_requested_dtype()],
            "post_check_onnx_graph": EG(
                [{"path": "Cast -> Pow:4", "counts": {"Cast": 1}}],
                no_unused_inputs=True,
            ),
        },
        {
            # A concrete integer exponent keeps the base dtype, as in JAX, even
            # when x64 makes np.int64 wider than the base.
            "testcase": "jnp_power_int32_base_concrete_int64_exponent",
            "callable": lambda x: jnp.power(x, np.int64(2)),
            "input_values": [np.array([3, -2, 7], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Pow:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "power_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.power(y, 2.0))(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "power_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.power(y, 2.0)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpPowerPlugin(_BaseJnpPow):
    _PRIM: ClassVar = _POWER_PRIM
    _FUNC_NAME: ClassVar[str] = "power"


@JnpPowerPlugin._PRIM.def_impl
def _power_impl(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    orig = get_orig_impl(JnpPowerPlugin._PRIM, JnpPowerPlugin._FUNC_NAME)
    return orig(x, y)


register_jvp_via_jax_jvp(JnpPowerPlugin._PRIM, _power_impl)


_POW_PRIM: Final = make_jnp_primitive("jax.numpy.pow")


@register_native_primitive(
    jaxpr_primitive=_POW_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.power.html",
    onnx=[{"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"}],
    since="0.8.0",
    context="primitives.jnp",
    component="pow",
    testcases=[
        {
            "testcase": "jnp_pow_vector",
            "callable": lambda x, y: jnp.pow(x, y),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Pow:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pow_jnp_pow",
            "callable": construct_and_call(_pow_fixture, dtype=with_requested_dtype()),
            "input_values": [],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [with_requested_dtype()],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 3.0}},
                        "path": "Pow:3",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pow_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.pow(y, 2.0))(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "pow_grad_issue_batch_diff_rules",
            "callable": lambda x: jax.grad(lambda y: jnp.sum(jnp.pow(y, 2.0)))(x),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpPowPlugin(_BaseJnpPow):
    _PRIM: ClassVar = _POW_PRIM
    _FUNC_NAME: ClassVar[str] = "pow"


@JnpPowPlugin._PRIM.def_impl
def _pow_impl(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    orig = get_orig_impl(JnpPowPlugin._PRIM, JnpPowPlugin._FUNC_NAME)
    return orig(x, y)


register_jvp_via_jax_jvp(JnpPowPlugin._PRIM, _pow_impl)


def _make_pow_batch_rule(prim: Primitive) -> Callable[..., Any]:
    def _batch_rule(args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any) -> Any:
        return broadcast_batcher_compat(prim, args, dims, **params)

    return _batch_rule


batching.primitive_batchers[JnpPowerPlugin._PRIM] = _make_pow_batch_rule(
    JnpPowerPlugin._PRIM
)
batching.primitive_batchers[JnpPowPlugin._PRIM] = _make_pow_batch_rule(
    JnpPowPlugin._PRIM
)
