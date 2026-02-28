# jax2onnx/plugins/jax/nn/scaled_dot_general.py

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, ClassVar, Final

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.lax.dot_general import DotGeneralPlugin
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_SDG_PRIM: Final[Primitive] = Primitive("jax.nn.scaled_dot_general")
_SDG_PRIM.multiple_results = False
_JAX_SDG_ORIG: Final = jax.nn.scaled_dot_general


def _normalize_dimension_numbers(
    dimension_numbers,
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]
]:
    (contract, batch) = dimension_numbers
    (lhs_contract, rhs_contract) = contract
    (lhs_batch, rhs_batch) = batch
    return (
        (
            tuple(int(x) for x in lhs_contract),
            tuple(int(x) for x in rhs_contract),
        ),
        (
            tuple(int(x) for x in lhs_batch),
            tuple(int(x) for x in rhs_batch),
        ),
    )


@register_primitive(
    jaxpr_primitive=_SDG_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.scaled_dot_general.html",
    onnx=[
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        },
    ],
    since="0.12.1",
    context="primitives.nn",
    component="scaled_dot_general",
    testcases=[
        {
            "testcase": "jaxnn_scaled_dot_general_basic",
            "callable": lambda lhs, rhs: jax.nn.scaled_dot_general(
                lhs,
                rhs,
                dimension_numbers=(((1,), (0,)), ((), ())),
            ),
            "input_shapes": [(2, 3), (3, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Gemm:2x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_scaled_dot_general_batched",
            "callable": lambda lhs, rhs: jax.nn.scaled_dot_general(
                lhs,
                rhs,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            ),
            "input_shapes": [(2, 3, 4), (2, 5, 4)],
            "post_check_onnx_graph": EG(
                ["Transpose:2x4x5 -> MatMul:2x3x5"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ScaledDotGeneralPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.scaled_dot_general`` (configs=None) via ``lax.dot_general``."""

    _PRIM: ClassVar[Primitive] = _SDG_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        lhs: jax.core.AbstractValue,
        rhs: jax.core.AbstractValue,
        *,
        dimension_numbers,
        preferred_element_type,
    ) -> jax.core.ShapedArray:
        lhs_spec = jax.ShapeDtypeStruct(lhs.shape, lhs.dtype)
        rhs_spec = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
        out = jax.eval_shape(
            lambda lhs_in, rhs_in: _JAX_SDG_ORIG(
                lhs_in,
                rhs_in,
                dimension_numbers=dimension_numbers,
                preferred_element_type=preferred_element_type,
                configs=None,
                implementation=None,
            ),
            lhs_spec,
            rhs_spec,
        )
        return jax.core.ShapedArray(out.shape, out.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        dimension_numbers = params.get("dimension_numbers")
        if dimension_numbers is None:
            raise ValueError("scaled_dot_general lowering requires dimension_numbers")

        proxy_eqn = SimpleNamespace(
            invars=eqn.invars,
            outvars=eqn.outvars,
            params={"dimension_numbers": dimension_numbers},
        )
        DotGeneralPlugin().lower(ctx, proxy_eqn)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.nn.scaled_dot_general not found")

            def _patched(
                lhs,
                rhs,
                dimension_numbers,
                preferred_element_type=np.float32,
                configs=None,
                implementation=None,
            ):
                if configs is not None or implementation is not None:
                    return orig(
                        lhs,
                        rhs,
                        dimension_numbers,
                        preferred_element_type=preferred_element_type,
                        configs=configs,
                        implementation=implementation,
                    )

                try:
                    dn = _normalize_dimension_numbers(dimension_numbers)
                except Exception:
                    return orig(
                        lhs,
                        rhs,
                        dimension_numbers,
                        preferred_element_type=preferred_element_type,
                        configs=None,
                        implementation=None,
                    )

                lhs_arr = jnp.asarray(lhs)
                rhs_arr = jnp.asarray(rhs)
                try:
                    pref_dtype = np.dtype(preferred_element_type)
                except Exception:
                    return orig(
                        lhs_arr,
                        rhs_arr,
                        dimension_numbers,
                        preferred_element_type=preferred_element_type,
                        configs=None,
                        implementation=None,
                    )

                if lhs_arr.dtype != pref_dtype:
                    lhs_arr = lhs_arr.astype(pref_dtype)
                if rhs_arr.dtype != pref_dtype:
                    rhs_arr = rhs_arr.astype(pref_dtype)

                return cls._PRIM.bind(
                    lhs_arr,
                    rhs_arr,
                    dimension_numbers=dn,
                    preferred_element_type=pref_dtype,
                )

            return _patched

        return [
            AssignSpec(
                "jax.nn", "scaled_dot_general_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="scaled_dot_general",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@ScaledDotGeneralPlugin._PRIM.def_impl
def _scaled_dot_general_impl(
    lhs: ArrayLike,
    rhs: ArrayLike,
    *,
    dimension_numbers,
    preferred_element_type,
) -> ArrayLike:
    return _JAX_SDG_ORIG(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        preferred_element_type=preferred_element_type,
        configs=None,
        implementation=None,
    )


dot_batch_rule: Callable[..., object] | None = batching.fancy_primitive_batchers.get(
    jax.lax.dot_general_p
)
if dot_batch_rule is not None:
    batching.fancy_primitive_batchers[ScaledDotGeneralPlugin._PRIM] = dot_batch_rule

register_jvp_via_jax_jvp(ScaledDotGeneralPlugin._PRIM, _scaled_dot_general_impl)
