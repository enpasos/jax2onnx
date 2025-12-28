# jax2onnx/plugins/flax/linen/dense_general.py

from __future__ import annotations

from typing import Callable, ClassVar, Final, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import linen as nn
from flax.linen import linear as linen_linear

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.flax.nnx import linear_general as nnx_linear_general
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)

EXPECT_GEMM_ONLY: Final = nnx_linear_general.EXPECT_GEMM_ONLY
EXPECT_RGR: Final = nnx_linear_general.EXPECT_RGR
EXPECT_DYNAMIC_RGR: Final = nnx_linear_general.EXPECT_DYNAMIC_RGR


def _is_trailing_axes(axis: Sequence[int], ndim: int) -> bool:
    if not axis:
        return False
    return tuple(axis) == tuple(range(ndim - len(axis), ndim))


@register_primitive(
    jaxpr_primitive="linen.dense_general",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.DenseGeneral",
    onnx=[
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Shape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Shape.html",
        },
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {
            "component": "CastLike",
            "doc": "https://onnx.ai/onnx/operators/onnx__CastLike.html",
        },
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="dense_general",
    testcases=[
        {
            "testcase": "dense_general_basic",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.DenseGeneral,
                input_shape=(1, 32),
                features=64,
                axis=-1,
                dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 32)],
            "expected_output_shapes": [("B", 64)],
            "post_check_onnx_graph": EXPECT_GEMM_ONLY,
        },
        {
            "testcase": "dense_general_multi_out",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.DenseGeneral,
                input_shape=(1, 256),
                features=(8, 32),
                axis=-1,
                dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 4, 256)],
            "expected_output_shapes": [(2, 4, 8, 32)],
            "post_check_onnx_graph": EXPECT_RGR,
        },
        {
            "testcase": "dense_general_contract_last_two",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.DenseGeneral,
                input_shape=(1, 4, 8, 32),
                features=256,
                axis=(-2, -1),
                dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 4, 8, 32)],
            "expected_output_shapes": [(2, 4, 256)],
            "post_check_onnx_graph": EXPECT_RGR,
        },
        {
            "testcase": "dense_general_dynamic_batch",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.DenseGeneral,
                input_shape=(1, 10, 128),
                features=64,
                axis=-1,
                dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 10, 128)],
            "expected_output_shapes": [("B", 10, 64)],
            "run_only_dynamic": True,
            "post_check_onnx_graph": EXPECT_DYNAMIC_RGR,
        },
        {
            "testcase": "dense_general_no_bias",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.DenseGeneral,
                input_shape=(1, 32),
                features=64,
                axis=-1,
                use_bias=False,
                dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 32)],
            "expected_output_shapes": [(3, 64)],
            "post_check_onnx_graph": EXPECT_GEMM_ONLY,
        },
    ],
)
class DenseGeneralPlugin(nnx_linear_general.LinearGeneralPlugin):
    """IR-only plugin for flax.linen.DenseGeneral → ONNX Gemm/Reshape."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.dense_general")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, kernel, bias, *, dimension_numbers):
        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
        lhs_contract = tuple(lhs_contract)
        rhs_contract = tuple(rhs_contract)
        lhs_batch = tuple(lhs_batch)
        rhs_batch = tuple(rhs_batch)

        lhs_other = [
            i for i in range(x.ndim) if i not in lhs_contract and i not in lhs_batch
        ]
        rhs_other = [
            i
            for i in range(kernel.ndim)
            if i not in rhs_contract and i not in rhs_batch
        ]
        out_shape = tuple(x.shape[i] for i in lhs_batch)
        out_shape += tuple(x.shape[i] for i in lhs_other)
        out_shape += tuple(kernel.shape[i] for i in rhs_other)
        return jax.core.ShapedArray(out_shape, x.dtype)

    @staticmethod
    def _make_patch(orig_fn: Callable):
        DenseGeneralPlugin._ORIGINAL_CALL = orig_fn
        prim = DenseGeneralPlugin._PRIM

        def patched(self, inputs):
            scope = getattr(self, "scope", None)
            if scope is None or not hasattr(scope, "variables"):
                return orig_fn(self, inputs)

            variables = scope.variables()
            params = variables.get("params", {})
            kernel = params.get("kernel")
            if kernel is None:
                return orig_fn(self, inputs)

            if getattr(self, "dot_general_cls", None) is not None:
                return orig_fn(self, inputs)
            if getattr(self, "dot_general", None) is not None:
                return orig_fn(self, inputs)

            features = linen_linear._canonicalize_tuple(self.features)
            axis = linen_linear._canonicalize_tuple(self.axis)
            batch_dims = linen_linear._canonicalize_tuple(self.batch_dims)
            if batch_dims:
                max_dim = int(np.max(batch_dims))
                if set(batch_dims) != set(range(max_dim + 1)):
                    return orig_fn(self, inputs)

            ndim = inputs.ndim
            axis = linen_linear._normalize_axes(axis, ndim)
            batch_dims = linen_linear._normalize_axes(batch_dims, ndim)
            if batch_dims:
                return orig_fn(self, inputs)
            if not _is_trailing_axes(axis, ndim):
                return orig_fn(self, inputs)

            use_bias = bool(getattr(self, "use_bias", True))
            bias = params.get("bias") if use_bias else None
            if bias is None:
                bias = jnp.zeros(features, dtype=inputs.dtype)

            rhs_contract = tuple(range(len(axis)))
            dn = ((axis, rhs_contract), ((), ()))
            return prim.bind(inputs, kernel, bias, dimension_numbers=dn)

        return patched

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec(
                "flax.linen",
                "dense_general_p",
                cls._PRIM,
                delete_if_missing=True,
            ),
            MonkeyPatchSpec(
                target="flax.linen.DenseGeneral",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, kernel, bias, dimension_numbers=None: cls.abstract_eval(
                    x, kernel, bias, dimension_numbers=dimension_numbers
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


@DenseGeneralPlugin._PRIM.def_impl
def _impl(x, kernel, bias, *, dimension_numbers):
    y = jax.lax.dot_general(x, kernel, dimension_numbers=dimension_numbers)
    if bias is not None:
        y = y + bias
    return y
