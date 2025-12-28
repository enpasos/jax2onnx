# jax2onnx/plugins/flax/linen/einsum.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final
import jax
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive
from flax import linen as nn
from flax.linen import module as linen_module
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)


EXPECT_EINSUM_ONLY: Final = EG(
    [
        (
            "Einsum",
            {
                "counts": {
                    "Einsum": 1,
                    "Add": 0,
                    "Reshape": 0,
                }
            },
        )
    ]
)

EXPECT_EINSUM_WITH_BIAS: Final = EG(
    [
        (
            "Einsum -> Add",
            {
                "counts": {
                    "Einsum": 1,
                    "Add": 1,
                }
            },
        )
    ]
)


@register_primitive(
    jaxpr_primitive="linen.einsum",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.Einsum",
    onnx=[
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        },
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="einsum",
    testcases=[
        {
            "testcase": "einsum_with_bias",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Einsum,
                input_shape=(1, 11, 2),
                shape=(8, 2, 4),
                einsum_str="nta,hab->nthb",
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(16, 11, 2)],
            "expected_output_shapes": [(16, 11, 8, 4)],
            "post_check_onnx_graph": EXPECT_EINSUM_WITH_BIAS,
        },
        {
            "testcase": "einsum_no_bias",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Einsum,
                input_shape=(1, 11, 2),
                shape=(8, 2, 4),
                einsum_str="nta,hab->nthb",
                use_bias=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(16, 11, 2)],
            "expected_output_shapes": [(16, 11, 8, 4)],
            "post_check_onnx_graph": EXPECT_EINSUM_ONLY,
        },
    ],
)
class EinsumPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("linen.einsum")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x,
        kernel,
        *maybe_bias,
        einsum_str: str,
        use_bias: bool,
        broadcasted_bias_shape: tuple[int, ...] | None = None,
        precision: Any | None = None,
        preferred_element_type: Any | None = None,
    ):
        bias = maybe_bias[0] if maybe_bias else None

        def _shape_fn(x_arg, kernel_arg, bias_arg=None):
            y = jnp.einsum(
                einsum_str,
                x_arg,
                kernel_arg,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )
            if use_bias and bias_arg is not None:
                if broadcasted_bias_shape is not None:
                    bias_arg = jnp.reshape(bias_arg, broadcasted_bias_shape)
                y = y + bias_arg
            return y

        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        kernel_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        if use_bias and bias is not None:
            bias_spec = jax.ShapeDtypeStruct(bias.shape, bias.dtype)
            out_spec = jax.eval_shape(_shape_fn, x_spec, kernel_spec, bias_spec)
        else:
            out_spec = jax.eval_shape(_shape_fn, x_spec, kernel_spec)

        if not isinstance(out_spec, jax.ShapeDtypeStruct):
            leaves = jax.tree_util.tree_leaves(out_spec)
            if len(leaves) == 1 and isinstance(leaves[0], jax.ShapeDtypeStruct):
                out_spec = leaves[0]
            else:
                raise TypeError("Unexpected output from linen.Einsum abstract eval")

        return core.ShapedArray(out_spec.shape, out_spec.dtype)

    def lower(self, ctx: Any, eqn):
        params = dict(getattr(eqn, "params", {}) or {})
        equation = params["einsum_str"]
        use_bias = bool(params.get("use_bias", False))
        broadcasted_bias_shape = params.get("broadcasted_bias_shape")

        invars = list(eqn.invars)
        out_var = eqn.outvars[0]

        x_var = invars[0]
        kernel_var = invars[1]
        bias_var = invars[2] if use_bias and len(invars) > 2 else None

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("einsum_x"))
        kernel_val = ctx.get_value_for_var(
            kernel_var, name_hint=ctx.fresh_name("einsum_kernel")
        )
        kernel_val = cast_param_like(
            ctx, kernel_val, x_val, name_hint="einsum_kernel_cast"
        )
        kernel_shape = tuple(getattr(getattr(kernel_var, "aval", None), "shape", ()))
        _stamp_type_and_shape(kernel_val, kernel_shape)
        _ensure_value_metadata(ctx, kernel_val)

        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("einsum_out")
        )
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for Einsum lowering")

        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("einsum_out")
        einsum_out = builder.Einsum(
            x_val,
            kernel_val,
            _outputs=[ctx.fresh_name("einsum_mid") if use_bias else out_name],
            equation=equation,
        )
        spec_type = getattr(out_spec, "type", None)
        if spec_type is not None:
            einsum_out.type = spec_type
        else:
            inferred_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
            if inferred_dtype is not None:
                einsum_out.type = ir.TensorType(inferred_dtype)
        _stamp_type_and_shape(einsum_out, out_shape)
        _ensure_value_metadata(ctx, einsum_out)

        if use_bias and bias_var is not None:
            bias_val = ctx.get_value_for_var(
                bias_var, name_hint=ctx.fresh_name("einsum_bias")
            )
            bias_val = cast_param_like(ctx, bias_val, x_val, "einsum_bias_cast")
            bias_shape = tuple(getattr(getattr(bias_var, "aval", None), "shape", ()))
            _stamp_type_and_shape(bias_val, bias_shape)
            _ensure_value_metadata(ctx, bias_val)

            if broadcasted_bias_shape is not None:
                broadcasted = tuple(int(d) for d in broadcasted_bias_shape)
                if broadcasted != bias_shape:
                    shape_const = _const_i64(
                        ctx, list(broadcasted), name_hint="einsum_bias_shape"
                    )
                    bias_val = builder.Reshape(
                        bias_val,
                        shape_const,
                        _outputs=[ctx.fresh_name("einsum_bias_reshape")],
                    )
                    if getattr(x_val, "type", None) is not None:
                        bias_val.type = x_val.type
                    _stamp_type_and_shape(bias_val, broadcasted)
                    _ensure_value_metadata(ctx, bias_val)

            result = builder.Add(
                einsum_out,
                bias_val,
                _outputs=[out_name],
            )
            if spec_type is not None:
                result.type = spec_type
            elif getattr(einsum_out, "type", None) is not None:
                result.type = einsum_out.type
            _stamp_type_and_shape(result, out_shape)
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)
        else:
            if getattr(einsum_out, "name", None) != out_name:
                einsum_out.name = out_name
            if spec_type is not None:
                einsum_out.type = spec_type
            ctx.bind_value_for_var(out_var, einsum_out)

    @staticmethod
    def _make_patch(orig_fn: Callable):
        EinsumPlugin._ORIGINAL_CALL = orig_fn
        prim = EinsumPlugin._PRIM

        def patched(self, inputs, einsum_str=None):
            scope = getattr(self, "scope", None)
            if scope is None or not hasattr(scope, "variables"):
                return orig_fn(self, inputs, einsum_str)

            variables = scope.variables()
            params = variables.get("params", {})
            kernel = params.get("kernel")
            bias = params.get("bias") if self.use_bias else None
            if kernel is None:
                return orig_fn(self, inputs, einsum_str)

            einsum_str = linen_module.merge_param(
                "einsum_str", self.einsum_str, einsum_str
            )
            einsum_str = str(einsum_str).replace(" ", "")
            if "->" not in einsum_str or einsum_str.count(",") != 1:
                raise ValueError(
                    '`einsum_str` equation must include "->" and exactly one comma.'
                )

            inputs, kernel, bias = self.promote_dtype(
                inputs,
                kernel,
                bias,
                dtype=getattr(self, "dtype", None),
            )

            broadcasted_bias_shape = None
            if self.use_bias:
                if bias is None:
                    return orig_fn(self, inputs, einsum_str)
                _, broadcasted_bias_shape = self._get_bias_shape(
                    einsum_str, inputs, kernel
                )
                broadcasted_bias_shape = tuple(int(d) for d in broadcasted_bias_shape)

            operands = [inputs, kernel]
            if self.use_bias and bias is not None:
                operands.append(bias)

            return prim.bind(
                *operands,
                einsum_str=einsum_str,
                use_bias=bool(self.use_bias),
                broadcasted_bias_shape=broadcasted_bias_shape,
                precision=getattr(self, "precision", None),
                preferred_element_type=getattr(self, "preferred_element_type", None),
            )

        return patched

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.linen", "einsum_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.linen.Einsum",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@EinsumPlugin._PRIM.def_impl
def _impl(
    x,
    kernel,
    *maybe_bias,
    einsum_str: str,
    use_bias: bool,
    broadcasted_bias_shape: tuple[int, ...] | None = None,
    precision: Any | None = None,
    preferred_element_type: Any | None = None,
):
    bias = maybe_bias[0] if maybe_bias else None
    y = jnp.einsum(
        einsum_str,
        x,
        kernel,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    if use_bias and bias is not None:
        if broadcasted_bias_shape is not None:
            bias = jnp.reshape(bias, broadcasted_bias_shape)
        y = y + bias
    return y
