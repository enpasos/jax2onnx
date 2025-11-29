# jax2onnx/plugins/flax/nnx/einsum.py

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Final, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax import core
from jax.extend.core import Primitive

from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.plugins.plugin_system import _IRBuildContext as IRBuildContext  # type: ignore


_EINSUM_MODULE_PRIM: Final[Primitive] = Primitive("nnx_einsum_module")
_EINSUM_MODULE_PRIM.multiple_results = False


class _MockParam:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, item):
        return self.value[item]


EXPECT_EINSUM_ONLY: Final = EG(
    [
        (
            "Einsum",
            {
                "counts": {
                    "Einsum": 1,
                    "Add": 0,
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
    jaxpr_primitive=_EINSUM_MODULE_PRIM.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Einsum",
    onnx=[
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        },
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    ],
    since="v0.4.2",
    context="primitives.nnx",
    component="einsum",
    testcases=[
        {
            "testcase": "einsum_module_with_bias",
            "callable": construct_and_call(
                nnx.Einsum,
                "nta,hab->nthb",
                (8, 2, 4),
                (8, 4),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(16, 11, 2)],
            "post_check_onnx_graph": EXPECT_EINSUM_WITH_BIAS,
        },
        {
            "testcase": "einsum_module_no_bias",
            "callable": construct_and_call(
                nnx.Einsum,
                "nta,hab->nthb",
                (8, 2, 4),
                None,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(16, 11, 2)],
            "post_check_onnx_graph": EXPECT_EINSUM_ONLY,
        },
    ],
)
class EinsumModulePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = _EINSUM_MODULE_PRIM
    _ORIG_CALL: ClassVar[Callable[[Any, Any], Any] | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x,
        kernel,
        *maybe_bias,
        einsum_str: str,
        use_bias_bool: bool,
        precision: Any | None = None,
        optimize: Any | None = None,
        preferred_element_type: Any | None = None,
        dtype: Any | None = None,
        param_dtype: Any | None = None,
    ):
        bias = maybe_bias[0] if maybe_bias else None

        def _shape_fn(x_arg, kernel_arg, bias_arg=None):
            result = jnp.einsum(
                einsum_str,
                x_arg,
                kernel_arg,
                precision=precision,
                optimize=optimize,
                preferred_element_type=preferred_element_type,
            )
            if use_bias_bool and bias_arg is not None:
                result = result + bias_arg
            return result

        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        kernel_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        if use_bias_bool and bias is not None:
            bias_spec = jax.ShapeDtypeStruct(bias.shape, bias.dtype)
            out_spec = jax.eval_shape(_shape_fn, x_spec, kernel_spec, bias_spec)
        else:
            out_spec = jax.eval_shape(_shape_fn, x_spec, kernel_spec)

        if not isinstance(out_spec, jax.ShapeDtypeStruct):
            leaves = jax.tree_util.tree_leaves(out_spec)
            if len(leaves) == 1 and isinstance(leaves[0], jax.ShapeDtypeStruct):
                out_spec = leaves[0]
            else:
                raise TypeError("Unexpected output from nnx.Einsum abstract eval")

        return core.ShapedArray(out_spec.shape, out_spec.dtype)

    def lower(self, ctx: "IRBuildContext", eqn):  # type: ignore[override]
        params = dict(getattr(eqn, "params", {}) or {})
        equation = params["einsum_str"]
        use_bias = bool(params.get("use_bias_bool", False))

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
        einsum_outputs = builder.Einsum(
            x_val,
            kernel_val,
            _outputs=[ctx.fresh_name("einsum_mid") if use_bias else out_name],
            equation=equation,
        )
        einsum_out = einsum_outputs
        spec_type = getattr(out_spec, "type", None)
        if spec_type is not None:
            einsum_out.type = spec_type
        _stamp_type_and_shape(einsum_out, out_shape)
        _ensure_value_metadata(ctx, einsum_out)

        if use_bias and bias_var is not None:
            bias_val = ctx.get_value_for_var(
                bias_var, name_hint=ctx.fresh_name("einsum_bias")
            )
            bias_val = cast_param_like(
                ctx, bias_val, x_val, name_hint="einsum_bias_cast"
            )
            bias_shape = tuple(getattr(getattr(bias_var, "aval", None), "shape", ()))
            _stamp_type_and_shape(bias_val, bias_shape)
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

    @classmethod
    def binding_specs(cls):
        def _make_patch(orig):
            cls._ORIG_CALL = orig

            def _patched(self: nnx.Einsum, x):
                kernel = self.kernel.value
                bias = self.bias.value if self.bias is not None else None
                operands = [x, kernel]
                if bias is not None:
                    operands.append(bias)
                return cls._PRIM.bind(
                    *operands,
                    einsum_str=str(self.einsum_str),
                    use_bias_bool=bias is not None,
                    precision=getattr(self, "precision", None),
                    dtype=getattr(self, "dtype", None),
                    param_dtype=getattr(self, "param_dtype", None),
                )

            return _patched

        return [
            AssignSpec(
                "flax.nnx", "einsum_module_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="flax.nnx.Einsum",
                attr="__call__",
                make_value=_make_patch,
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


def _runtime_einsum(
    x,
    kernel,
    bias: Optional[Any],
    *,
    einsum_str: str,
    precision: Any | None,
    optimize: Any | None,
    preferred_element_type: Any | None,
    dtype: Any | None,
    param_dtype: Any | None,
    orig_call: Callable[[Any, Any], Any] | None,
):
    if orig_call is None:
        raise RuntimeError("Original nnx.Einsum.__call__ not captured")

    def infer_bias_shape(self, x_shape, einsum_output_shape, input_for_bias=None):
        return getattr(self, "bias_shape", None)

    sanitized_bias_shape = getattr(bias, "shape", None) if bias is not None else None

    dummy = SimpleNamespace(
        kernel=_MockParam(kernel),
        bias=_MockParam(bias) if bias is not None else None,
        einsum_str=einsum_str,
        _einsum_str_check=lambda s: None,
        _infer_broadcasted_bias_shape=infer_bias_shape,
        precision=precision,
        dtype=dtype,
        param_dtype=param_dtype,
        bias_init=None,
        kernel_init=None,
        kernel_shape=getattr(kernel, "shape", None),
        bias_shape=sanitized_bias_shape,
        promote_dtype=lambda arr, **kw: arr,
        einsum_op=jnp.einsum,
        use_bias=bias is not None,
        optimize=optimize,
        preferred_element_type=preferred_element_type,
    )
    return orig_call(dummy, x)


@EinsumModulePlugin._PRIM.def_impl
def _einsum_impl(
    x,
    kernel,
    *maybe_bias,
    einsum_str: str,
    use_bias_bool: bool,
    precision: Any | None = None,
    optimize: Any | None = None,
    preferred_element_type: Any | None = None,
    dtype: Any | None = None,
    param_dtype: Any | None = None,
):
    bias = maybe_bias[0] if maybe_bias else None
    return _runtime_einsum(
        x,
        kernel,
        bias if use_bias_bool else None,
        einsum_str=einsum_str,
        precision=precision,
        optimize=optimize,
        preferred_element_type=preferred_element_type,
        dtype=dtype,
        param_dtype=param_dtype,
        orig_call=EinsumModulePlugin._ORIG_CALL,
    )


EinsumModulePlugin.ensure_abstract_eval_bound()
