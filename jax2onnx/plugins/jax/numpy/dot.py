# jax2onnx/plugins/jax/numpy/dot.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax._batching_utils import broadcast_batcher_compat
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_DOT_PRIM: Final = make_jnp_primitive("jax.numpy.dot")


def _dot_shape(
    a_shape: tuple[Any, ...],
    b_shape: tuple[Any, ...],
    a_dtype: Any,
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
    out_sharding: Any = None,
) -> tuple[tuple[Any, ...], Any]:
    spec_a = jax.ShapeDtypeStruct(a_shape, a_dtype)
    spec_b = jax.ShapeDtypeStruct(b_shape, a_dtype)
    orig = getattr(_DOT_PRIM, "__orig_impl__dot", jnp.dot)
    result = jax.eval_shape(
        lambda x, y: orig(
            x,
            y,
            precision=precision,
            preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        ),
        spec_a,
        spec_b,
    )
    return result.shape, result.dtype


@register_primitive(
    jaxpr_primitive=_DOT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dot.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        }
    ],
    since="0.12.4",
    context="primitives.jnp",
    component="dot",
    testcases=[
        {
            "testcase": "jnp_dot_vector_vector",
            "callable": lambda a, b: jnp.dot(a, b),
            "input_shapes": [(4,), (4,)],
            "post_check_onnx_graph": EG(["MatMul"], no_unused_inputs=True),
        },
        {
            "testcase": "jnp_dot_matrix_vector",
            "callable": lambda a, b: jnp.dot(a, b),
            "input_shapes": [(3, 4), (4,)],
            "post_check_onnx_graph": EG(["MatMul:3"], no_unused_inputs=True),
        },
        {
            "testcase": "jnp_dot_matrix_matrix",
            "callable": lambda a, b: jnp.dot(a, b),
            "input_shapes": [(3, 4), (4, 5)],
            "post_check_onnx_graph": EG(["MatMul:3x5"], no_unused_inputs=True),
        },
    ],
)
class JnpDotPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _DOT_PRIM
    _FUNC_NAME: ClassVar[str] = "dot"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        a: core.AbstractValue,
        b: core.AbstractValue,
        *,
        precision: Any = None,
        preferred_element_type: Any = None,
        out_sharding: Any = None,
    ) -> core.ShapedArray:
        shape, dtype = _dot_shape(
            a.shape,
            b.shape,
            a.dtype,
            precision=precision,
            preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        )
        return core.ShapedArray(shape, dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        a_var, b_var = eqn.invars
        out_var = eqn.outvars[0]

        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("dot_a"))
        b_val = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("dot_b"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("dot_out"))

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for dot lowering")

        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("MatMul")
        result = builder.MatMul(
            a_val,
            b_val,
            _outputs=[out_name],
        )

        out_type = getattr(out_spec, "type", None)
        if out_type is not None:
            result.type = out_type
        else:
            a_dtype = getattr(getattr(a_val, "type", None), "dtype", None)
            if a_dtype is not None:
                result.type = ir.TensorType(a_dtype)

        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jnp.dot not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                b: ArrayLike,
                *,
                precision: Any = None,
                preferred_element_type: Any = None,
                out_sharding: Any = None,
            ) -> ArrayLike:
                a_arr = jnp.asarray(a)
                b_arr = jnp.asarray(b)

                if a_arr.ndim > 2 or b_arr.ndim > 2:
                    return orig(
                        a_arr,
                        b_arr,
                        precision=precision,
                        preferred_element_type=preferred_element_type,
                        out_sharding=out_sharding,
                    )

                params: dict[str, Any] = {}
                if precision is not None:
                    params["precision"] = precision
                if preferred_element_type is not None:
                    params["preferred_element_type"] = preferred_element_type
                if out_sharding is not None:
                    try:
                        hash(out_sharding)
                    except TypeError:
                        out_sharding = None
                    if out_sharding is not None:
                        params["out_sharding"] = out_sharding

                return cls._PRIM.bind(a_arr, b_arr, **params)

            return _patched

        return [
            AssignSpec(
                "jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpDotPlugin._PRIM.def_impl
def _dot_impl(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
    out_sharding: Any = None,
) -> ArrayLike:
    orig = get_orig_impl(JnpDotPlugin._PRIM, JnpDotPlugin._FUNC_NAME)
    return orig(
        a,
        b,
        precision=precision,
        preferred_element_type=preferred_element_type,
        out_sharding=out_sharding,
    )


register_jvp_via_jax_jvp(JnpDotPlugin._PRIM, _dot_impl)

JnpDotPlugin._PRIM.def_abstract_eval(JnpDotPlugin.abstract_eval)


def _dot_batch_rule(
    args: tuple[Any, ...],
    dims: tuple[Any, ...],
    **params: Any,
) -> Any:
    return broadcast_batcher_compat(JnpDotPlugin._PRIM, args, dims, **params)


batching.primitive_batchers[JnpDotPlugin._PRIM] = _dot_batch_rule
