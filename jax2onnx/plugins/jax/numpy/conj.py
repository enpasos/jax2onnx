# jax2onnx/plugins/jax/numpy/conj.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

from jax import core
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import batching
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.plugins._complex_utils import (
    COMPLEX_DTYPES,
    conjugate_packed_tensor,
    ensure_packed_real_pair,
    is_packed_complex_tensor,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_CONJ_PRIM: Final = make_jnp_primitive("jax.numpy.conj")


@register_primitive(
    jaxpr_primitive=_CONJ_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.conj.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.10.1",
    context="primitives.jnp",
    component="conj",
    testcases=[
        {
            "testcase": "jnp_conj_real",
            "callable": lambda x: jnp.conj(x),
            "input_shapes": [(4,)],
            "post_check_onnx_graph": EG(["Identity:4"], no_unused_inputs=True),
        },
        {
            "testcase": "jnp_conj_complex64",
            "callable": lambda x: jnp.conj(x),
            "input_values": [np.array([1.0 + 2.0j, -0.75 + 0.5j], dtype=np.complex64)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Neg", "counts": {"Neg": 1}},
                    {"path": "Concat", "counts": {"Concat": 1}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "conj_vmap_batching",
            "callable": lambda x: jax.vmap(jnp.conj)(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpConjPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _CONJ_PRIM
    _FUNC_NAME: ClassVar[str] = "conj"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("conj_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("conj_out"))

        def _is_complex_var(var) -> bool:
            aval_dtype = getattr(getattr(var, "aval", None), "dtype", None)
            if aval_dtype is None:
                return False
            try:
                return np.issubdtype(np.dtype(aval_dtype), np.complexfloating)
            except TypeError:
                return False

        dtype = getattr(x_val, "dtype", None)
        complex_hint = (
            dtype in COMPLEX_DTYPES
            or is_packed_complex_tensor(x_val)
            or _is_complex_var(x_var)
        )

        if complex_hint:
            packed, base_dtype = ensure_packed_real_pair(ctx, x_val, name_hint="conj")
            target_dtype = packed.dtype or base_dtype
            output_name = getattr(out_spec, "name", None) or ctx.fresh_name("conj_out")
            conj_val = conjugate_packed_tensor(
                ctx,
                packed,
                target_dtype,
                prefix="conj",
                output_name=output_name,
            )
            out_spec.type = ir.TensorType(target_dtype)
            out_spec.dtype = target_dtype
            if getattr(conj_val, "shape", None) is not None:
                out_spec.shape = conj_val.shape
            _ensure_value_metadata(ctx, conj_val)
            ctx.bind_value_for_var(out_var, conj_val)
            return

        identity_val = ctx.builder.Identity(
            x_val,
            _outputs=[getattr(out_spec, "name", None) or ctx.fresh_name("conj_out")],
        )
        identity_val.type = getattr(out_spec, "type", None) or getattr(
            x_val, "type", None
        )
        if identity_val.type is None and dtype is not None:
            identity_val.type = ir.TensorType(dtype)
        output_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(identity_val, output_shape)
        _ensure_value_metadata(ctx, identity_val)
        ctx.bind_value_for_var(out_var, identity_val)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jnp.conj not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(x: ArrayLike) -> ArrayLike:
                return cls._PRIM.bind(x)

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


@JnpConjPlugin._PRIM.def_impl
def _conj_impl(x: ArrayLike) -> ArrayLike:
    orig = get_orig_impl(JnpConjPlugin._PRIM, JnpConjPlugin._FUNC_NAME)
    return orig(x)


JnpConjPlugin._PRIM.def_abstract_eval(
    lambda x: core.ShapedArray(getattr(x, "shape", ()), getattr(x, "dtype", None))
)


def _conj_batch_rule(args, dims, **params):
    (x,), (bdim,) = args, dims
    out = JnpConjPlugin._PRIM.bind(x, **params)
    return out, bdim


batching.primitive_batchers[JnpConjPlugin._PRIM] = _conj_batch_rule
