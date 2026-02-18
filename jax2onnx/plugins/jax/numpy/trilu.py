# jax2onnx/plugins/jax/numpy/trilu.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_TRILU_PRIM: Final = make_jnp_primitive("jax.numpy.trilu")
_ORIG_TRIU: Final = jnp.triu
_ORIG_TRIL: Final = jnp.tril
_ORIG_TRIU_SLOT: Final = "__orig_impl__triu"
_ORIG_TRIL_SLOT: Final = "__orig_impl__tril"


def _normalize_k(k: int | np.integer | ArrayLike) -> int:
    arr = np.asarray(k)
    if arr.ndim != 0:
        raise ValueError("jnp.triu/tril expects a scalar k offset")
    return int(arr.item())


@register_primitive(
    jaxpr_primitive=_TRILU_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.triu.html",
    onnx=[
        {
            "component": "Trilu",
            "doc": "https://onnx.ai/onnx/operators/onnx__Trilu.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="trilu",
    testcases=[
        {
            "testcase": "jnp_triu_basic",
            "callable": lambda x: jnp.triu(x),
            "input_shapes": [(4, 4)],
            "post_check_onnx_graph": EG(
                ["Trilu:4x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_tril_negative_k",
            "callable": lambda x: jnp.tril(x, k=-1),
            "input_shapes": [(3, 5)],
            "post_check_onnx_graph": EG(
                ["Trilu:3x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_triu_symbolic_batch",
            "callable": lambda x: jnp.triu(x, k=1),
            "input_shapes": [("B", 3, 3)],
            "post_check_onnx_graph": EG(
                ["Trilu:Bx3x3"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_trilu_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.tril(y, k=1))(x),
            "input_shapes": [(2, 4, 4)],
        },
    ],
)
class JnpTriluPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.numpy.triu``/``jax.numpy.tril`` to ONNX ``Trilu``."""

    _PRIM: ClassVar = _TRILU_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        k: int = 0,
        upper: bool = True,
    ) -> core.ShapedArray:
        del k, upper
        if len(x.shape) < 2:
            raise ValueError("jnp.triu/tril requires inputs with rank >= 2")
        return core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        k = _normalize_k(params.get("k", 0))
        upper = bool(params.get("upper", True))

        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_shape = tuple(getattr(x_var.aval, "shape", ()))
        if len(x_shape) < 2:
            raise ValueError("jnp.triu/tril lowering requires inputs with rank >= 2")

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("trilu_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("trilu_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("trilu_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("trilu_out")

        k_val = _const_i64(ctx, np.asarray(k, dtype=np.int64), "trilu_k")
        result = ctx.builder.Trilu(
            x_val,
            k_val,
            _outputs=[desired_name],
            upper=int(upper),
        )

        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif getattr(x_val, "type", None) is not None:
            result.type = x_val.type

        out_shape = tuple(getattr(out_var.aval, "shape", x_shape))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_triu(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.triu not found")
            setattr(cls._PRIM, _ORIG_TRIU_SLOT, orig)

            def _patched(m: ArrayLike, k: int = 0) -> jax.Array:
                return cls._PRIM.bind(jnp.asarray(m), k=_normalize_k(k), upper=True)

            return _patched

        def _make_tril(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.tril not found")
            setattr(cls._PRIM, _ORIG_TRIL_SLOT, orig)

            def _patched(m: ArrayLike, k: int = 0) -> jax.Array:
                return cls._PRIM.bind(jnp.asarray(m), k=_normalize_k(k), upper=False)

            return _patched

        return [
            AssignSpec("jax.numpy", "triu_p", cls._PRIM, delete_if_missing=True),
            AssignSpec("jax.numpy", "tril_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr="triu",
                make_value=_make_triu,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr="tril",
                make_value=_make_tril,
                delete_if_missing=False,
            ),
        ]


@JnpTriluPlugin._PRIM.def_impl
def _trilu_impl(m: ArrayLike, *, k: int = 0, upper: bool = True) -> jax.Array:
    orig = getattr(
        JnpTriluPlugin._PRIM,
        _ORIG_TRIU_SLOT if upper else _ORIG_TRIL_SLOT,
        None,
    )
    if orig is None:
        orig = _ORIG_TRIU if upper else _ORIG_TRIL
    return orig(m, k=k)


JnpTriluPlugin._PRIM.def_abstract_eval(JnpTriluPlugin.abstract_eval)


def _trilu_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[object, ...],
    *,
    k: int = 0,
    upper: bool = True,
) -> tuple[jax.Array, object]:
    (operand,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = JnpTriluPlugin._PRIM.bind(operand, k=k, upper=upper)
        return out, batching.not_mapped

    batch_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, batch_size)
    out = JnpTriluPlugin._PRIM.bind(operand, k=k, upper=upper)
    return out, 0


batching.primitive_batchers[JnpTriluPlugin._PRIM] = _trilu_batch_rule
