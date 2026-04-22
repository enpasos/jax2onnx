# jax2onnx/plugins/jax/numpy/ones.py

from __future__ import annotations

from collections.abc import Sequence as _Seq
from typing import Any, Callable, ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import ir_dtype_to_numpy, numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_ONES_PRIM: Final = make_jnp_primitive("jax.numpy.ones")


def _normalize_shape(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, _Seq) and not isinstance(shape, (str, bytes)):
        dims = tuple(int(d) for d in shape)
    else:
        dims = (int(shape),)
    if any(d < 0 for d in dims):
        raise ValueError(f"negative dimensions are not allowed: {dims}")
    return dims


@register_primitive(
    jaxpr_primitive=_ONES_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ones.html",
    onnx=[
        {
            "component": "ConstantOfShape",
            "doc": "https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html",
        }
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="ones",
    testcases=[
        {
            "testcase": "jnp_ones_2x3",
            "callable": lambda: jnp.ones((2, 3), dtype=jnp.float32),
            "input_values": [],
            "post_check_onnx_graph": EG(["ConstantOfShape:2x3"], no_unused_inputs=True),
        },
        {
            "testcase": "jnp_ones_bool",
            "callable": lambda: jnp.ones((4,), dtype=jnp.bool_),
            "input_values": [],
            "post_check_onnx_graph": EG(["ConstantOfShape:4"], no_unused_inputs=True),
        },
    ],
)
class JnpOnesPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ONES_PRIM
    _FUNC_NAME: ClassVar[str] = "ones"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        *,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | type,
    ) -> core.ShapedArray:
        dims = _normalize_shape(shape)
        return core.ShapedArray(dims, np.dtype(dtype))

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (out_var,) = eqn.outvars
        params = dict(getattr(eqn, "params", {}) or {})

        shape = _normalize_shape(params.get("shape"))
        req_dtype = np.dtype(params.get("dtype", np.float32))
        target_enum = numpy_dtype_to_ir(req_dtype)

        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("ones_out"))
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", shape))

        shape_tensor = _const_i64(ctx, np.asarray(shape, dtype=np.int64), "ones_shape")
        _stamp_type_and_shape(shape_tensor, (len(shape),))
        _ensure_value_metadata(ctx, shape_tensor)

        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("ones_out")
        one_np_dtype = ir_dtype_to_numpy(target_enum)
        if one_np_dtype is None:
            one_np_dtype = np.dtype(np.float32)
        result = ctx.builder.ConstantOfShape(
            shape_tensor,
            value=ir.tensor(np.asarray([1], dtype=one_np_dtype)),
            _outputs=[out_name],
        )
        result.type = ir.TensorType(target_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.ones not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                shape: Any,
                dtype: np.dtype[Any] | type | None = None,
                *,
                device: Any | None = None,
            ) -> jax.Array:
                if device is not None:
                    return orig(shape, dtype=dtype, device=device)

                try:
                    norm_shape = _normalize_shape(shape)
                    resolved_dtype = np.dtype(orig((1,), dtype=dtype).dtype)
                except Exception:
                    return orig(shape, dtype=dtype)

                return cls._PRIM.bind(shape=norm_shape, dtype=resolved_dtype)

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


@JnpOnesPlugin._PRIM.def_impl
def _ones_impl(
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type,
) -> jax.Array:
    orig = get_orig_impl(JnpOnesPlugin._PRIM, JnpOnesPlugin._FUNC_NAME)
    return orig(shape, dtype=dtype)


JnpOnesPlugin._PRIM.def_abstract_eval(JnpOnesPlugin.abstract_eval)
