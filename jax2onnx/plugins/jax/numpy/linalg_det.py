# jax2onnx/plugins/jax/numpy/linalg_det.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_LINALG_DET_PRIM: Final = make_jnp_primitive("jax.numpy.linalg.det")


@register_primitive(
    jaxpr_primitive=_LINALG_DET_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.det.html",
    onnx=[
        {
            "component": "Det",
            "doc": "https://onnx.ai/onnx/operators/onnx__Det.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_det",
    testcases=[
        {
            "testcase": "linalg_det_2x2",
            "callable": lambda x: jnp.linalg.det(x),
            "input_values": [np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
            "expected_output_shapes": [()],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Det"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linalg_det_batched",
            "callable": lambda x: jnp.linalg.det(x),
            "input_values": [
                np.asarray(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[2.0, 0.0], [0.0, 5.0]],
                    ],
                    dtype=np.float32,
                )
            ],
            "expected_output_shapes": [(2,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Det:2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpLinalgDetPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _LINALG_DET_PRIM
    _FUNC_NAME: ClassVar[str] = "det"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue) -> core.ShapedArray:
        storage_slot = f"__orig_impl__{JnpLinalgDetPlugin._FUNC_NAME}"
        orig = getattr(_LINALG_DET_PRIM, storage_slot, jnp.linalg.det)
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        result = jax.eval_shape(lambda arr: orig(arr), spec)
        return core.ShapedArray(result.shape, result.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("det_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("det_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("det_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("det_out")

        result = ctx.builder.Det(
            x_val,
            _outputs=[desired_name],
        )
        out_spec_type = getattr(out_spec, "type", None)
        if out_spec_type is not None:
            result.type = out_spec_type
        else:
            dtype = getattr(getattr(x_val, "type", None), "dtype", None)
            if dtype is not None:
                result.type = ir.TensorType(dtype)
        _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.linalg.det not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a: ArrayLike) -> jax.Array:
                rank = getattr(a, "ndim", None)
                shape = getattr(a, "shape", None)
                if not isinstance(rank, int) or rank < 2:
                    return orig(a)
                if (
                    shape is None
                    or len(shape) < 2
                    or not isinstance(shape[-1], (int, np.integer))
                    or not isinstance(shape[-2], (int, np.integer))
                    or int(shape[-1]) != int(shape[-2])
                ):
                    return orig(a)
                return cls._PRIM.bind(jnp.asarray(a))

            return _patched

        return [
            MonkeyPatchSpec(
                target="jax.numpy.linalg",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            )
        ]


@JnpLinalgDetPlugin._PRIM.def_impl
def _linalg_det_impl(x: ArrayLike) -> jax.Array:
    orig = get_orig_impl(JnpLinalgDetPlugin._PRIM, JnpLinalgDetPlugin._FUNC_NAME)
    return orig(x)


JnpLinalgDetPlugin._PRIM.def_abstract_eval(JnpLinalgDetPlugin.abstract_eval)
