# jax2onnx/plugins/jax/random/normal.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax.extend.core import Primitive
from numpy.typing import ArrayLike

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_RANDOM_NORMAL_PRIM: Final[Primitive] = Primitive("jax.random.normal")
_RANDOM_NORMAL_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_RANDOM_NORMAL_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.random.normal.html",
    onnx=[
        {
            "component": "RandomNormal",
            "doc": "https://onnx.ai/onnx/operators/onnx__RandomNormal.html",
        }
    ],
    since="0.12.1",
    context="primitives.random",
    component="random_normal",
    testcases=[
        {
            "testcase": "random_normal_f32_2x3",
            "callable": lambda: jax.random.normal(
                jax.random.PRNGKey(0), (2, 3), dtype=jnp.float32
            ),
            "input_shapes": [],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.dtype(np.float32)],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                ["RandomNormal:2x3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class RandomNormalPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.random.normal`` to ONNX ``RandomNormal``."""

    _PRIM: ClassVar[Primitive] = _RANDOM_NORMAL_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        key: jax.core.AbstractValue,
        *,
        shape: tuple[int, ...] = (),
        dtype: np.dtype = np.dtype(np.float32),
    ) -> jax.core.ShapedArray:
        del key
        out_shape = tuple(int(dim) for dim in shape)
        return jax.core.ShapedArray(out_shape, np.dtype(dtype))

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        (key_var,) = eqn.invars
        (out_var,) = eqn.outvars

        shape_param = tuple(int(dim) for dim in eqn.params.get("shape", ()))
        out_dtype = np.dtype(
            getattr(getattr(out_var, "aval", None), "dtype", np.float32)
        )
        dtype_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)

        # Materialize key so RNG plumbing remains explicit in the graph build.
        ctx.get_value_for_var(key_var, name_hint=ctx.fresh_name("random_normal_key"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("random_normal_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "random_normal_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("random_normal_out")

        result = ctx.builder.RandomNormal(
            _outputs=[desired_name],
            dtype=int(dtype_enum.value),
            shape=shape_param,
            seed=0.0,
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = ir.TensorType(dtype_enum)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            result.shape = ir.Shape(shape_param)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = "__orig_impl__random_normal"

        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.random.normal not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                key: ArrayLike,
                shape: tuple[int, ...] | list[int] = (),
                dtype: np.dtype | type = jnp.float32,
            ) -> ArrayLike:
                shape_tuple = tuple(int(dim) for dim in shape)
                return cls._PRIM.bind(
                    key,
                    shape=shape_tuple,
                    dtype=np.dtype(dtype),
                )

            return _patched

        return [
            AssignSpec(
                "jax.random",
                "normal_p",
                cls._PRIM,
                delete_if_missing=True,
            ),
            MonkeyPatchSpec(
                target="jax.random",
                attr="normal",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@RandomNormalPlugin._PRIM.def_impl
def _random_normal_impl(
    key: ArrayLike,
    *,
    shape: tuple[int, ...] = (),
    dtype: np.dtype = np.dtype(np.float32),
) -> ArrayLike:
    orig = getattr(
        RandomNormalPlugin._PRIM,
        "__orig_impl__random_normal",
        jax.random.normal,
    )
    return orig(key, shape=shape, dtype=dtype)
