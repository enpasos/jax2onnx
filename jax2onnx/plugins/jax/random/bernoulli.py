# jax2onnx/plugins/jax/random/bernoulli.py

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
from jax2onnx.plugins._axis0_utils import _np_dtype_for_enum
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_BERNOULLI_PRIM: Final[Primitive] = Primitive("jax.random.bernoulli")
_BERNOULLI_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_BERNOULLI_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.random.bernoulli.html",
    onnx=[
        {
            "component": "Bernoulli",
            "doc": "https://onnx.ai/onnx/operators/onnx__Bernoulli.html",
        }
    ],
    since="0.12.1",
    context="primitives.random",
    component="bernoulli",
    testcases=[
        {
            "testcase": "bernoulli_scalar_prob_shape",
            "callable": lambda p: jax.random.bernoulli(
                jax.random.PRNGKey(0), p, shape=(2, 3)
            ),
            "input_values": [np.asarray(0.0, dtype=np.float32)],
            "expected_output_shapes": [(2, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Bernoulli:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "bernoulli_tensor_prob",
            "callable": lambda p: jax.random.bernoulli(jax.random.PRNGKey(0), p),
            "input_values": [
                np.asarray(
                    [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
            ],
            "expected_output_shapes": [(2, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Bernoulli:2x3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class RandomBernoulliPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.random.bernoulli`` to ONNX ``Bernoulli``."""

    _PRIM: ClassVar[Primitive] = _BERNOULLI_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        key: jax.core.AbstractValue,
        p: jax.core.AbstractValue,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> jax.core.ShapedArray:
        del key
        if shape is None:
            out_shape = tuple(getattr(p, "shape", ()))
        else:
            out_shape = tuple(int(d) for d in shape)
        return jax.core.ShapedArray(out_shape, np.dtype(np.bool_))

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        key_var, p_var = eqn.invars
        (out_var,) = eqn.outvars
        shape_param = eqn.params.get("shape", None)

        # Materialize key so upstream RNG-related nodes are kept alive.
        ctx.get_value_for_var(key_var, name_hint=ctx.fresh_name("bernoulli_key"))

        p_val = ctx.get_value_for_var(p_var, name_hint=ctx.fresh_name("bernoulli_p"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("bernoulli_out")
        )

        p_dtype = getattr(getattr(p_val, "type", None), "dtype", None)
        p_input = p_val
        if p_dtype not in {
            ir.DataType.FLOAT16,
            ir.DataType.BFLOAT16,
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
        }:
            p_input = ctx.builder.Cast(
                p_val,
                _outputs=[ctx.fresh_name("bernoulli_prob_f32")],
                to=int(_dtype_to_ir(np.dtype(np.float32), False).value),
            )
            if getattr(p_val, "shape", None) is not None:
                p_input.shape = p_val.shape
            p_input.type = ir.TensorType(ir.DataType.FLOAT)

        if shape_param is not None:
            target_shape = tuple(int(dim) for dim in shape_param)
            shape_const = ctx.bind_const_for_var(
                object(), np.asarray(target_shape, dtype=np.int64)
            )
            expanded = ctx.builder.Expand(
                p_input,
                shape_const,
                _outputs=[ctx.fresh_name("bernoulli_prob_expand")],
            )
            expanded.type = p_input.type
            expanded.shape = ir.Shape(target_shape)
            p_input = expanded

        # ORT Bernoulli currently produces inverted booleans relative to JAX.
        # Feed (1 - p) so external semantics match jax.random.bernoulli.
        p_dtype_enum = getattr(getattr(p_input, "type", None), "dtype", None)
        one_dtype = _np_dtype_for_enum(p_dtype_enum) or np.dtype(np.float32)
        one_const = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name("bernoulli_one"),
            value=np.asarray(1.0, dtype=one_dtype),
        )
        prob_for_onnx = ctx.builder.Sub(
            one_const,
            p_input,
            _outputs=[ctx.fresh_name("bernoulli_prob_onnx")],
        )
        prob_for_onnx.type = p_input.type
        if getattr(p_input, "shape", None) is not None:
            prob_for_onnx.shape = p_input.shape

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "bernoulli_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("bernoulli_out")

        result = ctx.builder.Bernoulli(
            prob_for_onnx,
            _outputs=[desired_name],
            dtype=int(ir.DataType.BOOL.value),
            seed=0.0,
        )

        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = ir.TensorType(ir.DataType.BOOL)

        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        elif getattr(prob_for_onnx, "shape", None) is not None:
            result.shape = prob_for_onnx.shape

        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = "__orig_impl__bernoulli"

        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.random.bernoulli not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                key: ArrayLike,
                p: ArrayLike = 0.5,
                shape: tuple[int, ...] | list[int] | None = None,
            ) -> ArrayLike:
                if shape is None:
                    shape_param: tuple[int, ...] | None = None
                else:
                    shape_param = tuple(int(dim) for dim in shape)
                return cls._PRIM.bind(key, jnp.asarray(p), shape=shape_param)

            return _patched

        return [
            AssignSpec(
                "jax.random",
                "bernoulli_p",
                cls._PRIM,
                delete_if_missing=True,
            ),
            MonkeyPatchSpec(
                target="jax.random",
                attr="bernoulli",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@RandomBernoulliPlugin._PRIM.def_impl
def _bernoulli_impl(
    key: ArrayLike,
    p: ArrayLike = 0.5,
    *,
    shape: tuple[int, ...] | None = None,
) -> ArrayLike:
    orig = getattr(
        RandomBernoulliPlugin._PRIM,
        "__orig_impl__bernoulli",
        jax.random.bernoulli,
    )
    return orig(key, p=p, shape=shape)
